

from collections import Counter
from pathlib import Path
import time

import requests
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from bs4 import BeautifulSoup

try:
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
except ImportError:
    SentimentIntensityAnalyzer = None


OUTPUT_DIR = Path("internship_output")
OUTPUT_DIR.mkdir(exist_ok=True)


# ============================================================
# TASK 1: WEB SCRAPING
# Collect book title, price, rating, and availability from a
# public practice website.
# ============================================================

def scrape_books(max_pages=2):
    base_url = "https://books.toscrape.com/catalogue/page-{}.html"
    rows = []
    rating_map = {
        "One": 1, "Two": 2, "Three": 3, "Four": 4, "Five": 5
    }

    headers = {"User-Agent": "EducationalDataAnalysisProject/1.0"}

    for page in range(1, max_pages + 1):
        url = base_url.format(page)
        print(f"Scraping page {page}: {url}")

        try:
            response = requests.get(url, headers=headers, timeout=15)
            response.raise_for_status()
        except requests.RequestException as error:
            print(f"Could not retrieve page {page}: {error}")
            continue

        soup = BeautifulSoup(response.text, "html.parser")
        books = soup.select("article.product_pod")

        for book in books:
            title_element = book.select_one("h3 a")
            price_element = book.select_one(".price_color")
            availability_element = book.select_one(".availability")
            rating_element = book.select_one("p.star-rating")

            if not title_element or not price_element:
                continue

            rating_word = ""
            rating_value = None
            if rating_element:
                classes = rating_element.get("class", [])
                rating_word = next(
                    (item for item in classes if item != "star-rating"), ""
                )
                rating_value = rating_map.get(rating_word)

            price_text = price_element.get_text(strip=True)
            price_number = (
                price_text.replace("£", "").replace("Â", "").strip()
            )

            rows.append({
                "title": title_element.get("title", title_element.get_text(strip=True)),
                "price_gbp": pd.to_numeric(price_number, errors="coerce"),
                "rating": rating_value,
                "rating_label": rating_word,
                "availability": (
                    availability_element.get_text(" ", strip=True)
                    if availability_element else "Unknown"
                ),
                "product_url": requests.compat.urljoin(
                    "https://books.toscrape.com/catalogue/", title_element.get("href", "")
                )
            })

        # A short pause is polite when making multiple requests.
        time.sleep(1)

    df = pd.DataFrame(rows)
    if df.empty:
        print("No data was scraped. Check your internet connection or the website structure.")
        return df

    output_path = OUTPUT_DIR / "scraped_books.csv"
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} records to {output_path.resolve()}")
    return df


# ============================================================
# TASK 2: EXPLORATORY DATA ANALYSIS (EDA)
# Understand the structure, summarize variables, and check
# missing values and unusual values.
# ============================================================

def perform_eda(df):
    if df.empty:
        print("EDA skipped because the dataset is empty.")
        return

    print("\n" + "=" * 55)
    print("TASK 2: EXPLORATORY DATA ANALYSIS")
    print("=" * 55)

    print("\nFirst five rows:")
    print(df.head())

    print("\nDataset shape (rows, columns):")
    print(df.shape)

    print("\nColumn names:")
    print(df.columns.tolist())

    print("\nData types and non-null counts:")
    df.info()

    print("\nMissing values in each column:")
    print(df.isnull().sum())

    print("\nNumerical summary:")
    print(df[["price_gbp", "rating"]].describe())

    print("\nNumber of unique book titles:")
    print(df["title"].nunique())

    print("\nRating counts:")
    print(df["rating"].value_counts(dropna=False).sort_index())

    print("\nAvailability counts:")
    print(df["availability"].value_counts())

    # Check duplicate titles and potentially unusual prices.
    print("\nDuplicate title rows:", df["title"].duplicated().sum())
    print("Rows with missing or non-positive prices:",
          int(df["price_gbp"].isna().sum() + (df["price_gbp"] <= 0).sum()))

    # Example meaningful questions:
    print("\nQuestions explored:")
    print("1. What is the average book price?")
    print("2. Which star rating appears most often?")
    print("3. How are book prices distributed?")
    print("4. Are any prices or ratings missing?")

    print(f"\nAverage price: £{df['price_gbp'].mean():.2f}")
    print(f"Median price: £{df['price_gbp'].median():.2f}")
    print(f"Most common rating: {df['rating'].mode().iloc[0] if not df['rating'].mode().empty else 'N/A'}")


# ============================================================
# TASK 3: DATA VISUALIZATION
# Create charts that make the results easier to understand.
# ============================================================

def create_visualizations(df):
    if df.empty:
        print("Visualization skipped because the dataset is empty.")
        return

    print("\n" + "=" * 55)
    print("TASK 3: DATA VISUALIZATION")
    print("=" * 55)

    sns.set_theme()

    # Chart 1: Distribution of prices.
    plt.figure(figsize=(8, 5))
    sns.histplot(df["price_gbp"].dropna(), bins=10, kde=True)
    plt.title("Distribution of Book Prices")
    plt.xlabel("Price (£)")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "price_distribution.png", dpi=150)
    plt.show()

    # Chart 2: Number of books for each rating.
    rating_counts = df["rating"].value_counts().sort_index()
    plt.figure(figsize=(8, 5))
    sns.barplot(x=rating_counts.index.astype(str), y=rating_counts.values)
    plt.title("Number of Books by Star Rating")
    plt.xlabel("Star Rating")
    plt.ylabel("Number of Books")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "books_by_rating.png", dpi=150)
    plt.show()

    # Chart 3: Price by rating.
    plot_data = df.dropna(subset=["rating", "price_gbp"])
    if not plot_data.empty:
        plt.figure(figsize=(8, 5))
        sns.boxplot(data=plot_data, x="rating", y="price_gbp")
        plt.title("Book Price by Star Rating")
        plt.xlabel("Star Rating")
        plt.ylabel("Price (£)")
        plt.tight_layout()
        plt.savefig(OUTPUT_DIR / "price_by_rating.png", dpi=150)
        plt.show()

    print(f"Charts saved in: {OUTPUT_DIR.resolve()}")


# ============================================================
# TASK 4: SENTIMENT ANALYSIS
# The practice book website does not provide review text in
# its product cards, so this task uses a small example dataset.
# Replace the sample sentences with permitted review text for
# a real analysis.
# ============================================================

def perform_sentiment_analysis():
    print("\n" + "=" * 55)
    print("TASK 4: SENTIMENT ANALYSIS")
    print("=" * 55)

    sample_reviews = [
        "The story was wonderful and the characters were amazing.",
        "I enjoyed this book and would recommend it to everyone.",
        "The book was boring and the writing was disappointing.",
        "The plot was confusing and I did not enjoy reading it.",
        "The book was okay, with some good and some weak parts.",
        "Excellent writing and a very engaging story.",
        "The ending was poor and the book felt too slow.",
        "A useful book with clear explanations."
    ]

    if SentimentIntensityAnalyzer is None:
        print("Install vaderSentiment to run this section:")
        print("pip install vaderSentiment")
        print("\nSample review text is included in the source code.")
        return

    analyzer = SentimentIntensityAnalyzer()
    results = []

    for review in sample_reviews:
        compound = analyzer.polarity_scores(review)["compound"]
        if compound >= 0.05:
            label = "Positive"
        elif compound <= -0.05:
            label = "Negative"
        else:
            label = "Neutral"

        results.append({
            "review": review,
            "compound_score": compound,
            "sentiment": label
        })

    sentiment_df = pd.DataFrame(results)
    sentiment_df.to_csv(OUTPUT_DIR / "sentiment_results.csv", index=False)

    print("\nSentiment results:")
    print(sentiment_df.to_string(index=False))

    counts = sentiment_df["sentiment"].value_counts()
    plt.figure(figsize=(7, 5))
    sns.barplot(x=counts.index, y=counts.values)
    plt.title("Sentiment of Sample Book Reviews")
    plt.xlabel("Sentiment")
    plt.ylabel("Number of Reviews")
    plt.tight_layout()
    plt.savefig(OUTPUT_DIR / "sentiment_counts.png", dpi=150)
    plt.show()

    print(f"\nSentiment results saved to: {(OUTPUT_DIR / 'sentiment_results.csv').resolve()}")


def main():
    print("INTERNSHIP ASSIGNMENT: WEB SCRAPING, EDA, VISUALIZATION & SENTIMENT")
    books_df = scrape_books(max_pages=2)

    if not books_df.empty:
        perform_eda(books_df)
        create_visualizations(books_df)

    perform_sentiment_analysis()

    print("\nProject completed. Check the 'internship_output' folder for CSV files and charts.")
    print("If scraping fails, you can still run the EDA and visualization functions")
    print("after loading a suitable CSV dataset into a pandas DataFrame.")


if __name__ == "__main__":
    main()
