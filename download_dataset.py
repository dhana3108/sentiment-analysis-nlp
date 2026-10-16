"""
Dataset Acquisition Script for Sentiment Analysis
-------------------------------------------------
Downloads and prepares real-world Amazon Product Reviews for binary sentiment classification.
- Filters scores:
    * 4 and 5 stars -> 'positive'
    * 1 and 2 stars -> 'negative'
    * 3 stars -> dropped (neutral)
- Balances classes to prevent class imbalance issues.
- Cleans missing values and duplicate records.
- Saves the processed dataset to 'data/reviews.csv' with columns: ['review', 'sentiment'].
"""

import os
import pandas as pd
import numpy as np

DATA_URL = "https://raw.githubusercontent.com/shreya-maher/DSEM_Datasets/main/amazon_reviews.csv"
OUTPUT_FILE = os.path.join(os.path.dirname(__file__), "reviews.csv")


def acquire_dataset(sample_size_per_class: int = 5000, random_state: int = 42):
    """
    Downloads Amazon Product Reviews, filters into binary sentiment, balances classes,
    and saves to data/reviews.csv.
    """
    print("=" * 60)
    print("Downloading Amazon Product Reviews dataset from repository...")
    print(f"Source URL: {DATA_URL}")
    print("=" * 60)

    try:
        df_raw = pd.read_csv(DATA_URL, usecols=["Text", "Score"])
        print(f"Successfully downloaded raw dataset: {df_raw.shape[0]} reviews.")
    except Exception as e:
        print(f"Error downloading dataset: {e}")
        return False

    # Rename columns to standard names
    df_raw = df_raw.rename(columns={"Text": "review", "Score": "score"})

    # Drop missing values and duplicates
    df_raw = df_raw.dropna(subset=["review", "score"])
    df_raw = df_raw.drop_duplicates(subset=["review"])

    # Binary sentiment mapping:
    # 4 & 5 stars -> positive
    # 1 & 2 stars -> negative
    # 3 stars -> neutral (dropped for clean binary classification)
    positives = df_raw[df_raw["score"] >= 4].copy()
    positives["sentiment"] = "positive"

    negatives = df_raw[df_raw["score"] <= 2].copy()
    negatives["sentiment"] = "negative"

    print(f"Available Positive reviews (4-5 stars): {len(positives)}")
    print(f"Available Negative reviews (1-2 stars): {len(negatives)}")

    # Balance the dataset to avoid majority class bias
    n_samples = min(sample_size_per_class, len(positives), len(negatives))
    print(f"Sampling {n_samples} reviews per class (total {n_samples * 2} reviews)...")

    pos_sample = positives.sample(n=n_samples, random_state=random_state)
    neg_sample = negatives.sample(n=n_samples, random_state=random_state)

    # Combine and shuffle
    df_balanced = pd.concat([pos_sample, neg_sample], ignore_index=True)
    df_balanced = df_balanced.sample(frac=1.0, random_state=random_state).reset_index(drop=True)

    # Keep only the target columns: review and sentiment
    df_final = df_balanced[["review", "sentiment"]]

    # Save to CSV
    os.makedirs(os.path.dirname(OUTPUT_FILE), exist_ok=True)
    df_final.to_csv(OUTPUT_FILE, index=False, encoding="utf-8")

    print(f"\n[SUCCESS] Saved clean balanced dataset to: {OUTPUT_FILE}")
    print(f"Total Rows: {len(df_final)}")
    print("Sentiment Class Distribution:")
    print(df_final["sentiment"].value_counts())
    print("\nSample records:")
    print(df_final.head(3))
    return True


if __name__ == "__main__":
    acquire_dataset()
