"""
Model Training and Evaluation Pipeline
======================================
Sentiment Analysis of Product Reviews Using NLP and Machine Learning

This script performs the complete machine learning pipeline:
1. Loads and explores the product review dataset.
2. Cleans missing values, removes duplicates, and generates EDA distribution plots.
3. Preprocesses text reviews (negation-preserving cleaning).
4. Splits data into stratified train and test sets (80/20).
5. Fits a Scikit-learn TF-IDF Vectorizer strictly on training data (no data leakage).
6. Trains two classification models:
   - Logistic Regression
   - Multinomial Naive Bayes
7. Evaluates both models on held-out test data.
8. Compares metrics (Accuracy, Precision, Recall, F1-score, Confusion Matrix).
9. Selects and serializes the best model, vectorizer, and metadata via joblib.
10. Saves evaluation reports and figures to the reports/ directory.
"""

import os
import json
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for headless plot generation
import matplotlib.pyplot as plt
import seaborn as sns
import joblib

from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import MultinomialNB
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix,
)

from preprocess import clean_text

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_PATH = os.path.join(BASE_DIR, "data", "reviews.csv")
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

os.makedirs(MODELS_DIR, exist_ok=True)
os.makedirs(REPORTS_DIR, exist_ok=True)


def load_and_explore_data(filepath: str) -> pd.DataFrame:
    """
    Loads dataset from CSV and prints initial exploratory data analysis.
    If the file does not exist, triggers automated acquisition.
    """
    print("\n" + "=" * 70)
    print("STEP 1: DATASET LOADING AND EXPLORATORY DATA ANALYSIS (EDA)")
    print("=" * 70)

    if not os.path.exists(filepath):
        print(f"[!] Dataset not found at '{filepath}'. Running data acquisition...")
        from data.download_dataset import acquire_dataset
        success = acquire_dataset()
        if not success:
            raise FileNotFoundError(f"Failed to acquire dataset at {filepath}")

    df = pd.read_csv(filepath)
    print(f"Dataset successfully loaded from: {filepath}")
    print(f"Initial Shape: {df.shape[0]} rows, {df.shape[1]} columns")
    print(f"Columns: {list(df.columns)}")
    print(f"Data types:\n{df.dtypes}")

    # Standardize column names if needed
    col_map = {}
    for c in df.columns:
        if c.lower() in ["review", "text", "review_text", "reviews"]:
            col_map[c] = "review"
        elif c.lower() in ["sentiment", "label", "target", "score"]:
            col_map[c] = "sentiment"
    df = df.rename(columns=col_map)

    if "review" not in df.columns or "sentiment" not in df.columns:
        raise ValueError(
            f"Expected 'review' and 'sentiment' columns. Found: {list(df.columns)}"
        )

    # Missing value handling
    missing_count = df.isnull().sum().to_dict()
    print(f"Missing Values detected: {missing_count}")
    df = df.dropna(subset=["review", "sentiment"])

    # Duplicate handling
    duplicate_count = df.duplicated(subset=["review"]).sum()
    print(f"Duplicate reviews detected: {duplicate_count}")
    df = df.drop_duplicates(subset=["review"]).reset_index(drop=True)
    print(f"Cleaned dataset shape: {df.shape[0]} rows")

    # Map labels to consistent string format ('positive'/'negative')
    df["sentiment"] = df["sentiment"].astype(str).str.lower().str.strip()
    label_map = {
        "1": "positive",
        "0": "negative",
        "1.0": "positive",
        "0.0": "negative",
        "pos": "positive",
        "neg": "negative",
        "positive": "positive",
        "negative": "negative",
    }
    df["sentiment"] = df["sentiment"].map(label_map)
    df = df.dropna(subset=["sentiment"]).reset_index(drop=True)

    print("\nClass Distribution:")
    class_counts = df["sentiment"].value_counts()
    for sentiment, count in class_counts.items():
        pct = (count / len(df)) * 100
        print(f"  * {sentiment.capitalize()}: {count} ({pct:.2f}%)")

    # Generate EDA Visualizations
    generate_eda_plots(df)

    return df


def generate_eda_plots(df: pd.DataFrame):
    """
    Creates and saves exploratory data visualizations:
    - Sentiment class distribution bar chart
    - Review length (character and word counts) distributions
    """
    print("\nGenerating Exploratory Data Visualizations...")
    plot_df = df.copy()
    plot_df["char_length"] = plot_df["review"].apply(lambda x: len(str(x)))
    plot_df["word_count"] = plot_df["review"].apply(lambda x: len(str(x).split()))

    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    palette = {"positive": "#2ecc71", "negative": "#e74c3c"}

    # 1. Class Distribution Bar Chart
    sns.countplot(
        data=plot_df,
        x="sentiment",
        palette=palette,
        ax=axes[0],
        hue="sentiment",
        legend=False,
    )
    axes[0].set_title("Sentiment Class Distribution", fontsize=13, fontweight="bold")
    axes[0].set_xlabel("Sentiment", fontsize=11)
    axes[0].set_ylabel("Number of Reviews", fontsize=11)
    for p in axes[0].patches:
        height = p.get_height()
        axes[0].annotate(
            f"{int(height)}",
            (p.get_x() + p.get_width() / 2.0, height),
            ha="center",
            va="bottom",
            fontsize=10,
            fontweight="bold",
        )

    # 2. Word Count Distribution (Log scale or capped for clean viewing)
    sns.boxplot(
        data=plot_df,
        x="sentiment",
        y="word_count",
        palette=palette,
        ax=axes[1],
        hue="sentiment",
        legend=False,
        showfliers=False,
    )
    axes[1].set_title(
        "Review Word Count by Sentiment (Outliers Hidden)",
        fontsize=13,
        fontweight="bold",
    )
    axes[1].set_xlabel("Sentiment", fontsize=11)
    axes[1].set_ylabel("Word Count", fontsize=11)

    plt.tight_layout()
    chart_path = os.path.join(REPORTS_DIR, "sentiment_distribution.png")
    plt.savefig(chart_path, dpi=300)
    plt.close()
    print(f"[+] Saved EDA chart to: {chart_path}")


def prepare_features_and_split(df: pd.DataFrame):
    """
    Cleans review text using negation-preserving preprocessor,
    maps sentiment to binary integers (0 = negative, 1 = positive),
    and performs stratified train-test split (80% train, 20% test).
    """
    print("\n" + "=" * 70)
    print("STEP 2: TEXT PREPROCESSING & DATA SPLITTING")
    print("=" * 70)

    print("Applying text preprocessing pipeline (lowercasing, punctuation stripping, negation preservation)...")
    df["cleaned_review"] = df["review"].apply(clean_text)

    # Filter out empty cleaned reviews
    before_len = len(df)
    df = df[df["cleaned_review"].str.len() > 0].reset_index(drop=True)
    after_len = len(df)
    if before_len - after_len > 0:
        print(f"Dropped {before_len - after_len} empty reviews after preprocessing.")

    # Binary label encoding: negative = 0, positive = 1
    label_encoding = {"negative": 0, "positive": 1}
    df["label"] = df["sentiment"].map(label_encoding)

    X = df["cleaned_review"].values
    y = df["label"].values

    # Stratified Train-Test Split (80% training, 20% testing)
    X_train, X_test, y_train, y_test = train_test_split(
        X,
        y,
        test_size=0.20,
        random_state=42,
        stratify=y,
    )

    print(f"Training Samples: {len(X_train)} ({len(X_train)/len(X)*100:.1f}%)")
    print(f"Testing Samples:  {len(X_test)} ({len(X_test)/len(X)*100:.1f}%)")
    print(f"Train class balance: Neg={np.sum(y_train == 0)}, Pos={np.sum(y_train == 1)}")
    print(f"Test class balance:  Neg={np.sum(y_test == 0)}, Pos={np.sum(y_test == 1)}")

    return X_train, X_test, y_train, y_test


def extract_tfidf_features(X_train, X_test):
    """
    Extracts numerical TF-IDF features from text.
    CRITICAL: Fits ONLY on X_train to prevent data leakage!
    Uses n-gram range (1, 2) to capture critical bigrams like 'not good'.
    """
    print("\n" + "=" * 70)
    print("STEP 3: TF-IDF FEATURE EXTRACTION")
    print("=" * 70)

    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),  # unigrams and bigrams
        min_df=2,             # ignore terms appearing in fewer than 2 reviews
        sublinear_tf=True,    # apply sublinear scaling: 1 + log(tf)
    )

    print("Fitting TF-IDF Vectorizer strictly on X_train...")
    X_train_tfidf = vectorizer.fit_transform(X_train)
    print("Transforming X_test with fitted vectorizer (no data leakage)...")
    X_test_tfidf = vectorizer.transform(X_test)

    print(f"TF-IDF Feature Matrix (Train): {X_train_tfidf.shape}")
    print(f"TF-IDF Feature Matrix (Test):  {X_test_tfidf.shape}")
    print(f"Total Vocabulary Size: {len(vectorizer.vocabulary_)} n-grams")

    return vectorizer, X_train_tfidf, X_test_tfidf


def train_and_evaluate_models(X_train_tfidf, y_train, X_test_tfidf, y_test):
    """
    Trains and compares:
    1. Logistic Regression
    2. Multinomial Naive Bayes
    Calculates Accuracy, Precision, Recall, F1-Score, and Confusion Matrix.
    """
    print("\n" + "=" * 70)
    print("STEP 4: MACHINE LEARNING MODEL TRAINING & COMPARISON")
    print("=" * 70)

    models = {
        "Logistic Regression": LogisticRegression(
            C=1.0,
            max_iter=1000,
            random_state=42,
        ),
        "Multinomial Naive Bayes": MultinomialNB(
            alpha=1.0,
        ),
    }

    results = {}
    predictions = {}

    for name, model in models.items():
        print(f"\n---> Training {name}...")
        model.fit(X_train_tfidf, y_train)

        # Predictions
        y_pred = model.predict(X_test_tfidf)
        predictions[name] = y_pred

        # Metrics
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average="weighted")
        rec = recall_score(y_test, y_pred, average="weighted")
        f1 = f1_score(y_test, y_pred, average="weighted")
        cm = confusion_matrix(y_test, y_pred)
        report = classification_report(
            y_test,
            y_pred,
            target_names=["Negative", "Positive"],
            output_dict=True,
        )

        results[name] = {
            "model_object": model,
            "accuracy": float(acc),
            "precision": float(prec),
            "recall": float(rec),
            "f1_score": float(f1),
            "confusion_matrix": cm.tolist(),
            "classification_report": report,
        }

        print(f"Results for {name}:")
        print(f"  * Accuracy:  {acc * 100:.2f}%")
        print(f"  * Precision: {prec * 100:.2f}%")
        print(f"  * Recall:    {rec * 100:.2f}%")
        print(f"  * F1-Score:  {f1 * 100:.2f}%")

    # Generate Confusion Matrix Figure
    generate_confusion_matrix_plot(results, y_test)

    # Save detailed classification text report
    save_classification_text_report(results, y_test)

    return results


def generate_confusion_matrix_plot(results: dict, y_test: np.ndarray):
    """
    Generates and saves side-by-side Confusion Matrix heatmaps.
    """
    fig, axes = plt.subplots(1, 2, figsize=(13, 5))
    class_names = ["Negative", "Positive"]

    for idx, (name, metrics) in enumerate(results.items()):
        cm = np.array(metrics["confusion_matrix"])
        sns.heatmap(
            cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=class_names,
            yticklabels=class_names,
            cbar=False,
            ax=axes[idx],
            annot_kws={"size": 14, "weight": "bold"},
        )
        axes[idx].set_title(
            f"{name}\nAccuracy: {metrics['accuracy']*100:.2f}% | F1: {metrics['f1_score']*100:.2f}%",
            fontsize=12,
            fontweight="bold",
        )
        axes[idx].set_xlabel("Predicted Sentiment", fontsize=11)
        axes[idx].set_ylabel("Actual Sentiment", fontsize=11)

    plt.tight_layout()
    cm_path = os.path.join(REPORTS_DIR, "confusion_matrices.png")
    plt.savefig(cm_path, dpi=300)
    plt.close()
    print(f"\n[+] Saved Confusion Matrix figure to: {cm_path}")


def save_classification_text_report(results: dict, y_test: np.ndarray):
    """
    Saves a readable text report with classification metrics and comparative analysis.
    """
    report_path = os.path.join(REPORTS_DIR, "classification_report.txt")
    with open(report_path, "w", encoding="utf-8") as f:
        f.write("=" * 70 + "\n")
        f.write("MODEL EVALUATION & COMPARISON REPORT\n")
        f.write(f"Generated at: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write("=" * 70 + "\n\n")

        for name, metrics in results.items():
            f.write(f"MODEL: {name}\n")
            f.write("-" * 50 + "\n")
            f.write(f"Overall Accuracy:  {metrics['accuracy']:.4f} ({metrics['accuracy']*100:.2f}%)\n")
            f.write(f"Weighted Precision:{metrics['precision']:.4f}\n")
            f.write(f"Weighted Recall:   {metrics['recall']:.4f}\n")
            f.write(f"Weighted F1-Score: {metrics['f1_score']:.4f}\n\n")

            f.write("Per-Class Breakdown:\n")
            rep = metrics["classification_report"]
            for cls in ["Negative", "Positive"]:
                f.write(
                    f"  Class '{cls}': Precision={rep[cls]['precision']:.4f}, "
                    f"Recall={rep[cls]['recall']:.4f}, F1={rep[cls]['f1-score']:.4f}, "
                    f"Support={int(rep[cls]['support'])}\n"
                )

            cm = metrics["confusion_matrix"]
            f.write("\nConfusion Matrix (Rows: Actual [Neg, Pos], Cols: Predicted [Neg, Pos]):\n")
            f.write(f"  True Negative (TN):  {cm[0][0]}\n")
            f.write(f"  False Positive (FP): {cm[0][1]}\n")
            f.write(f"  False Negative (FN): {cm[1][0]}\n")
            f.write(f"  True Positive (TP):  {cm[1][1]}\n\n")
            f.write("=" * 70 + "\n\n")

    print(f"[+] Saved Classification Report to: {report_path}")


def select_and_save_best_model(results: dict, vectorizer: TfidfVectorizer):
    """
    Selects the best model based on F1-score (harmonic mean of precision and recall)
    and saves serialized pipeline artifacts to the models/ directory.
    """
    print("\n" + "=" * 70)
    print("STEP 5: MODEL SELECTION AND SERIALIZATION")
    print("=" * 70)

    # Print comparison table
    print("\nComparative Summary Table:")
    print("-" * 65)
    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10}")
    print("-" * 65)
    for name, metrics in results.items():
        print(
            f"{name:<28} | "
            f"{metrics['accuracy']*100:>8.2f}% | "
            f"{metrics['f1_score']*100:>8.2f}% | "
            f"{metrics['precision']*100:>8.2f}%"
        )
    print("-" * 65)

    # Model Selection Criterion: Highest weighted F1-Score
    best_model_name = max(results, key=lambda k: results[k]["f1_score"])
    best_model = results[best_model_name]["model_object"]
    best_metrics = results[best_model_name]

    print(f"\n[WINNER] Best Performing Model: '{best_model_name}'")
    print(f"Selection Criterion: Highest F1-Score ({best_metrics['f1_score']*100:.2f}%)")

    # Serialize Best Model and TF-IDF Vectorizer
    model_path = os.path.join(MODELS_DIR, "best_model.joblib")
    vectorizer_path = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
    metadata_path = os.path.join(MODELS_DIR, "model_metadata.joblib")

    joblib.dump(best_model, model_path)
    joblib.dump(vectorizer, vectorizer_path)

    # Save individual models too for exploration
    for name, metrics in results.items():
        clean_name = name.lower().replace(" ", "_")
        joblib.dump(metrics["model_object"], os.path.join(MODELS_DIR, f"{clean_name}.joblib"))

    # Save metadata dictionary
    metadata = {
        "best_model_name": best_model_name,
        "training_timestamp": datetime.now().isoformat(),
        "accuracy": best_metrics["accuracy"],
        "f1_score": best_metrics["f1_score"],
        "precision": best_metrics["precision"],
        "recall": best_metrics["recall"],
        "vocabulary_size": len(vectorizer.vocabulary_),
        "classes": ["negative", "positive"],
        "model_parameters": best_model.get_params(),
        "all_models_summary": {
            k: {
                "accuracy": v["accuracy"],
                "f1_score": v["f1_score"],
                "precision": v["precision"],
                "recall": v["recall"],
            }
            for k, v in results.items()
        },
    }
    joblib.dump(metadata, metadata_path)

    # Save machine-readable JSON evaluation metrics
    json_path = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "best_model": best_model_name,
                "timestamp": datetime.now().isoformat(),
                "models": {
                    k: {
                        "accuracy": v["accuracy"],
                        "f1_score": v["f1_score"],
                        "precision": v["precision"],
                        "recall": v["recall"],
                        "confusion_matrix": v["confusion_matrix"],
                        "classification_report": v["classification_report"],
                    }
                    for k, v in results.items()
                },
            },
            f,
            indent=4,
        )

    print(f"[+] Saved Best Model to:       {model_path}")
    print(f"[+] Saved TF-IDF Vectorizer to: {vectorizer_path}")
    print(f"[+] Saved Model Metadata to:   {metadata_path}")
    print(f"[+] Saved JSON Metrics to:     {json_path}")
    print("\nTraining and evaluation pipeline completed successfully!")


def main():
    # 1. Load and explore data
    df = load_and_explore_data(DATA_PATH)

    # 2. Text preprocessing & stratified train-test split
    X_train, X_test, y_train, y_test = prepare_features_and_split(df)

    # 3. TF-IDF feature extraction (fit strictly on train)
    vectorizer, X_train_tfidf, X_test_tfidf = extract_tfidf_features(X_train, X_test)

    # 4. Train and evaluate models
    results = train_and_evaluate_models(X_train_tfidf, y_train, X_test_tfidf, y_test)

    # 5. Select and save best model
    select_and_save_best_model(results, vectorizer)


if __name__ == "__main__":
    main()
