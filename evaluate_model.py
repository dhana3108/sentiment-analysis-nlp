"""
Model Evaluation and Metrics Interpretation Module
==================================================
Sentiment Analysis of Product Reviews Using NLP and Machine Learning

This script:
1. Loads the trained models and saved test evaluation reports.
2. Displays comprehensive performance metrics: Accuracy, Precision, Recall, F1-Score.
3. Prints per-class classification reports and confusion matrices.
4. Compares Logistic Regression vs. Multinomial Naive Bayes.
5. Explains every evaluation metric in simple, beginner-friendly terms.
"""

import os
import json
import joblib
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")
METRICS_JSON = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
REPORT_TXT = os.path.join(REPORTS_DIR, "classification_report.txt")


def explain_metrics():
    """
    Prints a clear, educational, beginner-friendly explanation
    of standard NLP evaluation metrics.
    """
    print("\n" + "=" * 75)
    print("BEGINNER'S GUIDE: UNDERSTANDING NLP EVALUATION METRICS")
    print("=" * 75)
    print("""
1. ACCURACY:
   - What it means: The percentage of total reviews correctly classified.
   - Formula: (True Positives + True Negatives) / (Total Predictions)
   - When to use: Great when positive and negative classes are balanced (50/50).
   - Limitation: Misleading if the dataset is heavily skewed (e.g., 95% positive).

2. PRECISION (Positive Predictive Value):
   - What it means: When the model predicts "Positive", how often is it right?
   - Formula: True Positives / (True Positives + False Positives)
   - Focus: Minimizing False Positives (preventing bad reviews from being marked good).

3. RECALL (Sensitivity / True Positive Rate):
   - What it means: Out of all actually Positive reviews, how many did the model find?
   - Formula: True Positives / (True Positives + False Negatives)
   - Focus: Minimizing False Negatives (avoiding missing good reviews).

4. F1-SCORE:
   - What it means: The harmonic mean of Precision and Recall.
   - Formula: 2 * (Precision * Recall) / (Precision + Recall)
   - Why it matters: Unlike simple average, the harmonic mean penalizes extreme imbalances.
     It is the gold-standard single metric for classification tasks.

5. CONFUSION MATRIX:
   A 2x2 grid comparing Ground Truth against Model Predictions:
   --------------------------------------------------------------
   |                 | Predicted Negative   | Predicted Positive |
   | Actual Negative | True Negative (TN)   | False Positive (FP)|
   | Actual Positive | False Negative (FN)  | True Positive (TP) |
   --------------------------------------------------------------
   - True Negative (TN): Negative review correctly identified as Negative.
   - False Positive (FP): Negative review mistakenly predicted as Positive.
   - False Negative (FN): Positive review mistakenly predicted as Negative.
   - True Positive (TP): Positive review correctly identified as Positive.
""")
    print("=" * 75)


def display_model_comparison():
    """
    Loads saved evaluation metrics and prints a clean comparison table.
    """
    if not os.path.exists(METRICS_JSON):
        print(f"[!] Evaluation metrics not found at '{METRICS_JSON}'.")
        print("Please run 'python train_model.py' first to train and evaluate the models.")
        return

    with open(METRICS_JSON, "r", encoding="utf-8") as f:
        data = json.load(f)

    models_data = data.get("models", {})
    best_model = data.get("best_model", "Unknown")

    print("\n" + "=" * 75)
    print("ACTUAL TEST-SET PERFORMANCE COMPARISON")
    print("=" * 75)
    print(f"{'Model Name':<28} | {'Accuracy':<10} | {'F1-Score':<10} | {'Precision':<10} | {'Recall':<10}")
    print("-" * 75)

    for name, m in models_data.items():
        print(
            f"{name:<28} | "
            f"{m['accuracy']*100:>8.2f}% | "
            f"{m['f1_score']*100:>8.2f}% | "
            f"{m['precision']*100:>8.2f}% | "
            f"{m['recall']*100:>8.2f}%"
        )
    print("-" * 75)

    print(f"\n[WINNER]: '{best_model}' achieved the top overall F1-Score.")

    # Print Confusion Matrix Breakdown
    print("\n" + "=" * 75)
    print("CONFUSION MATRIX BREAKDOWN (TEST SET: 2,000 SAMPLES)")
    print("=" * 75)
    for name, m in models_data.items():
        cm = m["confusion_matrix"]
        tn, fp = cm[0][0], cm[0][1]
        fn, tp = cm[1][0], cm[1][1]
        print(f"\n--- {name} ---")
        print(f"  True Negatives  (TN): {tn:>4} (Correctly predicted Negative)")
        print(f"  False Positives (FP): {fp:>4} (Negative review misclassified as Positive)")
        print(f"  False Negatives (FN): {fn:>4} (Positive review misclassified as Negative)")
        print(f"  True Positives  (TP): {tp:>4} (Correctly predicted Positive)")
        print(f"  Total Correct:        {tn + tp:>4} / {tn + fp + fn + tp} ({((tn+tp)/(tn+fp+fn+tp))*100:.2f}%)")

    # Print the full classification report text if available
    if os.path.exists(REPORT_TXT):
        print("\n" + "=" * 75)
        print("DETAILED CLASSIFICATION REPORT")
        print("=" * 75)
        with open(REPORT_TXT, "r", encoding="utf-8") as f:
            print(f.read())


def main():
    explain_metrics()
    display_model_comparison()


if __name__ == "__main__":
    main()
