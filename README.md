# Data Directory

This directory stores the datasets used for training and evaluating the sentiment analysis model.

## Dataset Details
- **Dataset Name:** Amazon Product Reviews Sentiment Dataset / IMDb Sentiment Dataset
- **Format:** CSV (`reviews.csv`)
- **Required Columns:**
  - `review`: Text content of the customer or product review.
  - `sentiment`: Target label indicating the review sentiment. Binary classification format:
    - `positive` (or `1`)
    - `negative` (or `0`)

## Dataset Acquisition
You can download or acquire a real-world product reviews dataset:
1. **Amazon Product Reviews (Cell Phones, Electronics, or Kitchen):** Available on Kaggle or Stanford SNAP.
2. **IMDb Movie Reviews:** A classic benchmark binary sentiment dataset with 50,000 reviews evenly split between positive and negative sentiment.

## Sample Dataset
For quick local verification, testing, and immediate college demonstration without requiring an external 100MB+ download, `data/reviews.csv` contains high-quality real product reviews covering multiple product categories (Electronics, Audio, Computers, Accessories, Household) with balanced positive and negative classes.

You can also drop any custom product review CSV here as long as it has `review` and `sentiment` columns!
