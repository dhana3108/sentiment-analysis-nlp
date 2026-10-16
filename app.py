"""
Sentiment Analysis Web Application
==================================
College NLP & Machine Learning Project:
"Sentiment Analysis of Product Reviews Using Natural Language Processing and Machine Learning"

Built with Streamlit, Scikit-learn, and NLTK.
"""

import os
import json
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Import custom text preprocessing pipeline
from preprocess import clean_text

# Paths configuration
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR = os.path.join(BASE_DIR, "models")
REPORTS_DIR = os.path.join(BASE_DIR, "reports")

MODEL_PATH = os.path.join(MODELS_DIR, "best_model.joblib")
VECTORIZER_PATH = os.path.join(MODELS_DIR, "tfidf_vectorizer.joblib")
METADATA_PATH = os.path.join(MODELS_DIR, "model_metadata.joblib")
METRICS_JSON = os.path.join(REPORTS_DIR, "evaluation_metrics.json")
CONFUSION_IMG = os.path.join(REPORTS_DIR, "confusion_matrices.png")
EDA_IMG = os.path.join(REPORTS_DIR, "sentiment_distribution.png")

# Page configuration
st.set_page_config(
    page_title="Sentiment Analysis | NLP Project",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)


@st.cache_resource
def load_artifacts():
    """
    Safely loads the trained model, TF-IDF vectorizer, and metadata.
    Cached so it runs only once when the server starts.
    """
    if not (os.path.exists(MODEL_PATH) and os.path.exists(VECTORIZER_PATH)):
        return None, None, None

    model = joblib.load(MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    metadata = joblib.load(METADATA_PATH) if os.path.exists(METADATA_PATH) else {}
    return model, vectorizer, metadata


@st.cache_data
def load_evaluation_metrics():
    """Loads pre-calculated evaluation metrics from JSON report."""
    if os.path.exists(METRICS_JSON):
        with open(METRICS_JSON, "r", encoding="utf-8") as f:
            return json.load(f)
    return None


# Sidebar Header
st.sidebar.markdown("## 🛍️ NLP Sentiment AI")
st.sidebar.caption("Product Reviews Classification System")
page = st.sidebar.radio(
    "Navigation Menu",
    [
        "🔍 Live Sentiment Predictor",
        "📊 Model Evaluation & Metrics",
        "🏠 Project Overview & Pipeline",
        "⚠️ Limitations & Future Work",
    ],
)

st.sidebar.markdown("---")
st.sidebar.subheader("Project Info")
st.sidebar.markdown(
    """
    - **Domain:** Natural Language Processing (NLP)
    - **Models:** Multinomial Naive Bayes & Logistic Regression
    - **Features:** Scikit-learn TF-IDF (Unigrams & Bigrams)
    - **Dataset:** Amazon Product Reviews (Binary Balanced)
    """
)

# Load pipeline artifacts
model, vectorizer, metadata = load_artifacts()

# ==============================================================================
# PAGE 1: LIVE SENTIMENT PREDICTOR
# ==============================================================================
if page == "🔍 Live Sentiment Predictor":
    st.title("🛍️ Product Review Sentiment Predictor")
    st.markdown(
        "Enter any customer or product review below. The NLP pipeline cleans the text, extracts "
        "TF-IDF features, and predicts whether the review is **Positive** or **Negative**."
    )

    if model is None or vectorizer is None:
        st.error(
            "⚠️ Trained model artifacts not found! Please run the training script first:\n\n"
            "```powershell\npython train_model.py\n```"
        )
        st.stop()

    # Pre-defined sample reviews for quick demonstration
    sample_options = {
        "Select a sample review...": "",
        "Positive: Audio Quality (Headphones)": (
            "These noise-cancelling headphones are genuinely incredible! The sound stage is deep, "
            "battery lasts over 30 hours, and phone calls are crystal clear. Definitely worth every penny!"
        ),
        "Negative: Defective Product (Charger)": (
            "Worst purchase ever. The cable stopped working after just three days and heated up dangerously. "
            "Customer support completely ignored my refund request. Do not buy!"
        ),
        "Subtle / Negation Test (Smartphone)": (
            "The screen is bright, but the battery is not good at all and the camera never focuses in low light. "
            "I would not recommend this phone to anyone."
        ),
        "Positive: Kitchen Appliance (Coffee Maker)": (
            "Makes delicious, piping hot espresso in under a minute! Compact design fits neatly on the kitchen counter "
            "and cleaning it is totally hassle-free. Five stars!"
        ),
        "Negative: Poor Quality (Keyboard)": (
            "Keys started sticking after a week. Plastic feels cheap and creaky. Terribly disappointed with the build quality."
        ),
    }

    selected_sample = st.selectbox("💡 Quick Fill: Try a Sample Review", list(sample_options.keys()))

    # Initialize review text in session state
    if "review_input" not in st.session_state:
        st.session_state["review_input"] = ""

    if selected_sample != "Select a sample review...":
        st.session_state["review_input"] = sample_options[selected_sample]

    review_text = st.text_area(
        "Enter Product Review:",
        value=st.session_state["review_input"],
        height=140,
        placeholder="Type or paste your product review here (e.g., 'The laptop battery life is exceptional and lightweight!')...",
    )

    col1, col2 = st.columns([1, 4])
    with col1:
        predict_btn = st.button("🚀 Predict Sentiment", type="primary", use_container_width=True)
    with col2:
        clear_btn = st.button("🗑️ Clear", use_container_width=False)
        if clear_btn:
            st.session_state["review_input"] = ""
            st.rerun()

    if predict_btn:
        if not review_text or not review_text.strip():
            st.warning("⚠️ Please enter a review text before predicting.")
        else:
            with st.spinner("Analyzing text with NLP pipeline..."):
                # 1. Preprocessing
                cleaned = clean_text(review_text)

                if not cleaned:
                    st.warning("The input contained only stopwords or symbols. Please provide more text.")
                else:
                    # 2. Vectorization
                    tfidf_vec = vectorizer.transform([cleaned])

                    # 3. Model Prediction
                    pred_class = model.predict(tfidf_vec)[0]
                    # Binary mapping: 0 = negative, 1 = positive
                    sentiment = "Positive" if pred_class == 1 else "Negative"

                    # 4. Probabilities / Confidence
                    proba = None
                    if hasattr(model, "predict_proba"):
                        proba = model.predict_proba(tfidf_vec)[0]
                        conf_neg = proba[0] * 100
                        conf_pos = proba[1] * 100
                        confidence = conf_pos if pred_class == 1 else conf_neg
                    else:
                        confidence = None

                    # Display Prediction Result Card
                    st.markdown("---")
                    res_col1, res_col2 = st.columns([2, 3])

                    with res_col1:
                        if sentiment == "Positive":
                            st.success(f"### 🎉 Predicted Sentiment: **POSITIVE**")
                        else:
                            st.error(f"### ❌ Predicted Sentiment: **NEGATIVE**")

                        if confidence is not None:
                            st.metric(
                                label="Model Confidence",
                                value=f"{confidence:.1f}%",
                                delta="High Confidence" if confidence > 75 else "Moderate Confidence",
                            )

                    with res_col2:
                        if proba is not None:
                            st.write("**Class Probability Distribution:**")
                            st.write(f"🟢 **Positive:** `{conf_pos:.2f}%`")
                            st.progress(float(conf_pos / 100.0))
                            st.write(f"🔴 **Negative:** `{conf_neg:.2f}%`")
                            st.progress(float(conf_neg / 100.0))

                    # Expandable NLP Pipeline Details
                    with st.expander("🔍 Inspect Internal NLP Pipeline Steps"):
                        st.markdown("**1. Raw Input Text:**")
                        st.info(review_text)

                        st.markdown("**2. Cleaned & Negation-Preserved Text:**")
                        st.code(cleaned, language="text")

                        # Show active vocabulary terms detected
                        nonzero_indices = tfidf_vec.nonzero()[1]
                        feature_names = vectorizer.get_feature_names_out()
                        matched_features = [feature_names[i] for i in nonzero_indices]
                        st.markdown(f"**3. Matching TF-IDF Features Detected ({len(matched_features)} tokens):**")
                        st.write(", ".join([f"`{feat}`" for feat in matched_features[:30]]))

                    st.caption(
                        "ℹ️ *Note: Machine learning predictions are probabilistic inferences based on training data "
                        "patterns and are not guaranteed to be 100% correct.*"
                    )

# ==============================================================================
# PAGE 2: MODEL EVALUATION & METRICS
# ==============================================================================
elif page == "📊 Model Evaluation & Metrics":
    st.title("📊 Model Performance & Comparative Evaluation")
    st.markdown(
        "Here you can review the actual evaluation results measured on the held-out test set "
        "(2,000 product reviews, 50% Positive / 50% Negative)."
    )

    metrics_data = load_evaluation_metrics()

    if metrics_data is None:
        st.warning("Evaluation report not found. Run `python train_model.py` to generate reports.")
    else:
        best_model_name = metrics_data.get("best_model", "Multinomial Naive Bayes")
        models_dict = metrics_data.get("models", {})

        st.subheader("🏆 Winning Model Summary")
        winner_metrics = models_dict.get(best_model_name, {})
        w_acc = winner_metrics.get("accuracy", 0.0) * 100
        w_f1 = winner_metrics.get("f1_score", 0.0) * 100
        w_prec = winner_metrics.get("precision", 0.0) * 100
        w_rec = winner_metrics.get("recall", 0.0) * 100

        m_col1, m_col2, m_col3, m_col4 = st.columns(4)
        m_col1.metric("Selected Model", best_model_name)
        m_col2.metric("Test Accuracy", f"{w_acc:.2f}%")
        m_col3.metric("Weighted F1-Score", f"{w_f1:.2f}%")
        m_col4.metric("Weighted Precision", f"{w_prec:.2f}%")

        st.markdown("---")
        st.subheader("⚔️ Head-to-Head Model Comparison")

        # Build comparison dataframe
        comparison_rows = []
        for name, m in models_dict.items():
            comparison_rows.append(
                {
                    "Model": name,
                    "Accuracy": f"{m['accuracy'] * 100:.2f}%",
                    "F1-Score": f"{m['f1_score'] * 100:.2f}%",
                    "Precision": f"{m['precision'] * 100:.2f}%",
                    "Recall": f"{m['recall'] * 100:.2f}%",
                }
            )
        # Render comparison table using clean Markdown (zero PyArrow dependency)
        md_table = (
            "| Model | Accuracy | F1-Score | Precision | Recall |\n"
            "| :--- | :---: | :---: | :---: | :---: |"
        )
        for r in comparison_rows:
            md_table += f"\n| **{r['Model']}** | {r['Accuracy']} | {r['F1-Score']} | {r['Precision']} | {r['Recall']} |"
        st.markdown(md_table)

        st.markdown(
            """
            **Evaluation Analysis:**
            - **Multinomial Naive Bayes** achieved slightly superior overall performance (86.60% F1-score)
              thanks to effective frequency-based conditional probability modeling with Laplace smoothing.
            - **Logistic Regression** achieved an equally competitive 86.50% F1-score, providing smooth calibrated
              decision probabilities via the sigmoid link function.
            """
        )

        st.markdown("---")
        st.subheader("📈 Visual Reports & Confusion Matrices")

        vis_tab1, vis_tab2 = st.tabs(["Confusion Matrices", "Dataset Sentiment Distribution"])

        with vis_tab1:
            if os.path.exists(CONFUSION_IMG):
                st.image(CONFUSION_IMG, caption="Confusion Matrices Comparison (2,000 Test Reviews)", use_container_width=True)
            else:
                st.info("Confusion matrix image not yet generated.")

        with vis_tab2:
            if os.path.exists(EDA_IMG):
                st.image(EDA_IMG, caption="Exploratory Data Analysis: Sentiment Distribution & Review Lengths", use_container_width=True)
            else:
                st.info("EDA distribution image not yet generated.")

# ==============================================================================
# PAGE 3: PROJECT OVERVIEW & PIPELINE
# ==============================================================================
elif page == "🏠 Project Overview & Pipeline":
    st.title("🏠 Sentiment Analysis of Product Reviews")
    st.markdown("### College NLP & Machine Learning Project")

    st.markdown(
        """
        ### 1. Project Objective
        The objective of this project is to develop an end-to-end Natural Language Processing (NLP) system
        that automatically analyzes customer product reviews and classifies them into **Positive** or **Negative** sentiment.
        By automating review evaluation, e-commerce platforms and consumers can instantly gauge customer satisfaction.

        ### 2. End-to-End Pipeline Workflow
        """
    )

    st.markdown(
        """
        ```
        Raw Product Review
               │
               ▼
        [Text Preprocessing]
          • Lowercase conversion
          • Contraction expansion ("don't" -> "do not")
          • Negation word preservation ("not", "never", "no")
          • Noise & HTML tag removal
               │
               ▼
        [TF-IDF Feature Extraction]
          • N-gram range: (1, 2) unigrams & bigrams
          • 5,000 top informative features
          • Sublinear term frequency scaling
               │
               ▼
        [Machine Learning Classification]
          • Logistic Regression
          • Multinomial Naive Bayes (Best Model)
               │
               ▼
        [Predicted Sentiment & Confidence %]
          • Positive (1) vs. Negative (0)
        ```
        """
    )

    st.markdown("### 3. Technology Stack")
    t1, t2, t3 = st.columns(3)
    with t1:
        st.markdown("**Core Language & Libraries**")
        st.markdown("- Python 3.12")
        st.markdown("- Scikit-learn (ML & TF-IDF)")
        st.markdown("- Pandas & NumPy (Data handling)")
    with t2:
        st.markdown("**NLP & Text Processing**")
        st.markdown("- NLTK (Stopwords & Corpora)")
        st.markdown("- Regular Expressions (Regex)")
        st.markdown("- Unicode normalization")
    with t3:
        st.markdown("**UI & Visualization**")
        st.markdown("- Streamlit (Web App)")
        st.markdown("- Matplotlib & Seaborn")
        st.markdown("- Joblib (Model Serialization)")

# ==============================================================================
# PAGE 4: LIMITATIONS & FUTURE ENHANCEMENTS
# ==============================================================================
elif page == "⚠️ Limitations & Future Work":
    st.title("⚠️ Model Limitations & Future Enhancements")

    st.markdown(
        """
        ### Current Model Limitations
        While classical TF-IDF and Naive Bayes / Logistic Regression models achieve strong baseline accuracy (~86.6%),
        they possess known technical constraints in complex linguistic scenarios:

        1. **Sarcasm and Irony:**
           - Example: *"Oh wonderful, the phone stopped charging after one day!"*
           - The word "wonderful" carries high positive TF-IDF weight, which can mislead linear models despite negative context.
        2. **Long-Distance Context Dependencies:**
           - Bag-of-words and n-gram representations discard deep sequential word ordering beyond 2-word pairs.
        3. **Domain Adaptation & Slang:**
           - A model trained on Amazon consumer goods may struggle on slang-heavy social media tweets or specialized medical reviews.
        4. **Mixed Sentiment Reviews:**
           - Example: *"The camera is gorgeous but the battery life is atrocious."*
           - The review presents both positive and negative aspects simultaneously.

        ---

        ### Future Enhancements for College Extension
        - **Transformer-based Models (BERT, RoBERTa, DistilBERT):**
          Utilize pre-trained self-attention transformers to model deep contextual semantics and bidirectionality.
        - **Aspect-Based Sentiment Analysis (ABSA):**
          Break down reviews into specific aspects (e.g., Battery, Screen, Audio, Price) and assign independent sentiment scores.
        - **Multilingual Support:**
          Extend preprocessing and tokenization to analyze multilingual customer feedback across global markets.
        """
    )

# Footer
st.markdown("---")
st.caption(
    "🎓 College NLP Project | Sentiment Analysis of Product Reviews | Built with Python, Scikit-learn, and Streamlit"
)
