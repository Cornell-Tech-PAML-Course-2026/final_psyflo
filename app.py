import json
import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import streamlit as st
from scipy import sparse


# ============================================================
# Model class definitions
# Required so pickle can load models trained in Colab
# ============================================================

class LogisticRegressionNumPy:
    def __init__(self, lr=0.1, n_iter=1000, lambda_reg=0.01):
        self.lr = lr
        self.n_iter = n_iter
        self.lambda_reg = lambda_reg
        self.weights = None
        self.bias = 0.0
        self.loss_history = []

    def sigmoid(self, z):
        return 1.0 / (1.0 + np.exp(-np.clip(z, -500, 500)))

    def predict_proba(self, X):
        z = X.dot(self.weights) + self.bias
        if sparse.issparse(z):
            z = np.asarray(z).flatten()
        return self.sigmoid(z)

    def predict(self, X, threshold=0.5):
        return (self.predict_proba(X) >= threshold).astype(int)


class MultinomialNaiveBayes:
    def __init__(self, alpha=1.0):
        self.alpha = alpha
        self.classes = None
        self.log_priors = {}
        self.log_likelihoods = {}

    def predict_proba(self, X):
        log_probs = []

        for c in self.classes:
            log_p = self.log_priors[c] + X.dot(self.log_likelihoods[c])
            if sparse.issparse(log_p):
                log_p = np.asarray(log_p).flatten()
            log_probs.append(log_p)

        log_probs = np.column_stack(log_probs)

        # Stable softmax
        log_probs -= log_probs.max(axis=1, keepdims=True)
        probs = np.exp(log_probs)
        probs /= probs.sum(axis=1, keepdims=True)

        return probs

    def predict(self, X, threshold=0.5):
        proba = self.predict_proba(X)[:, 1]
        return (proba >= threshold).astype(int)


# ============================================================
# Page config
# ============================================================

st.set_page_config(
    page_title="Crisis Language Detection",
    page_icon="🧠",
    layout="wide"
)


# ============================================================
# Load artifacts
# ============================================================

@st.cache_resource
def load_artifacts():
    model_path = Path("models/best_model.pkl")
    vectorizer_path = Path("models/tfidf_vectorizer.pkl")
    feature_path = Path("models/feature_names.npy")
    metadata_path = Path("models/model_metadata.json")

    with open(model_path, "rb") as f:
        model = pickle.load(f)

    with open(vectorizer_path, "rb") as f:
        tfidf = pickle.load(f)

    feature_names = np.load(feature_path, allow_pickle=True)

    if metadata_path.exists():
        with open(metadata_path, "r") as f:
            metadata = json.load(f)
    else:
        metadata = {
            "best_model": "Multinomial Naive Bayes",
            "default_threshold": 0.5,
            "recommended_screening_threshold": 0.3,
            "primary_metric": "Recall"
        }

    return model, tfidf, feature_names, metadata


model, tfidf, feature_names, metadata = load_artifacts()


# ============================================================
# Sidebar
# ============================================================

st.sidebar.title("⚙️ Controls")

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(metadata.get("default_threshold", 0.5)),
    step=0.05
)

st.sidebar.markdown("---")
st.sidebar.markdown("### Model")
st.sidebar.write(metadata.get("best_model", "Multinomial Naive Bayes"))

st.sidebar.markdown("### Primary Metric")
st.sidebar.write(metadata.get("primary_metric", "Recall"))

st.sidebar.markdown("---")
st.sidebar.markdown(
    "Lower threshold → higher recall and fewer missed high-risk posts."
)


# ============================================================
# Main UI
# ============================================================

st.title("🧠 Crisis Language Detection")
st.markdown("**Mental Health Risk Screening — INFO 5368, Cornell University**")

st.warning(
    "Disclaimer: This tool is for educational and experimental purposes only. "
    "It is not a clinical diagnosis, not medical advice, and not an emergency service."
)

st.markdown("---")

text_input = st.text_area(
    "Enter text to analyze",
    placeholder="Paste or type a Reddit post here...",
    height=170
)

analyze_btn = st.button("🔍 Analyze Text", type="primary")


# ============================================================
# Prediction
# ============================================================

if analyze_btn and text_input.strip():

    X = tfidf.transform([text_input])

    proba = model.predict_proba(X)

    if hasattr(proba, "ndim") and proba.ndim > 1:
        score = float(proba[0, 1])
    else:
        score = float(proba[0])

    label = "High Risk" if score >= threshold else "Low Risk"

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("### Risk Label")
        if label == "High Risk":
            st.error(f"🔴 **{label}**")
        else:
            st.success(f"🟢 **{label}**")

    with col2:
        st.markdown("### Risk Score")
        st.metric("Predicted suicide-risk probability", f"{score:.3f}")

    with col3:
        st.markdown("### Threshold")
        st.metric("Current threshold", f"{threshold:.2f}")

    st.progress(score)

    st.markdown("---")

    st.markdown("### Interpretation")

    if label == "High Risk":
        st.write(
            "The model classified this text as high risk because the predicted "
            "probability is greater than or equal to the selected threshold."
        )
    else:
        st.write(
            "The model classified this text as low risk because the predicted "
            "probability is below the selected threshold."
        )

    st.markdown("### Top TF-IDF Features in Input")

    vec = X.toarray()[0]
    active_idx = np.where(vec > 0)[0]

    if len(active_idx) > 0:
        top_idx = active_idx[np.argsort(vec[active_idx])[-10:][::-1]]

        rows = []
        for idx in top_idx:
            rows.append({
                "Feature": feature_names[idx],
                "TF-IDF Value": round(float(vec[idx]), 4)
            })

        feat_df = pd.DataFrame(rows)
        st.dataframe(feat_df, use_container_width=True)

    else:
        st.info("No major TF-IDF features found in the input.")

elif analyze_btn and not text_input.strip():
    st.warning("Please enter some text first.")


# ============================================================
# Footer
# ============================================================

st.markdown("---")
st.caption(
    "Built with Streamlit. Model trained from scratch using TF-IDF features and NumPy-based classifiers."
)