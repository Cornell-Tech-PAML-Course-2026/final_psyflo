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
    page_icon="✦",
    layout="wide"
)


# ============================================================
# Global styles from improved UI
# ============================================================

st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=DM+Serif+Display:ital@0;1&family=DM+Sans:wght@300;400;500&display=swap');

html, body, [class*="css"] {
    font-family: 'DM Sans', sans-serif;
    font-weight: 300;
    color: #2a2520;
}

[data-testid="stAppViewContainer"],
[data-testid="stAppViewContainer"] p,
[data-testid="stAppViewContainer"] span,
[data-testid="stAppViewContainer"] div,
[data-testid="stAppViewContainer"] label,
[data-testid="stAppViewContainer"] li {
    color: #2a2520;
}

.stApp {
    background-color: #f5f0e8;
    background-image:
        radial-gradient(ellipse 60% 50% at 80% 10%, rgba(220,190,230,0.35) 0%, transparent 60%),
        radial-gradient(ellipse 50% 45% at 15% 80%, rgba(180,210,240,0.30) 0%, transparent 55%),
        radial-gradient(ellipse 45% 40% at 55% 55%, rgba(255,220,180,0.28) 0%, transparent 50%);
}

[data-testid="stSidebar"] {
    background-color: #f5f0e8 !important;
    border-right: none !important;
    width: 260px !important;
    min-width: 260px !important;
    max-width: 260px !important;
}
[data-testid="stSidebarResizeHandle"] { display: none !important; }
[data-testid="stSidebarCollapseButton"] { display: none !important; }

.block-container {
    max-width: 100% !important;
    padding-left: 2rem !important;
    padding-right: 2rem !important;
    padding-top: 1rem !important;
}

[data-testid="stSidebarContent"] {
    background: rgba(255,255,255,0.45) !important;
    backdrop-filter: blur(12px) !important;
    -webkit-backdrop-filter: blur(12px) !important;
}

[data-testid="stSidebar"] * {
    font-family: 'DM Sans', sans-serif !important;
    color: #3a3530 !important;
}

[data-testid="stSidebar"] .stSlider label {
    font-size: 0.75rem !important;
    letter-spacing: 0.08em !important;
    text-transform: uppercase !important;
    color: #8a7f74 !important;
}

h1, h1 *, h1 span, h1 p {
    font-family: 'DM Serif Display', Georgia, serif !important;
    font-size: 3.2rem !important;
    font-weight: 400 !important;
    color: #1e1a16 !important;
    letter-spacing: -0.02em !important;
    line-height: 1.15 !important;
    margin-bottom: 0.2rem !important;
}

h3, h3 *, h3 span {
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.72rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.12em !important;
    text-transform: uppercase !important;
    color: #8a7f74 !important;
    margin-bottom: 0.75rem !important;
}

.subtitle {
    font-family: 'DM Sans', sans-serif;
    font-size: 0.85rem;
    color: #9a8f82;
    letter-spacing: 0.05em;
    margin-top: -0.5rem;
    margin-bottom: 2.5rem;
}

hr {
    border: none !important;
    border-top: 1px solid rgba(0,0,0,0.08) !important;
    margin: 2rem 0 !important;
}

.stTextArea textarea {
    background-color: rgba(255,255,255,0.65) !important;
    border: 1px solid rgba(0,0,0,0.10) !important;
    border-radius: 12px !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.92rem !important;
    color: #2a2520 !important;
    padding: 1rem 1.1rem !important;
    backdrop-filter: blur(8px);
}
.stTextArea textarea:focus {
    border-color: rgba(160,138,184,0.7) !important;
    box-shadow: 0 0 0 3px rgba(160,138,184,0.20) !important;
}
.stTextArea label {
    font-size: 0.72rem !important;
    letter-spacing: 0.10em !important;
    text-transform: uppercase !important;
    color: #8a7f74 !important;
    font-weight: 500 !important;
}

.stButton > button[kind="primary"],
button[data-testid="baseButton-primary"] {
    background: #1e1a16 !important;
    color: #ffffff !important;
    border: none !important;
    border-radius: 100px !important;
    padding: 0.6rem 2rem !important;
    font-family: 'DM Sans', sans-serif !important;
    font-size: 0.82rem !important;
    font-weight: 500 !important;
    letter-spacing: 0.08em !important;
}
.stButton > button[kind="primary"] *,
button[data-testid="baseButton-primary"] * {
    color: #ffffff !important;
}

.result-card {
    background: rgba(255,255,255,0.60);
    border: 1px solid rgba(0,0,0,0.07);
    border-radius: 16px;
    padding: 1.6rem 1.8rem;
    backdrop-filter: blur(10px);
    -webkit-backdrop-filter: blur(10px);
}

.dot-high { color: #dc2626 !important; }
.dot-low  { color: #2a7a50 !important; }

.badge-high {
    display: inline-block;
    background: rgba(220,80,80,0.10);
    color: #b03030;
    border: 1px solid rgba(180,60,60,0.25);
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}
.badge-low {
    display: inline-block;
    background: rgba(60,150,100,0.10);
    color: #2a7a50;
    border: 1px solid rgba(50,130,80,0.25);
    border-radius: 100px;
    padding: 0.4rem 1.2rem;
    font-size: 0.95rem;
    font-weight: 500;
    letter-spacing: 0.04em;
}

.score-number {
    font-family: 'DM Serif Display', Georgia, serif;
    font-size: 3rem;
    color: #1e1a16;
    line-height: 1;
    margin: 0.5rem 0 0.8rem 0;
}

.progress-track {
    height: 6px;
    background: rgba(0,0,0,0.08);
    border-radius: 100px;
    overflow: hidden;
    margin-top: 0.5rem;
}
.progress-fill-high {
    height: 100%;
    background: linear-gradient(90deg, #e8a0a0, #c05050);
    border-radius: 100px;
}
.progress-fill-low {
    height: 100%;
    background: linear-gradient(90deg, #a0d0b8, #3a9060);
    border-radius: 100px;
}

.stDataFrame {
    border-radius: 12px !important;
    overflow: hidden !important;
    border: 1px solid rgba(0,0,0,0.07) !important;
}

.stAlert {
    border-radius: 12px !important;
    border: 1px solid rgba(0,0,0,0.07) !important;
    background: rgba(255,255,255,0.5) !important;
}
.stAlert p, .stAlert div, .stAlert span {
    color: #3a3530 !important;
}

div[data-testid="stSlider"] div[role="slider"] {
    background-color: #a08ab8 !important;
    border-color: #a08ab8 !important;
}
div[data-testid="stSlider"] div[role="slider"]:focus,
div[data-testid="stSlider"] div[role="slider"]:active {
    box-shadow: 0 0 0 8px rgba(160, 138, 184, 0.25) !important;
    outline: none !important;
}

[data-testid="stHeader"] {
    background-color: #1e1a16 !important;
}

#MainMenu, footer { visibility: hidden; }
</style>
""", unsafe_allow_html=True)


# ============================================================
# Load final model artifacts
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

st.sidebar.markdown("<br>", unsafe_allow_html=True)
st.sidebar.markdown(
    '<p style="font-size:0.68rem;letter-spacing:0.14em;text-transform:uppercase;color:#9a8f82;margin-bottom:1.2rem;">Controls</p>',
    unsafe_allow_html=True
)

threshold = st.sidebar.slider(
    "Decision Threshold",
    min_value=0.1,
    max_value=0.9,
    value=float(metadata.get("default_threshold", 0.5)),
    step=0.05
)

st.sidebar.markdown("<hr style='border-top:1px solid rgba(0,0,0,0.08);margin:1.5rem 0;'>", unsafe_allow_html=True)

st.sidebar.markdown(
    f"""
    <p style="font-size:0.78rem;color:#9a8f82;line-height:1.7;">
    <strong>Model:</strong> {metadata.get("best_model", "Multinomial Naive Bayes")}<br>
    <strong>Primary metric:</strong> {metadata.get("primary_metric", "Recall")}<br><br>
    Lower threshold → higher Recall and fewer missed high-risk posts.
    </p>
    """,
    unsafe_allow_html=True
)


# ============================================================
# Main UI
# ============================================================

st.markdown("""
<br>
<h1 style="font-family:'DM Serif Display',Georgia,serif;font-size:3.2rem;font-weight:400;
           color:#1e1a16;letter-spacing:-0.02em;line-height:1.15;margin-bottom:0.2rem;">
    Crisis Language<br>Detection
</h1>
<p class="subtitle">Mental Health Risk Screening &nbsp;·&nbsp; INFO 5368, Cornell University</p>
""", unsafe_allow_html=True)

st.warning(
    "Disclaimer: This tool is for educational and experimental purposes only. "
    "It is not a clinical diagnosis, not medical advice, and not an emergency service."
)

st.markdown("---")

text_input = st.text_area(
    "Text to analyze",
    placeholder="Paste or type a Reddit post here…",
    height=160,
    label_visibility="visible"
)

st.markdown("<br>", unsafe_allow_html=True)
analyze_btn = st.button("Analyze Text →", type="primary")


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
    is_high = label == "High Risk"

    st.markdown("<br>", unsafe_allow_html=True)

    col1, col2 = st.columns(2, gap="large")

    with col1:
        badge_html = (
            '<span class="badge-high"><span class="dot-high">●</span> High Risk</span>'
            if is_high
            else '<span class="badge-low"><span class="dot-low">●</span> Low Risk</span>'
        )

        st.markdown(
            f"""
            <div class="result-card">
                <p style="font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#9a8f82;margin-bottom:0.8rem;">Risk Assessment</p>
                {badge_html}
                <p style="font-size:0.78rem;color:#9a8f82;margin-top:1rem;line-height:1.6;">
                    {"Linguistic patterns associated with crisis language detected." if is_high else "No significant crisis language patterns detected."}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    with col2:
        fill_class = "progress-fill-high" if is_high else "progress-fill-low"
        fill_pct = int(score * 100)

        st.markdown(
            f"""
            <div class="result-card">
                <p style="font-size:0.68rem;letter-spacing:0.12em;text-transform:uppercase;color:#9a8f82;margin-bottom:0rem;">Risk Score</p>
                <p class="score-number">{score:.3f}</p>
                <div class="progress-track">
                    <div class="{fill_class}" style="width:{fill_pct}%;"></div>
                </div>
                <p style="font-size:0.75rem;color:#b0a898;margin-top:0.6rem;">
                    Threshold set at {threshold:.2f} · Model: {metadata.get("best_model", "Multinomial Naive Bayes")}
                </p>
            </div>
            """,
            unsafe_allow_html=True
        )

    st.markdown("<br>", unsafe_allow_html=True)
    st.markdown("---")

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
    st.markdown("<br>", unsafe_allow_html=True)
    st.warning("Please enter some text before analyzing.")


st.markdown("---")
st.caption(
    "Built with Streamlit. Final model trained from scratch using TF-IDF features and NumPy-based classifiers."
)