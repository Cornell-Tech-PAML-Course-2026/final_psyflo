# Crisis Language Detection

Mental health crisis-risk screening system using TF-IDF features and machine learning models implemented from scratch with NumPy.

Developed for INFO 5368 — Cornell University.

Live App: https://crisis-language-detection-paml.streamlit.app/
GitHub Repository: https://github.com/lyxn16/PAML-Crisis-Language-Detection
---

## Project Overview

This project explores automatic detection of suicide-risk language in Reddit posts using classical machine learning approaches.

We implemented:

- Keyword Baseline
- Logistic Regression (from scratch)
- Multinomial Naive Bayes (from scratch)

using TF-IDF text representations.

The project focuses on high-recall crisis screening because false negatives (missing high-risk posts) are especially dangerous in mental health settings.

---

## Final Model

The final deployed model is:

- Multinomial Naive Bayes
- Alpha = 1.0

Selected because it achieved the strongest overall performance across Recall, F1-score, and AUC-ROC.

---

## How to Run

### Option 1 — Open Directly in Google Colab

The dataset is too large for GitHub and is automatically downloaded from Google Drive inside the notebook.

Open the notebook directly in Colab:

https://colab.research.google.com/drive/1lLF5JRxlvz9U6sow1j7QhzEIM1Nk3ZVL#scrollTo=B0Yn2BGHy07S

Then:

1. Go to **Runtime > Run all**
2. The dataset will automatically download using `gdown`
3. All experiments, figures, and evaluations will reproduce automatically

> If viewing the `.ipynb` directly on GitHub, pre-rendered outputs and visualizations are still visible without running the notebook.

---

## Repository Structure

```text
Crisis-Language-Detection/
│
├── app.py
├── README.md
├── requirements.txt
│
├── models/
│   ├── best_model.pkl
│   ├── tfidf_vectorizer.pkl
│   ├── feature_names.npy
│   └── model_metadata.json
│
├── notebooks/
│   └── final_experiments.ipynb
│
├── figures/
│   ├── roc_curve_comparison.png
│   ├── threshold_analysis.png
│   ├── top_features.png
│   ├── lr_loss_curve.png
│   └── confusion_matrix_*.png
│
├── results/
│   ├── final_model_comparison.csv
│   ├── lr_tuning_results.csv
│   └── nb_tuning_results.csv
```

---

## Dataset

The dataset is automatically downloaded inside the notebook using Google Drive + gdown.

The notebook automatically fetches:

```python
clean_data.csv
```

during execution.

External dataset folder:

https://drive.google.com/drive/folders/19-W7V1vS1zTFE18xpWKBjsr2BUlHW3g-

---

## Methodology

### 1. Data Preprocessing
- text cleaning
- normalization
- TF-IDF vectorization
- train/test splitting

### 2. Modeling
We compared:
- Keyword Baseline
- Logistic Regression
- Multinomial Naive Bayes

### 3. Evaluation
Models were evaluated using:
- Recall
- Precision
- F1-score
- AUC-ROC
- Accuracy

The project prioritized Recall because minimizing false negatives is especially important in crisis-risk screening.

### 4. Threshold Analysis
We analyzed how different decision thresholds affect Recall and Precision tradeoffs in clinical-style screening settings.

---

## Installation

Install dependencies:

```bash
pip install -r requirements.txt
```

---

## Running the Notebook

Open:

```text
notebooks/final_experiments.ipynb
```

Run all cells sequentially.

The notebook includes:
- preprocessing
- TF-IDF vectorization
- model training
- hyperparameter tuning
- evaluation
- ROC analysis
- threshold analysis
- feature importance analysis
- final model export

---

## Running the Streamlit App

Run locally:

```bash
streamlit run app.py
```

---

## Streamlit Features

The application supports:
- text-based suicide-risk prediction
- adjustable decision threshold
- risk probability scoring
- TF-IDF feature inspection
- recall-oriented screening analysis

---

## Evaluation Summary

| Model | Recall | F1-Score | AUC-ROC |
|---|---|---|---|
| Keyword Baseline | 0.369 | 0.525 | 0.666 |
| Logistic Regression | 0.831 | 0.866 | 0.944 |
| Naive Bayes | 0.947 | 0.914 | 0.973 |

Naive Bayes achieved the strongest overall performance and was selected as the final deployed model.

---

## Disclaimer

This project is for educational and research purposes only.

It is not a clinical diagnosis tool, medical advice system, or emergency-response service.

---

## License

This project was developed for educational purposes as part of the INFO 5368 PAML Final Project at Cornell University.