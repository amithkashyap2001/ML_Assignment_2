# ==========================================
# Adult Income Classification Web Application
# Developed by: Amith P Kashyap
# ==========================================

import streamlit as st
import pandas as pd
import numpy as np
import joblib
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)

# --------------------------------
# Page Configuration
# --------------------------------
st.set_page_config(
    page_title="Adult Income Classification - ML Assignment",
    layout="wide"
)

# --------------------------------
# Header Section
# --------------------------------
st.title("Adult Income Classification using Machine Learning")
st.markdown("**Developed by: Amith P Kashyap**")

st.markdown("""
This application predicts whether an individual's annual income exceeds $50K 
based on demographic and employment attributes.

The models were trained on the Adult Income dataset and evaluated using multiple
performance metrics including Accuracy, AUC, Precision, Recall, F1-score, and MCC.
""")

st.markdown("---")

# --------------------------------
# Load Saved Models
# --------------------------------
@st.cache_resource
def load_models():
    models = {
        "Logistic Regression": joblib.load("model/logistic.pkl"),
        "Decision Tree": joblib.load("model/decision_tree.pkl"),
        "K-Nearest Neighbors": joblib.load("model/knn.pkl"),
        "Naive Bayes (Gaussian)": joblib.load("model/naive_bayes.pkl"),
        "Random Forest (Ensemble)": joblib.load("model/random_forest.pkl"),
        "XGBoost (Ensemble)": joblib.load("model/xgboost.pkl")
    }
    return models

models = load_models()

# --------------------------------
# Explanation Section
# --------------------------------
with st.expander("About This Application"):
    st.markdown("""
    - Upload a test dataset in CSV format (same structure as training dataset).
    - Select one of the trained machine learning models.
    - The app will compute predictions and display evaluation metrics.
    - Confusion matrix and classification report are shown for performance analysis.
    """)

st.markdown("---")

# --------------------------------
# File Upload Section
# --------------------------------
st.header("1️⃣ Upload Test Dataset")

uploaded_file = st.file_uploader(
    "Upload CSV file containing test data",
    type=["csv"]
)

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    st.subheader("Preview of Uploaded Data")
    st.dataframe(df.head())

    # Separate features and target
    X = df.drop("income", axis=1)
    y = df["income"].map({"<=50K": 0, ">50K": 1})

    st.markdown("---")

    # --------------------------------
    # Model Selection
    # --------------------------------
    st.header("2️⃣ Select Machine Learning Model")

    selected_model_name = st.selectbox(
        "Choose a trained model for evaluation:",
        list(models.keys())
    )

    model = models[selected_model_name]

    # --------------------------------
    # Predictions
    # --------------------------------
    y_pred = model.predict(X)
    y_proba = model.predict_proba(X)[:, 1]

    # --------------------------------
    # Evaluation Metrics
    # --------------------------------
    st.markdown("---")
    st.header("3️⃣ Evaluation Metrics")

    accuracy = accuracy_score(y, y_pred)
    auc = roc_auc_score(y, y_proba)
    precision = precision_score(y, y_pred)
    recall = recall_score(y, y_pred)
    f1 = f1_score(y, y_pred)
    mcc = matthews_corrcoef(y, y_pred)

    metrics_df = pd.DataFrame({
        "Metric": ["Accuracy", "AUC", "Precision", "Recall", "F1 Score", "MCC"],
        "Value": [accuracy, auc, precision, recall, f1, mcc]
    })

    st.table(metrics_df.style.format({"Value": "{:.4f}"}))

    # --------------------------------
    # Confusion Matrix
    # --------------------------------
    st.markdown("---")
    st.header("4️⃣ Confusion Matrix")

    cm = confusion_matrix(y, y_pred)

    fig, ax = plt.subplots(figsize=(2.5,2.5))
    sns.heatmap(
        cm,
        annot=True,
        fmt="d",
        cmap="Blues",
        cbar=False,
        square=True,
        ax=ax
    )

    ax.set_xlabel("Predicted Label")
    ax.set_ylabel("Actual Label")
    ax.set_title("Confusion Matrix")

    st.pyplot(fig)

    # --------------------------------
    # Classification Report
    # --------------------------------
    st.markdown("---")
    st.header("5️⃣ Classification Report")

    report_dict = classification_report(y, y_pred, output_dict=True)
    report_df = pd.DataFrame(report_dict).transpose()

    st.dataframe(report_df.style.format("{:.2f}"))

else:
    st.info("Please upload a CSV file to start evaluation.")
