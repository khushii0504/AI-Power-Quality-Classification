import os
import sys
import streamlit as st
import numpy as np
import pandas as pd
import joblib
import plotly.express as px
import plotly.graph_objects as go

# -------------------------------------------------
# FIX: Add project root to Python path
# -------------------------------------------------
CURRENT_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.dirname(CURRENT_DIR)
sys.path.append(PROJECT_ROOT)

# Now safe to import from src
from src.feature_engineering import extract_features
from src.config import MODEL_PATH, SCALER_PATH

# -------------------------------------------------
# Streamlit Page Config
# -------------------------------------------------
st.set_page_config(
    page_title="Power Quality Disturbance Classifier",
    layout="wide"
)

st.title("⚡ AI-Based Power Quality Disturbance Classification")

# -------------------------------------------------
# Load Model & Scaler
# -------------------------------------------------
if not os.path.exists(MODEL_PATH) or not os.path.exists(SCALER_PATH):
    st.error("Model or Scaler not found. Run train.py first.")
    st.stop()

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)

# -------------------------------------------------
# Sidebar Upload
# -------------------------------------------------
st.sidebar.header("Upload Signal")
uploaded_file = st.sidebar.file_uploader(
    "Upload CSV file (Single Signal Row)",
    type=["csv"]
)

if uploaded_file:

    df = pd.read_csv(uploaded_file)
    signal = df.values.flatten()

    col1, col2 = st.columns(2)

    # ------------------------------
    # Waveform Plot
    # ------------------------------
    fig = go.Figure()
    fig.add_trace(go.Scatter(y=signal, mode='lines'))
    fig.update_layout(
        title="Signal Waveform",
        xaxis_title="Samples",
        yaxis_title="Amplitude"
    )

    col1.plotly_chart(fig, use_container_width=True)

    # ------------------------------
    # Feature Extraction
    # ------------------------------
    features = np.array(extract_features(signal)).reshape(1, -1)
    features_scaled = scaler.transform(features)

    prediction = model.predict(features_scaled)[0]
    probabilities = model.predict_proba(features_scaled)[0]

    # ------------------------------
    # Load Class Names
    # ------------------------------
    data_path = os.path.join(PROJECT_ROOT, "data", "raw", "XPQRS")
    class_files = sorted([f for f in os.listdir(data_path) if f.endswith(".csv")])
    class_names = [f.replace(".csv", "") for f in class_files]

    predicted_class = class_names[prediction]

    col2.subheader("Prediction Result")
    col2.success(f"Predicted Class: {predicted_class}")

    # ------------------------------
    # Probability Chart
    # ------------------------------
    prob_df = pd.DataFrame({
        "Class": class_names,
        "Probability": probabilities
    })

    fig2 = px.bar(
        prob_df,
        x="Class",
        y="Probability",
        title="Class Probability Distribution"
    )

    fig2.update_layout(xaxis_tickangle=-45)

    col2.plotly_chart(fig2, use_container_width=True)

else:
    st.info("Upload a CSV file containing a single signal row to classify.")

# -------------------------------------------------
# Confusion Matrix Display
# -------------------------------------------------
st.markdown("---")
st.header("Model Evaluation")

conf_path = os.path.join(PROJECT_ROOT, "results", "confusion_matrix.png")

if os.path.exists(conf_path):
    st.image(conf_path, caption="Confusion Matrix")
else:
    st.warning("Confusion matrix not found. Run train.py first.")