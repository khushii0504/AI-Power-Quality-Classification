# ⚡ AI-Based Power Quality Disturbance Classification

An end-to-end Machine Learning system for classifying power quality disturbances using electrical signal processing and wavelet-based feature extraction.

---

## 📌 Project Overview

This project implements a complete ML pipeline for detecting and classifying power quality disturbances such as:

- Voltage Sag
- Voltage Swell
- Harmonics
- Flicker
- Interruption
- Oscillatory Transient
- Notch
- Combined Disturbances

The system performs:

- Signal preprocessing
- Feature engineering (Electrical + Wavelet)
- Cross-validation
- Hyperparameter tuning
- Model comparison (Random Forest vs SVM)
- Confusion matrix evaluation
- Interactive Streamlit dashboard deployment

---

## 🧠 Technical Architecture

Dataset → Feature Engineering → Scaling →  
Model Selection (GridSearchCV) → Evaluation → Deployment

### Features Used
- RMS
- Peak value
- Crest factor
- THD
- Wavelet coefficients (mean, std, max, min)

---

## 📊 Model Performance

Cross-Validation Accuracy:

- Random Forest: **~87%**
- SVM: ~76%

Best model selected automatically via cross-validation.

---

## 🚀 Live Demo

(Will be added after Streamlit deployment)

---

## 🛠 Tech Stack

- Python 3.9+
- NumPy
- Pandas
- Scikit-learn
- PyWavelets
- Plotly
- Streamlit

---

## 📂 Project Structure
AI-Power-Quality-Classification/
│
├── app/
│ └── streamlit_app.py
├── src/
│ ├── feature_engineering.py
│ ├── model_training.py
│ ├── preprocessing.py
│ └── ...
├── models/
│ ├── best_model.pkl
│ └── scaler.pkl
├── results/
│ └── confusion_matrix.png
├── train.py
├── requirements.txt
└── README.md

---

## 💻 Local Setup

### 1. Clone Repository

```bash
git clone https://github.com/YOUR_USERNAME/AI-Power-Quality-Classification.git
cd AI-Power-Quality-Classification