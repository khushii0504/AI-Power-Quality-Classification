import os

# Project root directory
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))

# Updated to match your actual folder name: XPQRS
DATA_PATH = os.path.join(BASE_DIR, "data", "raw", "XPQRS")

MODEL_PATH = os.path.join(BASE_DIR, "models", "best_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "models", "scaler.pkl")
RESULTS_PATH = os.path.join(BASE_DIR, "results")