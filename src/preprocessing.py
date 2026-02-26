import os
import joblib
from sklearn.preprocessing import StandardScaler
from src.config import SCALER_PATH

def scale_features(X):

    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Ensure models directory exists
    os.makedirs(os.path.dirname(SCALER_PATH), exist_ok=True)

    joblib.dump(scaler, SCALER_PATH)

    return X_scaled