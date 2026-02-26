from src.data_loader import load_data
from src.feature_engineering import build_feature_matrix
from src.preprocessing import scale_features
from src.model_training import train_models
from src.evaluation import save_confusion_matrix

def main():
    X, y, label_map = load_data()

    X_feat = build_feature_matrix(X)
    X_scaled = scale_features(X_feat)

    model, comparison = train_models(X_scaled, y)

    print("Model Comparison (CV Accuracy):")
    for k, v in comparison.items():
        print(f"{k}: {v:.4f}")

    save_confusion_matrix(model, X_scaled, y, list(label_map.keys()))

    print("Training Complete.")

if __name__ == "__main__":
    main()