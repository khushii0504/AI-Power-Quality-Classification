import os
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix
from src.config import RESULTS_PATH

def save_confusion_matrix(model, X, y, label_names):

    preds = model.predict(X)
    cm = confusion_matrix(y, preds)

    os.makedirs(RESULTS_PATH, exist_ok=True)

    plt.figure(figsize=(8,6))
    sns.heatmap(cm, annot=True, fmt='d',
                xticklabels=label_names,
                yticklabels=label_names)
    plt.title("Confusion Matrix")
    plt.savefig(os.path.join(RESULTS_PATH, "confusion_matrix.png"))
    plt.close()

    return cm