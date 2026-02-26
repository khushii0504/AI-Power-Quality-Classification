import os
import pandas as pd
import numpy as np
from src.config import DATA_PATH

def load_data():

    if not os.path.exists(DATA_PATH):
        raise FileNotFoundError(
            f"Dataset folder not found at:\n{DATA_PATH}\n"
            "Check your folder name carefully."
        )

    X = []
    y = []
    label_map = {}

    files = sorted([f for f in os.listdir(DATA_PATH) if f.endswith(".csv")])

    if len(files) == 0:
        raise ValueError("No CSV files found inside XPQRS folder.")

    print("Found classes:")
    for idx, file in enumerate(files):
        print(f"{idx} -> {file}")

    for label, file in enumerate(files):
        class_name = file.replace(".csv", "")
        label_map[class_name] = label

        file_path = os.path.join(DATA_PATH, file)
        df = pd.read_csv(file_path)

        signals = df.values

        for row in signals:
            X.append(row)
            y.append(label)

    print("\nDataset Loaded Successfully")
    print("Total Samples:", len(X))
    print("Number of Classes:", len(label_map))

    return np.array(X), np.array(y), label_map