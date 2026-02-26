import os
import joblib
from src.config import MODEL_PATH
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

def train_models(X, y):

    models = {
        "RandomForest": {
            "model": RandomForestClassifier(),
            "params": {
                "n_estimators": [100, 200],
                "max_depth": [None, 10]
            }
        },
        "SVM": {
            "model": SVC(probability=True),
            "params": {
                "C": [0.1, 1],
                "kernel": ["rbf"]
            }
        }
    }

    best_model = None
    best_score = 0
    comparison_results = {}

    for name, config in models.items():

        grid = GridSearchCV(
            config["model"],
            config["params"],
            cv=5,
            scoring="accuracy",
            n_jobs=-1
        )

        grid.fit(X, y)

        cv_score = cross_val_score(
            grid.best_estimator_,
            X,
            y,
            cv=5
        ).mean()

        comparison_results[name] = cv_score

        if cv_score > best_score:
            best_score = cv_score
            best_model = grid.best_estimator_

    # Ensure models directory exists
    os.makedirs(os.path.dirname(MODEL_PATH), exist_ok=True)

    joblib.dump(best_model, MODEL_PATH)

    return best_model, comparison_results