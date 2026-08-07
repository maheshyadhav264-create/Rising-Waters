"""
Model Training & Evaluation - Flood Prediction Project
=========================================================
Trains a classifier on the preprocessed data and evaluates it using
accuracy, a classification report, and a confusion matrix.
The trained model is saved with Joblib for later real-time inference,
alongside the scaler and label encoder produced during preprocessing.
"""

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
)

from preprocessing import run_preprocessing


def train_and_evaluate():
    x_train, x_test, y_train, y_test, feature_names = run_preprocessing("dataset.csv")

    models = {
        "Logistic Regression": LogisticRegression(max_iter=1000, random_state=10),
        "Random Forest": RandomForestClassifier(n_estimators=200, random_state=10),
    }

    results = []
    best_model = None
    best_acc = -1
    best_name = None

    for name, model in models.items():
        model.fit(x_train, y_train)
        y_pred = model.predict(x_test)
        acc = accuracy_score(y_test, y_pred)
        report = classification_report(y_test, y_pred, digits=3)

        print(f"\n=== {name} ===")
        print(f"Accuracy: {acc:.4f}")
        print(report)

        results.append({"model": name, "accuracy": round(acc, 4)})

        if acc > best_acc:
            best_acc = acc
            best_model = model
            best_name = name
            best_report = report
            best_pred = y_pred

    # Save comparison table
    results_df = pd.DataFrame(results)
    results_df.to_csv("model_comparison.csv", index=False)

    # Save the classification report of the best model
    with open("classification_report.txt", "w") as f:
        f.write(f"Best model: {best_name}\n")
        f.write(f"Accuracy: {best_acc:.4f}\n\n")
        f.write(best_report)

    # Confusion matrix plot for the best model
    cm = confusion_matrix(y_test, best_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Flood", "Flood"])
    fig, ax = plt.subplots(figsize=(5, 4))
    disp.plot(ax=ax, cmap="Blues", colorbar=False)
    plt.title(f"Confusion Matrix - {best_name}")
    plt.tight_layout()
    plt.savefig("confusion_matrix.png", dpi=150)
    plt.close()

    # Feature importance plot (Random Forest only, if it's the best model or for reference)
    rf_model = models["Random Forest"]
    importances = pd.Series(rf_model.feature_importances_, index=feature_names).sort_values(ascending=True)
    plt.figure(figsize=(6, 5))
    importances.plot(kind="barh", color="#2b6cb0")
    plt.title("Feature Importance (Random Forest)")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("feature_importance.png", dpi=150)
    plt.close()

    # Save the best model
    joblib.dump(best_model, "flood_prediction_model.pkl")
    print(f"\nBest model ({best_name}) saved as flood_prediction_model.pkl")
    print("\nModel comparison:\n", results_df)

    return results_df, best_name, best_acc


if __name__ == "__main__":
    train_and_evaluate()
