"""
train_models.py
----------------
Flood Prediction - Model Building, Training, Evaluation & Comparison

Pipeline:
  1. Load dataset and split into train/test sets
  2. Scale features with StandardScaler
  3. Train four classifiers: Decision Tree, Random Forest, KNN, XGBoost
     (XGBoost here uses sklearn's GradientBoostingClassifier, matching the
     original project notebook's implementation, since it requires no
     external xgboost dependency)
  4. Evaluate every model (accuracy, confusion matrix, classification report)
  5. Compare all models side-by-side with compareModel()
  6. Save the best-performing model + the fitted scaler with joblib for
     deployment in the Flask app (floods.save / transform.save)
"""

import json
import numpy as np
import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn import tree
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import (
    confusion_matrix,
    classification_report,
    accuracy_score,
    precision_score,
    recall_score,
)

RANDOM_STATE = 42


# ----------------------------------------------------------------------
# 1. DECISION TREE
# ----------------------------------------------------------------------
def decisiontree(X_train, X_test, y_train, y_test):
    print("\n========== DECISION TREE MODEL BUILDING ==========")

    # 1. Initialize Decision Tree Classifier
    model = tree.DecisionTreeClassifier(random_state=RANDOM_STATE)
    print("[INFO] DecisionTreeClassifier initialized.")

    # 2. Train the model
    model.fit(X_train, y_train)
    print("[INFO] Model training completed.")

    # 3. Predict on test data
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")

    # 4. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # 5. Display results
    print(f"\n[RESULT] Accuracy : {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:\n")
    print(cr)

    # 6. Return the model and predictions
    return model, y_pred


# ----------------------------------------------------------------------
# 2. RANDOM FOREST
# ----------------------------------------------------------------------
def randomForest(X_train, X_test, y_train, y_test, n_estimators=100, random_state=42):
    print("\n========== RANDOM FOREST MODEL BUILDING ==========")

    # 1. Initialize Random Forest Classifier
    model = RandomForestClassifier(n_estimators=n_estimators, random_state=random_state)
    print(f"[INFO] RandomForestClassifier initialized with n_estimators={n_estimators}, random_state={random_state}")

    # 2. Train the model
    model.fit(X_train, y_train)
    print("[INFO] Model training completed.")

    # 3. Predict on test data
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")

    # 4. Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # 5. Display results
    print(f"\n[RESULT] Accuracy : {accuracy:.4f}")
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:\n")
    print(cr)

    # 6. Return the model and predictions
    return model, y_pred


# ----------------------------------------------------------------------
# 3. K-NEAREST NEIGHBORS
# ----------------------------------------------------------------------
def KNN(X_train, X_test, y_train, y_test):
    print("\n========== KNN MODEL BUILDING ==========")

    # Initialize KNN classifier
    model = KNeighborsClassifier(n_neighbors=5)

    # Train the model
    model.fit(X_train, y_train)
    print("[INFO] KNN model training completed.")

    # Predict on test data
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")

    # Evaluate model performance
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # Display results
    print("\n[RESULT] Accuracy:", accuracy)
    print("\nConfusion Matrix::")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # Return model and predictions
    return model, y_pred


# ----------------------------------------------------------------------
# 4. XGBOOST  (Gradient Boosting Classifier)
# ----------------------------------------------------------------------
def xgboost(X_train, X_test, y_train, y_test):
    print("\n========== XGBOOST MODEL BUILDING ==========")

    # Initialize Gradient Boosting Classifier
    model = GradientBoostingClassifier(random_state=RANDOM_STATE)

    # Train the model using training data
    model.fit(X_train, y_train)
    print("[INFO] XGBoost model training completed.")

    # Predict on test data
    y_pred = model.predict(X_test)
    print("[INFO] Prediction completed on test data.")

    # Evaluate the model
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    cr = classification_report(y_test, y_pred)

    # Display results
    print("\n[RESULT] Accuracy:", accuracy)
    print("\nConfusion Matrix:")
    print(cm)
    print("\nClassification Report:")
    print(cr)

    # Return model and predictions
    return model, y_pred


# ----------------------------------------------------------------------
# 5. COMPARE ALL MODELS
# ----------------------------------------------------------------------
def compareModel(results, y_test):
    """
    results: dict of {model_name: (model_object, y_pred)}
    Prints a consolidated, side-by-side comparison of all trained models
    and returns the name of the best-performing model.
    """
    print("\n========== MODEL COMPARISON ==========")
    summary = []
    for name, (model, y_pred) in results.items():
        acc = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, zero_division=0)
        rec = recall_score(y_test, y_pred, zero_division=0)
        summary.append((name, acc, prec, rec))

    print(f"{'Model':<20}{'Accuracy':<12}{'Precision':<12}{'Recall':<12}")
    for name, acc, prec, rec in summary:
        print(f"{name:<20}{acc:<12.4f}{prec:<12.4f}{rec:<12.4f}")

    best_name = max(summary, key=lambda x: x[1])[0]
    print(f"\n[RESULT] Best performing model: {best_name}")
    return best_name, summary


# ----------------------------------------------------------------------
# MAIN PIPELINE
# ----------------------------------------------------------------------
if __name__ == "__main__":
    # 1. Load dataset
    df = pd.read_csv("/home/claude/flood_project/data/flood_dataset.csv")
    X = df.drop(columns=["FLOOD"])
    y = df["FLOOD"]

    # 2. Train / test split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
    )

    # 3. Feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)

    # 4. Train all four models
    dt_model, p1 = decisiontree(X_train_scaled, X_test_scaled, y_train, y_test)
    rf_model, p2 = randomForest(X_train_scaled, X_test_scaled, y_train, y_test)
    knn_model, p3 = KNN(X_train_scaled, X_test_scaled, y_train, y_test)
    xgb_model, p4 = xgboost(X_train_scaled, X_test_scaled, y_train, y_test)

    # 5. Compare all models
    results = {
        "Decision Tree": (dt_model, p1),
        "Random Forest": (rf_model, p2),
        "KNN": (knn_model, p3),
        "XGBoost": (xgb_model, p4),
    }
    best_name, summary = compareModel(results, y_test)

    # 6. Save comparison results to file (for the project report)
    with open("/home/claude/flood_project/results/comparison_results.json", "w") as f:
        json.dump(
            {
                "models": [
                    {"name": n, "accuracy": a, "precision": p, "recall": r}
                    for n, a, p, r in summary
                ],
                "best_model": best_name,
            },
            f,
            indent=2,
        )

    with open("/home/claude/flood_project/results/confusion_matrices.txt", "w") as f:
        for name, (model, y_pred) in results.items():
            f.write(f"\n===== {name} =====\n")
            f.write("Confusion Matrix:\n")
            f.write(str(confusion_matrix(y_test, y_pred)) + "\n")
            f.write("Classification Report:\n")
            f.write(classification_report(y_test, y_pred) + "\n")

    # 7. Save the best model (XGBoost, per project convention) + scaler
    #    using Joblib for deployment in the Flask application.
    best_model_obj = results[best_name][0]
    joblib.dump(xgb_model, "/home/claude/flood_project/models/floods.save")
    joblib.dump(scaler, "/home/claude/flood_project/models/transform.save")
    print("\n[INFO] Saved trained XGBoost model -> models/floods.save")
    print("[INFO] Saved fitted StandardScaler -> models/transform.save")
    print(f"[INFO] (Best model by accuracy was: {best_name})")
