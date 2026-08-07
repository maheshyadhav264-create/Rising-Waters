"""
flood_prediction.py
--------------------
Flood Prediction using Machine Learning Classification.

Pipeline:
 1. Load & clean data (handle missing values)
 2. Exploratory Data Analysis (distributions, correlation heatmap)
 3. Feature engineering & train/test split
 4. Train & compare Logistic Regression, Random Forest, and
    Gradient Boosting classifiers
 5. Select the best model based on test-set F1 score
 6. Evaluate: accuracy, precision, recall, F1, ROC-AUC, confusion matrix
 7. Save all plots and a metrics summary for the report
"""

import pandas as pd
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, roc_curve, confusion_matrix, classification_report
)

sns.set_theme(style="whitegrid")
RANDOM_STATE = 42

# ------------------------------------------------------------------
# 1. Load & clean data
# ------------------------------------------------------------------
df = pd.read_csv("data/flood_dataset.csv")
print("Initial shape:", df.shape)
print("Missing values before cleaning:\n", df.isnull().sum()[df.isnull().sum() > 0])

df = df.fillna(df.mean(numeric_only=True))

feature_cols = ["TEMP", "HUMIDITY", "CLOUD_COVER", "JAN", "FEB", "MAR", "APR",
                 "MAY", "JUN", "JUL", "AUG", "SEP", "OCT", "NOV", "DEC",
                 "JUN_SEP", "ANNUAL"]
target_col = "FLOOD"

X = df[feature_cols]
y = df[target_col]

# ------------------------------------------------------------------
# 2. Exploratory Data Analysis
# ------------------------------------------------------------------
plt.figure(figsize=(10, 8))
corr = df[feature_cols + [target_col]].corr()
sns.heatmap(corr, cmap="coolwarm", center=0, annot=False, linewidths=0.3)
plt.title("Feature Correlation Heatmap")
plt.tight_layout()
plt.savefig("plots/correlation_heatmap.png", dpi=150)
plt.close()

fig, axes = plt.subplots(1, 3, figsize=(15, 4))
for ax, col in zip(axes, ["ANNUAL", "JUN_SEP", "HUMIDITY"]):
    sns.histplot(data=df, x=col, hue=target_col, bins=30, kde=True, ax=ax, palette=["#2E86AB", "#E63946"])
    ax.set_title(f"{col} distribution by Flood")
plt.tight_layout()
plt.savefig("plots/feature_distributions.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 4))
sns.countplot(data=df, x=target_col, palette=["#2E86AB", "#E63946"])
plt.title("Flood Class Balance")
plt.xlabel("Flood (0 = No, 1 = Yes)")
plt.tight_layout()
plt.savefig("plots/class_balance.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 3. Train/test split & scaling
# ------------------------------------------------------------------
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ------------------------------------------------------------------
# 4. Train & compare models
# ------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=RANDOM_STATE),
    "Random Forest": RandomForestClassifier(n_estimators=300, max_depth=8, random_state=RANDOM_STATE),
    "Gradient Boosting": GradientBoostingClassifier(random_state=RANDOM_STATE),
}

results = []
roc_data = {}
fitted_models = {}

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        y_pred = model.predict(X_test_scaled)
        y_proba = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        y_proba = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred)
    rec = recall_score(y_test, y_pred)
    f1 = f1_score(y_test, y_pred)
    auc = roc_auc_score(y_test, y_proba)

    results.append({"Model": name, "Accuracy": acc, "Precision": prec,
                     "Recall": rec, "F1 Score": f1, "ROC-AUC": auc})
    roc_data[name] = roc_curve(y_test, y_proba)
    fitted_models[name] = model

results_df = pd.DataFrame(results).sort_values("F1 Score", ascending=False).reset_index(drop=True)
print("\nModel comparison:\n", results_df)

best_model_name = results_df.iloc[0]["Model"]
best_model = fitted_models[best_model_name]
print(f"\nBest model selected: {best_model_name}")

# ------------------------------------------------------------------
# 5. Evaluation plots for the best model
# ------------------------------------------------------------------
if best_model_name == "Logistic Regression":
    y_pred_best = best_model.predict(X_test_scaled)
else:
    y_pred_best = best_model.predict(X_test)

cm = confusion_matrix(y_test, y_pred_best)
plt.figure(figsize=(5, 4))
sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
            xticklabels=["No Flood", "Flood"], yticklabels=["No Flood", "Flood"])
plt.title(f"Confusion Matrix — {best_model_name}")
plt.ylabel("Actual")
plt.xlabel("Predicted")
plt.tight_layout()
plt.savefig("plots/confusion_matrix.png", dpi=150)
plt.close()

plt.figure(figsize=(6, 5))
for name, (fpr, tpr, _) in roc_data.items():
    auc_val = results_df.loc[results_df["Model"] == name, "ROC-AUC"].values[0]
    plt.plot(fpr, tpr, label=f"{name} (AUC={auc_val:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — Model Comparison")
plt.legend()
plt.tight_layout()
plt.savefig("plots/roc_curves.png", dpi=150)
plt.close()

# Feature importance (tree-based models only)
if best_model_name in ("Random Forest", "Gradient Boosting"):
    importances = pd.Series(best_model.feature_importances_, index=feature_cols).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=importances.values, y=importances.index, palette="viridis")
    plt.title(f"Feature Importance — {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("plots/feature_importance.png", dpi=150)
    plt.close()

model_comparison_bar = results_df.set_index("Model")[["Accuracy", "Precision", "Recall", "F1 Score", "ROC-AUC"]]
model_comparison_bar.plot(kind="bar", figsize=(9, 5))
plt.title("Model Comparison Across Metrics")
plt.ylabel("Score")
plt.ylim(0, 1)
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig("plots/model_comparison.png", dpi=150)
plt.close()

# ------------------------------------------------------------------
# 6. Save metrics & classification report to disk for the report
# ------------------------------------------------------------------
results_df.to_csv("outputs/model_comparison_metrics.csv", index=False)

with open("outputs/classification_report.txt", "w") as f:
    f.write(f"Best Model: {best_model_name}\n\n")
    f.write(classification_report(y_test, y_pred_best, target_names=["No Flood", "Flood"]))

print("\nAll outputs saved to plots/ and outputs/")
