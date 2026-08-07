"""
FLOOD PREDICTION PROJECT
========================
End-to-end pipeline: data loading -> descriptive analysis -> univariate
analysis -> multivariate analysis -> preprocessing -> model training
(Logistic Regression, Random Forest, XGBoost) -> evaluation -> model export.

Run: python flood_prediction.py
Outputs are written to the 'outputs/' folder, trained model to 'model/'.
"""

import warnings
warnings.filterwarnings("ignore")

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import (accuracy_score, confusion_matrix,
                              classification_report, roc_auc_score, roc_curve)
from xgboost import XGBClassifier
import joblib

OUT = "outputs"
MODEL_DIR = "model"

# ---------------------------------------------------------------------------
# 1. IMPORT LIBRARIES  (see docstring above)
# ---------------------------------------------------------------------------

# ---------------------------------------------------------------------------
# 2. DATA LOADING
# ---------------------------------------------------------------------------
dataset = pd.read_excel("data/flood dataset.xlsx")
print("Shape:", dataset.shape)
print(dataset.head())

with open(f"{OUT}/01_info.txt", "w") as f:
    f.write("HEAD\n" + "=" * 60 + "\n")
    f.write(dataset.head().to_string() + "\n\n")
    f.write("SHAPE\n" + "=" * 60 + "\n")
    f.write(str(dataset.shape) + "\n\n")
    f.write("INFO\n" + "=" * 60 + "\n")
    import io
    buf = io.StringIO()
    dataset.info(buf=buf)
    f.write(buf.getvalue() + "\n\n")
    f.write("DESCRIBE\n" + "=" * 60 + "\n")
    f.write(dataset.describe().to_string() + "\n\n")
    f.write("MISSING VALUES\n" + "=" * 60 + "\n")
    f.write(dataset.isnull().sum().to_string() + "\n")

# ---------------------------------------------------------------------------
# 3. UNIVARIATE ANALYSIS -- Distribution plots + Box plots (outlier detection)
# ---------------------------------------------------------------------------
numeric_cols = dataset.select_dtypes(include=[np.number]).columns.tolist()
numeric_cols_no_target = [c for c in numeric_cols if c != "flood"]

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols_no_target):
    sns.histplot(dataset[col].dropna(), kde=True, ax=axes[i], color="#2E86AB")
    axes[i].set_title(f"Distribution of {col}")
for j in range(len(numeric_cols_no_target), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{OUT}/02_distributions.png", dpi=120)
plt.close()

fig, axes = plt.subplots(3, 4, figsize=(20, 12))
axes = axes.flatten()
for i, col in enumerate(numeric_cols_no_target):
    sns.boxplot(y=dataset[col], ax=axes[i], color="#F18F01")
    axes[i].set_title(f"Boxplot of {col}")
for j in range(len(numeric_cols_no_target), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig(f"{OUT}/03_boxplots.png", dpi=120)
plt.close()

# Outlier capping (IQR method) -- replaces extreme outliers with the
# upper/lower boundary instead of dropping rows.
def cap_outliers(df, cols):
    df = df.copy()
    for col in cols:
        q1, q3 = df[col].quantile(0.25), df[col].quantile(0.75)
        iqr = q3 - q1
        low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
        df[col] = df[col].clip(lower=low, upper=high)
    return df

# ---------------------------------------------------------------------------
# 4. MULTIVARIATE ANALYSIS -- Correlation heatmap
# ---------------------------------------------------------------------------
plt.figure(figsize=(12, 10))
sns.heatmap(dataset.corr(), annot=True, cmap="summer", linewidths=1,
            linecolor="k", square=True, mask=False, vmin=-1, vmax=1,
            cbar_kws={"orientation": "vertical"}, cbar=True, fmt=".2f")
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig(f"{OUT}/04_heatmap.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 5. PREPROCESSING
# ---------------------------------------------------------------------------
dataset = dataset.fillna(dataset.mean(numeric_only=True))
dataset = cap_outliers(dataset, numeric_cols_no_target)

X = dataset.drop(columns=["flood"])
y = dataset["flood"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)

# ---------------------------------------------------------------------------
# 6. MODEL TRAINING & EVALUATION
# ---------------------------------------------------------------------------
models = {
    "Logistic Regression": LogisticRegression(max_iter=1000, random_state=42),
    "Random Forest": RandomForestClassifier(n_estimators=200, random_state=42),
    "XGBoost": XGBClassifier(n_estimators=200, use_label_encoder=False,
                              eval_metric="logloss", random_state=42),
}

results = []
roc_data = {}
best_model, best_name, best_acc = None, None, -1

for name, model in models.items():
    if name == "Logistic Regression":
        model.fit(X_train_scaled, y_train)
        preds = model.predict(X_test_scaled)
        probs = model.predict_proba(X_test_scaled)[:, 1]
    else:
        model.fit(X_train, y_train)
        preds = model.predict(X_test)
        probs = model.predict_proba(X_test)[:, 1]

    acc = accuracy_score(y_test, preds)
    auc = roc_auc_score(y_test, probs)
    cm = confusion_matrix(y_test, preds)
    report = classification_report(y_test, preds)

    results.append({"Model": name, "Accuracy": round(acc, 4), "ROC_AUC": round(auc, 4)})
    roc_data[name] = roc_curve(y_test, probs)

    with open(f"{OUT}/05_{name.replace(' ', '_')}_report.txt", "w") as f:
        f.write(f"Model: {name}\n")
        f.write(f"Accuracy: {acc:.4f}\nROC-AUC: {auc:.4f}\n\n")
        f.write("Confusion Matrix:\n" + str(cm) + "\n\n")
        f.write("Classification Report:\n" + report)

    plt.figure(figsize=(5, 4))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    plt.title(f"Confusion Matrix - {name}")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.tight_layout()
    plt.savefig(f"{OUT}/06_cm_{name.replace(' ', '_')}.png", dpi=120)
    plt.close()

    if acc > best_acc:
        best_acc, best_name, best_model = acc, name, model

results_df = pd.DataFrame(results).sort_values("Accuracy", ascending=False)
results_df.to_csv(f"{OUT}/07_model_comparison.csv", index=False)
print(results_df)

# ROC curve comparison
plt.figure(figsize=(7, 6))
for name, (fpr, tpr, _) in roc_data.items():
    plt.plot(fpr, tpr, label=name)
plt.plot([0, 1], [0, 1], "k--", label="Random")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve Comparison")
plt.legend()
plt.tight_layout()
plt.savefig(f"{OUT}/08_roc_comparison.png", dpi=120)
plt.close()

# Feature importance (Random Forest / XGBoost -- whichever is best, fallback RF)
importance_model = best_model if hasattr(best_model, "feature_importances_") else models["Random Forest"]
importances = pd.Series(importance_model.feature_importances_, index=X.columns).sort_values(ascending=False)
plt.figure(figsize=(8, 6))
sns.barplot(x=importances.values, y=importances.index, palette="viridis")
plt.title(f"Feature Importance ({best_name if hasattr(best_model,'feature_importances_') else 'Random Forest'})")
plt.tight_layout()
plt.savefig(f"{OUT}/09_feature_importance.png", dpi=120)
plt.close()

# ---------------------------------------------------------------------------
# 7. SAVE BEST MODEL
# ---------------------------------------------------------------------------
joblib.dump(best_model, f"{MODEL_DIR}/flood_model.pkl")
joblib.dump(scaler, f"{MODEL_DIR}/scaler.pkl")
joblib.dump(list(X.columns), f"{MODEL_DIR}/feature_names.pkl")

with open(f"{OUT}/10_best_model.txt", "w") as f:
    f.write(f"Best Model: {best_name}\nAccuracy: {best_acc:.4f}\n")

print(f"\nBest model: {best_name} (Accuracy = {best_acc:.4f})")
print("Saved model to model/flood_model.pkl")
