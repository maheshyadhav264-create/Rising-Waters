"""
train_model.py
----------------
Generates a realistic synthetic dataset for flood prediction (since no
original dataset/notebook was supplied), trains a classification model
on five rainfall/cloud-cover based features, and saves:

    floods.save     -> trained model   (loaded by app.py)
    transform.save   -> fitted StandardScaler (loaded by app.py)

Features (matches the Predict Floods form in index.html):
    1. Cloud Cover        (%)
    2. Annual Rainfall     (mm)
    3. Jan-Feb Rainfall    (mm)
    4. March-May Rainfall  (mm)
    5. June-September Rainfall (mm)

Target:
    FLOOD (1) / NO FLOOD (0)

NOTE: Replace this synthetic-data step with your own historical
rainfall/flood dataset (e.g. a CSV of past years) for production use --
just point `df = pd.read_csv("your_data.csv")` at it and keep the same
column order.
"""

import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    accuracy_score, confusion_matrix, classification_report, roc_auc_score
)
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import joblib

RANDOM_STATE = 42
np.random.seed(RANDOM_STATE)

# --------------------------------------------------------------------
# 1. Synthetic dataset generation
# --------------------------------------------------------------------
N = 2000

cloud_cover = np.clip(np.random.normal(55, 20, N), 0, 100)
jan_feb = np.clip(np.random.gamma(2.0, 40, N), 0, None)
mar_may = np.clip(np.random.gamma(2.5, 60, N), 0, None)
jun_sep = np.clip(np.random.gamma(3.5, 150, N), 0, None)
annual_rainfall = jan_feb + mar_may + jun_sep + np.clip(np.random.gamma(2, 50, N), 0, None)

# Underlying "risk score" drives the probability of a flood -- heavier
# monsoon (Jun-Sep) rain and high cloud cover raise flood risk.
risk_score = (
    0.006 * jun_sep +
    0.004 * annual_rainfall +
    0.03 * cloud_cover +
    0.003 * mar_may -
    6.0
)
prob_flood = 1 / (1 + np.exp(-risk_score))
flood = np.random.binomial(1, prob_flood)

df = pd.DataFrame({
    "CLOUD_COVER": cloud_cover,
    "ANNUAL_RAINFALL": annual_rainfall,
    "JAN_FEB_RAINFALL": jan_feb,
    "MAR_MAY_RAINFALL": mar_may,
    "JUN_SEP_RAINFALL": jun_sep,
    "FLOOD": flood,
})

df.to_csv("/home/claude/floods_prediction/model_training/synthetic_flood_data.csv", index=False)
print("Class balance:\n", df["FLOOD"].value_counts(normalize=True))

# --------------------------------------------------------------------
# 2. Train / test split + scaling
# --------------------------------------------------------------------
X = df.drop(columns=["FLOOD"])
y = df["FLOOD"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=RANDOM_STATE, stratify=y
)

sc = StandardScaler()
X_train_sc = sc.fit_transform(X_train)
X_test_sc = sc.transform(X_test)

# --------------------------------------------------------------------
# 3. Train model
#    (GradientBoostingClassifier used as a drop-in stand-in for
#    XGBoost, which isn't installed in this environment. Swap in
#    `from xgboost import XGBClassifier` if you have it available --
#    the rest of the pipeline/app code does not need to change.)
# --------------------------------------------------------------------
model = GradientBoostingClassifier(
    n_estimators=200,
    learning_rate=0.08,
    max_depth=3,
    random_state=RANDOM_STATE,
)
model.fit(X_train_sc, y_train)

# --------------------------------------------------------------------
# 4. Evaluate
# --------------------------------------------------------------------
y_pred = model.predict(X_test_sc)
y_proba = model.predict_proba(X_test_sc)[:, 1]

acc = accuracy_score(y_test, y_pred)
auc = roc_auc_score(y_test, y_proba)
cm = confusion_matrix(y_test, y_pred)
report = classification_report(y_test, y_pred)

print(f"Accuracy: {acc:.3f}")
print(f"ROC AUC : {auc:.3f}")
print("Confusion matrix:\n", cm)
print(report)

with open("/home/claude/floods_prediction/model_training/metrics_report.txt", "w") as f:
    f.write(f"Accuracy: {acc:.3f}\nROC AUC : {auc:.3f}\n\n")
    f.write("Confusion Matrix:\n")
    f.write(str(cm) + "\n\n")
    f.write("Classification Report:\n")
    f.write(report)

# --------------------------------------------------------------------
# 5. Result plots (for the "results" section of the deliverable)
# --------------------------------------------------------------------
fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))

im = axes[0].imshow(cm, cmap="Blues")
axes[0].set_title("Confusion Matrix")
axes[0].set_xlabel("Predicted")
axes[0].set_ylabel("Actual")
axes[0].set_xticks([0, 1]); axes[0].set_xticklabels(["No Flood", "Flood"])
axes[0].set_yticks([0, 1]); axes[0].set_yticklabels(["No Flood", "Flood"])
for i in range(2):
    for j in range(2):
        axes[0].text(j, i, cm[i, j], ha="center", va="center",
                     color="white" if cm[i, j] > cm.max() / 2 else "black")

importances = model.feature_importances_
feat_names = X.columns
order = np.argsort(importances)
axes[1].barh(range(len(order)), importances[order], color="#2b6cb0")
axes[1].set_yticks(range(len(order)))
axes[1].set_yticklabels([feat_names[i] for i in order])
axes[1].set_title("Feature Importance")

plt.tight_layout()
plt.savefig("/home/claude/floods_prediction/screenshots/model_results.png", dpi=150)
print("Saved results plot.")

# --------------------------------------------------------------------
# 6. Persist model + scaler (consumed by app.py)
# --------------------------------------------------------------------
joblib.dump(model, "/home/claude/floods_prediction/floods.save")
joblib.dump(sc, "/home/claude/floods_prediction/transform.save")
print("Saved floods.save and transform.save")
