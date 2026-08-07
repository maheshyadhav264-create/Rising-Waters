import nbformat as nbf

nb = nbf.v4.new_notebook()
cells = []

def md(text):
    cells.append(nbf.v4.new_markdown_cell(text))

def code(text):
    cells.append(nbf.v4.new_code_cell(text))

# ============================== TITLE ==============================
md("""# 🌊 Flood Prediction — End-to-End Machine Learning Project

This notebook implements the full project workflow across all epics:

1. **Epic 1 — Data Collection**
2. **Epic 2 — Visualizing and Analysing the Data**
3. **Epic 3 — Data Pre-Processing**
4. **Epic 4 — Model Building**

The trained model produced here is consumed by the Flask web application in `app/` (**Epic 5**).
""")

# ============================== EPIC 1 ==============================
md("""## Epic 1 — Data Collection
### Story 1.1: Download the flood prediction dataset and load it into the notebook

The dataset (`flood_dataset.csv`) contains 6,000 records describing monsoon, hydrological,
terrain, land-use and socio-infrastructure factors, along with the binary target
`Flood_Occurred` (1 = flood occurred, 0 = no flood). It was assembled from public
flood-risk—factor definitions (rainfall, river discharge, water level, drainage,
historical flood counts, infrastructure/land-cover quality, etc.).
""")

code("""import pandas as pd

DATA_PATH = "../data/flood_dataset.csv"
df = pd.read_csv(DATA_PATH)
print("Dataset loaded successfully.")
print("Shape:", df.shape)
df.head()
""")

# ============================== EPIC 2 ==============================
md("""## Epic 2 — Visualizing and Analysing the Data
### Story 2.1: Import all required Python libraries""")

code("""import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from scipy import stats

pd.set_option("display.max_columns", 50)
sns.set_style("whitegrid")
plt.rcParams["figure.dpi"] = 100

import os
os.makedirs("../results", exist_ok=True)
print("Libraries imported.")
""")

md("### Story 2.2: Read and explore the dataset (structure, features, target)")

code("""print("Shape:", df.shape)
df.info()
""")

code("""df.describe(include='all').T
""")

code("""print("Target variable distribution:")
print(df['Flood_Occurred'].value_counts())
print(df['Flood_Occurred'].value_counts(normalize=True).round(3) * 100, "%")
""")

md("### Story 2.3: Univariate analysis — distribution of individual variables")

code("""numeric_cols = df.select_dtypes(include=[np.number]).columns.drop(['Record_ID', 'Flood_Occurred'])

fig, axes = plt.subplots(4, 3, figsize=(16, 14))
axes = axes.flatten()
for i, col in enumerate(numeric_cols):
    sns.histplot(df[col].dropna(), kde=True, ax=axes[i], color="#2b6cb0")
    axes[i].set_title(f"Distribution of {col}")
for j in range(len(numeric_cols), len(axes)):
    fig.delaxes(axes[j])
plt.tight_layout()
plt.savefig("../results/01_univariate_histograms.png", bbox_inches="tight")
plt.show()
""")

code("""categorical_cols = ['Land_Cover', 'Soil_Type', 'Infrastructure_Quality']

fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for i, col in enumerate(categorical_cols):
    sns.countplot(data=df, x=col, ax=axes[i], palette="Blues_d",
                   order=df[col].value_counts().index)
    axes[i].set_title(f"Count of {col}")
    axes[i].tick_params(axis='x', rotation=30)
plt.tight_layout()
plt.savefig("../results/02_univariate_categorical.png", bbox_inches="tight")
plt.show()
""")

code("""plt.figure(figsize=(5, 4.5))
sns.countplot(data=df, x="Flood_Occurred", palette=["#2b6cb0", "#e53e3e"])
plt.title("Target Variable Distribution — Flood Occurred")
plt.xticks([0, 1], ["No Flood (0)", "Flood (1)"])
plt.savefig("../results/03_target_distribution.png", bbox_inches="tight")
plt.show()
""")

md("### Story 2.4: Multivariate analysis — relationships between features")

code("""plt.figure(figsize=(12, 9))
corr = df[numeric_cols.tolist() + ['Flood_Occurred']].corr()
sns.heatmap(corr, annot=True, fmt=".2f", cmap="RdBu_r", center=0, linewidths=0.4)
plt.title("Correlation Heatmap")
plt.tight_layout()
plt.savefig("../results/04_correlation_heatmap.png", bbox_inches="tight")
plt.show()
""")

code("""top_corr = corr['Flood_Occurred'].drop('Flood_Occurred').sort_values(key=abs, ascending=False)
print("Features most correlated with Flood_Occurred:")
top_corr
""")

code("""fig, axes = plt.subplots(1, 3, figsize=(16, 4.5))
for i, col in enumerate(['Rainfall_mm', 'Water_Level_m', 'River_Discharge_m3s']):
    sns.boxplot(data=df, x='Flood_Occurred', y=col, ax=axes[i], palette=["#2b6cb0", "#e53e3e"])
    axes[i].set_title(f"{col} vs Flood_Occurred")
    axes[i].set_xticklabels(["No Flood", "Flood"])
plt.tight_layout()
plt.savefig("../results/05_multivariate_boxplots.png", bbox_inches="tight")
plt.show()
""")

code("""plt.figure(figsize=(6, 5))
ct = pd.crosstab(df['Land_Cover'], df['Flood_Occurred'], normalize='index') * 100
ct.plot(kind='bar', stacked=True, color=["#2b6cb0", "#e53e3e"], ax=plt.gca())
plt.ylabel("% of records")
plt.title("Flood Rate by Land Cover Type")
plt.legend(["No Flood", "Flood"])
plt.tight_layout()
plt.savefig("../results/06_landcover_vs_flood.png", bbox_inches="tight")
plt.show()
""")

md("### Story 2.5: Descriptive statistical analysis")

code("""desc_stats = df[numeric_cols].describe().T
desc_stats['skew'] = df[numeric_cols].skew()
desc_stats['kurtosis'] = df[numeric_cols].kurtosis()
desc_stats.to_csv("../results/07_descriptive_statistics.csv")
desc_stats
""")

# ============================== EPIC 3 ==============================
md("""## Epic 3 — Data Pre-Processing
### Story 3.1: Identify and handle missing values""")

code("""missing = df.isnull().sum()
missing_pct = (missing / len(df) * 100).round(2)
missing_summary = pd.DataFrame({'missing_count': missing, 'missing_pct': missing_pct})
missing_summary = missing_summary[missing_summary['missing_count'] > 0].sort_values('missing_count', ascending=False)
print(missing_summary)
""")

code("""df_clean = df.drop(columns=['Record_ID']).copy()

# Numeric columns -> median imputation (robust to outliers/skew)
num_missing_cols = df_clean.select_dtypes(include=[np.number]).columns[df_clean.select_dtypes(include=[np.number]).isnull().any()]
for col in num_missing_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].median())

# Categorical columns -> mode imputation
cat_missing_cols = df_clean.select_dtypes(include='object').columns[df_clean.select_dtypes(include='object').isnull().any()]
for col in cat_missing_cols:
    df_clean[col] = df_clean[col].fillna(df_clean[col].mode()[0])

print("Remaining missing values after imputation:", df_clean.isnull().sum().sum())
""")

md("### Story 3.2: Detect and treat outliers")

code("""def iqr_bounds(series, k=1.5):
    q1, q3 = series.quantile(0.25), series.quantile(0.75)
    iqr = q3 - q1
    return q1 - k * iqr, q3 + k * iqr

outlier_report = {}
for col in numeric_cols:
    low, high = iqr_bounds(df_clean[col])
    n_out = ((df_clean[col] < low) | (df_clean[col] > high)).sum()
    outlier_report[col] = n_out
outlier_report = pd.Series(outlier_report).sort_values(ascending=False)
print("Outlier counts per numeric feature (IQR method):")
print(outlier_report[outlier_report > 0])
""")

code("""# Cap outliers at the IQR fences (winsorization) - preserves records, reduces distortion
for col in numeric_cols:
    low, high = iqr_bounds(df_clean[col])
    df_clean[col] = df_clean[col].clip(lower=low, upper=high)

fig, axes = plt.subplots(1, 2, figsize=(11, 4.5))
sns.boxplot(data=df[['Rainfall_mm']], ax=axes[0], color="#e53e3e")
axes[0].set_title("Rainfall_mm — Before Treatment")
sns.boxplot(data=df_clean[['Rainfall_mm']], ax=axes[1], color="#2b6cb0")
axes[1].set_title("Rainfall_mm — After IQR Capping")
plt.tight_layout()
plt.savefig("../results/08_outlier_treatment.png", bbox_inches="tight")
plt.show()
""")

md("### Story 3.3: Convert categorical variables into numerical representations")

code("""from sklearn.preprocessing import LabelEncoder

categorical_cols_all = df_clean.select_dtypes(include='object').columns.tolist()
print("Categorical columns to encode:", categorical_cols_all)

label_encoders = {}
for col in categorical_cols_all:
    le = LabelEncoder()
    df_clean[col] = le.fit_transform(df_clean[col])
    label_encoders[col] = le
    print(f"{col}: {dict(zip(le.classes_, le.transform(le.classes_)))}")
""")

md("### Story 3.4: Split the dataset into training and testing subsets")

code("""from sklearn.model_selection import train_test_split

X = df_clean.drop(columns=['Flood_Occurred'])
y = df_clean['Flood_Occurred']

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)
print("Training set:", X_train.shape, " Testing set:", X_test.shape)
print("Train target balance:\\n", y_train.value_counts(normalize=True).round(3))
print("Test target balance:\\n", y_test.value_counts(normalize=True).round(3))
""")

md("### Story 3.5: Apply feature scaling")

code("""from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_train_scaled = pd.DataFrame(scaler.fit_transform(X_train), columns=X_train.columns, index=X_train.index)
X_test_scaled = pd.DataFrame(scaler.transform(X_test), columns=X_test.columns, index=X_test.index)

X_train_scaled.describe().T[['mean', 'std']].round(3).head()
""")

# ============================== EPIC 4 ==============================
md("""## Epic 4 — Model Building
### Story 4.1: Decision Tree""")

code("""from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (accuracy_score, precision_score, recall_score,
                              f1_score, roc_auc_score, confusion_matrix, classification_report)

results = {}
fitted_models = {}

def evaluate_model(name, model, X_tr, X_te, y_tr, y_te, needs_scaled=False):
    model.fit(X_tr, y_tr)
    y_pred = model.predict(X_te)
    y_proba = model.predict_proba(X_te)[:, 1] if hasattr(model, "predict_proba") else y_pred

    metrics = {
        "Accuracy": accuracy_score(y_te, y_pred),
        "Precision": precision_score(y_te, y_pred),
        "Recall": recall_score(y_te, y_pred),
        "F1_Score": f1_score(y_te, y_pred),
        "ROC_AUC": roc_auc_score(y_te, y_proba),
    }
    results[name] = metrics
    fitted_models[name] = model

    print(f"===== {name} =====")
    for k, v in metrics.items():
        print(f"{k}: {v:.4f}")
    print()
    print(classification_report(y_te, y_pred, target_names=["No Flood", "Flood"]))

    cm = confusion_matrix(y_te, y_pred)
    plt.figure(figsize=(4.2, 3.6))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues", cbar=False,
                xticklabels=["No Flood", "Flood"], yticklabels=["No Flood", "Flood"])
    plt.title(f"Confusion Matrix — {name}")
    plt.ylabel("Actual")
    plt.xlabel("Predicted")
    plt.tight_layout()
    safe_name = name.lower().replace(" ", "_")
    plt.savefig(f"../results/cm_{safe_name}.png", bbox_inches="tight")
    plt.show()
    return metrics

dt_model = DecisionTreeClassifier(max_depth=8, min_samples_leaf=10, random_state=42)
_ = evaluate_model("Decision Tree", dt_model, X_train, X_test, y_train, y_test)
""")

md("### Story 4.2: Random Forest")

code("""from sklearn.ensemble import RandomForestClassifier

rf_model = RandomForestClassifier(n_estimators=300, max_depth=12, min_samples_leaf=5,
                                   random_state=42, n_jobs=-1)
_ = evaluate_model("Random Forest", rf_model, X_train, X_test, y_train, y_test)
""")

md("### Story 4.3: K-Nearest Neighbors (KNN)\n\nKNN is distance-based, so it is trained on the **scaled** features.")

code("""from sklearn.neighbors import KNeighborsClassifier

knn_model = KNeighborsClassifier(n_neighbors=15, weights="distance")
_ = evaluate_model("KNN", knn_model, X_train_scaled, X_test_scaled, y_train, y_test)
""")

md("### Story 4.4: XGBoost")

code("""from xgboost import XGBClassifier

xgb_model = XGBClassifier(
    n_estimators=300, max_depth=5, learning_rate=0.05,
    subsample=0.85, colsample_bytree=0.85,
    eval_metric="logloss", random_state=42, n_jobs=-1
)
_ = evaluate_model("XGBoost", xgb_model, X_train, X_test, y_train, y_test)
""")

md("### Story 4.5: Compare the performance of all developed models")

code("""results_df = pd.DataFrame(results).T.sort_values("F1_Score", ascending=False)
results_df.to_csv("../results/09_model_comparison.csv")
results_df.round(4)
""")

code("""fig, ax = plt.subplots(figsize=(10, 5.5))
results_df[['Accuracy', 'Precision', 'Recall', 'F1_Score', 'ROC_AUC']].plot(
    kind='bar', ax=ax, colormap='viridis'
)
plt.title("Model Performance Comparison")
plt.ylabel("Score")
plt.xticks(rotation=0)
plt.ylim(0, 1.05)
plt.legend(loc="lower right")
plt.tight_layout()
plt.savefig("../results/10_model_comparison_chart.png", bbox_inches="tight")
plt.show()
""")

code("""from sklearn.metrics import roc_curve

plt.figure(figsize=(6.5, 5.5))
for name, model in fitted_models.items():
    X_te = X_test_scaled if name == "KNN" else X_test
    proba = model.predict_proba(X_te)[:, 1]
    fpr, tpr, _ = roc_curve(y_test, proba)
    plt.plot(fpr, tpr, label=f"{name} (AUC={results[name]['ROC_AUC']:.3f})")
plt.plot([0, 1], [0, 1], linestyle="--", color="gray")
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curves — All Models")
plt.legend()
plt.tight_layout()
plt.savefig("../results/11_roc_curves.png", bbox_inches="tight")
plt.show()
""")

md("""### Story 4.6: Select the best-performing model and save it for deployment

The model with the highest **F1-score** (balances precision and recall — important for
flood risk, where both false alarms and missed floods carry a cost) is selected as the
final production model.""")

code("""best_model_name = results_df.index[0]
best_model = fitted_models[best_model_name]
uses_scaled_input = (best_model_name == "KNN")

print(f"Best performing model: {best_model_name}")
print(results_df.loc[best_model_name].round(4))
""")

code("""if hasattr(best_model, "feature_importances_"):
    fi = pd.Series(best_model.feature_importances_, index=X_train.columns).sort_values(ascending=False)
    plt.figure(figsize=(8, 6))
    sns.barplot(x=fi.values[:12], y=fi.index[:12], palette="viridis")
    plt.title(f"Top Feature Importances — {best_model_name}")
    plt.xlabel("Importance")
    plt.tight_layout()
    plt.savefig("../results/12_feature_importance.png", bbox_inches="tight")
    plt.show()
    fi.to_csv("../results/13_feature_importance.csv")
""")

code("""import pickle
import json

MODEL_DIR = "../model"
import os
os.makedirs(MODEL_DIR, exist_ok=True)

with open(f"{MODEL_DIR}/flood_model.pkl", "wb") as f:
    pickle.dump(best_model, f)

with open(f"{MODEL_DIR}/scaler.pkl", "wb") as f:
    pickle.dump(scaler, f)

with open(f"{MODEL_DIR}/label_encoders.pkl", "wb") as f:
    pickle.dump(label_encoders, f)

model_meta = {
    "best_model_name": best_model_name,
    "uses_scaled_input": uses_scaled_input,
    "feature_order": list(X_train.columns),
    "categorical_columns": categorical_cols_all,
    "metrics": {k: round(v, 4) for k, v in results[best_model_name].items()},
    "all_model_results": {m: {k: round(v, 4) for k, v in mets.items()} for m, mets in results.items()},
}
with open(f"{MODEL_DIR}/model_metadata.json", "w") as f:
    json.dump(model_meta, f, indent=2)

print("Saved model, scaler, encoders and metadata to", MODEL_DIR)
print(json.dumps(model_meta, indent=2))
""")

nb['cells'] = cells
nb['metadata'] = {
    "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
    "language_info": {"name": "python", "version": "3.12"}
}

with open("/home/claude/flood_project/notebook/Flood_Prediction_Analysis.ipynb", "w") as f:
    nbf.write(nb, f)

print("Notebook written.")
