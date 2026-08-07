"""
Data Preprocessing Pipeline - Flood Prediction Project
=========================================================
Steps implemented (matching the project documentation):
  1. Load dataset
  2. Detect & handle missing values
  3. Detect & handle outliers (IQR capping)
  4. Encode categorical values (Label Encoding for 'sub')
  5. Split into independent (X) and dependent (y) variables
  6. Train-test split
  7. Feature scaling (StandardScaler) - fit on train, applied to train & test
  8. Save the fitted scaler and encoder for reuse at inference time
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import joblib

RANDOM_STATE = 10


def load_data(path="dataset.csv"):
    dataset = pd.read_csv(path)
    print(f"Dataset loaded: {dataset.shape[0]} rows, {dataset.shape[1]} columns")
    return dataset


def check_missing_values(dataset):
    print("\n--- Missing Value Check ---")
    print(dataset.isnull().sum())
    print("\nAny missing values present:", dataset.isnull().any().any())
    return dataset.isnull().sum()


def handle_missing_values(dataset, numeric_cols):
    # Impute numeric missing values with the column median (robust to outliers)
    for col in numeric_cols:
        if dataset[col].isnull().sum() > 0:
            median_val = dataset[col].median()
            dataset[col] = dataset[col].fillna(median_val)
    print("\nMissing values after imputation:", dataset.isnull().sum().sum())
    return dataset


def cap_outliers_iqr(dataset, numeric_cols):
    print("\n--- Outlier Capping (IQR method) ---")
    summary = []
    for col in numeric_cols:
        Q1 = dataset[col].quantile(0.25)
        Q3 = dataset[col].quantile(0.75)
        IQR = Q3 - Q1
        lower_bound = Q1 - 1.5 * IQR
        upper_bound = Q3 + 1.5 * IQR

        n_capped = ((dataset[col] < lower_bound) | (dataset[col] > upper_bound)).sum()
        dataset[col] = np.where(dataset[col] > upper_bound, upper_bound, dataset[col])
        dataset[col] = np.where(dataset[col] < lower_bound, lower_bound, dataset[col])

        summary.append((col, round(lower_bound, 2), round(upper_bound, 2), int(n_capped)))

    summary_df = pd.DataFrame(summary, columns=["column", "lower_bound", "upper_bound", "values_capped"])
    print(summary_df.to_string(index=False))
    return dataset, summary_df


def encode_categorical(dataset, cat_col="sub"):
    print(f"\n--- Label Encoding categorical column: '{cat_col}' ---")
    le = LabelEncoder()
    dataset[cat_col] = le.fit_transform(dataset[cat_col])
    mapping = dict(zip(le.classes_, le.transform(le.classes_)))
    print("Encoding map:", mapping)
    return dataset, le


def split_features_target(dataset, target_col="flood"):
    X = dataset.drop(target_col, axis=1)
    y = dataset[target_col]
    print(f"\nIndependent variables (X): {X.shape}")
    print(f"Dependent/target variable (y): {y.shape}")
    return X, y


def run_preprocessing(path="dataset.csv"):
    dataset = load_data(path)
    check_missing_values(dataset)

    numeric_cols = ["Temp", "Humidity", "Cloud Cover", "ANNUAL",
                     "Jan-Feb", "Mar-May", "Jun-Sep", "Oct-Dec", "avgjune"]

    dataset = handle_missing_values(dataset, numeric_cols)
    dataset, outlier_summary = cap_outliers_iqr(dataset, numeric_cols)
    dataset, label_encoder = encode_categorical(dataset, "sub")

    X, y = split_features_target(dataset, "flood")

    x_train, x_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=RANDOM_STATE, stratify=y
    )
    print(f"\nTrain set: {x_train.shape}, Test set: {x_test.shape}")

    sc = StandardScaler()
    x_train_scaled = sc.fit_transform(x_train)
    x_test_scaled = sc.transform(x_test)  # transform only (not fit) on test data

    joblib.dump(sc, "scaler.pkl")
    joblib.dump(label_encoder, "label_encoder.pkl")
    print("\nSaved scaler.pkl and label_encoder.pkl")

    outlier_summary.to_csv("outlier_summary.csv", index=False)

    return x_train_scaled, x_test_scaled, y_train, y_test, X.columns.tolist()


if __name__ == "__main__":
    run_preprocessing()
