FLOOD PREDICTION PROJECT - PACKAGE CONTENTS
=============================================

IMPORTANT NOTE ON THE DATASET:
Only screenshots of code and column names were provided (no actual data
file). A synthetic dataset (code/dataset.csv) with the same column schema
was generated so the full pipeline could be built and run end-to-end.
To get results on your real data: replace code/dataset.csv with your
actual file (same column names) and re-run the three scripts below in
order.

/code
  generate_dataset.py   - creates the synthetic dataset (skip if you have real data)
  preprocessing.py      - missing values, outlier capping, encoding, split, scaling
  train_model.py        - trains Logistic Regression + Random Forest, evaluates, saves best model
  dataset.csv           - the dataset used for this run

  Run order:
    python generate_dataset.py   (optional - only if you need synthetic data)
    python train_model.py        (this also runs preprocessing.py internally)

/document
  Flood_Prediction_Project_Report.docx
      Full write-up: missing value handling, outlier (IQR) capping, categorical
      encoding, X/y split, train-test split, feature scaling, model training,
      evaluation results, confusion matrix, feature importance, and conclusion.

/results
  model_comparison.csv         - accuracy of each model tested
  classification_report.txt    - precision/recall/F1 for the best model
  outlier_summary.csv          - IQR bounds and number of values capped per column
  confusion_matrix.png         - confusion matrix of the best model (Random Forest)
  feature_importance.png       - feature importance ranking
  flood_prediction_model.pkl   - the trained, saved model (Joblib)
  scaler.pkl                   - fitted StandardScaler (for real-time inference)
  label_encoder.pkl            - fitted LabelEncoder for the 'sub' column

RESULT SUMMARY
  Best model: Random Forest
  Test Accuracy: 78.67%
