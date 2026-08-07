FLOOD PREDICTION PROJECT — PACKAGE CONTENTS
=============================================

document/
    Flood_Prediction_Project_Report.docx   <- Full written project report
                                               (introduction, EDA, methodology,
                                               results, deployment, conclusion)

code/                                      <- Fully self-contained, runnable project
    make_dataset.py     Generates the synthetic dataset (data/flood dataset.xlsx)
    flood_prediction.py Full pipeline: EDA -> preprocessing -> model training
                         -> evaluation -> saves outputs/ and model/
    app.py               Flask web app that serves the trained model
    templates/index.html Front-end form for the Flask app
    data/                Dataset used (flood dataset.xlsx)
    model/               Trained model + scaler + feature list (.pkl, via joblib)
    outputs/             All plots, reports, and the model comparison table

outputs/  and  model/  and  data/ (top level)
    Copies of the same result files for convenience, without needing to
    open the code/ folder.

HOW TO RUN
==========
1. (Optional — a dataset is already included) Regenerate the dataset:
       python make_dataset.py

2. Run the full analysis + train models:
       python flood_prediction.py
   This re-creates every chart in outputs/ and re-trains/saves the model
   in model/.

3. Launch the prediction web app:
       python app.py
   Then open http://127.0.0.1:5000 in your browser and enter readings
   for Temp, Humidity, Cloud Cover, ANNUAL, Jan-Feb, Mar-May, Jun-Sep,
   Oct-Dec, avgjune, and sub to get a flood / no-flood prediction.

REQUIREMENTS
============
pip install numpy pandas matplotlib seaborn scikit-learn xgboost flask joblib openpyxl

RESULTS SUMMARY
================
Model                  Accuracy    ROC-AUC
Random Forest           0.8500      0.9275   <- selected as final model
Logistic Regression     0.8333      0.9319
XGBoost                 0.8333      0.9141

Monsoon-season rainfall (Jun-Sep), ANNUAL rainfall, and avgjune were the
strongest predictors of flood occurrence.
