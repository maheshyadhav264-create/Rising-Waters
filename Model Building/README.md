# Flood Prediction System

End-to-end machine learning project that trains, compares, and deploys
classification models to predict flood occurrence from hydrological and
weather readings.

## 1. Project Structure

```
flood_project/
├── data/
│   ├── generate_dataset.py     # builds the synthetic training dataset
│   └── flood_dataset.csv       # generated dataset (1,200 rows, 8 features)
├── models/
│   ├── floods.save             # trained XGBoost (GradientBoosting) model, joblib
│   └── transform.save          # fitted StandardScaler, joblib
├── results/
│   ├── training_log.txt        # full console output of the training run
│   ├── comparison_results.json # accuracy / precision / recall per model
│   └── confusion_matrices.txt  # confusion matrix + classification report per model
├── templates/
│   └── index.html              # Flask front-end form
├── train_models.py             # decisiontree(), randomForest(), KNN(), xgboost(), compareModel()
├── app.py                      # Flask deployment app
├── requirements.txt
└── README.md
```

> **Note on the dataset:** no source dataset was supplied for this project,
> so `data/generate_dataset.py` builds a synthetic but realistic
> flood-risk dataset (rainfall, water level, humidity, temperature, river
> discharge, elevation, drainage quality, upstream dam release) with a
> noisy, feature-dependent FLOOD label. Drop in the real project CSV as
> `data/flood_dataset.csv` (same column names, or update `FEATURE_ORDER`
> in `app.py`) and re-run `train_models.py` — nothing else changes.

## 2. Setup

```bash
pip install -r requirements.txt
```

## 3. Train the Models

```bash
python train_models.py
```

This will:
1. Load `data/flood_dataset.csv` and split it 80/20 into train/test sets.
2. Scale features with `StandardScaler`.
3. Train four classifiers — Decision Tree, Random Forest, KNN, XGBoost.
4. Print accuracy, confusion matrix, and classification report for each.
5. Run `compareModel()` to rank all four side-by-side.
6. Save the XGBoost model to `models/floods.save` and the scaler to
   `models/transform.save` with Joblib, and write results to `results/`.

## 4. Run the Web App

```bash
python app.py
```

Then open `http://127.0.0.1:5000` in a browser, enter the eight input
readings, and submit to get a flood / no-flood prediction with the
model's confidence score.

## 5. Model Comparison (this run)

| Model         | Accuracy | Precision | Recall |
|---------------|----------|-----------|--------|
| Decision Tree | 0.7750   | 0.6324    | 0.5972 |
| Random Forest | 0.8542   | 0.8364    | 0.6389 |
| KNN           | 0.8417   | 0.8148    | 0.6111 |
| **XGBoost**   | **0.8750** | **0.8500** | **0.7083** |

XGBoost (implemented here with `GradientBoostingClassifier`, which
requires no external `xgboost` binary) is the best performer and is the
model saved for deployment, consistent with the original project design.

## 6. Deployment Notes

- `floods.save` and `transform.save` are loaded once at Flask startup.
- Incoming form values are scaled with the *same* fitted scaler used
  during training before being passed to the model — this consistency is
  essential; predicting on unscaled input will produce wrong results.
- For production, run behind a WSGI server (e.g. `gunicorn app:app`)
  instead of the Flask development server.
