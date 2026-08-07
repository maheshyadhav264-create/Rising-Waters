# 🌊 Flood Prediction — End-to-End ML Project

A complete machine learning project that predicts flood risk from meteorological,
hydrological, terrain and infrastructure factors — from data collection through
to a deployed Flask web application.

## Project structure

```
flood_project/
├── data/
│   ├── generate_dataset.py       # Epic 1 - builds the flood prediction dataset
│   └── flood_dataset.csv         # 6,000-record dataset (15 features + target)
├── notebook/
│   └── Flood_Prediction_Analysis.ipynb   # Epics 2-4: EDA, preprocessing, modeling
├── model/
│   ├── flood_model.pkl            # Best model (selected in Epic 4, Story 6)
│   ├── scaler.pkl                 # StandardScaler fitted on training data
│   ├── label_encoders.pkl         # LabelEncoders for categorical features
│   └── model_metadata.json        # Feature order, chosen model, metrics
├── app/                           # Epic 5 - Flask web application
│   ├── app.py
│   ├── templates/  (index.html, result.html, about.html, base.html)
│   └── static/     (style.css, img/)
├── results/                       # Saved charts & tables from the notebook run
├── docs/
│   └── Flood_Prediction_Project_Report.docx   # Full written project report
├── requirements.txt
└── README.md
```

## Epic → deliverable map

| Epic | Story | Where it lives |
|---|---|---|
| 1. Data Collection | 1.1 Download/load dataset | `data/generate_dataset.py`, `data/flood_dataset.csv`, notebook §Epic 1 |
| 2. Visualizing & Analysing Data | 2.1–2.5 | notebook §Epic 2, charts in `results/01`–`07` |
| 3. Data Pre-Processing | 3.1–3.5 | notebook §Epic 3 |
| 4. Model Building | 4.1–4.6 | notebook §Epic 4, `model/`, `results/09`–`13` |
| 5. Application Building | 5.1–5.3 | `app/` (Flask app + HTML templates) |

## How to run

### 1. Install dependencies
```bash
pip install -r requirements.txt
```

### 2. (Optional) Regenerate the dataset
The dataset is already included at `data/flood_dataset.csv`. To rebuild it:
```bash
cd data && python generate_dataset.py
```

### 3. (Optional) Re-run the analysis / retrain the model
Open and run all cells in `notebook/Flood_Prediction_Analysis.ipynb` (Jupyter/JupyterLab),
or from the command line:
```bash
cd notebook
jupyter nbconvert --to notebook --execute --inplace Flood_Prediction_Analysis.ipynb
```
This re-creates everything in `model/` and `results/`.

### 4. Run the web application
```bash
cd app
python app.py
```
Then open **http://127.0.0.1:5000** in a browser.

- **Assessment page (`/`)** — a grouped form for the 15 input factors; submitting it
  runs the trained model and shows a result.
- **Result page (`/predict`)** — flood probability, risk level (Low / Moderate / High)
  shown as a gauge, and a readout of the submitted values.
- **Model Report page (`/about`)** — comparison table and charts for all four trained models.

## Dataset

`flood_dataset.csv` (6,000 rows, 15 predictive features + `Flood_Occurred` target) was
generated to mirror the structure of public flood-risk datasets: it combines
meteorological readings (monsoon intensity, rainfall, temperature, humidity),
hydrological readings (river discharge, water level, drainage quality), terrain/land-use
factors (elevation, land cover, soil type) and infrastructure/history factors
(population density, infrastructure quality, historical flood count, deforestation and
climate-change indices). The target is derived from a weighted, noise-perturbed flood-risk
function, giving a realistic ~64/36 class split and genuine (not perfectly separable)
signal for the models to learn — the same setup you'd see analysing a real flood dataset.
Missing values and outliers were deliberately injected so Epic 3's cleaning steps have
real work to do.

## Model results

| Model | Accuracy | Precision | Recall | F1-Score | ROC-AUC |
|---|---|---|---|---|---|
| Decision Tree | 0.754 | 0.683 | 0.593 | 0.634 | 0.777 |
| Random Forest | 0.816 | 0.865 | 0.579 | 0.694 | 0.889 |
| KNN | 0.768 | 0.777 | 0.500 | 0.609 | 0.829 |
| **XGBoost (selected)** | **0.838** | **0.825** | **0.699** | **0.757** | **0.906** |

XGBoost was selected (Epic 4, Story 6) as it had the best F1-score and ROC-AUC, giving the
best balance between catching real flood events and avoiding false alarms.

## Notes

- The dataset is synthetically generated (see `data/generate_dataset.py`) rather than
  downloaded from an external source, since this environment cannot reach dataset hosting
  sites — but it is deliberately built to have the same structure, feature types, and
  realistic noise/imperfect-separability as a genuine flood-prediction dataset, so every
  notebook step (EDA, cleaning, encoding, scaling, modeling) does real, meaningful work.
- This is a demonstration project — the model and app are not intended for real-world
  emergency/disaster-response decisions.
