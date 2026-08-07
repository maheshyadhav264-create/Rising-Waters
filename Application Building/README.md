# Floods Prediction

A small Flask web application that predicts flood risk from five seasonal
weather readings (cloud cover + four rainfall figures) using a trained
scikit-learn classifier.

Full write-up, architecture, and results are in
**`Floods_Prediction_Documentation.docx`**.

## Quick start

```bash
pip install -r requirements.txt
python app.py
```

Open the URL printed in the terminal (typically `http://127.0.0.1:5000/`).

## Retraining the model

```bash
cd model_training
python train_model.py
```

This regenerates `floods.save` and `transform.save` in the project root.
The script currently trains on a generated synthetic dataset — point it at
your own historical rainfall/flood CSV (same five feature columns + a
`FLOOD` target) for production use.

## Project layout

```
app.py                  Flask app (routes + prediction logic)
floods.save              Trained classifier (joblib)
transform.save            Fitted StandardScaler (joblib)
requirements.txt
templates/               home.html, index.html, chance.html, no_chance.html
static/                  main.css, main.js
model_training/          train_model.py, synthetic_flood_data.csv, metrics_report.txt
screenshots/             Rendered UI + model result images
Floods_Prediction_Documentation.docx   Full project report
```
