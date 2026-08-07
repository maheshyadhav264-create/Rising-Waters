"""
Epic 5 - Story 2: Build the Flask application and integrate the trained flood
prediction model.

Loads the model / scaler / label encoders / metadata produced by the notebook
(Epic 4) and serves a small web app: a form to enter the flood-risk factors,
and a result page showing the prediction + probability.
"""
import json
import os
import pickle

import numpy as np
import pandas as pd
from flask import Flask, render_template, request

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_DIR = os.path.join(BASE_DIR, "..", "model")

app = Flask(__name__)

# ---------------------------------------------------------------------------
# Load trained artifacts once, at startup
# ---------------------------------------------------------------------------
with open(os.path.join(MODEL_DIR, "flood_model.pkl"), "rb") as f:
    model = pickle.load(f)

with open(os.path.join(MODEL_DIR, "scaler.pkl"), "rb") as f:
    scaler = pickle.load(f)

with open(os.path.join(MODEL_DIR, "label_encoders.pkl"), "rb") as f:
    label_encoders = pickle.load(f)

with open(os.path.join(MODEL_DIR, "model_metadata.json"), "r") as f:
    metadata = json.load(f)

FEATURE_ORDER = metadata["feature_order"]
CATEGORICAL_COLS = metadata["categorical_columns"]
USES_SCALED_INPUT = metadata["uses_scaled_input"]
BEST_MODEL_NAME = metadata["best_model_name"]
MODEL_METRICS = metadata["metrics"]

CATEGORY_OPTIONS = {col: list(label_encoders[col].classes_) for col in CATEGORICAL_COLS}

NUMERIC_FIELD_CONFIG = {
    "MonsoonIntensity":        {"label": "Monsoon Intensity (0-10)",           "min": 0,  "max": 10,   "step": 0.1, "default": 5},
    "Rainfall_mm":             {"label": "Rainfall (mm)",                       "min": 0,  "max": 600,  "step": 1,   "default": 150},
    "Temperature_C":           {"label": "Temperature (°C)",                    "min": 10, "max": 45,   "step": 0.1, "default": 27},
    "Humidity_pct":            {"label": "Humidity (%)",                        "min": 20, "max": 100,  "step": 0.1, "default": 70},
    "River_Discharge_m3s":     {"label": "River Discharge (m³/s)",              "min": 0,  "max": 3000, "step": 1,   "default": 400},
    "Water_Level_m":           {"label": "Water Level (m)",                     "min": 0,  "max": 15,   "step": 0.1, "default": 4.5},
    "Elevation_m":             {"label": "Elevation (m)",                       "min": 0,  "max": 1200, "step": 1,   "default": 150},
    "Population_Density":      {"label": "Population Density (people/km²)",     "min": 5,  "max": 12000,"step": 1,   "default": 800},
    "Drainage_Quality_Score":  {"label": "Drainage Quality Score (0-10)",       "min": 0,  "max": 10,   "step": 0.1, "default": 5},
    "Historical_Floods_Count": {"label": "Historical Floods (past record)",     "min": 0,  "max": 15,   "step": 1,   "default": 1},
    "Deforestation_Index":     {"label": "Deforestation Index (0-10)",          "min": 0,  "max": 10,   "step": 0.1, "default": 5},
    "Climate_Change_Index":    {"label": "Climate Change Index (0-10)",         "min": 0,  "max": 10,   "step": 0.1, "default": 5},
}


FIELD_GROUPS = [
    {"title": "Meteorological", "fields": ["MonsoonIntensity", "Rainfall_mm", "Temperature_C", "Humidity_pct"]},
    {"title": "Hydrological", "fields": ["River_Discharge_m3s", "Water_Level_m", "Drainage_Quality_Score"]},
    {"title": "Terrain & Land Use", "fields": ["Elevation_m", "Land_Cover", "Soil_Type"]},
    {"title": "Infrastructure & History", "fields": ["Population_Density", "Infrastructure_Quality",
                                                       "Historical_Floods_Count", "Deforestation_Index",
                                                       "Climate_Change_Index"]},
]


def build_field(name):
    if name in CATEGORICAL_COLS:
        return {"name": name, "type": "select", "label": name.replace("_", " "),
                "options": CATEGORY_OPTIONS[name]}
    cfg = NUMERIC_FIELD_CONFIG.get(name, {"label": name, "min": 0, "max": 100, "step": 1, "default": 0})
    return {"name": name, "type": "number", "label": cfg["label"],
            "min": cfg["min"], "max": cfg["max"], "step": cfg["step"], "default": cfg["default"]}


def build_form_groups():
    """Grouped field descriptors used to render the HTML form section by section."""
    return [{"title": g["title"], "fields": [build_field(n) for n in g["fields"]]} for g in FIELD_GROUPS]


@app.route("/", methods=["GET"])
def index():
    return render_template("index.html", groups=build_form_groups(), active="home",
                            model_name=BEST_MODEL_NAME, metrics=MODEL_METRICS)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        row = {}
        for name in FEATURE_ORDER:
            raw_value = request.form.get(name)
            if name in CATEGORICAL_COLS:
                row[name] = raw_value
            else:
                row[name] = float(raw_value)

        input_df = pd.DataFrame([row], columns=FEATURE_ORDER)

        # Encode categoricals with the SAME label encoders fit during training
        for col in CATEGORICAL_COLS:
            le = label_encoders[col]
            input_df[col] = le.transform(input_df[col])

        model_input = input_df
        if USES_SCALED_INPUT:
            model_input = pd.DataFrame(scaler.transform(input_df), columns=input_df.columns)

        prediction = int(model.predict(model_input)[0])
        probability = float(model.predict_proba(model_input)[0][1])

        if probability >= 0.7:
            risk_level, risk_class = "High Risk", "risk-high"
        elif probability >= 0.4:
            risk_level, risk_class = "Moderate Risk", "risk-moderate"
        else:
            risk_level, risk_class = "Low Risk", "risk-low"

        return render_template(
            "result.html",
            active="home",
            prediction=prediction,
            probability=round(probability * 100, 1),
            risk_level=risk_level,
            risk_class=risk_class,
            model_name=BEST_MODEL_NAME,
            input_values=row,
        )
    except Exception as exc:  # pragma: no cover - defensive UI error path
        return render_template("result.html", active="home", error=str(exc))


@app.route("/about", methods=["GET"])
def about():
    return render_template("about.html", active="about", model_name=BEST_MODEL_NAME, metrics=MODEL_METRICS,
                            all_results=metadata["all_model_results"])


if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=5000)
