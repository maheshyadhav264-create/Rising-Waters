"""
Flask web app that serves the trained flood-prediction model.
Run: python app.py   then open http://127.0.0.1:5000
"""
from flask import Flask, render_template, request
import joblib
import numpy as np
import os

app = Flask(__name__)

MODEL_PATH = os.path.join("model", "flood_model.pkl")
SCALER_PATH = os.path.join("model", "scaler.pkl")
FEATURES_PATH = os.path.join("model", "feature_names.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
feature_names = joblib.load(FEATURES_PATH)


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", features=feature_names, result=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        values = [float(request.form.get(f, 0)) for f in feature_names]
        arr = np.array(values).reshape(1, -1)

        # Random Forest / XGBoost were trained on raw (unscaled) features;
        # Logistic Regression needs the scaler. This model-agnostic check
        # keeps the app working regardless of which model "flood_model.pkl" is.
        if hasattr(model, "coef_"):
            arr_in = scaler.transform(arr)
        else:
            arr_in = arr

        pred = model.predict(arr_in)[0]
        prob = model.predict_proba(arr_in)[0][1]
        result = {
            "label": "Flood Likely" if pred == 1 else "No Flood Expected",
            "probability": round(float(prob) * 100, 2)
        }
    except Exception as e:
        result = {"label": f"Error: {e}", "probability": None}

    return render_template("index.html", features=feature_names, result=result)


if __name__ == "__main__":
    app.run(debug=True)
