"""
app.py
------
Flask web application for real-time Flood Prediction.

Loads the pre-trained XGBoost model (floods.save) and the fitted
StandardScaler (transform.save), then serves a simple form where a user
can enter the flood-risk features and receive an instant prediction.
"""

from flask import Flask, render_template, request
import joblib
import numpy as np

app = Flask(__name__)

# Load model + scaler once at startup
model = joblib.load("models/floods.save")
scaler = joblib.load("models/transform.save")

FEATURE_ORDER = [
    "rainfall_mm",
    "water_level_m",
    "humidity_pct",
    "temperature_c",
    "river_discharge_m3s",
    "elevation_m",
    "drainage_quality",
    "upstream_dam_release",
]


@app.route("/", methods=["GET"])
def home():
    return render_template("index.html", prediction=None)


@app.route("/predict", methods=["POST"])
def predict():
    try:
        # 1. Collect user input in the correct feature order
        values = [float(request.form[feat]) for feat in FEATURE_ORDER]
        input_array = np.array(values).reshape(1, -1)

        # 2. Apply the same scaling used during training
        scaled_input = scaler.transform(input_array)

        # 3. Predict using the saved XGBoost model
        pred = model.predict(scaled_input)[0]
        proba = model.predict_proba(scaled_input)[0][1]

        result = "FLOOD LIKELY" if pred == 1 else "NO FLOOD EXPECTED"
        return render_template(
            "index.html",
            prediction=result,
            probability=f"{proba * 100:.1f}%",
        )
    except Exception as e:
        return render_template("index.html", prediction=f"Error: {e}")


if __name__ == "__main__":
    app.run(debug=True)
