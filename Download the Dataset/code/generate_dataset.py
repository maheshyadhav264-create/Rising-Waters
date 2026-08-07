"""
generate_dataset.py
--------------------
Generates a synthetic meteorological dataset for flood prediction,
matching the feature structure referenced by the user (Temp, Humidity,
Cloud Cover, monthly rainfall Jun-Sep, ANNUAL rainfall, etc.), since the
original Kaggle dataset (arbethi/rainfall-dataset) requires an
authenticated Kaggle session and could not be downloaded directly.

The generation logic encodes realistic meteorological relationships:
 - Higher humidity & cloud cover -> higher rainfall
 - Monsoon months (Jun-Sep) contribute most of ANNUAL rainfall
 - FLOOD occurs when ANNUAL rainfall and JUN-SEP rainfall exceed
   region-adjusted thresholds, with some noise for realism
"""

import numpy as np
import pandas as pd

np.random.seed(42)
N = 1200

# --- Base meteorological features ---
temp = np.round(np.random.normal(29, 2.2, N), 1).clip(18, 38)
humidity = np.round(np.random.normal(74, 6, N), 0).clip(45, 98)
cloud_cover = np.round(np.random.normal(38, 8, N), 0).clip(10, 90)

# Monthly rainfall (mm) — monsoon months (Jun-Sep) driven by humidity/cloud cover
def month_rain(base, hum, cc, noise_scale):
    val = base + 2.2 * (hum - 74) + 1.8 * (cc - 38) + np.random.normal(0, noise_scale, N)
    return val.clip(0, None)

jan = month_rain(15, humidity, cloud_cover, 8)
feb = month_rain(18, humidity, cloud_cover, 8)
mar = month_rain(22, humidity, cloud_cover, 10)
apr = month_rain(35, humidity, cloud_cover, 12)
may = month_rain(70, humidity, cloud_cover, 20)
jun = month_rain(260, humidity, cloud_cover, 55)
jul = month_rain(310, humidity, cloud_cover, 60)
aug = month_rain(290, humidity, cloud_cover, 58)
sep = month_rain(200, humidity, cloud_cover, 45)
oct_ = month_rain(90, humidity, cloud_cover, 25)
nov = month_rain(30, humidity, cloud_cover, 12)
dec = month_rain(18, humidity, cloud_cover, 8)

jun_sep = jun + jul + aug + sep
annual = jan+feb+mar+apr+may+jun+jul+aug+sep+oct_+nov+dec

# --- Flood label ---
# Flood risk rises sharply once JUN-SEP rainfall and ANNUAL rainfall both
# cross high thresholds; humidity/cloud cover add compounding risk.
risk_score = (
    0.0035 * (annual - annual.mean()) +
    0.004 * (jun_sep - jun_sep.mean()) +
    0.03 * (humidity - humidity.mean()) +
    0.02 * (cloud_cover - cloud_cover.mean()) +
    np.random.normal(0, 1.0, N)
)
prob_flood = 1 / (1 + np.exp(-risk_score))
flood = (prob_flood > 0.5).astype(int)

df = pd.DataFrame({
    "TEMP": temp,
    "HUMIDITY": humidity,
    "CLOUD_COVER": cloud_cover,
    "JAN": jan.round(1), "FEB": feb.round(1), "MAR": mar.round(1), "APR": apr.round(1),
    "MAY": may.round(1), "JUN": jun.round(1), "JUL": jul.round(1), "AUG": aug.round(1),
    "SEP": sep.round(1), "OCT": oct_.round(1), "NOV": nov.round(1), "DEC": dec.round(1),
    "JUN_SEP": jun_sep.round(1),
    "ANNUAL": annual.round(1),
    "FLOOD": flood
})

# Inject a small amount of missing data to mirror real-world datasets
for col in ["HUMIDITY", "CLOUD_COVER", "ANNUAL"]:
    idx = np.random.choice(df.index, size=int(0.02 * N), replace=False)
    df.loc[idx, col] = np.nan

df.to_csv("data/flood_dataset.csv", index=False)
print("Saved data/flood_dataset.csv with shape:", df.shape)
print(df["FLOOD"].value_counts(normalize=True))
