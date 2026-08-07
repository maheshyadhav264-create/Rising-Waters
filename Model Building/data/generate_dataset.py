"""
generate_dataset.py
--------------------
Generates a synthetic but realistic flood-prediction dataset.

NOTE: No original dataset was supplied with this project, so this script
builds a stand-in dataset using domain-plausible feature ranges (rainfall,
water level, humidity, temperature, river discharge, etc.) with a labeled
FLOOD target that depends on those features in a realistic, noisy way.
Replace this file / the resulting CSV with the real project dataset when
available; the rest of the pipeline (train_models.py, app.py) does not
need to change as long as column names match.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200

rainfall_mm = np.random.gamma(shape=2.0, scale=40, size=N)          # 0 - ~400 mm
water_level_m = np.random.normal(4.0, 1.5, N).clip(0.5, 10)         # river level
humidity_pct = np.random.normal(70, 12, N).clip(20, 100)
temperature_c = np.random.normal(28, 4, N).clip(10, 45)
river_discharge = np.random.gamma(2.0, 150, N)                      # m3/s
elevation_m = np.random.normal(50, 30, N).clip(0, 300)
drainage_quality = np.random.randint(1, 6, N)                       # 1 (poor) - 5 (excellent)
upstream_dam_release = np.random.gamma(1.5, 20, N)

# latent "flood risk score" combining the drivers (higher = more risk)
risk_score = (
    0.045 * rainfall_mm
    + 0.9 * water_level_m
    + 0.02 * humidity_pct
    + 0.01 * river_discharge
    - 0.03 * elevation_m
    - 0.6 * drainage_quality
    + 0.05 * upstream_dam_release
    + np.random.normal(0, 2.0, N)  # noise
)

threshold = np.percentile(risk_score, 70)  # ~30% flood-positive class (imbalanced, like the real world)
flood = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "rainfall_mm": rainfall_mm.round(2),
    "water_level_m": water_level_m.round(2),
    "humidity_pct": humidity_pct.round(2),
    "temperature_c": temperature_c.round(2),
    "river_discharge_m3s": river_discharge.round(2),
    "elevation_m": elevation_m.round(2),
    "drainage_quality": drainage_quality,
    "upstream_dam_release": upstream_dam_release.round(2),
    "FLOOD": flood,
})

df.to_csv("/home/claude/flood_project/data/flood_dataset.csv", index=False)
print("[INFO] Dataset generated:", df.shape)
print(df["FLOOD"].value_counts())
