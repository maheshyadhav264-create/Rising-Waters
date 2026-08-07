"""
Epic 1 - Story 1: Data Collection
--------------------------------------------------------------------
Builds the Flood Prediction dataset used throughout this project.

The dataset simulates the kind of hydrological / environmental / socio-
economic factors that public flood-prediction datasets (e.g. Kaggle's
"Flood Prediction Dataset") contain: rainfall, river discharge, water
level, terrain, land cover, soil type, infrastructure quality, historic
flood record, etc. The target `Flood_Occurred` is generated from a
physically-plausible weighted risk function plus random noise, so the
downstream ML models have real, learnable signal (not pure noise) but
also realistic overlap/imperfect separability between classes.
--------------------------------------------------------------------
"""
import numpy as np
import pandas as pd

np.random.seed(42)
N = 6000

land_covers = ["Forest", "Urban", "Agricultural", "Wetland", "Barren"]
soil_types = ["Clay", "Sandy", "Loamy", "Silty"]
infra_quality = ["Poor", "Average", "Good"]

df = pd.DataFrame({
    "MonsoonIntensity": np.clip(np.random.normal(5, 2.2, N), 0, 10).round(2),
    "Rainfall_mm": np.clip(np.random.gamma(4, 40, N), 0, 600).round(1),
    "Temperature_C": np.clip(np.random.normal(27, 4, N), 10, 45).round(1),
    "Humidity_pct": np.clip(np.random.normal(70, 12, N), 20, 100).round(1),
    "River_Discharge_m3s": np.clip(np.random.gamma(3, 150, N), 0, 3000).round(1),
    "Water_Level_m": np.clip(np.random.normal(4.5, 2.0, N), 0, 15).round(2),
    "Elevation_m": np.clip(np.random.exponential(150, N), 0, 1200).round(1),
    "Land_Cover": np.random.choice(land_covers, N, p=[0.22, 0.28, 0.30, 0.12, 0.08]),
    "Soil_Type": np.random.choice(soil_types, N, p=[0.30, 0.25, 0.30, 0.15]),
    "Population_Density": np.clip(np.random.gamma(2, 400, N), 5, 12000).round(0),
    "Infrastructure_Quality": np.random.choice(infra_quality, N, p=[0.30, 0.45, 0.25]),
    "Drainage_Quality_Score": np.clip(np.random.normal(5, 2.3, N), 0, 10).round(2),
    "Historical_Floods_Count": np.random.poisson(1.1, N),
    "Deforestation_Index": np.clip(np.random.normal(5, 2.5, N), 0, 10).round(2),
    "Climate_Change_Index": np.clip(np.random.normal(5, 2.0, N), 0, 10).round(2),
})

# ---- introduce a modest amount of realistic missingness ----
for col, frac in [("Rainfall_mm", 0.03), ("Water_Level_m", 0.025),
                   ("Drainage_Quality_Score", 0.02), ("Infrastructure_Quality", 0.015),
                   ("Humidity_pct", 0.02)]:
    idx = np.random.choice(N, size=int(N * frac), replace=False)
    df.loc[idx, col] = np.nan

# ---- introduce a small number of extreme outliers (data-entry / sensor noise) ----
out_idx = np.random.choice(N, size=25, replace=False)
df.loc[out_idx, "Rainfall_mm"] = df.loc[out_idx, "Rainfall_mm"] * np.random.uniform(3, 6, 25)
out_idx2 = np.random.choice(N, size=15, replace=False)
df.loc[out_idx2, "River_Discharge_m3s"] = df.loc[out_idx2, "River_Discharge_m3s"] * np.random.uniform(3, 5, 15)

# ---- build a physically-plausible flood risk score to derive the target ----
infra_map = {"Poor": 8, "Average": 4, "Good": 1}
land_map = {"Urban": 6, "Barren": 5, "Agricultural": 3, "Wetland": 4, "Forest": 0}
soil_map = {"Clay": 6, "Silty": 4, "Loamy": 2, "Sandy": 0}

infra_filled = df["Infrastructure_Quality"].map(infra_map)
infra_filled = infra_filled.fillna(infra_filled.median())

risk = (
    0.28 * df["Rainfall_mm"].fillna(df["Rainfall_mm"].median()) / 100
    + 0.9 * df["MonsoonIntensity"]
    + 0.55 * df["River_Discharge_m3s"].fillna(df["River_Discharge_m3s"].median()) / 100
    + 1.1 * df["Water_Level_m"].fillna(df["Water_Level_m"].median())
    - 0.02 * df["Elevation_m"]
    + 0.7 * infra_filled
    + 0.5 * df["Land_Cover"].map(land_map)
    + 0.4 * df["Soil_Type"].map(soil_map)
    - 0.9 * df["Drainage_Quality_Score"].fillna(df["Drainage_Quality_Score"].median())
    + 1.3 * df["Historical_Floods_Count"]
    + 0.35 * df["Deforestation_Index"]
    + 0.3 * df["Climate_Change_Index"]
    + 0.0008 * df["Population_Density"]
)

risk_noisy = risk + np.random.normal(0, np.nanstd(risk) * 0.55, N)
threshold = np.nanpercentile(risk_noisy, 64)  # ~36% positive class -> realistic imbalance
df["Flood_Occurred"] = (risk_noisy > threshold).astype(int)

df = df.sample(frac=1, random_state=7).reset_index(drop=True)
df.insert(0, "Record_ID", range(1, N + 1))

df.to_csv("/home/claude/flood_project/data/flood_dataset.csv", index=False)
print("Dataset shape:", df.shape)
print(df["Flood_Occurred"].value_counts(normalize=True))
print(df.isna().sum()[df.isna().sum() > 0])
