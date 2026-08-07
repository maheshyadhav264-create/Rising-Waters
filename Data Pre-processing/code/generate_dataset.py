"""
Generates a synthetic flood-prediction dataset matching the column schema
used in the project (Temp, Humidity, Cloud Cover, ANNUAL, Jan-Feb, Mar-May,
Jun-Sep, Oct-Dec, avgjune, sub, flood).

NOTE: The original dataset file was not provided/uploaded, only screenshots
of code and column names. This script creates a realistic synthetic dataset
with the same structure so the full pipeline can be built, run, and
demonstrated end-to-end. Replace this file's output (dataset.csv) with the
real dataset to get results on actual data -- the rest of the pipeline
(preprocessing.py, train_model.py) will work unchanged as long as column
names match.
"""

import numpy as np
import pandas as pd

np.random.seed(42)

N = 1200

subdivisions = [
    "Assam & Meghalaya", "Bihar", "Coastal Andhra Pradesh", "Coastal Karnataka",
    "East Rajasthan", "Gangetic West Bengal", "Kerala", "Konkan & Goa",
    "Madhya Maharashtra", "Vidarbha"
]

sub = np.random.choice(subdivisions, size=N)

temp = np.random.normal(27, 3.5, N).round(1)
humidity = np.clip(np.random.normal(70, 12, N), 20, 100).round(1)
cloud_cover = np.clip(np.random.normal(5, 2, N), 0, 8).round(1)

jan_feb = np.clip(np.random.normal(20, 15, N), 0, None).round(1)
mar_may = np.clip(np.random.normal(60, 30, N), 0, None).round(1)
jun_sep = np.clip(np.random.normal(900, 300, N), 0, None).round(1)
oct_dec = np.clip(np.random.normal(80, 40, N), 0, None).round(1)

annual = (jan_feb + mar_may + jun_sep + oct_dec).round(1)
avgjune = np.clip(jun_sep / 4 + np.random.normal(0, 20, N), 0, None).round(1)

# Introduce some missing values to demonstrate missing-value handling
for col in [temp, humidity, cloud_cover]:
    idx = np.random.choice(N, size=int(N * 0.02), replace=False)
    col[idx] = np.nan

# Introduce a few extreme outliers to demonstrate IQR capping
outlier_idx = np.random.choice(N, size=15, replace=False)
jun_sep[outlier_idx] = jun_sep[outlier_idx] * np.random.uniform(3, 5, size=15)

# Flood risk driven mainly by monsoon (Jun-Sep) rainfall + humidity, plus noise
risk_score = (
    0.0025 * jun_sep + 0.03 * humidity + 0.02 * annual / 10
    - 0.5 * temp / 27 + np.random.normal(0, 1.5, N)
)
threshold = np.nanpercentile(risk_score, 65)
flood = (risk_score > threshold).astype(int)

df = pd.DataFrame({
    "Temp": temp,
    "Humidity": humidity,
    "Cloud Cover": cloud_cover,
    "ANNUAL": annual,
    "Jan-Feb": jan_feb,
    "Mar-May": mar_may,
    "Jun-Sep": jun_sep,
    "Oct-Dec": oct_dec,
    "avgjune": avgjune,
    "sub": sub,
    "flood": flood,
})

df.to_csv("dataset.csv", index=False)
print("Synthetic dataset created:", df.shape)
print(df.head())
print("\nFlood class balance:\n", df["flood"].value_counts())
