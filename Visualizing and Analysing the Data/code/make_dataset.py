"""
Generates a synthetic 'flood dataset.xlsx' consistent with the columns/ranges
visible in the user's notebook screenshots:
Temp, Humidity, Cloud Cover, ANNUAL, Jan-Feb, Mar-May, Jun-Sep, Oct-Dec,
avgjune, sub, flood
"""
import numpy as np
import pandas as pd

np.random.seed(42)
n = 300

temp = np.round(np.random.normal(29.5, 1.2, n), 0).clip(25, 34)
humidity = np.round(np.random.normal(74, 4, n), 0).clip(60, 90)
cloud_cover = np.round(np.random.normal(38, 6, n), 0).clip(20, 55)

jan_feb = np.round(np.random.gamma(2.2, 12, n), 1).clip(2, 90)
mar_may = np.round(np.random.normal(330, 45, n), 1).clip(150, 450)
jun_sep = np.round(np.random.normal(2150, 260, n), 1).clip(1400, 2800)
oct_dec = np.round(np.random.normal(520, 130, n), 1).clip(150, 900)

annual = np.round(jan_feb + mar_may + jun_sep + oct_dec, 1)
avgjune = np.round(jun_sep / 8 + np.random.normal(0, 15, n), 1).clip(80, 400)
sub = np.round(oct_dec + jan_feb + np.random.normal(0, 40, n), 1).clip(150, 950)

# Flood risk driven mainly by heavy monsoon rainfall (Jun-Sep) and cloud cover,
# with some noise, converted to a binary label via a logistic threshold.
z = (jun_sep - 2150) / 260 * 1.4 + (cloud_cover - 38) / 6 * 0.6 + \
    (annual - annual.mean()) / annual.std() * 0.5 + np.random.normal(0, 1, n)
prob = 1 / (1 + np.exp(-z))
flood = (prob > 0.55).astype(int)

df = pd.DataFrame({
    "Temp": temp.astype(int),
    "Humidity": humidity.astype(int),
    "Cloud Cover": cloud_cover.astype(int),
    "ANNUAL": annual,
    "Jan-Feb": jan_feb,
    "Mar-May": mar_may,
    "Jun-Sep": jun_sep,
    "Oct-Dec": oct_dec,
    "avgjune": avgjune,
    "sub": sub,
    "flood": flood
})

# introduce a few missing values to demonstrate preprocessing (fillna with mean)
for col in ["Temp", "Humidity", "Cloud Cover"]:
    idx = np.random.choice(df.index, size=5, replace=False)
    df.loc[idx, col] = np.nan

df.to_excel("data/flood dataset.xlsx", index=False)
print(df.head())
print(df.shape)
print(df["flood"].value_counts())
