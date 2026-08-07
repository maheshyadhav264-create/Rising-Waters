# HDI Prediction System (Sample)

def predict_hdi(life_expectancy, mean_schooling, expected_schooling, gni):

```
score = (
    (life_expectancy / 85) * 0.3 +
    (mean_schooling / 15) * 0.2 +
    (expected_schooling / 18) * 0.2 +
    (gni / 100000) * 0.3
)

if score >= 0.8:
    category = "Very High"
elif score >= 0.7:
    category = "High"
elif score >= 0.55:
    category = "Medium"
else:
    category = "Low"

return round(score, 2), category
```

country = "India"

score, result = predict_hdi(
life_expectancy=70,
mean_schooling=7,
expected_schooling=12,
gni=7000
)

print("Country:", country)
print("Predicted HDI Score:", score)
print("Category:", result)
