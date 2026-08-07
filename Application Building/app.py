from flask import Flask, render_template, request
# used to run/serve our application
# render_template is used for rendering the html pages
# import load from joblib to load the saved model file
from joblib import load
import pandas as pd

# Create Flask app
app = Flask(__name__)

# load model file
model = load('floods.save')
sc = load('transform.save')

# Order must match the columns the scaler/model were trained on
FEATURE_COLUMNS = [
    "CLOUD_COVER",
    "ANNUAL_RAINFALL",
    "JAN_FEB_RAINFALL",
    "MAR_MAY_RAINFALL",
    "JUN_SEP_RAINFALL",
]


@app.route('/')  # rendering the html template
def home():
    return render_template('home.html')


@app.route('/Predict')  # rendering the html template
def index():
    return render_template('index.html')


@app.route('/predict', methods=['POST'])
def predict():
    # 1. Pull the five form fields sent by index.html
    cloud_cover = float(request.form['cloud_cover'])
    annual_rainfall = float(request.form['annual_rainfall'])
    jan_feb_rainfall = float(request.form['jan_feb_rainfall'])
    mar_may_rainfall = float(request.form['mar_may_rainfall'])
    jun_sep_rainfall = float(request.form['jun_sep_rainfall'])

    # 2. Structure the input into a DataFrame with the five features
    input_df = pd.DataFrame(
        [[cloud_cover, annual_rainfall, jan_feb_rainfall,
          mar_may_rainfall, jun_sep_rainfall]],
        columns=FEATURE_COLUMNS,
    )

    # 3. Scale the inputs with the same StandardScaler used at train time
    input_scaled = sc.transform(input_df)

    # 4. Predict with the saved model
    prediction = model.predict(input_scaled)[0]
    probability = model.predict_proba(input_scaled)[0][1]

    # 5. Route to the appropriate result page
    if prediction == 1:
        return render_template('chance.html', probability=round(probability * 100, 1))
    else:
        return render_template('no_chance.html', probability=round((1 - probability) * 100, 1))


if __name__ == '__main__':
    app.run(debug=True)
