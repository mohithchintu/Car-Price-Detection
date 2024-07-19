from flask import Flask, render_template
from flask import request
import numpy as np
import pandas as pd
import pickle

car = pd.read_csv('cleaned_car.csv')
app = Flask(__name__)
model = pickle.load(open("LinearRegressionModel.pkl", "rb"))

@app.route('/')
def index():
    car_companies = sorted(car['company'].unique())
    car_models = sorted(car['name'].unique())
    years = sorted(car['year'].unique(), reverse=True)
    fuel_type = car['fuel_type'].unique()
    data = {
        'companies': car_companies,
        'models': car_models,
        'years': years,
        'fuel_types': fuel_type,
    }
    return render_template('index.html', data = data)
@@app.route('/predict', methods=['POST'])
def predict():
    company = request.form.get('company')
    car_model = request.form.get('model')
    year = int(request.form.get('year'))
    fuel_type = request.form.get('fuel_type')
    kms_driven = int(request.form.get('kms_driven'))

    # Check if any of the required fields are missing
    if not (company and car_model and year and fuel_type and kms_driven):
        return 'Insufficient Details'

    # Check if the car model and company exist in the dataset
    if car_model not in car['name'].values or company not in car['company'].values:
        return 'Invalid Car Model or Company'

    # Check if the year and kms_driven are within expected ranges
    if year not in car['year'].values or kms_driven <= 0:
        return 'Invalid Year or Kms Driven'

    # Create a DataFrame with the input data
    data = pd.DataFrame([[car_model, company, year, fuel_type, kms_driven]],
                        columns=['name', 'company', 'year', 'fuel_type', 'kms_driven'])

    # Predict using the model
    prediction = model.predict(data)

    # Round the prediction to two decimal places
    prediction = np.round(prediction[0], 2)

    return str(prediction)


if __name__ == '__main__':
    app.run(debug=True)
