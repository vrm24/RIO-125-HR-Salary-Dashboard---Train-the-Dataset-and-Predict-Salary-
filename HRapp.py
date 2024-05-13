# HRapp.py
from flask import Flask, render_template, request
import pickle

app = Flask(__name__)

# Load the model
model = pickle.load(open('HRModel.pkl', 'rb'))

import pandas as pd


# Load the dataset
dataset = pd.read_csv('salary_data.csv')

# Extract columns and unique values
columns = dataset.columns.tolist()
unique_values = {col: dataset[col].unique().tolist() for col in columns}

@app.route('/')
def home():
    return render_template('index1.html', columns=columns, unique_values=unique_values)


import traceback
from sklearn.preprocessing import OneHotEncoder
from sklearn.compose import ColumnTransformer
from sklearn.preprocessing import StandardScaler

@app.route('/predict', methods=['POST', 'GET'])
def predict():
    if request.method == 'POST':
        try:
            # Define the column transformer for one-hot encoding
            categorical_features = ['workclass', 'education', 'marital_status', 'occupation', 'relationship', 'race', 'sex', 'native_country']
            categorical_transformer = OneHotEncoder(handle_unknown='ignore')

            # Define the column transformer for scaling numerical features
            numerical_features = ['age', 'hours_per_week']
            numerical_transformer = StandardScaler()

            # Combine transformers
            preprocessor = ColumnTransformer(
                transformers=[
                    ('cat', categorical_transformer, categorical_features),
                    ('num', numerical_transformer, numerical_features)
                ])

            # Fit the preprocessor with dummy data (you can replace this with actual data)
            dummy_data = [[0, 'Private', 'Bachelors', 'Married-civ-spouse', 'Tech-support', 'Husband', 'White', 'Male', 40, 'United-States']]
            preprocessor.fit(dummy_data)

            # Get input values from the form
            age = float(request.form['age'])  # Convert to float
            workclass = request.form['workclass']
            education = request.form['education']
            marital_status = request.form['marital_status']
            occupation = request.form['occupation']
            relationship = request.form['relationship']
            race = request.form['race']
            sex = request.form['sex']
            hours_per_week = float(request.form['hours_per_week'])  # Convert to float
            native_country = request.form['native_country']

            # Preprocess the input data
            input_data = [[age, workclass, education, marital_status, occupation, relationship, race, sex, hours_per_week, native_country]]
            input_data_preprocessed = preprocessor.transform(input_data)

            # Predict using the preprocessed data
            result = model.predict(input_data_preprocessed)[0]

            return render_template('index1.html', result=result)

        except Exception as e:
            traceback.print_exc()  # Print the traceback to console for debugging
            return 'An error occurred: ' + str(e), 500  # Return an error response with status code 500

    else:
        # If the request method is not POST, return an empty result
        result = ''
        return render_template('index1.html', result=result)

if __name__ == '__main__':
    app.run(port=3333, debug=True)  # Enable debugging mode








