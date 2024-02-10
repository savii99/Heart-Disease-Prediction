from flask import Flask, render_template, request
import pickle
import numpy as np

app = Flask(__name__)

# Load the pre-trained Random Forest model
with open('random_forest_model.pkl', 'rb') as model_file:
    model = pickle.load(model_file)

with open('standard_scaler.pkl', 'rb') as model_file1:
    model1 = pickle.load(model_file1)

# Home page route
@app.route('/')
def home():
    return render_template('index.html')

# Prediction route
@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get user input from the form
        features = [float(request.form.get('age')),
                    float(request.form.get('sex')),
                    float(request.form.get('cp')),
                    float(request.form.get('trestbps')),
                    float(request.form.get('chol')),
                    float(request.form.get('fbs')),
                    float(request.form.get('restecg')),
                    float(request.form.get('thalach')),
                    float(request.form.get('exang')),
                    float(request.form.get('oldpeak')),
                    float(request.form.get('slope')),
                    float(request.form.get('ca')),
                    float(request.form.get('thal'))]

        # Transform the input features using the scaler
        features_scaled = model1.transform([features])

        # Make a prediction
        prediction = model.predict(features_scaled)[0]

        # Display the prediction
        if prediction == 1:
            result_text = "The patient is likely to have a heart disease."
        else:
            result_text = "The patient is unlikely to have a heart disease."

        return render_template('index.html', prediction=result_text)

    except Exception as e:
        # Handle exceptions, print for debugging purposes
        print(f"Error: {str(e)}")
        return render_template('index.html', prediction="Error occurred, please check input")


if __name__ == '__main__':
    app.run(debug=True)

