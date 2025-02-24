from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd
import numpy as np
import joblib
import os

app = Flask(__name__, template_folder='../templates')

# Load the trained model and scaler
try:
    with open('models/model.pkl', 'rb') as f:
        model = pickle.load(f)
    scaler = joblib.load('models/scaler.pkl')
except Exception as e:
    print(f"Error loading model or scaler: {str(e)}")
    model = None
    scaler = None

def preprocess_input(data):
    """Preprocess input data similar to training pipeline"""
    try:
        # Convert input to DataFrame
        df = pd.DataFrame([data])
        
        # Handle missing values
        zero_columns = ['Glucose', 'BloodPressure', 'SkinThickness', 'Insulin', 'BMI']
        df[zero_columns] = df[zero_columns].replace(0, np.nan)
        
        # Fill missing values with medians (using predefined values from training)
        medians = {
            'Glucose': 117.0,
            'BloodPressure': 72.0,
            'SkinThickness': 23.0,
            'Insulin': 30.5,
            'BMI': 32.0
        }
        
        for column in zero_columns:
            df[column] = df[column].fillna(medians[column])
        
        # Scale features
        if scaler is not None:
            df_scaled = scaler.transform(df)
            df_scaled = pd.DataFrame(df_scaled, columns=df.columns)
            return df_scaled
        return df
    
    except Exception as e:
        raise ValueError(f"Error in preprocessing: {str(e)}")

def validate_input(data):
    """Validate input data"""
    required_fields = [
        'Pregnancies', 'Glucose', 'BloodPressure', 'SkinThickness',
        'Insulin', 'BMI', 'DiabetesPedigreeFunction', 'Age'
    ]
    
    # Check if all fields are present
    for field in required_fields:
        if field not in data:
            raise ValueError(f"Missing required field: {field}")
    
    # Validate ranges
    validations = {
        'Pregnancies': (0, 20),
        'Glucose': (0, 500),
        'BloodPressure': (0, 300),
        'SkinThickness': (0, 100),
        'Insulin': (0, 1000),
        'BMI': (0, 100),
        'DiabetesPedigreeFunction': (0, 3),
        'Age': (0, 120)
    }
    
    for field, (min_val, max_val) in validations.items():
        value = float(data[field])
        if not min_val <= value <= max_val:
            raise ValueError(f"{field} must be between {min_val} and {max_val}")

@app.route('/')
def home():
    if model is None:
        return render_template('error.html', error="Model not loaded. Please check server logs.")
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Get data from the form
        data = {
            'Pregnancies': float(request.form['Pregnancies']),
            'Glucose': float(request.form['Glucose']),
            'BloodPressure': float(request.form['BloodPressure']),
            'SkinThickness': float(request.form['SkinThickness']),
            'Insulin': float(request.form['Insulin']),
            'BMI': float(request.form['BMI']),
            'DiabetesPedigreeFunction': float(request.form['DiabetesPedigreeFunction']),
            'Age': float(request.form['Age'])
        }
        
        # Validate input
        validate_input(data)
        
        # Preprocess input
        input_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Get confidence score
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
        
        # Prepare result
        result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
        
        return render_template('index.html', 
                             prediction_text=f'Patient is {result}',
                             confidence=f'Confidence: {confidence:.2%}',
                             input_values=data)
    
    except ValueError as ve:
        error_message = str(ve)
        return render_template('index.html', error=error_message, input_values=data)
    
    except Exception as e:
        error_message = f"An error occurred: {str(e)}"
        return render_template('index.html', error=error_message)

@app.route('/api/predict', methods=['POST'])
def predict_api():
    try:
        # Get JSON data
        data = request.get_json()
        
        # Validate input
        validate_input(data)
        
        # Preprocess input
        input_data = preprocess_input(data)
        
        # Make prediction
        prediction = model.predict(input_data)
        prediction_proba = model.predict_proba(input_data)
        
        # Get confidence score
        confidence = prediction_proba[0][1] if prediction[0] == 1 else prediction_proba[0][0]
        
        return jsonify({
            'status': 'success',
            'prediction': int(prediction[0]),
            'prediction_label': "Diabetic" if prediction[0] == 1 else "Not Diabetic",
            'confidence': float(confidence)
        })
    
    except Exception as e:
        return jsonify({
            'status': 'error',
            'message': str(e)
        }), 400

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=True)