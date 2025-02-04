from flask import Flask, request, jsonify, render_template
import pickle
import pandas as pd

app = Flask(__name__, template_folder='../templates')

# Load the trained model
with open('models/model.pkl', 'rb') as f:
    model = pickle.load(f)

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

    # Convert data into a DataFrame
    input_data = pd.DataFrame([data])

    # Make prediction
    prediction = model.predict(input_data)

    # Return the result
    result = "Diabetic" if prediction[0] == 1 else "Not Diabetic"
    return render_template('index.html', prediction_text=f'Patient is {result}')

if __name__ == '__main__':
    app.run(debug=True)