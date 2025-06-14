import requests 
from flask import Flask, render_template, request 
import joblib 
import pandas as pd 
import numpy as np 

# Initialize the Flask app 
app = Flask(__name__) 

# Load the trained models 
try: 
    cardio_model = joblib.load("model_cardio_h.joblib") 
    diabetes_model = joblib.load('model_diabetes_h.joblib') 
    stroke_model = joblib.load("model_stroke_h.joblib") 
    print("Models loaded successfully.") 
except FileNotFoundError as e: 
    print(f"Error loading model: {e}. Please ensure model files are in the correct path.") 
    exit() 

# Define mappings 
gender_mapping = {'Male': 1, 'Female': 0} 
smoking_status_mapping = {'No': 0, 'Yes': 1, 'Former smoker': 2} 
marital_status_mapping = {'Divorced': 0, 'Single': 0, 'Married': 1, 'Widow': 2} 
working_status_mapping = { 
    'Homemaker': 0, 'Unemployed': 0, 'Retired': 0, 
    'Private': 1, 'Self-employed': 1, 'Student': 2, 
    'Working': 3, 'Public': 4 
} 
binary_mapping = {'Yes': 1, 'No': 0} 

def calculate_bmi(weight, height): 
    if height == 0: 
        return 0.0 
    return weight / (height ** 2) 

def convert_float32_to_float(data): 
    for key, value in data.items(): 
        if isinstance(value, np.float32): 
            data[key] = float(value) 
    return data 

def preprocess_input_for_cardio(gender, age, weight, height, hypertension, heart_disease, glucose_val, smoking_status, alcohol): 
    gender_encoded = gender_mapping[gender] 
    smoking_status_encoded = smoking_status_mapping[smoking_status] 
    alcohol_encoded = binary_mapping[alcohol] 
    heart_disease_encoded = binary_mapping[heart_disease] 
    bmi = calculate_bmi(weight, height) 
    ap_hi, ap_lo = (140, 90) if hypertension == 'Yes' else (120, 80) 

    return pd.DataFrame({ 
        'Age': [age], 
        'Gender': [gender_encoded], 
        'BMI': [bmi], 
        'Height': [height], 
        'ap_hi': [ap_hi], 
        'ap_lo': [ap_lo], 
        'High Cholesterol': [heart_disease_encoded], 
        'Glucose': [glucose_val], 
        'Smoking Status': [smoking_status_encoded], 
        'Alcohol Intake': [alcohol_encoded], 
    }) 

def preprocess_input_for_diabetes(gender, age, heart_disease, smoking_status, weight, height, glucose_val): 
    gender_encoded = gender_mapping[gender] 
    heart_disease_encoded = binary_mapping[heart_disease] 
    smoking_status_encoded = smoking_status_mapping[smoking_status] 
    bmi = calculate_bmi(weight, height) 

    return pd.DataFrame({ 
        'Gender': [gender_encoded], 
        'Age': [age], 
        'Heart Disease': [heart_disease_encoded], 
        'Smoking Status': [smoking_status_encoded], 
        'Weight': [weight], 
        'Height': [height], 
        'BMI': [bmi], 
        'Glucose': [glucose_val], 
    }) 

def preprocess_input_for_stroke(gender, age, heart_disease, marital_status, working_status, glucose_val, weight, height, smoking_status): 
    gender_encoded = gender_mapping[gender] 
    heart_disease_encoded = binary_mapping[heart_disease] 
    marital_status_encoded = marital_status_mapping[marital_status] 
    working_status_encoded = working_status_mapping[working_status] 
    smoking_status_encoded = smoking_status_mapping[smoking_status] 
    bmi = calculate_bmi(weight, height) 

    return pd.DataFrame({ 
        'Gender': [gender_encoded], 
        'Age': [age], 
        'Heart Disease': [heart_disease_encoded], 
        'Marital Status': [marital_status_encoded], 
        'Working Status': [working_status_encoded], 
        'Glucose': [glucose_val], 
        'Weight': [weight], 
        'Height': [height], 
        'BMI': [bmi], 
        'Smoking Status': [smoking_status_encoded], 
    }) 

def make_predictions(model, input_data): 
    predicted_prob = model.predict_proba(input_data)[:, 1] 
    return predicted_prob[0] * 100 

@app.route('/', methods=['GET']) 
def index(): 
    return render_template('index.html') 

@app.route("/predict", methods=["POST"]) 
def predict(): 
    if request.method == "POST": 
        name = request.form['name'] 
        address = request.form['address'] 
        contact_number = request.form['contact_number'] 
        gender = request.form['gender'] 
        age = int(request.form['age']) 
        weight = float(request.form['weight']) 
        height = float(request.form['height']) 
        marital_status = request.form['marital_status'] 
        working_status = request.form['working_status'] 
        exercise = request.form.get('exercise') 
        hypertension = request.form['hypertension'] 
        heart_disease = request.form['heart_disease'] 
        diabetes = request.form['diabetes'] 
        blood_glucose_level = float(request.form['blood_glucose_level']) 
        alcohol = request.form.get('alcohol') 
        smoking_status = request.form['smoking_status'] 

        bmi = round(calculate_bmi(weight, height), 1) 

        cardio_input = preprocess_input_for_cardio(gender, age, weight, height, hypertension, heart_disease, blood_glucose_level, smoking_status, alcohol) 
        diabetes_input = preprocess_input_for_diabetes(gender, age, heart_disease, smoking_status, weight, height, blood_glucose_level) 
        stroke_input = preprocess_input_for_stroke(gender, age, heart_disease, marital_status, working_status, blood_glucose_level, weight, height, smoking_status) 

        cardio_prob = round(make_predictions(cardio_model, cardio_input), 2) 
        diabetes_prob = round(make_predictions(diabetes_model, diabetes_input), 2) 
        stroke_prob = round(make_predictions(stroke_model, stroke_input), 2) 

        data = { 
            'name': name, 
            'address': address, 
            'contact_number': contact_number, 
            'gender': gender, 
            'age': age, 
            'weight': weight, 
            'height': height, 
            'bmi': bmi, 
            'marital_status': marital_status, 
            'working_status': working_status, 
            'exercise': exercise, 
            'hypertension': hypertension, 
            'heart_disease': heart_disease, 
            'diabetes': diabetes, 
            'blood_glucose_level': blood_glucose_level, 
            'alcohol': alcohol, 
            'smoking_status': smoking_status, 
            'cardio_prob': cardio_prob, 
            'diabetes_prob': diabetes_prob, 
            'stroke_prob': stroke_prob 
        } 

        data = convert_float32_to_float(data) 

        url = 'https://script.google.com/macros/s/AKfycbx4uDdoLcWeHbGC-K0QkgzUz-h-e5VTHjl2mQKH4vtRt9GUzKWekcFtz3kurfgGv48eOg/exec' 

        try: 
            response = requests.post(url, json=data) 
            print(f"Response status code: {response.status_code}") 
            print(f"Response content: {response.text}") 
            if response.status_code == 200: 
                print("Data successfully sent to Google Sheets.") 
            else: 
                print("Failed to send data to Google Sheets.") 
        except requests.exceptions.RequestException as e: 
            print(f"Request to Google Sheets failed: {e}") 

        return render_template("index.html", 
                                gender=gender, 
                                age=age, 
                                weight=weight, 
                                height=height, 
                                bmi=bmi, 
                                marital_status_status=marital_status, 
                                working_status=working_status, 
                                exercise=exercise, 
                                hypertension=hypertension, 
                                heart_disease=heart_disease, 
                                diabetes=diabetes, 
                                blood_glucose_level=blood_glucose_level, 
                                alcohol=alcohol, 
                                smoking_status=smoking_status, 
                                cardio_prob=cardio_prob, 
                                diabetes_prob=diabetes_prob, 
                                stroke_prob=stroke_prob) 

    return render_template("index.html") 

if __name__ == "__main__": 
    app.run(debug=True)