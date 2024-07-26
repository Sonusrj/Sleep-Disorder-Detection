from flask import Flask, request, render_template
import pickle
import numpy as np
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

app = Flask(__name__)

# Load the trained model and preprocessing objects
try:
    with open('model.pkl', 'rb') as model_file:
        model = pickle.load(model_file)
except Exception as e:
    print(f"Error loading the model: {e}")
    model = None

# Load LabelEncoders and Scaler if available
try:
    gender_encoder = pickle.load(open('Gender_label_encoder.pkl', 'rb'))
    occupation_encoder = pickle.load(open('Occupation_label_encoder.pkl', 'rb'))
    bmi_encoder = pickle.load(open('BMI_Category_label_encoder.pkl', 'rb'))
    scaler = pickle.load(open('minmax_scaler_split.pkl', 'rb'))
except Exception as e:
    print(f"Error loading preprocessing objects: {e}")
    gender_encoder = occupation_encoder = bmi_encoder = scaler = None

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        try:
            # Get form data
            age = int(request.form['age'])
            gender = request.form['gender']
            occupation = request.form['occupation']
            sleep_duration = float(request.form['sleep_duration'])
            quality_of_sleep = int(request.form['quality_of_sleep'])
            physical_activity_level = int(request.form['physical_activity_level'])
            stress_level = int(request.form['stress_level'])
            heart_rate = int(request.form['heart_rate'])
            daily_steps = int(request.form['daily_steps'])
            systolic = int(request.form['systolic'])
            diastolic = int(request.form['diastolic'])
            bmi_category = request.form['bmi_category']

            # Encode categorical variables
            if gender_encoder:
                gender_num = gender_encoder.transform([gender])[0]
            else:
                gender_num = 0 if gender == 'Male' else 1
            
            if occupation_encoder:
                occupation_num = occupation_encoder.transform([occupation])[0]
            else:
                occupation_mapping = {'Doctor': 0, 'Teacher': 1, 'Nurse': 2, 'Engineer': 3, 
                                      'Accountant': 4, 'Lawyer': 5, 'Salesperson': 6, 'Others': 7}
                occupation_num = occupation_mapping.get(occupation, 7)
            
            if bmi_encoder:
                bmi_category_num = bmi_encoder.transform([bmi_category])[0]
            else:
                bmi_category_mapping = {'Normal': 0, 'Overweight': 1, 'Obese': 2}
                bmi_category_num = bmi_category_mapping.get(bmi_category, 0)

            # Create feature array
            numerical_features = [
                age, sleep_duration, quality_of_sleep, physical_activity_level, 
                stress_level, heart_rate, daily_steps, systolic, diastolic
            ]

            # Ensure input has the correct number of features
            complete_features = np.zeros((1, 12))
            complete_features[0, :9] = numerical_features  # Assuming the first 9 are numerical features

            # Scale numerical features if scaler is available
            if scaler:
                scaled_features = scaler.transform(complete_features).flatten()
            else:
                scaled_features = complete_features.flatten()

            # Combine features
            features = np.array([
                gender_num,
                scaled_features[0],  # age_scaled
                occupation_num,
                scaled_features[1],  # sleep_duration_scaled
                scaled_features[2],  # quality_of_sleep_scaled
                scaled_features[3],  # physical_activity_level_scaled
                scaled_features[4],  # stress_level_scaled
                bmi_category_num,
                scaled_features[5],  # heart_rate_scaled
                scaled_features[6],  # daily_steps_scaled
                scaled_features[7],  # systolic_scaled
                scaled_features[8]   # diastolic_scaled
            ])

            # Ensure model is correctly loaded
            if model is None or not hasattr(model, 'predict'):
                return render_template('index.html', prediction='Error: Model is not correctly loaded.')

            # Make prediction
            prediction = model.predict(features.reshape(1, -1))
            
            # Map prediction to readable format
            disorders = {0: 'No Disorder', 1: 'Insomnia', 2: 'Sleep Apnea'}
            prediction_text = disorders.get(prediction[0], 'Unknown')

            # Provide personalized sleep tips
            tips = {
                'No Disorder': 'Maintain a healthy sleep routine.',
                'Insomnia': 'Try to establish a regular bedtime and avoid caffeine before sleep.',
                'Sleep Apnea': 'Consult a healthcare provider for a proper diagnosis and treatment.'
            }
            
            return render_template('index.html', prediction=prediction_text, tips=tips.get(prediction_text, ''))
        except Exception as e:
            return render_template('index.html', prediction=f'Error: {e}')

if __name__ == '__main__':
    app.run("0.0.0.0")
