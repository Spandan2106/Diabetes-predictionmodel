import streamlit as st
import pickle
import numpy as np

# Set page configuration
st.set_page_config(page_title="Diabetes Prediction", layout="centered")

# Load the saved model and scaler
try:
    model = pickle.load(open('diabetes_model.sav', 'rb'))
    scaler = pickle.load(open('scaler.sav', 'rb'))
except FileNotFoundError:
    st.error("Model files not found. Please run 'main.py' first to generate 'diabetes_model.sav' and 'scaler.sav'.")
    st.stop()

st.title('🩺 Diabetes Prediction Web App')
st.write('Enter the medical details below to predict the diabetes status.')

# Layout for input fields
col1, col2 = st.columns(2)

with col1:
    Pregnancies = st.text_input('Number of Pregnancies', '0')
    Glucose = st.text_input('Glucose Level', '0')
    BloodPressure = st.text_input('Blood Pressure value', '0')
    SkinThickness = st.text_input('Skin Thickness value', '0')

with col2:
    Insulin = st.text_input('Insulin Level', '0')
    BMI = st.text_input('BMI value', '0')
    DiabetesPedigreeFunction = st.text_input('Diabetes Pedigree Function', '0')
    Age = st.text_input('Age of the Person', '0')

# Prediction logic
if st.button('Diabetes Test Result'):
    try:
        # Convert inputs to float
        input_data = [
            float(Pregnancies), float(Glucose), float(BloodPressure), float(SkinThickness),
            float(Insulin), float(BMI), float(DiabetesPedigreeFunction), float(Age)
        ]
        
        # Prepare data
        input_as_numpy = np.asarray(input_data)
        input_data_reshaped = input_as_numpy.reshape(1, -1)
        
        # Standardize
        std_data = scaler.transform(input_data_reshaped)
        
        # Predict
        prediction = model.predict(std_data)
        
        if prediction[0] == 0:
            st.success('The person is Non-Diabetic')
        else:
            st.error('The person is Diabetic')
            
    except ValueError:
        st.warning("Please enter valid numerical values.")