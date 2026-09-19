import streamlit as st
import pandas as pd
import joblib

# 1. Page Config (Clean, no emojis)
st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")

# Inject custom CSS to remove rounded corners for a raw, technical UI
st.markdown("""
    <style>
    div[data-baseweb="input"] > div {
        border-radius: 0px !important;
    }
    .stButton>button {
        border-radius: 0px !important;
    }
    div[data-baseweb="select"] > div {
        border-radius: 0px !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

model = load_model()

st.title("Diabetes Risk Assessment System")
st.markdown("Enter medical metrics below to evaluate the diabetes risk level.")
st.divider()

# 3. Input Form
col1, col2, col3 = st.columns(3)

with col1:
    st.subheader("Body Metrics")
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
    waist = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=150.0, value=80.0)
    sleep = st.slider("Sleep Hours/Night", 0.0, 24.0, 7.0)

with col2:
    st.subheader("Blood Pressure")
    bp_sys = st.number_input("Systolic BP", min_value=70, max_value=200, value=120)
    bp_dia = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

with col3:
    st.subheader("Blood Sugar")
    blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=95)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5)

# 4. Prediction Logic
if st.button("Analyze Risk", type="primary", use_container_width=True):
    input_data = pd.DataFrame(columns=model.feature_names_in_)
    input_data.loc[0] = 0 
    
    input_data['age'] = age
    input_data['bmi'] = bmi
    input_data['waist_circumference_cm'] = waist
    input_data['hours_sleep_per_night'] = sleep
    input_data['blood_pressure_systolic'] = bp_sys
    input_data['blood_pressure_diastolic'] = bp_dia
    input_data['stress_level'] = stress
    input_data['fasting_blood_sugar'] = blood_sugar
    input_data['hba1c_level'] = hba1c
    
    prediction = model.predict(input_data)[0]
    
    st.divider()
    if prediction == 0:
        st.success("RESULT: LOW RISK - Metrics are within a safe range.")
    elif prediction == 1:
        st.warning("RESULT: MODERATE RISK - Consider adjusting diet and lifestyle.")
    else:
        st.error("RESULT: HIGH RISK - Please consult a healthcare professional for a detailed examination.")
    