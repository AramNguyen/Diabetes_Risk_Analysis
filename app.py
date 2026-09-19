import streamlit as st
import pandas as pd
import joblib

# 1. Page Config
st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")

# CSS "Đập đi xây lại" - Giao diện trắng bóc, không khung viền
st.markdown("""
    <style>
    /* Ép nền trắng, chữ đen, giấu thanh header/footer mặc định */
    .stApp { background-color: #FFFFFF !important; color: #000000 !important; }
    header { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    
    /* Xóa nền và khung của ô nhập liệu, chỉ để lại gạch chân */
    div[data-baseweb="input"] > div {
        border: none !important;
        border-bottom: 1px solid #000000 !important;
        border-radius: 0px !important;
        background-color: transparent !important;
        box-shadow: none !important;
    }
    
    /* Xóa nền khối slider */
    div[data-baseweb="slider"] {
        background-color: transparent !important;
    }

    /* Nút bấm thô sơ: viền đen, nền trắng */
    .stButton>button {
        border: 1px solid #000000 !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 0px !important;
        box-shadow: none !important;
        font-weight: bold !important;
    }
    .stButton>button:hover {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

# 2. Load Model
@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

model = load_model()

# Header tối giản
st.markdown("### DIABETES RISK ASSESSMENT SYSTEM")
st.markdown("Enter medical metrics below.")
st.write("") # Dòng trống thay cho đường kẻ

# 3. Input Form
col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5)
    waist = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=150.0, value=80.0)
    sleep = st.slider("Sleep Hours/Night", 0.0, 24.0, 7.0)

with col2:
    bp_sys = st.number_input("Systolic BP", min_value=70, max_value=200, value=120)
    bp_dia = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

with col3:
    blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=95)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5)

st.write("") 

# 4. Prediction Logic
if st.button("ANALYZE RISK", use_container_width=True):
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
    
    st.write("")
    # Dùng markdown in chữ thô thay vì dùng hộp màu st.success/st.error
    if prediction == 0:
        st.markdown("**RESULT: LOW RISK** - Metrics are within a safe range.")
    elif prediction == 1:
        st.markdown("**RESULT: MODERATE RISK** - Consider adjusting diet and lifestyle.")
    else:
        st.markdown("**RESULT: HIGH RISK** - Please consult a healthcare professional.")