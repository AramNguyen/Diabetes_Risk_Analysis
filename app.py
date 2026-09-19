import streamlit as st
import pandas as pd
import joblib

st.set_page_config(page_title="Diabetes Risk Assessment", layout="wide")

st.markdown("""
    <style>
    /* Ép nền trắng toàn bộ ứng dụng */
    .stApp { background-color: #FFFFFF !important; }
    
    /* Giấu header và footer mặc định của Streamlit */
    header { visibility: hidden !important; }
    footer { visibility: hidden !important; }
    
    /* Triệt tiêu mọi chữ in đậm VÀ ÉP MÀU ĐEN CHO MỌI LOẠI CHỮ */
    * {
        font-weight: normal !important;
        color: #000000 !important;
    }

    
    /* Canh me đúng cái khung nhập số: Nền trắng, viền đen thô, bo góc 0px */
    div[data-baseweb="input"] > div, 
    div[data-testid="stNumberInputContainer"] {
        background-color: #FFFFFF !important;
        border: 1px solid #000000 !important; 
        border-radius: 0px !important; 
    }
    
    /* Ép chữ gõ vào phải là màu đen */
    input {
        color: #000000 !important;
    }

    /* Bắn bỏ mọi nút bấm ẩn bên trong khung nhập số (phòng hờ) */
    div[data-testid="stNumberInputContainer"] button {
        display: none !important;
    }
    
    /* Nút Analyze: Vuông vức, nền trắng viền đen */
    .stButton>button {
        border: 1px solid #000000 !important;
        background-color: #FFFFFF !important;
        color: #000000 !important;
        border-radius: 0px !important;
    }
    .stButton>button:hover {
        background-color: #000000 !important;
        color: #FFFFFF !important;
    }
    </style>
""", unsafe_allow_html=True)

@st.cache_resource
def load_model():
    return joblib.load('diabetes_model.pkl')

model = load_model()

st.write("DIABETES RISK ASSESSMENT SYSTEM")
st.write("Enter medical metrics below.")
st.write("")

col1, col2, col3 = st.columns(3)

with col1:
    age = st.number_input("Age", min_value=1, max_value=120, value=30, step=None)
    bmi = st.number_input("BMI", min_value=10.0, max_value=60.0, value=22.5, step=None)
    waist = st.number_input("Waist Circumference (cm)", min_value=40.0, max_value=150.0, value=80.0, step=None)
    sleep = st.slider("Sleep Hours/Night", 0.0, 24.0, 7.0)

with col2:
    bp_sys = st.number_input("Systolic BP", min_value=70, max_value=200, value=120, step=None)
    bp_dia = st.number_input("Diastolic BP", min_value=40, max_value=130, value=80, step=None)
    stress = st.slider("Stress Level (1-10)", 1, 10, 5)

with col3:
    blood_sugar = st.number_input("Fasting Blood Sugar (mg/dL)", min_value=50, max_value=300, value=95, step=None)
    hba1c = st.number_input("HbA1c Level (%)", min_value=3.0, max_value=15.0, value=5.5, step=None)

st.write("") 

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
    if prediction == 0:
        st.write("RESULT: LOW RISK")
    elif prediction == 1:
        st.write("RESULT: MODERATE RISK")
    else:
        st.write("RESULT: HIGH RISK")