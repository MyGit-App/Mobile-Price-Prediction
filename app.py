import streamlit as st
import joblib
import numpy as np
import pandas as pd

# ---------------------------------------------------------
# Load model + scaler
# ---------------------------------------------------------
model = joblib.load("lasso_regression_model.pkl")
scaler = joblib.load("scaler.pkl")

# ---------------------------------------------------------
# Exact feature order used during training
# ---------------------------------------------------------
FEATURE_ORDER = [
    'Sale', 'weight', 'resolution', 'ppi',
    'cpu_core', 'cpu_freq', 'internal_mem', 'ram',
    'RearCam', 'Front_Cam', 'battery', 'thickness'
]

# ---------------------------------------------------------
# Streamlit UI
# ---------------------------------------------------------
st.set_page_config(page_title="Mobile Price Predictor", layout="wide")
st.title("📱 Mobile Price Prediction")
st.markdown("Predict mobile phone prices using the trained Ridge Regression model")

st.sidebar.header("Enter Mobile Features")

col1, col2 = st.columns(2)

with col1:
    sale = st.number_input('Sale (units)', min_value=0, max_value=20000, value=600, step=1)
    weight = st.slider('Weight (g)', min_value=50, max_value=800, value=170, step=1)
    resolution = st.slider('Resolution (inches)', min_value=1.0, max_value=12.5, value=5.2, step=0.1)
    ppi = st.slider('PPI (pixel density)', min_value=50, max_value=900, value=335, step=1)
    cpu_core = st.slider('CPU cores', min_value=1, max_value=12, value=4, step=1)
    cpu_freq = st.slider('CPU freq (GHz)', min_value=0.1, max_value=4.0, value=1.5, step=0.1)

with col2:
    internal_mem = st.select_slider('Internal memory (GB)', options=[4, 8, 16, 32, 64, 128, 256], value=32)
    ram = st.select_slider('RAM (GB)', options=[1, 2, 3, 4, 6, 8, 12, 16], value=2)
    rear_cam = st.slider('Rear Camera (MP)', min_value=0, max_value=108, value=12, step=1)
    front_cam = st.slider('Front Camera (MP)', min_value=0, max_value=64, value=5, step=1)
    battery = st.slider('Battery (mAh)', min_value=500, max_value=10000, value=2800, step=50)
    thickness = st.slider('Thickness (mm)', min_value=3.0, max_value=25.0, value=8.9, step=0.1)

# ---------------------------------------------------------
# Build DataFrame in correct order
# ---------------------------------------------------------
features_input = pd.DataFrame([{
    "Sale": sale,
    "weight": weight,
    "resolution": resolution,
    "ppi": ppi,
    "cpu_core": cpu_core,
    "cpu_freq": cpu_freq,
    "internal_mem": internal_mem,
    "ram": ram,
    "RearCam": rear_cam,
    "Front_Cam": front_cam,
    "battery": battery,
    "thickness": thickness
}])[FEATURE_ORDER]

# ---------------------------------------------------------
# Prediction
# ---------------------------------------------------------
if st.button("🔮 Predict Price"):
    
    try:
        scaled = scaler.transform(features_input)
        prediction = model.predict(scaled)[0]
        
        st.success(f"Predicted Price: ₹{prediction:,.2f}")

    except Exception as e:
        st.error(f"Error during prediction: {e}")
