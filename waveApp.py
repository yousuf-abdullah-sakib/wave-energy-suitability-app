import streamlit as st
import numpy as np
import pickle
from tensorflow.keras.models import load_model

# Load model
model = load_model("ModelANN_LSI_Model_500.keras", compile=False)

# Load scaler
with open("scaler_params.pkl", "rb") as f:
    scaler_params = pickle.load(f)

def norm(val, xmin, xmax):
    if xmax == xmin:
        return 0
    if val > xmax:
        return 1.0
    elif val < xmin:
        return 0.0
    else:
        return (val - xmin) / (xmax - xmin)

st.markdown(
    "<h3 style='text-align:center; color:#0E6BA8;'>🌊 Wave Energy Suitability Predictor</h1>",
    unsafe_allow_html=True
)
st.write("Enter Raw Values:")

# Input fields
WaveHeight = st.number_input("Wave Height (m)")
WavePeriod = st.number_input("Wave Period (s)")
Bathymetry = st.number_input("Bathymetry (m)")
ShoreDistance = st.number_input("Distance to Shore (km)")
WindSpeed = st.number_input("Wind Speed (m/s)")
ShipDensity = st.number_input("Ship Density (k)")
Salinity = st.number_input("Salinity (psu)")

if st.button("Predict"):
    inputs = [
        WaveHeight, WavePeriod, Bathymetry,
        ShoreDistance, WindSpeed, ShipDensity, Salinity
    ]

    norm_inputs = []
    keys = list(scaler_params.keys())

    for val, k in zip(inputs, keys):
        xmin, xmax = scaler_params[k]
        norm_inputs.append(norm(val, xmin, xmax))

    X = np.array(norm_inputs).reshape(1, -1)
    pred = model.predict(X)[0][0]

    st.subheader("Prediction Result")

    if pred >= 0.80:
        st.success(f"🌟 Highly Suitable (LSI: {pred:.4f})")
    elif pred >= 0.60:
        st.info(f"✅ Moderately Suitable (LSI: {pred:.4f})")
    elif pred >= 0.40:
        st.warning(f"⚠️ Marginally Suitable (LSI: {pred:.4f})")
    else:
        st.error(f"❌ Unsuitable (LSI: {pred:.4f})")

# Footer (only once)
st.markdown("---")
st.markdown(
    """
    <div style="text-align:center; font-size:14px; color:gray;">
        🌊 Developed by <b>Yousuf Abdullah Sakib</b><br>
        Department of Oceanography, University of Dhaka
    </div>
    """,
    unsafe_allow_html=True
)
