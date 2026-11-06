%%writefile app.py
import streamlit as st
import pandas as pd
import numpy as np
import joblib

st.set_page_config(page_title="Breast Cancer Predictor (PCA)", layout="wide")

st.title("🩺 Breast Cancer Diagnosis Prediction App (PCA Inputs)")
st.write("""
This app uses a Support Vector Machine (SVC) classifier trained on Principal Components
to predict whether a breast mass is **Malignant (M)** or **Benign (B)**.
""")

# Load saved SVC model
@st.cache_resource
def load_model():
    model = joblib.load('svc_model.sav')
    return model

try:
    model = load_model()
except Exception as e:
    st.error(f"Error loading models: {e}. Please make sure you have 'svc_model.sav' in the directory.")
    st.stop()

# Define the 17 principal components
pca_features = [f'PC{i+1}' for i in range(17)]

st.sidebar.header("🔬 Input Principal Component Values")
st.sidebar.write("Please input the values for the 17 Principal Components:")

# Group inputs dynamically into 3 columns for a clean UI layout
input_data = {}
cols = st.columns(3)
for idx, col in enumerate(pca_features):
    col_target = cols[idx % 3]
    val = col_target.number_input(f"{col}", value=0.0, format="%.4f")
    input_data[col] = val

if st.button("Run Diagnostic Prediction", type="primary"):
    # Convert inputs directly to a numpy array matching the 17 components expected by the SVC model
    input_features = np.array([[input_data[pc] for pc in pca_features]])

    # Predict directly using the loaded classifier
    prediction = model.predict(input_features)[0]

    st.subheader("Prediction Results")
    if prediction == 1:
        st.error("🚨 **Diagnosis Prediction: Malignant (M)**")
        st.write("The model suggests a high likelihood that the tumor is malignant.")
    else:
        st.success("🟢 **Diagnosis Prediction: Benign (B)**")
        st.write("The model suggests a high likelihood that the tumor is benign.")
