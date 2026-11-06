import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(
    page_title="Breast Cancer Prediction App",
    page_icon="🎗️",
    layout="centered"
)

# Load pre-trained Scaler, PCA, and Classifier models
@st.cache_resource
def load_models():
    scaler = joblib.load('scaler.sav')
    pca = joblib.load('pca_transformer.sav')
    classifier = joblib.load('svc_model.sav')
    return scaler, pca, classifier

try:
    scaler_model, pca_model, classifier_model = load_models()
    models_loaded = True
except Exception as e:
    models_loaded = False
    st.error(f"Error loading models: {e}. Please make sure 'scaler.sav', 'pca_transformer.sav', and 'svc_model.sav' are present in the directory.")

# App Header
st.title("🎗️ Breast Cancer Prediction Platform")
st.write("""
This application uses a Machine Learning pipeline containing standard scaling, **Principal Component Analysis (PCA)** 
for dimension reduction, and a **Support Vector Machine (SVC)** classifier to predict whether a breast mass tumor is **Malignant** or **Benign**.
""")

if models_loaded:
    st.sidebar.header("User Input Features")
    
    # Setup a simple way for users to manually input some primary parameters
    st.subheader("Input Mass Measurements manually:")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        radius_mean = st.number_input("Radius Mean", min_value=0.0, max_value=40.0, value=14.0)
        texture_mean = st.number_input("Texture Mean", min_value=0.0, max_value=40.0, value=19.0)
        perimeter_mean = st.number_input("Perimeter Mean", min_value=0.0, max_value=200.0, value=92.0)
    with col2:
        area_mean = st.number_input("Area Mean", min_value=0.0, max_value=2500.0, value=650.0)
        smoothness_mean = st.number_input("Smoothness Mean", min_value=0.0, max_value=0.5, value=0.1)
        compactness_mean = st.number_input("Compactness Mean", min_value=0.0, max_value=0.5, value=0.1)
    with col3:
        concavity_mean = st.number_input("Concavity Mean", min_value=0.0, max_value=0.5, value=0.09)
        concave_points_mean = st.number_input("Concave Points Mean", min_value=0.0, max_value=0.5, value=0.05)
        symmetry_mean = st.number_input("Symmetry Mean", min_value=0.0, max_value=0.5, value=0.18)

    # Create a full mock vector of 30 raw features required by our scaler/PCA pipeline.
    # Using mean values for the variables not configured directly via inputs to avoid mismatch.
    raw_input_data = np.zeros((1, 30))
    raw_input_data[0, 0] = radius_mean
    raw_input_data[0, 1] = texture_mean
    raw_input_data[0, 2] = perimeter_mean
    raw_input_data[0, 3] = area_mean
    raw_input_data[0, 4] = smoothness_mean
    raw_input_data[0, 5] = compactness_mean
    raw_input_data[0, 6] = concavity_mean
    raw_input_data[0, 7] = concave_points_mean
    raw_input_data[0, 8] = symmetry_mean
    # The remaining feature index lines are initialized to 0
    
    if st.button("Predict Diagnosis"):
        # Scale the raw input features first
        scaled_input = scaler_model.transform(raw_input_data)
        
        # Apply PCA reduction down to the 17 principal components
        pca_input = pca_model.transform(scaled_input)
        
        # Make prediction using our classifier model
        prediction = classifier_model.predict(pca_input)[0]
        
        st.subheader("Prediction Results")
        if prediction == 1 or prediction == 'M':
            st.error("⚠️ **Prediction: Malignant (M)**")
            st.write("The pipeline indicates a high probability of malignancy.")
        else:
            st.success("✅ **Prediction: Benign (B)**")
            st.write("The pipeline indicates a high probability of benign behavior.")
