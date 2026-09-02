
import streamlit as st
import pandas as pd
import numpy as np
import joblib

# Set Page Config
st.set_page_config(page_title="Patient Persona Classifier", page_icon="🏥", layout="centered")

# 1. Load Saved Joblib Artifacts
@st.cache_resource
def load_artifacts():
    scaler = joblib.load('scaler.joblib')
    model = joblib.load('kmeans_k3.joblib')
    return scaler, model

try:
    scaler, kmeans_k3 = load_artifacts()
    model_loaded = True
except Exception as e:
    model_loaded = False
    st.error(f"Error loading models: {e}. Ensure 'scaler.joblib' and 'kmeans_k3.joblib' exist.")

# 2. Persona Mapping (Exact match from your notebook)
persona_map = {
    0: "Low-Risk / Healthy",
    1: "High-Complexity / Inpatient",
    2: "Frequent Monitoring / High-BMI"
}

persona_descriptions = {
    "Low-Risk / Healthy": "Younger cohort, low spending, low visit frequency, and shorter stay durations.",
    "High-Complexity / Inpatient": "Older cohort, high chronic condition count, longer stay duration, and highest total spending.",
    "Frequent Monitoring / High-BMI": "Higher BMI, frequent annual visits, moderate spending requiring routine monitoring."
}

# 3. Streamlit Interface
st.title("🏥 Patient Clustering & Persona Classifier")
st.write("Enter patient metrics to assign them to one of the K=3 hospital personas.")

# Sidebar Inputs (Matching profile_cols)
st.sidebar.header("Patient Feature Input")

age = st.sidebar.number_input("Age", min_value=0, max_value=110, value=35)
chronic_condition = st.sidebar.number_input("Chronic Condition Count", min_value=0, max_value=10, value=1)
bmi = st.sidebar.number_input("BMI", min_value=10.0, max_value=70.0, value=28.0, step=0.1)
annual_visits = st.sidebar.number_input("Annual Visits", min_value=0, max_value=50, value=2)
avg_stay_duration = st.sidebar.number_input("Avg Stay Duration (days)", min_value=0.0, max_value=60.0, value=2.0, step=0.5)
total_spending = st.sidebar.number_input("Total Spending ($)", min_value=0.0, max_value=100000.0, value=900.0, step=50.0)

# Build DataFrame in the exact feature order used during training
input_df = pd.DataFrame([{
    'Age': age,
    'Chronic_Condition': chronic_condition,
    'BMI': bmi,
    'Annual_Visits': annual_visits,
    'Avg_Stay_Duration': avg_stay_duration,
    'Total_Spending': total_spending
}])

st.subheader("Input Patient Metrics")
st.dataframe(input_df)

if model_loaded:
    if st.button("Classify Patient"):
        # Scale inputs using saved scaler
        scaled_input = scaler.transform(input_df)
        
        # Predict cluster ID
        cluster_id = int(kmeans_k3.predict(scaled_input)[0])
        persona_name = persona_map.get(cluster_id, "Unknown Persona")
        
        # Output Results
        st.markdown("---")
        st.subheader("Classification Result")
        st.metric(label="Predicted Cluster ID", value=f"Cluster {cluster_id}")
        
        if cluster_id == 0:
            st.success(f"**Assigned Persona:** {persona_name}")
        elif cluster_id == 1:
            st.error(f"**Assigned Persona:** {persona_name}")
        else:
            st.warning(f"**Assigned Persona:** {persona_name}")

        st.info(f"**Persona Overview:** {persona_descriptions[persona_name]}")
