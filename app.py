import streamlit as st
import pandas as pd
import pickle
import os
import joblib

# -------------------------------
# Load model and scaler
# -------------------------------
import joblib

BASE_DIR = os.path.dirname(__file__)

model = joblib.load(os.path.join(BASE_DIR, "model", "loan_model.pkl"))
scaler = joblib.load(os.path.join(BASE_DIR, "model", "scaler.pkl"))


# -------------------------------
# Streamlit App UI
# -------------------------------
st.title("🏦 Loan Approval Prediction App")
st.write("Enter applicant details to check loan eligibility:")

# Input fields
gender = st.selectbox("Gender", ["Male", "Female"])
married = st.selectbox("Married", ["Yes", "No"])
dependents = st.selectbox("Dependents", [0, 1, 2, 3])
education = st.selectbox("Education", ["Graduate", "Not Graduate"])
self_emp = st.selectbox("Self Employed", ["Yes", "No"])
app_income = st.number_input("Applicant Income", min_value=0)
coapp_income = st.number_input("Coapplicant Income", min_value=0)
loan_amount = st.number_input("Loan Amount (in thousands)", min_value=0)
loan_term = st.selectbox("Loan Term (days)", [360, 180, 120, 60])
credit_hist = st.selectbox("Credit History", [1, 0])
property_area = st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])

# -------------------------------
# Encode categorical inputs
# -------------------------------
gender_map = {"Male": 1, "Female": 0}
married_map = {"Yes": 1, "No": 0}
education_map = {"Graduate": 1, "Not Graduate": 0}
selfemp_map = {"Yes": 1, "No": 0}
property_map = {"Urban": 2, "Semiurban": 1, "Rural": 0}

input_data = pd.DataFrame([{
    "Gender": gender_map[gender],
    "Married": married_map[married],
    "Dependents": dependents,
    "Education": education_map[education],
    "Self_Employed": selfemp_map[self_emp],
    "ApplicantIncome": app_income,
    "CoapplicantIncome": coapp_income,
    "LoanAmount": loan_amount,
    "Loan_Amount_Term": loan_term,
    "Credit_History": credit_hist,
    "Property_Area": property_map[property_area]
}])

# -------------------------------
# Scale input & Predict
# -------------------------------
input_scaled = scaler.transform(input_data)

if st.button("Check Eligibility"):
    prediction = model.predict(input_scaled)[0]
    if prediction == 1:
        st.success("✅ Loan Eligible: Approved")
    else:
        st.error("❌ Loan Not Eligible: Rejected")
