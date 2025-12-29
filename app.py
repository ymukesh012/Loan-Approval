import streamlit as st
import pandas as pd
import numpy as np
import joblib
import pickle

# ------------------- Load Model and Preprocessors -------------------

model = joblib.load("loan_model.pkl")
label_encoders = joblib.load("label_encoders.pkl")
scaler = joblib.load("scaler.pkl")

with open("property_cols.pkl", "rb") as f:
    property_cols = pickle.load(f)


st.markdown(
    """
    <h1 style='text-align: center; color: #1f77b4;'>Check Loan Eligibility</h1>
    """,
    unsafe_allow_html=True
)
st.write("Enter your details to see loan approval status.")
# ------------------- User Input -------------------
def user_input():
   
    data = {
        "Gender": st.selectbox("Gender", label_encoders['Gender'].classes_),
        "Married": st.selectbox("Married", label_encoders['Married'].classes_),
        "Dependents": st.selectbox("Dependents", ["0", "1", "2", "3+"]),
        "Education": st.selectbox("Education", label_encoders['Education'].classes_),
        "Self_Employed": st.selectbox("Self Employed", label_encoders['Self_Employed'].classes_),
        "ApplicantIncome": st.number_input("Applicant Income", value=0),
        "CoapplicantIncome": st.number_input("Coapplicant Income", value=0),
        "LoanAmount": st.number_input("Loan Amount (in thousands)", value=1),
        "Loan_Amount_Term": st.number_input("Loan Term (months)", value=360),
        "Credit_History": st.selectbox("Credit History", [1.0, 0.0]),
        "Property_Area": st.selectbox("Property Area", ["Urban", "Semiurban", "Rural"])
    }
    return pd.DataFrame([data])

input_df = user_input()

# ------------------- Preprocessing -------------------
def preprocess(df):
    # 1. Label Encoding (Categorical)
    binary_cols = ['Gender', 'Married', 'Education', 'Self_Employed']
    for col in binary_cols:
        le = label_encoders[col]
        # Safety check for unseen labels
        df[col] = df[col].astype(str).apply(lambda x: x if x in le.classes_ else le.classes_[0])
        df[col] = le.transform(df[col])

    # 2. Dependents (Manual mapping used in your notebook)
    df['Dependents'] = df['Dependents'].replace('3+', 3).astype(int)

    # 3. One-hot encode Property_Area
    df = pd.get_dummies(df, columns=['Property_Area'], drop_first=True)
    
    for col in property_cols:
        if col not in df.columns:
            df[col] = 0

    # 4. Feature engineering
    df["TotalIncome"] = df["ApplicantIncome"] + df["CoapplicantIncome"]
    df["Income_Loan_Ratio"] = df["TotalIncome"] / df["LoanAmount"]

    # 5. Scale numeric features
    numeric_features = ["ApplicantIncome", "CoapplicantIncome", "LoanAmount", "Loan_Amount_Term", "TotalIncome", "Income_Loan_Ratio"]
    df[numeric_features] = scaler.transform(df[numeric_features])

    # 6. FIX: Dynamically reorder columns to match the model's training order
    # This uses the model's internal memory of the feature names
    df = df[model.feature_names_in_]

    return df

# ------------------- Prediction -------------------
if st.button("Predict Loan Status"):
    processed_input = preprocess(input_df)
    prediction = model.predict(processed_input)
    
    st.subheader("Result")
    if prediction[0] == 1:
        st.success("Loan Approved ✅")
    else:
        st.error("Loan Not Approved ❌")