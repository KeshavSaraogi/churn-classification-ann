import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from sklearn.preprocessing import StandardScaler

st.title("📊 Customer Churn Prediction with ANN")

try:
    print("🔄 Loading model_rebuilt.keras...")
    model = load_model("model_fixed.keras", compile=False)
    print("✅ Model loaded successfully in app.py!")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")
    print(f"❌ Error loading model: {e}")

st.write("App is running. Model loaded successfully!")

# Load the scaler
try:
    with open("scaler.pkl", "rb") as scaler_file:
        scaler = pickle.load(scaler_file)
    print("✅ Scaler loaded successfully!")
except Exception as e:
    st.error(f"❌ Error loading scaler: {e}")
    print(f"❌ Error loading scaler: {e}")

# User Inputs
age = st.slider("🎂 Age", 18, 100, 30)
balance = st.number_input("💰 Balance", min_value=0.0, format="%.2f", value=10000.0)
creditScore = st.number_input("🏦 Credit Score", min_value=300, max_value=850, value=600)
estimatedSalary = st.number_input("💵 Estimated Salary", min_value=0.0, format="%.2f", value=50000.0)
tenure = st.slider("📆 Tenure (Years)", 0, 20, 5)
numOfProducts = st.slider("📦 Number of Products", 1, 4, 1)
hasCreditCards = st.selectbox("💳 Has Credit Card", [0, 1])
isActiveMember = st.selectbox("🔄 Is Active Member", [0, 1])

try:
    inputData = np.array([[creditScore, age, tenure, balance, numOfProducts, hasCreditCards, isActiveMember, estimatedSalary]])
    inputDataScaled = scaler.transform(inputData)

    prediction = model.predict(inputDataScaled)
    predictionProbability = prediction[0][0]

    st.subheader("🔮 Prediction Result")
    if predictionProbability > 0.5:
        st.error("🚨 The Customer Is Likely To Churn!")
    else:
        st.success("✅ The Customer Is Not Likely To Churn.")

    st.write(f"🔢 Churn Probability: **{predictionProbability:.2%}**")

except Exception as e:
    st.error(f"❌ Error during processing: {e}")
