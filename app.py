import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import legacy
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# 🎯 Load the Model without compiling to prevent optimizer issues
try:
    model = load_model("model_rebuilt.keras", compile=False)
    model.compile(optimizer=legacy.Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])
    st.success("✅ Model Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading model: {e}")

# 🎯 Load Encoders and Scaler
try:
    with open('geoOHE.pkl', 'rb') as file:
        geoOHE = pickle.load(file)

    with open('genderEncoder.pkl', 'rb') as file:
        genderEncoder = pickle.load(file) 

    with open('scaler.pkl', 'rb') as file:
        scaler = pickle.load(file)

    st.success("✅ Encoders and Scaler Loaded Successfully")
except Exception as e:
    st.error(f"❌ Error loading encoders or scaler: {e}")

# 🏠 Streamlit App Title
st.title('📊 Customer Churn Prediction')

# 🌍 Geographical Location Selection
if hasattr(geoOHE, "categories_"):
    geography = st.selectbox('🌍 Geographical Location', geoOHE.categories_[0])
else:
    st.error("❌ Error: Geography encoder is not loaded correctly.")

# 🧑‍🤝‍🧑 Gender Selection
if hasattr(genderEncoder, "classes_"):
    gender = st.selectbox('🧑‍🤝‍🧑 Gender', genderEncoder.classes_)
else:
    st.error("❌ Error: Gender encoder is not loaded correctly.")

# 📥 User Input Fields
age             = st.slider('🎂 Age', 18, 100, 30)
balance         = st.number_input('💰 Balance', min_value=0.0, format="%.2f", value=10000.0)
creditScore     = st.number_input('🏦 Credit Score', min_value=300, max_value=850, value=600)
estimatedSalary = st.number_input('💵 Estimated Salary', min_value=0.0, format="%.2f", value=50000.0)
tenure          = st.slider('📆 Tenure (Years)', 0, 20, 5)
numOfProducts   = st.slider('📦 Number of Products', 1, 4, 1)
hasCreditCards  = st.selectbox('💳 Has Credit Card', [0, 1])
isActiveMember  = st.selectbox('🔄 Is Active Member', [0, 1])  

# 🏗️ Data Preprocessing
try:
    inputData = pd.DataFrame({
        'CreditScore': [creditScore],
        'Gender': [genderEncoder.transform([gender])[0]],
        'Age': [age],
        'Tenure': [tenure],
        'Balance': [balance],
        'NumOfProducts': [numOfProducts],
        'HasCrCard': [hasCreditCards],
        'IsActiveMember': [isActiveMember],
        'EstimatedSalary': [estimatedSalary]
    })

    # 🏗️ One-Hot Encode 'Geography'
    geoEncoded = geoOHE.transform([[geography]]).toarray()
    geoEncodedDF = pd.DataFrame(geoEncoded, columns=geoOHE.get_feature_names_out(['Geography'])) 

    # 🏗️ Combine OHE column with input data
    inputData = pd.concat([inputData.reset_index(drop=True), geoEncodedDF], axis=1)

    # 🔍 Check for Missing Values
    if inputData.isnull().sum().sum() > 0:
        st.error("❌ Error: Missing values detected in input data.")

    # 📊 Scale the Input Data
    inputDataScaled = scaler.transform(inputData)

    # 🏆 Predict Churn
    prediction = model.predict(inputDataScaled)
    predictionProbability = prediction[0][0]

    # 🏁 Display Prediction Result
    st.subheader("🔮 Prediction Result")
    if predictionProbability > 0.5:
        st.error('🚨 The Customer Is Likely To Churn!')
    else:
        st.success('✅ The Customer Is Not Likely To Churn.')

    st.write(f"🔢 Churn Probability: **{predictionProbability:.2%}**")

except Exception as e:
    st.error(f"❌ Error during processing: {e}")
