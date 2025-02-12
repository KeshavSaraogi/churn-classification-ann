import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from tensorflow.keras.models import load_model
from tensorflow.keras.optimizers import Adam
from sklearn.preprocessing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the Model without compiling to prevent optimizer issues
model = load_model("model_rebuilt.keras")

# Recompile the model with Adam optimizer
model.compile(optimizer=Adam(learning_rate=0.001), loss='binary_crossentropy', metrics=['accuracy'])

# Load Encoders and Scaler
with open('geoOHE.pkl', 'rb') as file:
    geoOHE = pickle.load(file)

with open('genderEncoder.pkl', 'rb') as file:
    genderEncoder = pickle.load(file) 

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 

# Streamlit App
st.title('📊 Customer Churn Prediction')

# User-Input
geography       = st.selectbox('🌍 Geographical Location', geoOHE.categories_[0])
gender          = st.selectbox('🧑‍🤝‍🧑 Gender', genderEncoder.classes_)
age             = st.slider('🎂 Age', 18, 100)
balance         = st.number_input('💰 Balance', min_value=0.0, format="%.2f")
creditScore     = st.number_input('🏦 Credit Score', min_value=300, max_value=850)
estimatedSalary = st.number_input('💵 Estimated Salary', min_value=0.0, format="%.2f")
tenure          = st.slider('📆 Tenure (Years)', 0, 20)
numOfProducts   = st.slider('📦 Number of Products', 1, 4)
hasCreditCards  = st.selectbox('💳 Has Credit Card', [0, 1])
isActiveMember  = st.selectbox('🔄 Is Active Member', [0, 1])  

# Input Data
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

# One-Hot Encode 'Geography'
geoEncoded = geoOHE.transform([[geography]]).toarray()
geoEncodedDF = pd.DataFrame(geoEncoded, columns=geoOHE.get_feature_names_out(['Geography'])) 

# Combine OHE column with input data
inputData = pd.concat([inputData.reset_index(drop=True), geoEncodedDF], axis=1)

# Scale the Input Data
inputDataScaled = scaler.transform(inputData)

# Prediction Churn
prediction = model.predict(inputDataScaled)
predictionProbability = prediction[0][0]

# Display Prediction Result
st.subheader("🔮 Prediction Result")
if predictionProbability > 0.5:
    st.error('🚨 The Customer Is Likely To Churn!')
else:
    st.success('✅ The Customer Is Not Likely To Churn.')
    
st.write(f"🔢 Churn Probability: **{predictionProbability:.2%}**")
