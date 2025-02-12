import pickle
import numpy as np
import pandas as pd
import streamlit as st
import tensorflow as tf
from sklearn.processing import StandardScaler, LabelEncoder, OneHotEncoder

# Load the Model, Scaler and the One-Hot-Encoder
model = tf.keras.models.load_model('model.h5')

with open('geoOHE.pkl', 'rb') as file:
    geoOHE = pickle.load(file)

with open('genderEncoder.pkl', 'rb') as file:
    genderEncoder = pickle.load(file) 

with open('scaler.pkl', 'rb') as file:
    scaler = pickle.load(file) 
    

# Streamlit App 
st.title('Customer Churn Prediction')

# User-Input
geography       = st.selectbox('Geographical Location'  , geoOHE.categories_[0])
gender          = st.selectbox('Gender'                 , genderEncoder.classes_)
age             = st.slider('Age'                       , 18, 100)
balance         = st.number_input('Balance')
creditScore     = st.number_input('Credit Score')
estimatedSalary = st.number_input('Estimated Salary')
tenure          = st.slider('Tenure'                    , 0, 20)
numOfProducts   = st.slider('Number of Products'        , 1, 4)
hasCreditCards  = st.selectbox('Has Credit Card'        , [0, 1])
isActiveMember  = st.selectBox('Is Active Member'       , [0, 1])


# Input Data
inputData = pd.DataFrame({
    'CreditScore'       : [creditScore],
    'Gender'            : [genderEncoder.transform([gender])[0]],
    'Age'               : [age],
    'Tenure'            : [tenure],
    'Balance'           : [balance],
    'numOfProducts'     : [numOfProducts],
    'hasCreditCards'    : [hasCreditCards],
    'IsActiveMember'    : [isActiveMember],
    'EstimatedSalary'   : [estimatedSalary]
})

# OHE 'Geography'
geoEncoded      = geoOHE.trnasform([[geography]]).toarray()
geoEncodedDF    = pd.DataFrame(geoEncoded, columns = geoOHE.get_feature_names_out(['Geography'])) 

# Combine OHE column with the input data
inputData = pd.concat([inputData.reset_index(drop = True), geoEncodedDF], axis = 1)

# Scale the Input Data
inputDataScaled = scaler.transform(inputData)

# Prediction Churn
prediction = model.predict(inputDataScaled)
predictionProbability = prediction[0][0]

if predictionProbability > 0.5:
    print('The Customer Is Likely To Churn')
else:
    print('The Customer Is Not Likely To Churn')