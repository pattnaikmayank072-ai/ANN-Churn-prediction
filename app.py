import streamlit as st
import tensorflow as tf
import pickle
import pandas as pd
import numpy as np
from sklearn.preprocessing  import StandardScaler, OneHotEncoder, LabelEncoder

#Load the trained model
model = tf.keras.models.load_model('model.h5')

#Load the encoders and scaler
with open('onehot_encoder_geo.pkl', 'rb') as f:
    onehot_encoder_geo = pickle.load(f)

with open('label_encoder_gender.pkl', 'rb') as f:
    label_encoder_gender = pickle.load(f)

with open('scaler.pkl', 'rb') as f:
    scaler = pickle.load(f)    


## streamlit app
st.title("Customer Churn Prediction")

#user input
geography = st.selectbox("Select Geography", ["France", "Spain", "Germany"])
gender = st.selectbox("Select Gender", ["Male", "Female"])
age = st.slider("Enter Age", min_value=18, max_value=100, value=30)
balance = st.number_input("Enter Balance", min_value=0.0, value=1000.0)
credit_score = st.number_input("Enter Credit Score", min_value=300, max_value=850, value=600)
estimated_salary = st.number_input("Enter Estimated Salary", min_value=0.0, value=50000.0)
tenure = st.slider("Enter Tenure", min_value=0, max_value=10, value=5)
num_of_products = st.slider("Enter Number of Products", min_value=1, max_value=4, value=1)
has_cr_card = st.selectbox("Has Credit Card?", ["Yes", "No"])
is_active_member = st.selectbox("Is Active Member?", ["Yes", "No"])


#prepare input data
input_data = pd.DataFrame({
    "CreditScore": [credit_score],
    "Geography": [geography],
    "Gender": [gender],
    "Age": [age],
    "Tenure": [tenure],
    "Balance": [balance],
    "NumOfProducts": [num_of_products],
    "HasCrCard": [1 if has_cr_card == "Yes" else 0],
    "IsActiveMember": [1 if is_active_member == "Yes" else 0],
    "EstimatedSalary": [estimated_salary]
})

input_data['Gender'] = label_encoder_gender.transform(input_data['Gender'])

#one-hot encode 'Geography'
geo_encoded = onehot_encoder_geo.transform(input_data[['Geography']]).toarray()
geo_encoded_df = pd.DataFrame(geo_encoded, columns=onehot_encoder_geo.get_feature_names_out(['Geography']))


#combine the input data with the one-hot encoded geography
input_data = pd.concat([input_data.drop('Geography', axis=1), geo_encoded_df], axis=1)

input_data = input_data.loc[:, ~input_data.columns.duplicated()]
input_data = input_data.rename(columns={
    'Geography_France': 'France',
    'Geography_Germany': 'Germany',
    'Geography_Spain': 'Spain'
})
input_data = input_data[scaler.feature_names_in_]

#scale the input data
input_data_scaled = scaler.transform(input_data)

#prediction churn
prediction = model.predict(input_data_scaled)
prediction_probability = prediction[0][0]

st.write(f"Prediction Probability: {prediction_probability:.2f}")

if prediction_probability > 0.5:
    st.write("The customer is likely to churn.")
else:
    st.write("The customer is unlikely to churn.")



