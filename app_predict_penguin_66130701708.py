%%writefile Predict_Species_Penguin.py

import streamlit as st
import pandas as pd
import pickle

# Load the trained KNN model
with open('model_penguin_66130701708.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the penguin dataset for reference
df = pd.read_csv('penguins_size.csv')

# Streamlit App
st.title('Penguin Species Predictor')

# User Input Form
st.header('Input Penguin Measurements to Predict Species')

island = st.selectbox('Island', df['island'].unique())
culmen_length = st.slider('Culmen Length (mm)', float(df['culmen_length_mm'].min()), float(df['culmen_length_mm'].max()))
culmen_depth = st.slider('Culmen Depth (mm)', float(df['culmen_depth_mm'].min()), float(df['culmen_depth_mm'].max()))
flipper_length = st.slider('Flipper Length (mm)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
body_mass = st.slider('Body Mass (g)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))
sex = st.selectbox('Sex', ['MALE', 'FEMALE'])

# Create a DataFrame for the user input
user_input = pd.DataFrame({
    'island': [island],
    'culmen_length_mm': [culmen_length],
    'culmen_depth_mm': [culmen_depth],
    'flipper_length_mm': [flipper_length],
    'body_mass_g': [body_mass],
    'sex': [sex]
})

# Categorical Data Encoding using pd.get_dummies
user_input = pd.get_dummies(user_input, columns=['island', 'sex'])

# Align columns with the model's expected input
model_columns = pd.get_dummies(df, columns=['island', 'sex']).columns.drop('species')
for col in model_columns:
    if col not in user_input.columns:
        user_input[col] = 0  # Add missing columns with zeros

user_input = user_input[model_columns]  # Reorder columns

# Predict the species using the model
prediction = model.predict(user_input)

# Display the prediction result
st.subheader('Prediction Result:')
st.write('Predicted Species:', prediction[0])
