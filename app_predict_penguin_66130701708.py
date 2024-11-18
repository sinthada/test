

import streamlit as st
import pandas as pd
import pickle
from sklearn.preprocessing import LabelEncoder

# Load the trained KNN model (and possibly other objects)
with open('model_penguin_66130701708.pkl', 'rb') as file:
    loaded_objects = pickle.load(file)
    model = loaded_objects[0]  # Adjust the index if necessary

# Load the penguin dataset for reference
df = pd.read_csv('penguins_size.csv')

# Initialize label encoders (assuming they were used during training)
island_encoder = LabelEncoder()
sex_encoder = LabelEncoder()

# Fit encoders on the original data
island_encoder.fit(df['island'])
sex_encoder.fit(df['sex'].dropna())

# Streamlit App
st.title('Penguin Species Predictor')

# User Input Form
st.header('Input Penguin Measurements to Predict Species')

island = st.selectbox('Island', island_encoder.classes_)
culmen_length = st.slider('Culmen Length (mm)', float(df['culmen_length_mm'].min()), float(df['culmen_length_mm'].max()))
culmen_depth = st.slider('Culmen Depth (mm)', float(df['culmen_depth_mm'].min()), float(df['culmen_depth_mm'].max()))
flipper_length = st.slider('Flipper Length (mm)', float(df['flipper_length_mm'].min()), float(df['flipper_length_mm'].max()))
body_mass = st.slider('Body Mass (g)', float(df['body_mass_g'].min()), float(df['body_mass_g'].max()))
sex = st.selectbox('Sex', sex_encoder.classes_)

# Encode categorical variables
island_encoded = island_encoder.transform([island])[0]
sex_encoded = sex_encoder.transform([sex])[0]

# Create a DataFrame for the user input
user_input = pd.DataFrame({
    'island': [island_encoded],
    'culmen_length_mm': [culmen_length],
    'culmen_depth_mm': [culmen_depth],
    'flipper_length_mm': [flipper_length],
    'body_mass_g': [body_mass],
    'sex': [sex_encoded]
})

# Predict the species using the model
prediction = model.predict(user_input)

# Display the prediction result
st.subheader('Prediction Result:')
st.write('Predicted Species:', prediction[0])
