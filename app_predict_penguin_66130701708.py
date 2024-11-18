import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the trained model
with open('model_penguin_66130701708.pkl', 'rb') as file:
    model = pickle.load(file)

# Load the penguin dataset
df = pd.read_csv('penguins_size.csv')

# Streamlit App
st.title('Penguin Species Predictor')

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Predict Penguin Species', 'Visualize Data']
selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# Tab selection logic
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

# Tab 1: Predict Penguin Species
if st.session_state.tab_selected == 0:
    st.header('Predict Penguin Species')

    # User Input Form
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

    # Categorical Data Encoding
    user_input['island'] = island_encoder.transform(user_input['island'])
    user_input['sex'] = sex_encoder.transform(user_input['sex'])

    # Predicting the species
    prediction = model.predict(user_input)

    # Display Result
    st.subheader('Prediction Result:')
    st.write('Predicted Species:', prediction[0])

# Tab 2: Visualize Data
elif st.session_state.tab_selected == 1:
    st.header('Visualize Data')

    # Select a feature for visualization
    feature = st.selectbox('Select Feature to Visualize:', df.columns)

    # Plot distribution of selected feature by species
    fig, ax = plt.subplots(figsize=(14, 8))
    sns.histplot(data=df, x=feature, hue='species', multiple='stack', palette='viridis')
    plt.title(f'Distribution of {feature} by Species')
    plt.xlabel(feature)
    plt.ylabel('Count')
    st.pyplot(fig)
