
import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

# Load the trained model (and possibly other items)
with open('model_penguin_66130701708.pkl', 'rb') as file:
    # If the pickle file contains multiple objects, unpack them
    loaded_objects = pickle.load(file)
    
    # Assuming the model is the first item in the tuple
    model = loaded_objects[0]  # Adjust index if necessary

# Load the penguin dataset
df = pd.read_csv('penguins_size.csv')

# Streamlit App
st.title('Penguin Species Predictor')

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Predict Penguin Species', 'Visualize Data', 'Predict from CSV']
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
    user_input = pd.get_dummies(user_input, columns=['island', 'sex'])

    # Align columns with the model's expected input
    # Use the columns from the dataset as a reference
    model_columns = pd.get_dummies(df, columns=['island', 'sex']).columns.drop('species')
    for col in model_columns:
        if col not in user_input.columns:
            user_input[col] = 0  # Add missing columns with zeros
    user_input = user_input[model_columns]  # Reorder columns

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

# Tab 3: Predict from CSV
elif st.session_state.tab_selected == 2:
    st.header('Predict from CSV')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])

    if uploaded_file is not None:
        # Read CSV file
        csv_df = pd.read_csv(uploaded_file)
        csv_df = csv_df.dropna()  # Handle missing values by dropping rows

        # Prepare data for prediction
        csv_df_encoded = pd.get_dummies(csv_df, columns=['island', 'sex'])

        # Align columns with the model's expected input
        for col in model_columns:
            if col not in csv_df_encoded.columns:
                csv_df_encoded[col] = 0
        csv_df_encoded = csv_df_encoded[model_columns]

        # Predict species for each row
        predictions = model.predict(csv_df_encoded)
        csv_df['Predicted Species'] = predictions

        # Display the DataFrame with predictions
        st.subheader('Predicted Results:')
        st.write(csv_df)

        # Visualize predictions
        st.subheader('Visualize Predictions')
        feature_for_visualization = st.selectbox('Select Feature for Visualization:', csv_df.columns)

        # Plot predictions
        fig, ax = plt.subplots(figsize=(14, 8))
        sns.histplot(data=csv_df, x=feature_for_visualization, hue='Predicted Species', multiple='stack', palette='viridis')
        plt.title(f'Predicted Species Distribution by {feature_for_visualization}')
        plt.xlabel(feature_for_visualization)
        plt.ylabel('Count')
        st.pyplot(fig)

