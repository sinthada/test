import streamlit as st
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
%matplotlib inline
import seaborn as sns
import pickle

# Load model and encoders
with open('model_penguin_66130701708.pkl', 'rb') as file:
     model, species_encoder, island_encoder ,sex_encoder = pickle.load(file)

# Load your DataFrame
# Replace 'your_data.csv' with the actual file name or URL
df = pd.read_csv('/content/penguins_size.csv.csv')

# Streamlit App
st.title('app_predict_penguin_66130701708')

# Define a session state to remember tab selections
if 'tab_selected' not in st.session_state:
    st.session_state.tab_selected = 0

# Create tabs for prediction and visualization
tabs = ['Predict Penguin_Species', 'Visualize Data', 'Predict from CSV']
selected_tab = st.radio('Select Tab:', tabs, index=st.session_state.tab_selected)

# Tab selection logic
if selected_tab != st.session_state.tab_selected:
    st.session_state.tab_selected = tabs.index(selected_tab)

# Tab 1: Predict Penguin_Species
if st.session_state.tab_selected == 0:
    st.header('Predict Penguin_Species')

    # User Input Form
    x_new['culmen_length_mm'] = [40.0]
    x_new['culmen_depth_mm'] = [22.5]
    x_new['flipper_length_mm'] = [200.0]
    x_new['body_mass_g'] = [4550]
    
    # Create a DataFrame for the user input
    user_input = pd.DataFrame({
        'culmen_length_mm': [40.0],
        'culmen_depth_mm': [22.5],
        'flipper_length_mm': [200.0],
        'body_mass_g': [4550]
    })

    # Predicting
    prediction = model.predict(user_input)

    # Display Result
    st.subheader('Prediction Result')
    st.write(f'Predicted Penguin_Species: {prediction[0]}')

# Tab 2: Visualize Data
elif st.session_state.tab_selected == 1:
    st.header('Visualize Data')

    # Select condition feature
    condition_feature = st.selectbox('Select Condition Feature:', df.columns)

    # Set default condition values
    default_condition_values = ['Select All'] + df[condition_feature].unique().tolist()

    # Select condition values
    condition_values = st.multiselect('Select Condition Values:', default_condition_values)

    # Handle 'Select All' choice
    if 'Select All' in condition_values:
        condition_values = df[condition_feature].unique().tolist()

    if len(condition_values) > 0:
        # Filter DataFrame based on selected condition
        filtered_df = df[df[condition_feature].isin(condition_values)]

        # Create a countplot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=filtered_df, x=condition_feature)
        plt.title(f'Countplot of {condition_feature}')
        plt.xticks(rotation=45)
        st.pyplot(plt)

# Tab 3: Predict from CSV
elif st.session_state.tab_selected == 2:
    st.header('Predict from CSV')

    # Upload CSV file
    uploaded_file = st.file_uploader("Upload CSV file", type=["csv"])
    # uploaded_file
    
    if uploaded_file is not None:
        # Read CSV file
        csv_df_org = pd.read_csv(uploaded_file)
        csv_df_org = csv_df_org.dropna()
        # csv_df_org.columns
        
        csv_df = csv_df_org.copy()
        
        # Predicting
        predictions = model.predict(csv_df)

        # Add prediction to the DataFrame
        csv_df_org['Predicted Penguin_Species:'] = predictions

        # Display the DataFrame with predictions
        st.subheader('Predicted Results:')
        st.write(csv_df_org)

        # Visualize predictions based on a selected feature
        st.subheader('Visualize Predictions')

        # Select feature for visualization
        feature_for_visualization = st.selectbox('Select Feature for Visualization:', csv_df_org.columns)

        # Create a countplot
        plt.figure(figsize=(10, 6))
        sns.countplot(data=filtered_df, x=condition_feature)
        plt.title(f'Countplot of {condition_feature}')
        plt.xticks(rotation=45)
        st.pyplot(plt)
