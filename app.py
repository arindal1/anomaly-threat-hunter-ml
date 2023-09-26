import streamlit as st
import pickle
import numpy as np

# Load the Random Forest model
with open('selected_randomforest.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the features used by the model
selected_features = ['dst_bytes', 'unknown_feature', 'dst_host_count', 'dst_host_srv_count', 'dst_host_same_srv_rate', 'dst_host_srv_serror_rate', 'service_http', 'service_private', 'neptune_neptune', 'neptune_normal']

st.title('Intrusion Detection System')

# Get user input for selected features
user_input = {}
for feature in selected_features:
    user_input[feature] = st.slider(f'Enter {feature}', min_value=0.0, max_value=1.0, step=0.01)

# Predict upon button press
if st.button('Predict'):
    # Convert user input to a numpy array
    input_array = np.array([user_input[feature] for feature in selected_features]).reshape(1, -1)
    
    # Make predictions
    prediction = model.predict(input_array)

    # Display the prediction
    if prediction == 0:
        st.success('Prediction: Normal traffic (class 0)')
    else:
        st.error('Prediction: Anomaly detected (class 1)')
