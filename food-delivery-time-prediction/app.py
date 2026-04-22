import streamlit as st
import pickle
import numpy as np
import pandas as pd

# Load model
model = pickle.load(open("xgboost_model.pkl", "rb"))

st.title("🚀 Food Delivery Time Predictor")

# Inputs (same features used in training!)
# Input widgets for numerical features
distance = st.number_input("Distance (km)", min_value=0.0, value=7.93, step=0.1)
preparation_time = st.number_input("Preparation Time (minutes)", min_value=5, max_value=29, value=12, step=1)
courier_experience = st.slider("Courier Experience (years)", min_value=0.0, max_value=9.0, value=1.0, step=0.5)

# Input widgets for categorical features (using selectbox)
weather_options = ['Clear', 'Foggy', 'Rainy', 'Snowy', 'Windy']
selected_weather = st.selectbox("Weather", weather_options, index=weather_options.index('Clear'))

traffic_options = ['High', 'Low', 'Medium']
selected_traffic = st.selectbox("Traffic Level", traffic_options, index=traffic_options.index('Medium'))

time_of_day_options = ['Afternoon', 'Evening', 'Morning', 'Night']
selected_time_of_day = st.selectbox("Time of Day", time_of_day_options, index=time_of_day_options.index('Afternoon'))

vehicle_type_options = ['Bike', 'Car', 'Scooter']
selected_vehicle_type = st.selectbox("Vehicle Type", vehicle_type_options, index=vehicle_type_options.index('Bike'))

if st.button("Predict"):    
    # Create a dictionary to hold all feature values, initialized to False for one-hot encoded
    features_dict = {
        'Distance_km': distance,
        'Preparation_Time_min': preparation_time,
        'Courier_Experience_yrs': courier_experience,
        'Weather_Foggy': False,
        'Weather_Rainy': False,
        'Weather_Snowy': False,
        'Weather_Windy': False,
        'Traffic_Level_Low': False,
        'Traffic_Level_Medium': False,
        'Time_of_Day_Evening': False,
        'Time_of_Day_Morning': False,
        'Time_of_Day_Night': False,
        'Vehicle_Type_Car': False,
        'Vehicle_Type_Scooter': False,
    }

    # Set boolean flags based on user selection for Weather
    if selected_weather == 'Foggy':
        features_dict['Weather_Foggy'] = True
    elif selected_weather == 'Rainy':
        features_dict['Weather_Rainy'] = True
    elif selected_weather == 'Snowy':
        features_dict['Weather_Snowy'] = True
    elif selected_weather == 'Windy':
        features_dict['Weather_Windy'] = True

    # Set boolean flags based on user selection for Traffic_Level
    if selected_traffic == 'Low':
        features_dict['Traffic_Level_Low'] = True
    elif selected_traffic == 'Medium':
        features_dict['Traffic_Level_Medium'] = True

    # Set boolean flags based on user selection for Time_of_Day
    if selected_time_of_day == 'Evening':
        features_dict['Time_of_Day_Evening'] = True
    elif selected_time_of_day == 'Morning':
        features_dict['Time_of_Day_Morning'] = True
    elif selected_time_of_day == 'Night':
        features_dict['Time_of_Day_Night'] = True

    # Set boolean flags based on user selection for Vehicle_Type
    if selected_vehicle_type == 'Car':
        features_dict['Vehicle_Type_Car'] = True
    elif selected_vehicle_type == 'Scooter':
        features_dict['Vehicle_Type_Scooter'] = True

    # Ensure the order of features matches the training data
    feature_order = [
        'Distance_km', 'Preparation_Time_min', 'Courier_Experience_yrs',
        'Weather_Foggy', 'Weather_Rainy', 'Weather_Snowy', 'Weather_Windy',
        'Traffic_Level_Low', 'Traffic_Level_Medium',
        'Time_of_Day_Evening', 'Time_of_Day_Morning', 'Time_of_Day_Night',
        'Vehicle_Type_Car', 'Vehicle_Type_Scooter'
    ]

    # Create a DataFrame from the features_dict, ensuring the correct order of columns
    features_df = pd.DataFrame([features_dict], columns=feature_order)

    # Make prediction
    prediction = model.predict(features_df)

    st.success(f"Estimated Delivery Time: {prediction[0]:.2f} minutes")