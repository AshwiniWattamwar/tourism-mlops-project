
import os
import streamlit as st
import pandas as pd
import joblib

MODEL_PATH = os.path.join(
    os.path.dirname(__file__),
    "best_model.pkl"
)

model = joblib.load(MODEL_PATH)

st.title("Wellness Tourism Package Prediction")

# User Inputs

age = st.number_input(
    "Age",
    min_value=18,
    max_value=80,
    value=35
)

monthly_income = st.number_input(
    "Monthly Income",
    value=50000
)

number_of_trips = st.number_input(
    "Number Of Trips",
    value=2
)

passport = st.selectbox(
    "Passport",
    [0, 1]
)

own_car = st.selectbox(
    "Own Car",
    [0, 1]
)

# Create DataFrame

input_df = pd.DataFrame({
    "Age": [age],
    "MonthlyIncome": [monthly_income],
    "NumberOfTrips": [number_of_trips],
    "Passport": [passport],
    "OwnCar": [own_car]
})

st.write("Input Data")
st.dataframe(input_df)

if st.button("Predict"):
    st.success("Prediction generated successfully")
