
import streamlit as st
import pandas as pd
import joblib

model = joblib.load("best_model.pkl")

st.title("Wellness Tourism Package Prediction")

age = st.number_input("Age", 18, 80, 35)
city_tier = st.selectbox("City Tier",[1,2,3])

monthly_income = st.number_input(
    "Monthly Income",
    value=25000
)

passport = st.selectbox(
    "Passport",
    [0,1]
)

own_car = st.selectbox(
    "Own Car",
    [0,1]
)

type_contact = st.selectbox(
    "Type Of Contact",
    ["Self Enquiry","Company Invited"]
)

occupation = st.selectbox(
    "Occupation",
    ["Salaried","Small Business","Large Business","Free Lancer"]
)

gender = st.selectbox(
    "Gender",
    ["Male","Female"]
)

input_df = pd.DataFrame({
    "Age":[age],
    "TypeofContact":[type_contact],
    "CityTier":[city_tier],
    "Occupation":[occupation],
    "Gender":[gender],
    "NumberOfPersonVisiting":[2],
    "PreferredPropertyStar":[3],
    "MaritalStatus":["Married"],
    "NumberOfTrips":[2],
    "Passport":[passport],
    "OwnCar":[own_car],
    "NumberOfChildrenVisiting":[1],
    "Designation":["Executive"],
    "MonthlyIncome":[monthly_income],
    "PitchSatisfactionScore":[3],
    "ProductPitched":["Basic"],
    "NumberOfFollowups":[2],
    "DurationOfPitch":[15]
})

st.write("Input Data")
st.dataframe(input_df)

if st.button("Predict"):

    prediction = model.predict(input_df)

    if prediction[0] == 1:

        st.success(
            "Customer is likely to purchase the Wellness Tourism Package."
        )

    else:

        st.error(
            "Customer is unlikely to purchase the Wellness Tourism Package."
        )
