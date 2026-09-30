
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

st.write("Model Loaded Successfully")
