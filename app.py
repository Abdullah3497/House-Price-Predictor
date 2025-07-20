# app.py

import streamlit as st
import joblib
import numpy as np

# 📦 Load model and encoders
model = joblib.load("xgb_model.pkl")
property_type_encoder = joblib.load("property_type_encoder.pkl")
city_encoder = joblib.load("city_encoder.pkl")
location_encoder = joblib.load("location_encoder.pkl")
purpose_encoder = joblib.load("purpose_encoder.pkl")

st.title("🏡 House Price Prediction App")

st.markdown("Enter the property details to predict the **price in PKR**:")

# 📥 User Inputs
property_type = st.selectbox("Property Type", property_type_encoder.classes_)
city = st.selectbox("City", city_encoder.classes_)
location = st.selectbox("Location", location_encoder.classes_)
purpose = st.selectbox("Purpose", purpose_encoder.classes_)
bedrooms = st.number_input("Bedrooms", min_value=1, max_value=20, value=3)
baths = st.number_input("Baths", min_value=1, max_value=20, value=2)
area = st.number_input("Area (Marla)", min_value=1.0, max_value=1000.0, value=5.0)

# 🔍 Predict
if st.button("Predict Price"):
    try:
        input_data = [
            area,
            bedrooms,
            property_type_encoder.transform([property_type])[0],
            city_encoder.transform([city])[0],
            baths,
            location_encoder.transform([location])[0],
            purpose_encoder.transform([purpose])[0]
        ]

        prediction = model.predict([input_data])[0]
        st.success(f"💰 Estimated Price: {prediction:,.0f} PKR")
    except Exception as e:
        st.error(f"❌ Error: {e}")
