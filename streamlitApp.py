# Import the required libraries
import streamlit as st
import pickle
import numpy as np
from PIL import Image
import pandas as pd

# Set the page configuration
st.set_page_config(
    page_title="Timelytics - OTD Prediction",
    page_icon="⏱️",
    layout="wide"
)

# Display the title and captions
st.title("Timelytics: Optimize your supply chain with advanced forecasting techniques")

st.caption(
    "Timelytics is an ensemble model that utilizes three powerful machine learning algorithms - XGBoost, Random Forests, and Support Vector Machines (SVM) - to accurately forecast Order to Delivery (OTD) times."
)

st.caption(
    "With Timelytics, businesses can identify potential bottlenecks and delays in their supply chain and take proactive measures to address them, reducing lead times and improving delivery times."
)

# Load the trained ensemble model
@st.cache_resource
def load_model():
    with open('./voting_model.pkl', 'rb') as file:
        model = pickle.load(file)
    return model

voting_model = load_model()

# Prediction function
def predict_otd(
    purchase_dow,
    purchase_month,
    year,
    product_size_cm3,
    product_weight_g,
    geolocation_state_customer,
    geolocation_state_seller,
    distance
):
    prediction = voting_model.predict(
        np.array([
            [
                purchase_dow,
                purchase_month,
                year,
                product_size_cm3,
                product_weight_g,
                geolocation_state_customer,
                geolocation_state_seller,
                distance
            ]
        ])
    )
    return round(prediction[0])

# Input parameters in sidebar
with st.sidebar:
    img = Image.open('./assets/supply_chain_optimisation.jpg')
    st.image(img)
    st.header("Input Parameters")
    
    purchase_dow = st.selectbox(
        "Purchased Day of the Week",
        options=list(range(7)),
        format_func=lambda x: ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"][x],
        index=3
    )
    
    purchase_month = st.selectbox(
        "Purchased Month",
        options=list(range(1, 13)),
        format_func=lambda x: [
            "January", "February", "March", "April", "May", "June",
            "July", "August", "September", "October", "November", "December"
        ][x-1],
        index=0
    )
    
    year = st.number_input("Purchased Year", min_value=2015, max_value=2025, value=2018)
    
    product_size_cm3 = st.number_input(
        "Product Size (cm³)",
        min_value=0,
        value=9328
    )
    
    product_weight_g = st.number_input(
        "Product Weight (grams)",
        min_value=0,
        value=1800
    )
    
    geolocation_state_customer = st.number_input(
        "Customer State Code",
        min_value=0,
        max_value=30,
        value=10
    )
    
    geolocation_state_seller = st.number_input(
        "Seller State Code",
        min_value=0,
        max_value=30,
        value=20
    )
    
    distance = st.number_input(
        "Distance (km)",
        min_value=0.0,
        value=475.35,
        step=0.1
    )
    
    submit = st.button("Predict Delivery Time")

# Main content area
if submit:
    with st.spinner("Calculating delivery time..."):
        prediction = predict_otd(
            purchase_dow,
            purchase_month,
            year,
            product_size_cm3,
            product_weight_g,
            geolocation_state_customer,
            geolocation_state_seller,
            distance
        )
    
    st.success(f"Predicted Order-to-Delivery Time: {prediction} days")

# Sample dataset display
st.header("Sample Dataset")
sample_data = {
    "Day of Week": ["Monday (0)", "Thursday (3)", "Tuesday (1)"],
    "Month": ["June (6)", "March (3)", "January (1)"],
    "Year": [2018, 2017, 2018],
    "Size (cm³)": [37206, 63714, 54816],
    "Weight (g)": [16250, 7249, 9600],
    "Customer State": [25, 25, 25],
    "Seller State": [20, 7, 20],
    "Distance (km)": [247.94, 250.35, 4.915],
    "Actual OTD (days)": [15, 12, 3]
}

df = pd.DataFrame(sample_data)
st.dataframe(df, hide_index=True)

# Additional information
st.markdown("""
### How to Use:
1. Adjust the input parameters in the sidebar
2. Click 'Predict Delivery Time' button
3. View the predicted Order-to-Delivery time

### State Codes:
- 0-30: Brazilian states (SP=25, RJ=20, etc.)
""")