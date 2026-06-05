import streamlit as st
import pandas as pd
import joblib

# Load Model & Encoders

model = joblib.load("british_airways_rf_model.pkl")
encoders = joblib.load("label_encoders.pkl")

st.set_page_config(
    page_title="British Airways Booking Predictor",
    page_icon="✈️",
    layout="wide"
)

st.title("✈️ British Airways Customer Booking Prediction")
st.markdown("Predict whether a customer is likely to complete a booking.")

st.markdown("---")

# User Inputs

col1, col2 = st.columns(2)

with col1:

    num_passengers = st.number_input(
        "Number of Passengers",
        min_value=1,
        max_value=10,
        value=1
    )

    sales_channel = st.selectbox(
        "Sales Channel",
        encoders["sales_channel"].classes_
    )

    trip_type = st.selectbox(
        "Trip Type",
        encoders["trip_type"].classes_
    )

    purchase_lead = st.number_input(
        "Purchase Lead (Days)",
        min_value=0,
        value=30
    )

    length_of_stay = st.number_input(
        "Length of Stay",
        min_value=1,
        value=7
    )

    flight_hour = st.slider(
        "Flight Hour",
        0,
        23,
        12
    )

    flight_day = st.selectbox(
        "Flight Day",
        encoders["flight_day"].classes_
    )

with col2:

    route = st.selectbox(
        "Route",
        encoders["route"].classes_
    )

    booking_origin = st.selectbox(
        "Booking Origin",
        encoders["booking_origin"].classes_
    )

    wants_extra_baggage = st.selectbox(
        "Extra Baggage",
        [0, 1]
    )

    wants_preferred_seat = st.selectbox(
        "Preferred Seat",
        [0, 1]
    )

    wants_in_flight_meals = st.selectbox(
        "In Flight Meals",
        [0, 1]
    )

    flight_duration = st.number_input(
        "Flight Duration (Hours)",
        min_value=0.0,
        value=6.5
    )

# Encode Inputs

sales_channel_enc = encoders["sales_channel"].transform([sales_channel])[0]
trip_type_enc = encoders["trip_type"].transform([trip_type])[0]
flight_day_enc = encoders["flight_day"].transform([flight_day])[0]
route_enc = encoders["route"].transform([route])[0]
booking_origin_enc = encoders["booking_origin"].transform([booking_origin])[0]

# Prediction

if st.button("Predict Booking"):

    input_data = pd.DataFrame({

        "num_passengers":[num_passengers],
        "sales_channel":[sales_channel_enc],
        "trip_type":[trip_type_enc],
        "purchase_lead":[purchase_lead],
        "length_of_stay":[length_of_stay],
        "flight_hour":[flight_hour],
        "flight_day":[flight_day_enc],
        "route":[route_enc],
        "booking_origin":[booking_origin_enc],
        "wants_extra_baggage":[wants_extra_baggage],
        "wants_preferred_seat":[wants_preferred_seat],
        "wants_in_flight_meals":[wants_in_flight_meals],
        "flight_duration":[flight_duration]

    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown("---")

    st.subheader("Prediction Result")

    if prediction == 1:

        st.success(
            f"✅ Customer Likely to Complete Booking\n\nProbability: {probability:.2%}"
        )

    else:

        st.error(
            f"❌ Customer Unlikely to Complete Booking\n\nProbability: {probability:.2%}"
        )

st.markdown("---")
st.caption("Built by Nasimuddin Ansari | British Airways Customer Booking Prediction")