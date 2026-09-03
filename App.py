import streamlit as st
import pandas as pd
import joblib

model = joblib.load("models/british_airways_rf_model.pkl")
encoders = joblib.load("models/label_encoders.pkl")

st.set_page_config(
    page_title="British Airways Booking Predictor",
    page_icon="✈️",
    layout="wide"
)

st.markdown("""
<style>
    .main {
        padding-top: 2rem;
    }

    .hero {
        padding: 10px 0 25px 0;
    }

    .hero h1 {
        font-size: 42px;
        font-weight: 700;
        margin-bottom: 8px;
    }

    .hero p {
        font-size: 17px;
        color: #aeb4c0;
    }

    .section-title {
        font-size: 27px;
        font-weight: 650;
        margin-top: 15px;
        margin-bottom: 5px;
    }

    .section-subtitle {
        color: #aeb4c0;
        margin-bottom: 25px;
    }

    .info-box {
        border: 1px solid #3a3d46;
        border-radius: 12px;
        padding: 22px;
        margin-bottom: 25px;
    }

    .result-box {
        border: 1px solid #3a3d46;
        border-radius: 12px;
        padding: 25px;
        margin-top: 25px;
        text-align: center;
    }

    .result-title {
        font-size: 25px;
        font-weight: 650;
        margin-bottom: 10px;
    }

    .probability {
        font-size: 34px;
        font-weight: 700;
        margin: 10px 0;
    }

    .footer {
        text-align: center;
        color: #8f949e;
        font-size: 14px;
        padding: 35px 0 10px 0;
    }

    div.stButton > button {
        width: 100%;
        height: 52px;
        font-size: 17px;
        font-weight: 600;
        border-radius: 9px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown("""
<div class="hero">
    <h1>British Airways Customer Booking Prediction</h1>
    <p>
        A machine learning application that predicts whether a customer
        is likely to complete a flight booking.
    </p>
</div>
""", unsafe_allow_html=True)

st.divider()

m1, m2, m3 = st.columns(3)

with m1:
    st.caption("MACHINE LEARNING MODEL")
    st.subheader("Random Forest")

with m2:
    st.caption("PREDICTION TYPE")
    st.subheader("Binary Classification")

with m3:
    st.caption("INPUT FEATURES")
    st.subheader("13")

st.divider()

st.markdown(
    '<div class="section-title">Customer & Booking Details</div>',
    unsafe_allow_html=True
)

st.markdown(
    '<div class="section-subtitle">Enter the booking information to generate a prediction.</div>',
    unsafe_allow_html=True
)

st.markdown('<div class="info-box">', unsafe_allow_html=True)

st.markdown("### Customer Information")

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

    trip_type = st.selectbox(
        "Trip Type",
        encoders["trip_type"].classes_
    )

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
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    wants_preferred_seat = st.selectbox(
        "Preferred Seat",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    wants_in_flight_meals = st.selectbox(
        "In-Flight Meals",
        [0, 1],
        format_func=lambda x: "Yes" if x == 1 else "No"
    )

    flight_duration = st.number_input(
        "Flight Duration (Hours)",
        min_value=0.0,
        value=6.5,
        step=0.5
    )

st.markdown("</div>", unsafe_allow_html=True)

st.write("")

if st.button("Predict Booking"):

    sales_channel_enc = encoders["sales_channel"].transform(
        [sales_channel]
    )[0]

    trip_type_enc = encoders["trip_type"].transform(
        [trip_type]
    )[0]

    flight_day_enc = encoders["flight_day"].transform(
        [flight_day]
    )[0]

    route_enc = encoders["route"].transform(
        [route]
    )[0]

    booking_origin_enc = encoders["booking_origin"].transform(
        [booking_origin]
    )[0]

    input_data = pd.DataFrame({
        "num_passengers": [num_passengers],
        "sales_channel": [sales_channel_enc],
        "trip_type": [trip_type_enc],
        "purchase_lead": [purchase_lead],
        "length_of_stay": [length_of_stay],
        "flight_hour": [flight_hour],
        "flight_day": [flight_day_enc],
        "route": [route_enc],
        "booking_origin": [booking_origin_enc],
        "wants_extra_baggage": [wants_extra_baggage],
        "wants_preferred_seat": [wants_preferred_seat],
        "wants_in_flight_meals": [wants_in_flight_meals],
        "flight_duration": [flight_duration]
    })

    prediction = model.predict(input_data)[0]
    probability = model.predict_proba(input_data)[0][1]

    st.markdown('<div class="result-box">', unsafe_allow_html=True)

    st.markdown(
        '<div class="result-title">Prediction Result</div>',
        unsafe_allow_html=True
    )

    if prediction == 1:

        st.success("Customer is likely to complete the booking.")

        st.markdown(
            f'<div class="probability">{probability:.2%}</div>',
            unsafe_allow_html=True
        )

        st.caption("Booking Completion Probability")

    else:

        st.warning("Customer is unlikely to complete the booking.")

        st.markdown(
            f'<div class="probability">{probability:.2%}</div>',
            unsafe_allow_html=True
        )

        st.caption("Booking Completion Probability")

    st.markdown("</div>", unsafe_allow_html=True)

st.markdown("""
<div class="footer">
    Made by <b>Nasimuddin Ansari</b><br>
    British Airways Customer Booking Prediction
</div>
""", unsafe_allow_html=True)
