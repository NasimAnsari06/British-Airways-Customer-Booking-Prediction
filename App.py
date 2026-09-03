import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIGURATION
# ============================================================

st.set_page_config(
    page_title="British Airways Booking Predictor",
    page_icon="✈",
    layout="wide",
    initial_sidebar_state="collapsed"
)


# ============================================================
# LOAD MODEL AND ENCODERS
# ============================================================

@st.cache_resource
def load_model():
    model = joblib.load("models/british_airways_rf_model.pkl")
    encoders = joblib.load("models/label_encoders.pkl")
    return model, encoders


try:
    model, encoders = load_model()

except Exception as e:
    st.error("Unable to load the trained model or encoders.")
    st.info(
        "Please make sure the following files are available in the models folder:"
    )
    st.code(
        "models/british_airways_rf_model.pkl\n"
        "models/label_encoders.pkl"
    )
    st.stop()


# ============================================================
# HEADER
# ============================================================

st.title("British Airways Customer Booking Prediction")

st.write(
    "A machine learning application that predicts whether a customer "
    "is likely to complete a flight booking based on customer and "
    "booking-related attributes."
)

st.divider()


# ============================================================
# PROJECT OVERVIEW
# ============================================================

info_col1, info_col2, info_col3 = st.columns(3)

with info_col1:
    st.metric(
        label="Machine Learning Model",
        value="Random Forest"
    )

with info_col2:
    st.metric(
        label="Prediction Type",
        value="Binary Classification"
    )

with info_col3:
    st.metric(
        label="Input Features",
        value="13"
    )


st.divider()


# ============================================================
# INPUT FORM
# ============================================================

st.subheader("Customer & Booking Details")

st.write(
    "Enter the customer's booking information below and click "
    "**Predict Booking** to generate the prediction."
)


with st.form("booking_prediction_form"):

    # --------------------------------------------------------
    # CUSTOMER INFORMATION
    # --------------------------------------------------------

    st.markdown("#### Customer Information")

    customer_col1, customer_col2 = st.columns(2)

    with customer_col1:

        num_passengers = st.number_input(
            "Number of Passengers",
            min_value=1,
            max_value=10,
            value=1,
            step=1,
            help="Number of passengers included in the booking."
        )

        sales_channel = st.selectbox(
            "Sales Channel",
            encoders["sales_channel"].classes_,
            help="Channel through which the booking was made."
        )

    with customer_col2:

        trip_type = st.selectbox(
            "Trip Type",
            encoders["trip_type"].classes_,
            help="Type of trip selected by the customer."
        )

        booking_origin = st.selectbox(
            "Booking Origin",
            encoders["booking_origin"].classes_,
            help="Country or location from which the booking originated."
        )


    st.divider()


    # --------------------------------------------------------
    # TRAVEL INFORMATION
    # --------------------------------------------------------

    st.markdown("#### Travel Information")

    travel_col1, travel_col2 = st.columns(2)

    with travel_col1:

        route = st.selectbox(
            "Route",
            encoders["route"].classes_,
            help="Flight route selected by the customer."
        )

        flight_day = st.selectbox(
            "Flight Day",
            encoders["flight_day"].classes_,
            help="Day of the week on which the flight is scheduled."
        )

        flight_hour = st.slider(
            "Flight Hour",
            min_value=0,
            max_value=23,
            value=12,
            help="Scheduled departure hour."
        )

    with travel_col2:

        flight_duration = st.number_input(
            "Flight Duration (Hours)",
            min_value=0.0,
            max_value=50.0,
            value=6.5,
            step=0.5,
            help="Expected duration of the flight in hours."
        )

        purchase_lead = st.number_input(
            "Purchase Lead (Days)",
            min_value=0,
            max_value=1000,
            value=30,
            step=1,
            help="Number of days between booking and flight."
        )

        length_of_stay = st.number_input(
            "Length of Stay",
            min_value=1,
            max_value=365,
            value=7,
            step=1,
            help="Number of days the customer plans to stay."
        )


    st.divider()


    # --------------------------------------------------------
    # CUSTOMER PREFERENCES
    # --------------------------------------------------------

    st.markdown("#### Customer Preferences")

    preference_col1, preference_col2, preference_col3 = st.columns(3)

    with preference_col1:

        wants_extra_baggage = st.selectbox(
            "Extra Baggage",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether the customer requested extra baggage."
        )

    with preference_col2:

        wants_preferred_seat = st.selectbox(
            "Preferred Seat",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether the customer requested a preferred seat."
        )

    with preference_col3:

        wants_in_flight_meals = st.selectbox(
            "In-Flight Meals",
            [0, 1],
            format_func=lambda x: "Yes" if x == 1 else "No",
            help="Whether the customer requested an in-flight meal."
        )


    st.divider()


    # --------------------------------------------------------
    # PREDICTION BUTTON
    # --------------------------------------------------------

    predict_button = st.form_submit_button(
        "Predict Booking",
        use_container_width=True
    )


# ============================================================
# PREDICTION
# ============================================================

if predict_button:

    try:

        # ----------------------------------------------------
        # ENCODE CATEGORICAL FEATURES
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # CREATE INPUT DATAFRAME
        # ----------------------------------------------------

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


        # ----------------------------------------------------
        # MODEL PREDICTION
        # ----------------------------------------------------

        with st.spinner("Generating prediction..."):

            prediction = model.predict(input_data)[0]

            probability = model.predict_proba(
                input_data
            )[0][1]


        # ====================================================
        # RESULT SECTION
        # ====================================================

        st.divider()

        st.subheader("Prediction Result")


        # ----------------------------------------------------
        # RESULT COLUMNS
        # ----------------------------------------------------

        result_col1, result_col2 = st.columns([1.5, 1])


        with result_col1:

            if prediction == 1:

                st.success(
                    "Customer is likely to complete the booking."
                )

            else:

                st.error(
                    "Customer is unlikely to complete the booking."
                )


            st.write("Booking Completion Probability")

            st.progress(
                int(probability * 100)
            )

            st.write(
                f"**Probability: {probability:.2%}**"
            )


        with result_col2:

            if probability >= 0.75:

                confidence = "High"

            elif probability >= 0.50:

                confidence = "Moderate"

            else:

                confidence = "Low"


            st.metric(
                label="Prediction",
                value="Likely" if prediction == 1 else "Unlikely"
            )

            st.metric(
                label="Model Confidence",
                value=confidence
            )


        # ====================================================
        # INPUT SUMMARY
        # ====================================================

        st.divider()

        st.subheader("Prediction Input Summary")

        summary_col1, summary_col2 = st.columns(2)


        with summary_col1:

            st.write("**Passenger Details**")

            st.write(
                f"Passengers: {num_passengers}"
            )

            st.write(
                f"Sales Channel: {sales_channel}"
            )

            st.write(
                f"Trip Type: {trip_type}"
            )

            st.write(
                f"Booking Origin: {booking_origin}"
            )


        with summary_col2:

            st.write("**Flight Details**")

            st.write(
                f"Route: {route}"
            )

            st.write(
                f"Flight Day: {flight_day}"
            )

            st.write(
                f"Flight Hour: {flight_hour}:00"
            )

            st.write(
                f"Flight Duration: {flight_duration:.1f} hours"
            )


        # ====================================================
        # TECHNICAL INPUT DATA
        # ====================================================

        with st.expander("View Model Input Data"):

            st.dataframe(
                input_data,
                use_container_width=True,
                hide_index=True
            )


    except Exception as e:

        st.error(
            "An error occurred while generating the prediction."
        )

        st.exception(e)


# ============================================================
# MODEL INFORMATION
# ============================================================

st.divider()

with st.expander("About This Project"):

    st.write(
        """
        This project uses a Random Forest classification model to predict
        whether a customer is likely to complete a British Airways booking.

        The prediction is based on customer behaviour, booking information,
        travel details, route information and additional service preferences.

        Categorical variables are transformed using the saved label encoders
        before being passed to the trained machine learning model.
        """
    )


# ============================================================
# FOOTER
# ============================================================

st.divider()

st.caption(
    "British Airways Customer Booking Prediction | "
    "Machine Learning Project"
)
