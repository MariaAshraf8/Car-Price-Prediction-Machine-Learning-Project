import streamlit as st
import pandas as pd
import joblib


# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Car Price Predictor",
    page_icon="",
    layout="wide"
)


# ============================================================
# LOAD MODEL
# ============================================================

loaded_model = joblib.load(
    "car_price_prediction_model.pkl"
)


# ============================================================
# TITLE
# ============================================================

st.title("Car Price Prediction")

st.write(
    "Enter the details of a car to predict its estimated selling price."
)

st.info(
    "The model uses Gradient Boosting Regression to predict car prices."
)


# ============================================================
# CAR INFORMATION
# ============================================================

st.subheader("Car Information")


col1, col2, col3 = st.columns(3)


# ------------------------------------------------------------
# MAKE
# ------------------------------------------------------------

with col1:

    make = st.selectbox(
        "Car Make",
        [
            "Toyota",
            "Nissan",
            "Kia",
            "Hyundai",
            "BMW",
            "Suzuki",
            "Honda",
            "Ford",
            "Volkswagen",
            "Mercedes"
        ]
    )


# ------------------------------------------------------------
# MODEL
# ------------------------------------------------------------

with col2:

    model = st.selectbox(
        "Car Model",
        [
            "Yaris",
            "Juke",
            "Stonic",
            "i10",
            "X1",
            "Alto",
            "Fortuner",
            "Patrol",
            "1 Series"
        ]
    )


# ------------------------------------------------------------
# YEAR
# ------------------------------------------------------------

with col3:

    year = st.number_input(
        "Manufacturing Year",
        min_value=2000,
        max_value=2025,
        value=2018,
        step=1
    )


col4, col5, col6 = st.columns(3)


# ------------------------------------------------------------
# ENGINE SIZE
# ------------------------------------------------------------

with col4:

    engine_size = st.number_input(
        "Engine Size",
        min_value=0.5,
        max_value=6.0,
        value=2.0,
        step=0.1
    )


# ------------------------------------------------------------
# MILEAGE
# ------------------------------------------------------------

with col5:

    mileage = st.number_input(
        "Mileage",
        min_value=0,
        max_value=300000,
        value=80000,
        step=1000
    )


# ------------------------------------------------------------
# FUEL TYPE
# ------------------------------------------------------------

with col6:

    fuel_type = st.selectbox(
        "Fuel Type",
        [
            "Petrol",
            "Diesel",
            "Hybrid"
        ]
    )


# ------------------------------------------------------------
# TRANSMISSION
# ------------------------------------------------------------

transmission = st.selectbox(
    "Transmission",
    [
        "Manual",
        "Automatic"
    ]
)


# ============================================================
# CREATE INPUT DATAFRAME
# ============================================================

input_data = pd.DataFrame({

    "Make": [make],

    "Model": [model],

    "Year": [year],

    "Engine Size": [engine_size],

    "Mileage": [mileage],

    "Fuel Type": [fuel_type],

    "Transmission": [transmission]
})


# ============================================================
# PREDICTION
# ============================================================

st.divider()


if st.button(
    "Predict Car Price",
    use_container_width=True
):

    prediction = loaded_model.predict(
        input_data
    )

    predicted_price = prediction[0]

    # Prevent negative price
    predicted_price = max(
        0,
        predicted_price
    )


    # ========================================================
    # RESULT
    # ========================================================

    st.subheader("Predicted Car Price")

    st.metric(
        "Estimated Price",
        f"{predicted_price:,.0f}"
    )


    st.success(
        f"Estimated Car Price: {predicted_price:,.0f}"
    )


# ============================================================
# ABOUT PROJECT
# ============================================================

st.divider()


with st.expander("About Project"):

    st.write(
        """
        This project predicts the selling price of a used car
        using machine learning.

        The model uses information such as car make, model,
        manufacturing year, engine size, mileage, fuel type,
        and transmission.

        Multiple regression algorithms were compared and
        Gradient Boosting Regression was selected as the
        final model based on model evaluation results.
        """
    )


# ============================================================
# MODEL INFORMATION
# ============================================================

with st.expander("Model Information"):

    st.write(
        """
        Model: Gradient Boosting Regressor

        Preprocessing:
        StandardScaler for numerical features

        OneHotEncoder for categorical features

        Evaluation Metrics:
        MAE
        RMSE
        R²
        """
    )


# ============================================================
# FEATURES USED
# ============================================================

with st.expander("Features Used by Model"):

    st.write(
        """
        • Make
        • Model
        • Year
        • Engine Size
        • Mileage
        • Fuel Type
        • Transmission
        """
    )