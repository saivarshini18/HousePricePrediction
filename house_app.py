import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

# ---------------- PAGE CONFIG ---------------- #
st.set_page_config(
    page_title="House Price Prediction",
    page_icon="🏠",
    layout="wide"
)

# ---------------- LOAD MODEL ---------------- #
model = joblib.load("model/model.pkl")
le_location = joblib.load("model/le_location.pkl")
le_condition = joblib.load("model/le_condition.pkl")
le_garage = joblib.load("model/le_garage.pkl")

# ---------------- LOAD DATA ---------------- #
data = pd.read_csv(
    "data/HousingPricePredictionDataset.csv",
    sep="\t"
)

# Remove Id column if present
if "Id" in data.columns:
    data = data.drop("Id", axis=1)

# ---------------- TITLE ---------------- #
st.title("🏠 House Price Prediction & Analytics Dashboard")

st.write("""
This project predicts house prices based on area, bedrooms,
bathrooms, floors, year built, location, condition,
and garage availability.
""")

# ---------------- DASHBOARD METRICS ---------------- #
st.subheader("📊 Real Estate Dashboard")

col1, col2, col3, col4 = st.columns(4)

with col1:
    st.metric("Total Houses", len(data))

with col2:
    st.metric(
        "Average Price",
        f"₹ {data['Price'].mean():,.0f}"
    )

with col3:
    st.metric(
        "Highest Price",
        f"₹ {data['Price'].max():,.0f}"
    )

with col4:
    st.metric(
        "Lowest Price",
        f"₹ {data['Price'].min():,.0f}"
    )

# ---------------- DATASET PREVIEW ---------------- #
st.subheader("📋 Dataset Preview")
st.dataframe(data.head())

# ---------------- CITY ANALYTICS ---------------- #
st.subheader("🏙️ City Analytics")

selected_city = st.selectbox(
    "Select City",
    sorted(data["Location"].unique())
)

city_data = data[
    data["Location"] == selected_city
]

c1, c2, c3, c4 = st.columns(4)

with c1:
    st.metric(
        "Properties",
        len(city_data)
    )

with c2:
    st.metric(
        "Average Price",
        f"₹ {city_data['Price'].mean():,.0f}"
    )

with c3:
    st.metric(
        "Highest Price",
        f"₹ {city_data['Price'].max():,.0f}"
    )

with c4:
    st.metric(
        "Average Area",
        f"{city_data['Area'].mean():,.0f} sq.ft"
    )

# ---------------- PRICE DISTRIBUTION ---------------- #
st.subheader("📈 Price Distribution")

fig, ax = plt.subplots(figsize=(8, 4))
ax.hist(
    data["Price"],
    bins=20
)
ax.set_xlabel("House Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

# ---------------- AREA VS PRICE ---------------- #
st.subheader("📈 Area vs Price")

fig2, ax2 = plt.subplots(figsize=(8, 4))

scatter = ax2.scatter(
    data["Area"],
    data["Price"]
)

ax2.set_xlabel("Area")
ax2.set_ylabel("Price")

st.pyplot(fig2)

# ---------------- AVERAGE PRICE BY CITY ---------------- #
st.subheader("🏙️ Average Price by City")

avg_price = (
    data.groupby("Location")["Price"]
    .mean()
    .sort_values()
)

fig3, ax3 = plt.subplots(figsize=(10, 5))

avg_price.plot(
    kind="bar",
    ax=ax3
)

ax3.set_xlabel("Location")
ax3.set_ylabel("Average Price")

st.pyplot(fig3)

# ---------------- PREDICTION SECTION ---------------- #
st.subheader("🔮 Predict House Price")

col1, col2 = st.columns(2)

with col1:
    area = st.number_input(
        "Area",
        500,
        10000,
        1500
    )

    bedrooms = st.slider(
        "Bedrooms",
        1,
        10,
        3
    )

    bathrooms = st.slider(
        "Bathrooms",
        1,
        5,
        2
    )

    floors = st.slider(
        "Floors",
        1,
        5,
        2
    )

with col2:
    year = st.number_input(
        "Year Built",
        1900,
        2026,
        2000
    )

    location = st.selectbox(
        "Location",
        list(le_location.classes_)
    )

    condition = st.selectbox(
        "Condition",
        list(le_condition.classes_)
    )

    garage = st.selectbox(
        "Garage",
        list(le_garage.classes_)
    )

location_enc = le_location.transform(
    [location]
)[0]

condition_enc = le_condition.transform(
    [condition]
)[0]

garage_enc = le_garage.transform(
    [garage]
)[0]

if st.button("Predict Price"):

    input_data = np.array([
        [
            area,
            bedrooms,
            bathrooms,
            floors,
            year,
            location_enc,
            condition_enc,
            garage_enc
        ]
    ])

    prediction = model.predict(input_data)[0]

    st.success(
        f"💰 Predicted House Price: ₹ {prediction:,.2f}"
    )

    city_avg = city_data["Price"].mean()

    st.info(
        f"Average Price in {selected_city}: ₹ {city_avg:,.0f}"
    )

    difference = (
        prediction - city_avg
    ) / city_avg * 100

    if difference > 0:
        st.success(
            f"This property is {difference:.1f}% above the average price in {selected_city}."
        )
    else:
        st.warning(
            f"This property is {abs(difference):.1f}% below the average price in {selected_city}."
        )

# ---------------- MODEL EXPLANATION ---------------- #
st.subheader("📌 Model Information")

st.write("""
**Algorithm Used:** Random Forest Regressor

**Features Used:**
- Area
- Bedrooms
- Bathrooms
- Floors
- Year Built
- Location
- Condition
- Garage Availability

**Evaluation Metrics:**
- MAE (Mean Absolute Error)
- R² Score

The model learns patterns from historical house data and predicts prices for new properties.
City-wise analytics are also provided to compare predicted prices with actual market averages.
""")