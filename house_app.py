import streamlit as st
import joblib
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

model = joblib.load("model/model.pkl")
le_location = joblib.load("model/le_location.pkl")
le_condition = joblib.load("model/le_condition.pkl")
le_garage = joblib.load("model/le_garage.pkl")

st.set_page_config(page_title="House Price Prediction", page_icon="🏠")

st.title("🏠 House Price Prediction using Machine Learning")

st.write("""
This project predicts the price of a house based on area, bedrooms, bathrooms,
floors, year built, location, condition, and garage availability.
""")

data = pd.read_csv("data/House Price Prediction Dataset.csv")

st.subheader("📊 Dataset Preview")
st.dataframe(data.head())

st.subheader("📈 Price Distribution")
fig, ax = plt.subplots()
ax.hist(data["Price"], bins=20)
ax.set_xlabel("House Price")
ax.set_ylabel("Frequency")
st.pyplot(fig)

st.subheader("📈 Area vs Price")
fig2, ax2 = plt.subplots()
ax2.scatter(data["Area"], data["Price"])
ax2.set_xlabel("Area")
ax2.set_ylabel("Price")
st.pyplot(fig2)

st.subheader("🔮 Predict House Price")

area = st.number_input("Area", 500, 10000, 1500)
bedrooms = st.slider("Bedrooms", 1, 10, 3)
bathrooms = st.slider("Bathrooms", 1, 5, 2)
floors = st.slider("Floors", 1, 5, 2)
year = st.number_input("Year Built", 1900, 2026, 2000)

location = st.selectbox("Location", list(le_location.classes_))
condition = st.selectbox("Condition", list(le_condition.classes_))
garage = st.selectbox("Garage", list(le_garage.classes_))

location_enc = le_location.transform([location])[0]
condition_enc = le_condition.transform([condition])[0]
garage_enc = le_garage.transform([garage])[0]

if st.button("Predict Price"):
    input_data = np.array([[area, bedrooms, bathrooms, floors,
                            year, location_enc, condition_enc, garage_enc]])

    prediction = model.predict(input_data)

    st.success(f"💰 Predicted House Price: ₹ {prediction[0]:,.2f}")

st.subheader("📌 Model Accuracy Explanation")

st.write("""
The model is trained using Random Forest Regressor.  
The dataset is divided into training and testing data.  
The model learns patterns from the training data and predicts prices for new inputs.

Evaluation metrics used:
- MAE shows the average prediction error.
- R² Score shows how well the model explains the price variation.

If the R² score is low or negative, it means the dataset may need more records,
better features, or improved preprocessing.
""")