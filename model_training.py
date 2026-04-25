import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import LabelEncoder

# Load dataset
data = pd.read_csv("data/House Price Prediction Dataset.csv")

# Drop ID column
data = data.drop("Id", axis=1)

# Encode categorical columns
le_location = LabelEncoder()
le_condition = LabelEncoder()
le_garage = LabelEncoder()

data["Location"] = le_location.fit_transform(data["Location"])
data["Condition"] = le_condition.fit_transform(data["Condition"])
data["Garage"] = le_garage.fit_transform(data["Garage"])

# Features & Target
X = data.drop("Price", axis=1)
y = data["Price"]

# Split data
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Train model
model = RandomForestRegressor(n_estimators=100, random_state=42)
model.fit(X_train, y_train)

# Predictions
y_pred = model.predict(X_test)

# Evaluation
print("MAE:", mean_absolute_error(y_test, y_pred))
print("R2 Score:", r2_score(y_test, y_pred))

# Save model & encoders
joblib.dump(model, "model/model.pkl")
joblib.dump(le_location, "model/le_location.pkl")
joblib.dump(le_condition, "model/le_condition.pkl")
joblib.dump(le_garage, "model/le_garage.pkl")

print("✅ Model trained and saved successfully!")