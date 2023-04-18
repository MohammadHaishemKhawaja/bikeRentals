import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression

# Load data
bike_rentals = pd.read_csv("bike_rentals.csv")

# Preprocess data
bike_rentals["temp_scaled"] = bike_rentals["temp"] / bike_rentals["temp"].max()
bike_rentals["atemp_scaled"] = bike_rentals["atemp"] / bike_rentals["atemp"].max()
bike_rentals["humidity_scaled"] = bike_rentals["humidity"] / bike_rentals["humidity"].max()
bike_rentals["windspeed_scaled"] = bike_rentals["windspeed"] / bike_rentals["windspeed"].max()

season_dummies = pd.get_dummies(bike_rentals["season"], prefix="season")
weather_dummies = pd.get_dummies(bike_rentals["weather"], prefix="weather")
bike_rentals = pd.concat([bike_rentals, season_dummies, weather_dummies], axis=1)

# Split data into training and test sets
X = bike_rentals.drop(["cnt"], axis=1)
y = bike_rentals["cnt"]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Train a linear regression model
linreg = LinearRegression()
linreg.fit(X_train, y_train)

# Evaluate the model on the test set
y_pred = linreg.predict(X_test)
score = linreg.score(X_test, y_test)
print(f"R-squared score: {score:.2f}")

# Analyze performance and interpret coefficients
plt.scatter(y_test, y_pred)
plt.xlabel("Actual bike rentals")
plt.ylabel("Predicted bike rentals")
plt.title("Actual vs. Predicted Bike Rentals")
plt.show()

coefficients = pd.DataFrame({"feature": X.columns, "coefficient": linreg.coef_})
coefficients = coefficients.sort_values(by="coefficient", ascending=False)
print(coefficients)

# Use the model to make predictions on new data
new_data = pd.DataFrame({
    "season_1": [0],
    "season_2": [1],
    "season_3": [0],
    "season_4": [0],
    "weather_1": [1],
    "weather_2": [0],
    "weather_3": [0],
    "temp_scaled": [0.7],
    "atemp_scaled": [0.65],
    "humidity_scaled": [0.7],
    "windspeed_scaled": [0.3],
    "hour": [10],
    "holiday": [0],
    "workingday": [1]
})
prediction = linreg.predict(new_data)
print(f"Predicted bike rentals: {prediction[0]:.2f}")
