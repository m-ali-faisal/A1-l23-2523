print("helloooo")

import pandas as pd
import joblib

from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score
from sklearn.preprocessing import MinMaxScaler

data = pd.read_csv("data/flight_dataset.csv")

print("loaded")
print(data.shape)


data = pd.get_dummies(
    data,
    columns=["Airline", "Source", "Destination"]
)


X = data.drop("Price", axis=1)

scaler = MinMaxScaler()
X = scaler.fit_transform(X)

y = data["Price"]


X_train, X_test, y_train, y_test = train_test_split(
    X,
    y,
    test_size=0.2,
    random_state=42
)


model = RandomForestRegressor(
    n_estimators=100,
    random_state=42
)


print("Trainin..............")

model.fit(X_train, y_train)


predictions = model.predict(X_test)


mae = mean_absolute_error(y_test, predictions)
r2 = r2_score(y_test, predictions)

print("Mean Absolute Error:", mae)
print("R2 Score:", r2)


joblib.dump(
    model,
    "model/flight_price_model.joblib"
)

print("DONE")
