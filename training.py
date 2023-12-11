# Import necessary libraries
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error
import numpy as np

# Load the preprocessed data
data = pd.read_csv("C:\Users\tarun\OneDrive\Desktop\House Prediction 1\train.csv")

# Select the features (X) and the target variable (y)
features = ['GrLivArea', 'BedroomAbvGr', 'FullBath']
target = 'SalePrice'

X = data[features]
y = data[target]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Initialize the linear regression model
model = LinearRegression()

# Train the model on the training set
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = np.sqrt(mse)

print(f"Mean Squared Error on the test set: {mse}")
print(f"Mean Absolute Error on the test set: {mae}")
print(f"Root Mean Squared Error on the test set: {rmse}")

import joblib
joblib.dump(rf_model, 'dermatology_model.pkl')
joblib.dump(label_encoder, 'label_encoder.pkl')
