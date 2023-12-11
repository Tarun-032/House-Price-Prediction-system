import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, mean_absolute_error

# Load the dataset
df = pd.read_csv("C:\Users\tarun\OneDrive\Desktop\House Prediction 1\train.csv")

# Select relevant columns for the model
selected_features = ['GrLivArea', 'BedroomAbvGr', 'FullBath', 'SalePrice']
df = df[selected_features]

# Display the first few rows of the dataset
print("Original dataset:")
print(df.head())

# Save the processed DataFrame to a CSV file
df.to_csv("train_processed.csv", index=False)

# Split the data into features (X) and the target variable (y)
X = df.drop("SalePrice", axis=1)
y = df["SalePrice"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Build and train the linear regression model
model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the test set
y_pred = model.predict(X_test)

# Evaluate the model
mse = mean_squared_error(y_test, y_pred)
mae = mean_absolute_error(y_test, y_pred)
rmse = mean_squared_error(y_test, y_pred, squared=False)  # Root Mean Squared Error

print(f"\nMean Squared Error on the test set: {mse}")
print(f"Mean Absolute Error on the test set: {mae}")
print(f"Root Mean Squared Error on the test set: {rmse}")
