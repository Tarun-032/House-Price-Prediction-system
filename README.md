# House Price Prediction System

## Overview

This project implements a House Price Prediction System, allowing users to predict house prices based on various features. The system uses machine learning models, including linear regression, to make predictions.

## Files

- **app.py**: Flask web application handling user interactions and predictions.
- **data_description.txt**: Description of the dataset used for training and testing.
- **houseprediction_model.pkl**: Pickle file containing the trained house price prediction model.
- **label_encoder.pkl**: Pickle file for label encoding categorical variables.
- **linear_regression_model.pkl**: Pickle file containing the trained linear regression model.
- **predictions.csv**: CSV file containing sample predictions.
- **preprocessing.py**: Python script for data preprocessing.
- **sample_submission.csv**: Sample CSV file for submission.
- **test.csv**: Test dataset for evaluation.
- **train.csv**: Training dataset for model training.
- **training.ipynb**: Jupyter Notebook containing the model training code.
- **testing.ipynb**: Jupyter Notebook for testing and evaluating the models.

## Features

- **Predictions**: Users can input house features to receive predictions for the house prices.
- **Data Analysis**: Data analysis and preprocessing steps are included in the Jupyter Notebooks.
- **Models**: Trained machine learning models for prediction.

## Technologies Used

- **Python**: Used for scripting, model training, and web application development.
- **Flask**: Web framework for creating the prediction application.
- **Scikit-Learn**: Python library for machine learning models.
- **Pandas, NumPy**: Data manipulation and analysis.
- **Jupyter Notebooks**: Used for exploratory data analysis and model training.

## Installation and Usage

1. Clone the repository to your local machine.
2. Navigate to the project directory.
3. Run `python app.py` to start the Flask web application.
4. Open a web browser and go to `http://localhost:5000` to access the House Price Prediction System.

## Future Improvements

- Integration of frontend and backend components for a more seamless user experience.
- Enhanced model evaluation and performance.
- Improved UI/UX design for better user interaction.

