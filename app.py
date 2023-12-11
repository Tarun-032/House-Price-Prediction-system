from flask import Flask, render_template, request
import pandas as pd
from joblib import load

app = Flask(__name__, template_folder='templates', static_folder='static')

# Load the label encoder and linear regression model
label_encoder = load('label_encoder.pkl')
model = load('linear_regression_model.pkl')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['GET', 'POST'])
def predict():
    if request.method == 'POST':
        # Get form inputs
        GrLivArea = float(request.form['GrLivArea'])
        BedroomAbvGr = float(request.form['BedroomAbvGr'])
        FullBath = float(request.form['FullBath'])

        # Create a DataFrame with the user input
        user_input = pd.DataFrame({
            'GrLivArea': [GrLivArea],
            'BedroomAbvGr': [BedroomAbvGr],
            'FullBath': [FullBath]
        })

        # Make predictions using the model
        prediction = model.predict(user_input)[0]

        # Pass the prediction to the result.html template
        return render_template('result.html', prediction=prediction)

    return render_template('index.html')

if __name__ == '__main__':
    app.run(debug=True)
