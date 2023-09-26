from flask import Flask, render_template, request
import pickle
import numpy as np
import pandas as pd

app = Flask(__name__)

# Load the saved Linear Regression model
with open('model.pkl', 'rb') as model_file:
    reg_model = pickle.load(model_file)

# Load the insurance data from CSV
df = pd.read_csv('insu.csv')

@app.route('/')
def home():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if request.method == 'POST':
        age = float(request.form['age'])
        height = float(request.form['height'])
        weight = float(request.form['weight'])

        # Use the loaded model to make predictions
        input_data = np.array([[age, height, weight]])
        predicted_premium = reg_model.predict(input_data)

        return render_template('index.html', prediction=f'Predicted Premium: Rs {predicted_premium[0]:.2f}')

if __name__ == '__main__':
    app.run(debug=True)
