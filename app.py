from flask import Flask, render_template, request
import numpy as np
import pandas as pd
import yfinance as yf
import joblib
from tensorflow.keras.models import load_model
from datetime import datetime, timedelta

app = Flask(__name__)
model = load_model('btc_lstm_model.h5')
scaler = joblib.load('scaler.pkl')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    # Get target future date from user input
    future_date_str = request.form['future_date']
    future_date = datetime.strptime(future_date_str, "%Y-%m-%d")
    today = datetime.today()
    
    # Calculate how many days ahead
    n_days = (future_date - today).days
    if n_days <= 0:
        return render_template('index.html', error="Please enter a future date.")

    # Load recent BTC data
    df = yf.download('BTC-USD', period='100d')
    df = df[['Close', 'Volume']]
    df['MA20'] = df['Close'].rolling(window=20).mean()
    df['MA50'] = df['Close'].rolling(window=50).mean()
    df.dropna(inplace=True)

    scaled_data = scaler.transform(df)
    input_sequence = scaled_data[-60:].tolist()

    # Predict n_days into the future
    for _ in range(n_days):
        X_input = np.array([input_sequence[-60:]])
        next_pred = model.predict(X_input)[0][0]
        
        # Add dummy values for other features to keep dimensionality
        input_sequence.append([next_pred, 0, 0, 0])

    # Inverse scale predicted close price
    predicted_price = scaler.inverse_transform([[next_pred, 0, 0, 0]])[0][0]
    predicted_price = round(predicted_price, 2)

    return render_template('index.html', price=predicted_price, date=future_date_str)

if __name__ == '__main__':
    app.run(debug=True)
