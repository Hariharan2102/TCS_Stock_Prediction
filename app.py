from flask import Flask, request, jsonify
import joblib
import numpy as np

app = Flask(__name__)

# Load model and scaler
model = joblib.load("tcs_lr_model.pkl")
scaler = joblib.load("tcs_scaler.pkl")

@app.route("/")
def home():
    return "TCS Stock Price Prediction API is running"

@app.route("/predict", methods=["POST"])
def predict():
    data = request.json["features"]
    
    input_scaled = scaler.transform([data])
    prediction = model.predict(input_scaled)
    
    return jsonify({
        "predicted_next_day_price": float(prediction[0])
    })

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
