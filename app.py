from flask import Flask, request, jsonify
import joblib
import numpy as np
import os
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load trained model
model = joblib.load("water_model.pkl")

@app.route('/')
def home():
    return jsonify({
        "message": "Water Quality Prediction API is running"
    })

@app.route('/predict', methods=['POST'])
def predict():
    data = request.get_json()

    # Extract features in the SAME order as model training
    features = [
        data["ph"],
        data["Hardness"],
        data["Solids"],
        data["Chloramines"],
        data["Sulfate"],
        data["Conductivity"],
        data["Organic_carbon"],
        data["Trihalomethanes"],
        data["Turbidity"]
    ]

    final_features = np.array(features).reshape(1, -1)
    prediction = model.predict(final_features)

    result = "SAFE for Drinking" if prediction[0] == 1 else "NOT SAFE for Drinking"

    return jsonify({
        "prediction": int(prediction[0]),
        "result": result
    })

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    app.run(host="0.0.0.0", port=port)
