from flask import Flask, request, jsonify
from flask_cors import CORS
import pickle
import numpy as np
import os

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)
CORS(app)

# Define the expected field order
FEATURE_ORDER = [
    "Age"
    "BMI",
    "Smoking",
    "PhysicalActivity",
    "DietQuality",
    "SleepQuality",
    "FamilyHistoryAsthma",
    "HistoryOfAllergies",
    "HayFever",
    "GastroesophagealReflux",
    "Wheezing",
    "ShortnessOfBreath",
    "ChestTightness",
    "Coughing",
    "ExerciseInduced"
]


@app.route("/")
def home():
    return "Model is running"

@app.route("/predict", methods=["POST"])
def predict():
    try:
        data = request.get_json()

        # Extract features in correct order
        features = [data.get(key, 0) for key in FEATURE_ORDER]
        features = np.array(features).reshape(1, -1)

        prediction = model.predict(features)
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

#if __name__ == "__main__":
 #   app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))