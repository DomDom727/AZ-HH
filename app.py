from flask import Flask, request, jsonify
import pickle
import numpy as np
import os

# Load the model
with open("model.pkl", "rb") as f:
    model = pickle.load(f)

app = Flask(__name__)

@app.route("/")
def home():
    return "Model API is running."

@app.route("/predict", methods=["POST"])
def predict():
    try:
        # Expecting a JSON payload with a key "features"
        data = request.get_json()
        features = np.array(data["features"]).reshape(1, -1)
        
        # Predict using the loaded model
        prediction = model.predict(features)
        
        return jsonify({"prediction": prediction.tolist()})
    except Exception as e:
        return jsonify({"error": str(e)}), 400

if __name__ == "__main__":
    app.run(debug=False, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))

    