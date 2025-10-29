# app.py
from flask import Flask, request, jsonify
import joblib
import numpy as np
import os

app = Flask(__name__)

# adjust path if needed
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "src", "fraud_model.pkl")
SCALER_PATH = os.path.join(BASE_DIR, "src", "scaler.pkl")

model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)  # if you used one

@app.route("/")
def home():
    return "Fraud Detection API is up."

@app.route("/predict", methods=["POST"])
def predict():
    """
    Expects JSON:
    { "features": [v1, v2, v3, ...] }
    or for batch:
    { "features": [[...], [...]] }
    """
    data = request.get_json()
    if not data or "features" not in data:
        return jsonify({"error": "JSON with 'features' required"}), 400

    X = np.array(data["features"])
    # if single row, reshape
    if X.ndim == 1:
        X = X.reshape(1, -1)

    # apply scaler if used
    try:
        X = scaler.transform(X)
    except Exception:
        pass

    preds = model.predict(X)
    probs = None
    try:
        probs = model.predict_proba(X).tolist()
    except Exception:
        pass

    return jsonify({"predictions": preds.tolist(), "probabilities": probs})

if __name__ == "__main__":
    app.run(debug=True, host="0.0.0.0", port=int(os.environ.get("PORT", 5000)))
