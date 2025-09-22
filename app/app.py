# app/app.py
import os
from flask import Flask, request, jsonify
import joblib
import logging

# Configure logger
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load model path from env var or default
MODEL_PATH = os.environ.get('MODEL_PATH', 'models/model_v1.pkl')

# Load model at startup (this is quick for small models)
model = joblib.load(MODEL_PATH)

app = Flask(__name__)

@app.route('/health', methods=['GET'])
def health():
    # Simple health check used by load balancers
    return jsonify({'status': 'ok'})

@app.route('/predict', methods=['POST'])
def predict():
    """
    Expects JSON: {"features": [{...}, {...}] } or {"features": [[...] , [...]]}
    We'll accept either a list of dicts or a 2D list
    """
    req = request.get_json(force=True)
    if 'features' not in req:
        return jsonify({'error': "request JSON must contain 'features'"}), 400

    X = req['features']
    try:
        # model expects a DataFrame or 2D list with same columns as training
        # If user passed list of dicts, convert to DataFrame to preserve columns
        import pandas as pd
        if isinstance(X, list) and len(X) > 0 and isinstance(X[0], dict):
            X_df = pd.DataFrame(X)
        else:
            X_df = pd.DataFrame(X, columns=model.named_steps['preproc'].transformers_[0][2] + ['amount','is_foreign','prev_frauds','hour'])
        preds = model.predict(X_df)
        return jsonify({'predictions': preds.tolist()})
    except Exception as e:
        logger.exception("Prediction error")
        return jsonify({'error': str(e)}), 500

if __name__ == '__main__':
    # Dev server — use gunicorn for production
    app.run(host='0.0.0.0', port=5000, debug=True)

## Explanation:
# This Flask app serves a pre-trained ML model for fraud detection.
# It has two endpoints:
# - /health: simple GET to check if service is running
# - /predict: POST endpoint that accepts JSON with features and returns predictions
# The model is loaded once at startup for efficiency.
# The /predict endpoint handles input as either a list of dicts (with feature names)
# or a 2D list (with fixed column order). It uses pandas to ensure correct feature alignment.
# Error handling and logging are included for robustness.
# Note: In production, use a WSGI server like gunicorn to run this app.

#MODEL_PATH from env var: lets you swap models without changing code.

#joblib.load: load saved pipeline; it's ready to predict.

#/health: simple endpoint so container orchestrators can check if service is alive.

#/predict: accepts JSON payload, builds a DataFrame if necessary, calls model.predict, returns JSON. We handle errors with try/except and log exceptions.

## In production you will run this with gunicorn (we’ll show in Dockerfile).