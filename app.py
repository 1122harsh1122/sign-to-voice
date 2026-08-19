import os
import sys
import numpy as np
from flask import Flask, request, jsonify, render_template
from flask_cors import CORS

# Resolve absolute paths
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MODEL_PATH = os.path.join(BASE_DIR, "models", "keras_model.h5")
LABELS_PATH = os.path.join(BASE_DIR, "models", "labels_keras.npy")

app = Flask(__name__, static_folder="static", template_folder="templates")
CORS(app)

# Global variables for model and labels
model = None
labels = None
CONFIDENCE_THRESHOLD = 0.9

def load_sign_model():
    global model, labels
    if model is None:
        import tensorflow as tf
        print(f"🔄 Loading Keras model from: {MODEL_PATH}")
        model = tf.keras.models.load_model(MODEL_PATH)
        print("✅ Model loaded successfully.")
    
    if labels is None:
        print(f"🔄 Loading labels from: {LABELS_PATH}")
        labels = np.load(LABELS_PATH, allow_pickle=True)
        print(f"✅ Loaded {len(labels)} labels: {labels}")

# Pre-load on startup
try:
    load_sign_model()
except Exception as e:
    print(f"⚠️ Model startup load warning: {e}")

@app.route("/")
def index():
    return render_template("index.html")

@app.route("/predict", methods=["POST"])
def predict():
    global model, labels
    if model is None or labels is None:
        try:
            load_sign_model()
        except Exception as e:
            return jsonify({"status": "error", "message": f"Model loading failed: {str(e)}"}), 500

    try:
        data = request.get_json(force=True)
        if not data or "landmarks" not in data:
            return jsonify({"status": "error", "message": "Missing 'landmarks' key in JSON payload"}), 400

        raw_landmarks = data["landmarks"]
        
        # Flatten if nested (e.g. [[x, y, z], ...])
        if isinstance(raw_landmarks, list) and len(raw_landmarks) > 0 and isinstance(raw_landmarks[0], list):
            flat_landmarks = [coord for pt in raw_landmarks for coord in pt]
        else:
            flat_landmarks = raw_landmarks

        if len(flat_landmarks) != 63:
            return jsonify({
                "status": "error",
                "message": f"Expected 63 landmark coordinates (21 x 3), got {len(flat_landmarks)}"
            }), 400

        # Exact prediction logic from predict_keras_live.py
        input_array = np.array(flat_landmarks, dtype=np.float32).reshape(1, -1)
        pred_probs = model.predict(input_array, verbose=0)
        prediction_index = int(np.argmax(pred_probs))
        confidence = float(pred_probs[0][prediction_index])

        prediction = None
        if confidence > CONFIDENCE_THRESHOLD:
            prediction = str(labels[prediction_index])

        return jsonify({
            "status": "success",
            "prediction": prediction,
            "confidence": round(confidence, 4),
            "label": str(labels[prediction_index]),
            "threshold": CONFIDENCE_THRESHOLD
        })

    except Exception as e:
        return jsonify({"status": "error", "message": str(e)}), 500

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "healthy",
        "model_loaded": model is not None,
        "labels_count": len(labels) if labels is not None else 0,
        "threshold": CONFIDENCE_THRESHOLD
    })

@app.route("/api/labels", methods=["GET"])
def get_labels():
    if labels is not None:
        return jsonify({"labels": labels.tolist()})
    return jsonify({"labels": []})

if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    print(f"🚀 Starting Sign-to-Voice Flask App on http://127.0.0.1:{port}")
    app.run(host="0.0.0.0", port=port, debug=False)
