import json
import numpy as np
import os
import sys

# Ensure root workspace directory is in sys.path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if BASE_DIR not in sys.path:
    sys.path.insert(0, BASE_DIR)

def test_app_and_math():
    print("🧪 Testing Flask app and math verification...")

    # 1. Test importing app
    from app import app, CONFIDENCE_THRESHOLD, model, labels
    print(f"✅ App imported. Model status: {model is not None}, Labels: {len(labels) if labels is not None else 0}")

    client = app.test_client()

    # 2. Test GET /
    res_index = client.get("/")
    assert res_index.status_code == 200, f"GET / failed with {res_index.status_code}"
    print("✅ GET / returns 200 OK")

    # 3. Test GET /health
    res_health = client.get("/health")
    assert res_health.status_code == 200
    health_data = res_health.get_json()
    assert health_data["status"] == "healthy"
    print(f"✅ GET /health returns {health_data}")

    # 4. Test GET /api/labels
    res_labels = client.get("/api/labels")
    assert res_labels.status_code == 200
    labels_data = res_labels.get_json()
    assert len(labels_data["labels"]) == 28
    print(f"✅ GET /api/labels returns {len(labels_data['labels'])} labels")

    # 5. Test POST /predict with dummy landmarks
    # Test all zeros
    dummy_zeros = [0.0] * 63
    res_predict = client.post("/predict", json={"landmarks": dummy_zeros})
    assert res_predict.status_code == 200
    predict_data = res_predict.get_json()
    assert predict_data["status"] == "success"
    print(f"✅ POST /predict (zeros) response: {predict_data}")

    # Test bad shape
    res_bad = client.post("/predict", json={"landmarks": [0.0] * 10})
    assert res_bad.status_code == 400
    print(f"✅ POST /predict (invalid length) correctly rejected with 400")

    # 6. Test consistency between Python Keras model and static/model_weights.json
    weights_path = os.path.join(BASE_DIR, "static", "model_weights.json")
    with open(weights_path, "r") as f:
        weights_data = json.load(f)

    # Generate a realistic random test vector
    np.random.seed(42)
    sample_input = np.random.uniform(0.1, 0.9, 63).astype(np.float32)

    # Python Keras forward pass
    keras_pred = model.predict(sample_input.reshape(1, -1), verbose=0)[0]
    keras_idx = int(np.argmax(keras_pred))

    # Emulate JS forward pass in Python using the exported weights
    cur = sample_input.tolist()
    for layer in weights_data:
        W = np.array(layer["weights"])
        b = np.array(layer["bias"])
        out = np.dot(cur, W) + b
        if layer["activation"] == "relu":
            out = np.maximum(0, out)
        elif layer["activation"] == "softmax":
            exp_out = np.exp(out - np.max(out))
            out = exp_out / np.sum(exp_out)
        cur = out

    js_pred = cur
    js_idx = int(np.argmax(js_pred))

    # Verify predictions match exactly
    np.testing.assert_allclose(keras_pred, js_pred, rtol=1e-4, atol=1e-4)
    assert keras_idx == js_idx, f"Argmax mismatch: Keras={keras_idx}, JS={js_idx}"
    print(f"✅ Mathematical equivalence verified! Top class match: {labels[keras_idx]} (prob: {keras_pred[keras_idx]:.4f})")

    print("\n🎉 ALL TESTS PASSED SUCCESSFULLY!")

if __name__ == "__main__":
    test_app_and_math()
