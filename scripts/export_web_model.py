import os
import json
import numpy as np
import tensorflow as tf

def export_model():
    model_path = os.path.join("models", "keras_model.h5")
    labels_path = os.path.join("models", "labels_keras.npy")
    static_dir = "static"
    os.makedirs(static_dir, exist_ok=True)

    print("Loading model and labels...")
    model = tf.keras.models.load_model(model_path)
    labels = np.load(labels_path, allow_pickle=True)

    # Save labels as JSON
    labels_list = labels.tolist()
    with open(os.path.join(static_dir, "labels.json"), "w") as f:
        json.dump(labels_list, f, indent=2)
    print(f"✅ Saved static/labels.json ({len(labels_list)} classes)")

    # Extract Dense layer weights & biases
    # The architecture is: Dense(128, relu) -> Dropout -> Dense(64, relu) -> Dropout -> Dense(28, softmax)
    weights_data = []
    for layer in model.layers:
        weights = layer.get_weights()
        if len(weights) == 2:  # Weight matrix and Bias vector
            W, b = weights
            weights_data.append({
                "name": layer.name,
                "activation": layer.activation.__name__ if hasattr(layer, "activation") else "linear",
                "weights": W.tolist(),  # shape: (in_dim, out_dim)
                "bias": b.tolist()      # shape: (out_dim,)
            })
            print(f"Layer {layer.name}: W shape {W.shape}, bias shape {b.shape}, activation: {layer.activation.__name__}")

    with open(os.path.join(static_dir, "model_weights.json"), "w") as f:
        json.dump(weights_data, f)
    print(f"✅ Saved static/model_weights.json ({len(weights_data)} dense layers)")

if __name__ == "__main__":
    export_model()
