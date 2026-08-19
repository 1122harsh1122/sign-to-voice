# ✋ Sign to Voice AI — Real-Time ASL Translation

A full-stack, real-time American Sign Language (ASL) to Voice translation system powered by MediaPipe Hands and a Keras Neural Network classifier.

---

## 🌟 Key Features

- **MediaPipe Hands in JavaScript**: Extracts 21 3D hand landmarks (63 coordinates) in real-time directly inside the browser.
- **Flask REST API (`/predict`)**: Wraps your trained `keras_model.h5` with exact 90% confidence thresholding and 28 gesture classes.
- **Dual Inference Mode**:
  - ⚡ **In-Browser Mode (Default)**: Instant $<2\text{ms}$ inference using client-side neural forward pass (zero server latency, works offline, 100% free Vercel hosting).
  - 🌐 **Server API Mode**: Streams landmark vectors via HTTP POST to the Flask backend `/predict` endpoint.
- **Sentence Builder**: Accumulates recognized signs with 1.5-second debounce delay and supports `space`, `del` (backspace), and `nothing` gestures.
- **Web Speech API**: Browser-native voice output with adjustable speed, pitch, voice selection, and optional Auto-Speak mode.
- **Clean, Minimal UI**: Professional dark-slate dashboard with camera overlay, skeleton toggle, live confidence gauge, and ASL alphabet reference sheet.

---

## 📁 Project Structure

```
├── app.py                     # Flask web server & /predict API
├── models/
│   ├── keras_model.h5         # Trained Keras Sequential model
│   └── labels_keras.npy       # 28 class labels
├── static/
│   ├── app.js                 # MediaPipe Hands, dual-mode inference & TTS logic
│   ├── labels.json            # Web labels asset
│   └── model_weights.json     # Precomputed weights for client-side inference
├── templates/
│   └── index.html             # Clean web frontend
├── scripts/
│   ├── export_web_model.py    # Script to regenerate static JSON model weights
│   ├── test_app.py            # Automated test suite
│   ├── train_keras_model.py   # Model training script
│   └── predict_keras_live.py  # Original desktop prediction script
├── requirements.txt           # Python dependencies
├── vercel.json                # Vercel serverless deployment config
└── .vercelignore              # Ignore heavy datasets during deployment
```

---

## 🚀 Quick Start (Local Run)

### 1. Install Dependencies
```bash
pip install -r requirements.txt
```

### 2. Start the Application
```bash
python app.py
```

### 3. Open in Browser
Visit **`http://localhost:5000`** in Chrome, Edge, Firefox, or Safari, and allow camera access.

---

## ⌨️ Keyboard Shortcuts

| Key | Action |
| :--- | :--- |
| <kbd>Space</kbd> | Insert space in sentence |
| <kbd>Backspace</kbd> | Delete last character |
| <kbd>S</kbd> | Speak full sentence |
| <kbd>C</kbd> | Clear sentence |
| <kbd>P</kbd> | Pause / Resume camera detection |

---

## ☁️ Deploying to Vercel

1. Push this repository to GitHub (ensure `data/` and large `.csv` files are ignored via `.gitignore` / `.vercelignore`).
2. Go to [Vercel Dashboard](https://vercel.com) and click **"Add New Project"**.
3. Select your repository and click **Deploy**.
4. The frontend runs with zero latency in the browser using MediaPipe and TensorFlow.js forward pass!
