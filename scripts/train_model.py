import pandas as pd
from sklearn.preprocessing import LabelEncoder
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import joblib
import numpy as np
import os

# Paths to CSV files
public_csv = "data/public_landmarks.csv"
custom_csv = "data/custom_landmarks.csv"

# Load datasets
dfs = []
if os.path.exists(public_csv):
    dfs.append(pd.read_csv(public_csv))
if os.path.exists(custom_csv):
    dfs.append(pd.read_csv(custom_csv))

# Combine datasets
df = pd.concat(dfs, ignore_index=True)

# Separate features and labels
X = df.drop("label", axis=1)
y = df["label"]

# Encode labels
le = LabelEncoder()
y_encoded = le.fit_transform(y)

# Save labels to file for later decoding
np.save("models/labels.npy", le.classes_)

# Split dataset
X_train, X_test, y_train, y_test = train_test_split(X, y_encoded, test_size=0.2, random_state=42)

# Train model
model = RandomForestClassifier(n_estimators=150, random_state=42)
model.fit(X_train, y_train)

# Evaluate
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"✅ Model trained with accuracy: {accuracy * 100:.2f}%")

# Save model
os.makedirs("models", exist_ok=True)
joblib.dump(model, "models/gesture_model.pkl")
print("💾 Model saved to models/gesture_model.pkl")
