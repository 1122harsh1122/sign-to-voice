# extract_landmarks.py
import mediapipe as mp
import cv2
import os
import pandas as pd

# Initialize MediaPipe
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=True, max_num_hands=1)

# Dataset paths
base_dirs = [
    ("data/public", "public_landmarks.csv"),
    ("data/custom", "custom_landmarks.csv") 
]

for base_path, output_csv in base_dirs:
    data, labels = [], []
    classes = sorted(os.listdir(base_path))
    print(f"\n📁 Processing: {base_path}")

    for label in classes:
        folder_path = os.path.join(base_path, label)
        if not os.path.isdir(folder_path):
            continue

        image_files = os.listdir(folder_path)
        print(f"🔤 Label '{label}': {len(image_files)} images")

        for i, img_name in enumerate(image_files):
            img_path = os.path.join(folder_path, img_name)
            img = cv2.imread(img_path)
            if img is None:
                print(f"⚠️ Skipping unreadable image: {img_path}")
                continue

            img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
            results = hands.process(img_rgb)

            if results.multi_hand_landmarks:
                landmarks = results.multi_hand_landmarks[0]
                lm_list = []
                for lm in landmarks.landmark:
                    lm_list.extend([lm.x, lm.y, lm.z])  # x, y, z
                data.append(lm_list)
                labels.append(label)

            if (i + 1) % 40 == 0 or (i + 1) == len(image_files):
                print(f"✅ {i + 1}/{len(image_files)} images done for '{label}'")

    # Save as CSV
    df = pd.DataFrame(data)
    df['label'] = labels
    df.to_csv(output_csv, index=False)
    print(f"📂 Saved landmark data to {output_csv}")
