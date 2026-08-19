import cv2
import mediapipe as mp
import numpy as np
import tensorflow as tf
import pyttsx3
import time

# Load Keras model and label mappings
model = tf.keras.models.load_model("models/keras_model.h5")
labels = np.load("models/labels_keras.npy", allow_pickle=True)

# MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# Text-to-Speech
engine = pyttsx3.init()

# Webcam
cap = cv2.VideoCapture(0)

last_prediction = None
last_pred_time = 0
sentence = ""
prediction_delay = 1.5  # seconds between different predictions

print("\n Starting Live Prediction with Keras Model...")

while True:
    ret, frame = cap.read()
    if not ret:
        break

    frame = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    result = hands.process(img_rgb)

    prediction = None

    if result.multi_hand_landmarks:
        hand_landmarks = result.multi_hand_landmarks[0]
        landmark_points = []

        for lm in hand_landmarks.landmark:
            landmark_points.extend([lm.x, lm.y, lm.z])

        if len(landmark_points) == 63:
            input_array = np.array(landmark_points).reshape(1, -1)
            pred_probs = model.predict(input_array, verbose=0)
            prediction_index = np.argmax(pred_probs)
            confidence = pred_probs[0][prediction_index]

            if confidence > 0.9:
                prediction = labels[prediction_index]

                if prediction != last_prediction and (time.time() - last_pred_time) > prediction_delay:
                    sentence += prediction
                    last_prediction = prediction
                    last_pred_time = time.time()

    # Display current predicted letter and sentence
    cv2.putText(frame, f"Last: {last_prediction if last_prediction else ''}", (10, 40),
                cv2.FONT_HERSHEY_SIMPLEX, 1.1, (0, 255, 0), 2)
    cv2.putText(frame, f"Sentence: {sentence}", (10, 80),
                cv2.FONT_HERSHEY_SIMPLEX, 0.9, (255, 255, 0), 2)

    cv2.imshow("Sign to Voice - Keras", frame)

    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('s'):  # Speak sentence
        if sentence:
            engine.say(sentence)
            engine.runAndWait()
            sentence = ""
    elif key == ord('b'):  # Backspace
        sentence = sentence[:-1]
    elif key== 32:    #space
        sentence += " "
    elif key == ord('c'):  # Clear all
        sentence = ""

cap.release()
cv2.destroyAllWindows()
