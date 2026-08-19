import cv2
import mediapipe as mp
import joblib
import numpy as np
import pyttsx3
import time

# Load the trained model
model = joblib.load("models/gesture_model.pkl")

# Load class labels (ensure these are the words corresponding to the gestures)
labels = np.load("models/labels.npy", allow_pickle=True)

# Initialize MediaPipe Hands
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(max_num_hands=1)

# Initialize Text-to-Speech engine
engine = pyttsx3.init()

# Initialize webcam
cap = cv2.VideoCapture(0)

# Initialize variables for predictions
last_prediction = None
frame_counter = 0
speak_delay = 30  # Delay for speaking prediction
sentence = ""  # The accumulated sentence
backspace_gesture = "backspace"  # Define the gesture for backspace
clear_gesture = "clear"  # Define the gesture for clear
last_hand_detected_time = time.time()  # To track when the last hand was detected

while True:
    ret, frame = cap.read()
    if not ret:
        break

    img = cv2.flip(frame, 1)
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = hands.process(img_rgb)

    # Check if hands are detected
    hand_detected = False
    if results.multi_hand_landmarks:
        hand_detected = True
        for hand_landmarks in results.multi_hand_landmarks:
            landmarks = []
            for lm in hand_landmarks.landmark:
                landmarks.extend([lm.x, lm.y, lm.z])  # Use x, y, z for more accuracy

            if len(landmarks) == 63:  # 21 landmarks * 3 (x, y, z)
                prediction_index = model.predict([landmarks])[0]
                prediction = labels[prediction_index]
                frame_counter += 1

                # Speak only after enough frames (to avoid repetitive speech)
                if prediction != last_prediction and frame_counter > speak_delay:
                    print(f"🖐️ Recognized: {prediction}")
                    # Handle backspace and clear gestures
                    if prediction == backspace_gesture and len(sentence) > 0:
                        sentence = sentence[:-1]  # Remove last character
                        print(f"🖋️ Sentence after backspace: {sentence}")
                    elif prediction == clear_gesture:
                        sentence = ""  # Clear sentence
                        print(f"🧹 Sentence cleared.")
                    elif prediction != "space":  # Add only non-space gestures to sentence
                        sentence += prediction  # Add the recognized word to the sentence
                    else:
                        # Add space between words if it's the "space" gesture
                        sentence += " "

                    last_prediction = prediction
                    frame_counter = 0  # Reset the frame counter after a valid prediction

                # Show current word or action
                cv2.putText(img, f"Word: {prediction}", (10, 50),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)

    # If no hand is detected, don't add anything to the sentence
    if not hand_detected:
        last_hand_detected_time = time.time()  # Update the last hand detection time

    # Display accumulated sentence
    cv2.putText(img, f"Sentence: {sentence.strip()}", (10, 90),
                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

    # Show webcam feed
    cv2.imshow("Sign to Voice - Sentence Builder", img)

    # Handle key presses for speech or exit
    key = cv2.waitKey(1)
    if key == 27:  # ESC to exit
        break
    elif key == ord('s'):  # Press 's' to speak full sentence
        if sentence.strip():
            print(f"🗣️ Speaking: {sentence.strip()}")
            engine.say(sentence.strip())
            engine.runAndWait()
            sentence = ""  # Clear the sentence after speaking

cap.release()
cv2.destroyAllWindows()
