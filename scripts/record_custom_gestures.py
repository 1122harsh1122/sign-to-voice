import cv2
import os

gesture_name = input("Enter gesture name: ").strip()
save_dir = f'data/custom/{gesture_name}'
os.makedirs(save_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
count = 0
max_images = 200

print("📸 Press SPACE to capture image, ESC to exit.")

while True:
    ret, frame = cap.read()
    if not ret:
        continue

    cv2.imshow("Recording Gesture", frame)
    key = cv2.waitKey(10)

    if key == 27:  # ESC key
        print("🚪 Exiting...")
        break
    elif key == ord(' '):  # Space key
        img_path = os.path.join(save_dir, f"{count}.jpg")
        cv2.imwrite(img_path, frame)
        print(f"✅ Saved: {img_path}")
        count += 1
        if count >= max_images:
            print("🎉 Done capturing!")
            break

cap.release()
cv2.destroyAllWindows()
