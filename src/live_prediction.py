import time
import pickle

import cv2
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

HAND_LANDMARKER_PATH = "models/hand_landmarker.task"
ASL_MODEL_PATH = "models/asl_alphabet_model.pkl"

with open(ASL_MODEL_PATH, "rb") as file:
    asl_model = pickle.load(file)
    
def landmarks_to_row(hand_landmarks):
    row = []

    for landmark in hand_landmarks:
        row.extend([landmark.x, landmark.y, landmark.z])

    return row

base_options = python.BaseOptions(model_asset_path=HAND_LANDMARKER_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=1,
)

landmarker = vision.HandLandmarker.create_from_options(options)

cap = cv2.VideoCapture(0, cv2.CAP_DSHOW)

if not cap.isOpened():
    print("Could not open webcam")
    exit()
    
while True:
    ret, frame = cap.read()

    if not ret:
        print("Failed to grab frame")
        break

    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_frame)

    timestamp_ms = int(time.time() * 1000)
    result = landmarker.detect_for_video(mp_image, timestamp_ms)

    prediction = None

    if result.hand_landmarks:
        hand_landmarks = result.hand_landmarks[0]

        row = landmarks_to_row(hand_landmarks)

        prediction = asl_model.predict([row])[0]

    if prediction:
        cv2.putText(frame, f"Prediction: {prediction}", (20, 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()