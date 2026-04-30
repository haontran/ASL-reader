import time
import pickle
from collections import deque, Counter
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

def normalize_landmarks(row):
    wrist_x = row[0]
    wrist_y = row[1]
    wrist_z = row[2]

    normalized = []

    for i in range(0, len(row), 3):
        x = row[i] - wrist_x
        y = row[i + 1] - wrist_y
        z = row[i + 2] - wrist_z

        normalized.extend([x, y, z])

    max_value = max(abs(value) for value in normalized)

    if max_value == 0:
        return normalized

    normalized = [value / max_value for value in normalized]

    return normalized

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
    
prediction_history = deque(maxlen=10)
    
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
        normalized_row = normalize_landmarks(row)

        prediction = asl_model.predict([normalized_row])[0] 
        prediction_history.append(prediction)
        
        most_common = Counter(prediction_history).most_common(1)[0][0]

    if prediction_history:
        cv2.putText(frame, f"Prediction: {most_common}", (20, 50),
                cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("ASL Prediction", frame)

    key = cv2.waitKey(1) & 0xFF
    if key == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()