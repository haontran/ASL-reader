import time
import cv2
import csv
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision

DATA_FILE = "data/hand_landmarks.csv"
MODEL_PATH = "models/hand_landmarker.task"

file = open(DATA_FILE, mode="a", newline="")
writer = csv.writer(file)

last_save_time = 0
SAVE_DELAY = 1.0
sample_counts = {}

def draw_landmarks(frame, hand_landmarks):
    height, width, _ = frame.shape
    
    connections = [
        (0, 1), (1, 2), (2, 3), (3, 4),       # thumb
        (0, 5), (5, 6), (6, 7), (7, 8),       # index
        (0, 9), (9, 10), (10, 11), (11, 12),  # middle
        (0, 13), (13, 14), (14, 15), (15, 16),# ring
        (0, 17), (17, 18), (18, 19), (19, 20),# pinky
        (5, 9), (9, 13), (13, 17)             # palm
    ]
    
    points = []

    for landmark in hand_landmarks:
        x = int(landmark.x * width)
        y = int(landmark.y * height)
        points.append((x, y))

    for start, end in connections:
        cv2.line(frame, points[start], points[end], (255, 255, 255), 2)

    for point in points:
        cv2.circle(frame, point, 5, (0, 255, 0), -1)
        
def landmarks_to_row(hand_landmarks):
    row = []
    
    for landmark in hand_landmarks:
        row.extend([landmark.x, landmark.y, landmark.z])
    
    return row

base_options = python.BaseOptions(model_asset_path=MODEL_PATH)

options = vision.HandLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,
    num_hands=2,
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

    current_hand_landmarks = None

    if result.hand_landmarks:
        current_hand_landmarks = result.hand_landmarks[0]

        for hand_landmarks in result.hand_landmarks:
            draw_landmarks(frame, hand_landmarks)

    cv2.imshow("Hand Detection", frame)

    key = cv2.waitKey(1) & 0xFF

    current_time = time.time()

    if current_hand_landmarks and 97 <= key <= 122:
        if current_time - last_save_time > SAVE_DELAY:
            label = chr(key).upper()
            row = [label] + landmarks_to_row(current_hand_landmarks)
            writer.writerow(row)
            sample_counts[label] = sample_counts.get(label, 0) + 1
            print(f"Saved sample for {label}. Count this run: {sample_counts[label]}")
            last_save_time = current_time

    if key == 27:
        break

cap.release()
landmarker.close()
cv2.destroyAllWindows()
file.close()
