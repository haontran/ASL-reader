import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

def normalize_landmarks(row):
    row = row.to_numpy()

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

df = pd.read_csv("data/hand_landmarks.csv", header=None)

raw_features = df.iloc[:, 1:]
labels = df.iloc[:, 0]

features = raw_features.apply(normalize_landmarks, axis=1, result_type="expand")

X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)

model = RandomForestClassifier(n_estimators=100)
model.fit(X_train, y_train)

print("Model trained")

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Accuracy:", accuracy)

with open("models/asl_alphabet_model.pkl", "wb") as file:
    pickle.dump(model, file)

print("Model saved to models/asl_alphabet_model.pkl")