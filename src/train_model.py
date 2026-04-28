import pandas as pd
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

df = pd.read_csv("data/hand_landmarks.csv", header=None)

features = df.iloc[:, 1:]
labels = df.iloc[:, 0]

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