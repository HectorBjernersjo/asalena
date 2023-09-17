# detector.py
from pathlib import Path
import pickle
import face_recognition
from collections import Counter
import cv2
import time

DEFAULT_ENCODINGS_PATH = Path("face_recognition/encodingsold.pkl")

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


def recognize_multiple_faces(input_image, loaded_encodings, input_face_locations):
    input_face_encodings = face_recognition.face_encodings(input_image, input_face_locations)

    names = []

    for unknown_encoding in input_face_encodings:
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        names.append(name)

    return names


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]
    
def recognize_one_face(input_image, encodings, face_location):
    input_face_encodings = face_recognition.face_encodings(input_image, [face_location])
    face_encoding = input_face_encodings[0]

    boolean_matches = face_recognition.compare_faces(encodings["encodings"], face_encoding)
    votes = Counter(
        name
        for match, name in zip(boolean_matches, encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]

    return "Unknown"

IMAGE_LOCATION = "adam.jpg"
cv2_image = cv2.imread(IMAGE_LOCATION)
rgb_image = cv2.cvtColor(cv2_image, cv2.COLOR_BGR2RGB)

encodings = pickle.load(open(DEFAULT_ENCODINGS_PATH, "rb"))

face_locations = face_recognition.face_locations(rgb_image)

face_location = face_locations[1]

start_time= time.time()
for i in range(10):
    name = recognize_one_face(rgb_image, encodings, face_location)
print(f"Time per image: {(time.time() - start_time)/10} seconds")

print(name)