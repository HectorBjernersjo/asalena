import cv2
import face_recognition
import pickle
from collections import Counter
import time

ENCODINGS_PATH = "face_recognition/encodingsold.pkl"
loaded_encodings = pickle.load(open(ENCODINGS_PATH, "rb"))

def recognize_face(input_image, face_location) -> str:
    left, top, right, bottom = face_location
    face_image = input_image[top:bottom, left:right]

    # face_locations = [face_location]

    cv2.imshow("Face", face_image)

    face_locations = [(0, right-left, bottom-top, 0)]  # Convert (x, y, w, h) to (top, right, bottom, left)
    input_face_encodings = face_recognition.face_encodings(input_image, face_locations)

    start_face_encodings = time.time()
    input_face_encodings = face_recognition.face_encodings(
        input_image, face_locations
    )

    if not input_face_encodings:
        return "No face"

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


IMG_PATH = "adam.jpg"
image = cv2.imread(IMG_PATH)

# Find all the faces in the image
face_locations = face_recognition.face_locations(image)

for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)



    person = recognize_face(image, (left, top, right, bottom))
    cv2.putText(image, person, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
cv2.imshow("Faces found", image)
cv2.waitKey(0)