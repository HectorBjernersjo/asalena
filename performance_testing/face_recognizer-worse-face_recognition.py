import face_recognition
import pickle
import cv2
from collections import Counter
from pathlib import Path

DEFAULT_ENCODINGS_PATH = Path("encodingsoriginal.pkl")  # same as the second script
ENCODINGS_PATH = DEFAULT_ENCODINGS_PATH

# Load known face encodings
loaded_encodings = pickle.load(open(ENCODINGS_PATH, "rb"))

def recognize_face(input_image, face_location) -> str:
    left, top, right, bottom = face_location
    face_image = input_image[top:bottom, left:right]

    cv2.imshow("Face", face_image)

    face_locations = [(0, right-left, bottom-top, 0)]  # Convert (x, y, w, h) to (top, right, bottom, left)
    input_face_encodings = face_recognition.face_encodings(input_image, face_locations)

    if not input_face_encodings:
        return "No face"

    # Use the same recognition method as in the second script
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], input_face_encodings[0])
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)

    if votes:
        return votes.most_common(1)[0][0]
    else:
        return "Unknown"

IMG_PATH = "adam.jpg"
image = cv2.imread(IMG_PATH)

# Find all the faces in the image using HOG (can be changed to "cnn" or "haar" as per the second script)
face_locations = face_recognition.face_locations(image, model="hog")

for face_location in face_locations:
    top, right, bottom, left = face_location
    person = recognize_face(image, (left, top, right, bottom))

    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
    cv2.putText(image, person, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

cv2.imshow("Faces found", image)
cv2.waitKey(0)
