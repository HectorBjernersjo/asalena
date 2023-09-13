# detector.py
# from pathlib import Path
import pickle
import face_recognition
from collections import Counter
# from PIL import Image, ImageDraw
# from tqdm import tqdm
import cv2
import time
# import sys

# print(sys.version)

DEFAULT_ENCODINGS_PATH = "face_recognition/encodings.pkl"

encodings_location=DEFAULT_ENCODINGS_PATH
with open(encodings_location, "rb") as f:
        loaded_encodings = pickle.load(f)

def recognize_face(input_image, face_location) -> str:
    # Load the known encodings
    
    # Load the input image
    # input_image = face_recognition.load_image_file(image_location)

    # Extract the face from the provided location
    top, left, bottom, right = face_location
    face_image = input_image[top:bottom, left:right]

    cv2.imshow("Face", face_image)

    # Get the face encodings of the input image
    face_locations = [(0, right-left, bottom-top, 0)]  # Convert (x, y, w, h) to (top, right, bottom, left)
    input_face_encodings = face_recognition.face_encodings(input_image, face_locations)
    # If no faces are found, return a message
    if not input_face_encodings:
        return "No face found in the image."

    # Compare the input face with the known encodings
    boolean_matches = face_recognition.compare_faces(loaded_encodings["encodings"], input_face_encodings[0])
    votes = Counter(name for match, name in zip(boolean_matches, loaded_encodings["names"]) if match)

    # Return the recognized name or 'Unknown' if no match is found
    if votes:
        return votes.most_common(1)[0][0]
    else:
        return "Unknown"

if __name__ == "__main__":
    # image_path = "face_recognition/onlyfaces/IMG_4525.jpg"
    image_path = "adam.jpg"
    image = cv2.imread(image_path)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    face_locations = face_recognition.face_locations(image)
    print(face_locations)
    for face_location in face_locations:
        top, right, bottom, left = face_location
        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        person = recognize_face(image, (top, left, bottom, right))
        cv2.putText(image, person, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)
    cv2.imshow("Faces found", image)
    cv2.waitKey(0)