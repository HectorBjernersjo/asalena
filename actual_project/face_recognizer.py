import cv2
import face_recognition
import pickle
from collections import Counter

def recognize_face(input_image, face_location, encodings) -> str:
    # loaded_encodings = pickle.load(open(encodings_path, "rb"))
    left, top, right, bottom = face_location
    face_image = input_image[top:bottom, left:right]

    # face_locations = [face_location]

    cv2.imshow("Face", face_image)

    face_locations = [(0, right-left, bottom-top, 0)]  # Convert (x, y, w, h) to (top, right, bottom, left)
    input_face_encodings = face_recognition.face_encodings(input_image, face_locations)


    if not input_face_encodings:
        return "No face"

    # Compare the input face with the known encodings
    boolean_matches = face_recognition.compare_faces(encodings["encodings"], input_face_encodings[0])
    votes = Counter(name for match, name in zip(boolean_matches, encodings["names"]) if match)

    # Return the recognized name or 'Unknown' if no match is found
    if votes:
        return votes.most_common(1)[0][0]
    else:
        return "Unknown"