import cv2
import face_recognition
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
    

if __name__ == "__main__":
    import os
    import pickle
    import time
    test_images_path = "zotherimages"
    encodings_path = "encodingsoriginal.pkl"
    encodings = pickle.load(open(encodings_path, "rb"))

    for filename in os.listdir(test_images_path):
        if filename == "desktop.ini":
            continue
        input_image = cv2.imread(f"{test_images_path}/{filename}")
        face_locations = face_recognition.face_locations(input_image)
        for (top, right, bottom, left) in face_locations:
            start_time = time.time()
            recognized_name = recognize_face(input_image, (left, top, right, bottom), encodings)
            print(f"Time to recognize face: {time.time() - start_time} seconds")
            print(f"{filename}: {recognized_name}")
            cv2.rectangle(input_image, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.putText(input_image, recognized_name, (left, top - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        cv2.imshow("image", input_image)
        cv2.waitKey(0)