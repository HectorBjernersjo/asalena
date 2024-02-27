import cv2
import numpy as np
import tensorflow as tf
from collections import Counter
from pathlib import Path
import pickle
import time
import face_recognition

DEFAULT_MODEL_PATH = Path("face_recognition_model.h5")  # Path to your trained model
MODEL_PATH = DEFAULT_MODEL_PATH

# Load the trained model
model = tf.keras.models.load_model(MODEL_PATH)


# Load class labels
with open('class_labels.pkl', 'rb') as f:
    class_labels_dict = pickle.load(f)
class_labels = list(class_labels_dict.keys())

def preprocess_face(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    return face_pixels

def recognize_face(input_image, face_location) -> str:
    left, top, right, bottom = face_location
    face_image = input_image[top:bottom, left:right]
    face_image = cv2.resize(face_image, (160, 160))  # Resize to the input size of the model
    face_image = preprocess_face(face_image)
    face_image = np.expand_dims(face_image, axis=0)

    cv2.imshow("Face", face_image[0])


    starttr = time.time()
    predictions = model.predict(face_image)
    print(f"Predicted in {time.time() - starttr} seconds")
    predicted_class = class_labels[np.argmax(predictions)]

    return predicted_class

testimage = "onlyfaces/elon_musk/testelon.jpg"
test_image = cv2.imread(testimage)
recognize_face(test_image, (0, 0, test_image.shape[1], test_image.shape[0]))

image_paths = list(Path("ztestimages").glob("*.jpg"))

for image_path in image_paths:
    image = cv2.imread(str(image_path))

    # if image is larger than 1920x1080, resize it to 1920x1080 but keep the aspect ratio
    aspect_ratio = image.shape[1] / image.shape[0]
    if image.shape[1] > 1920 or image.shape[0] > 1080:
        if aspect_ratio > 16/9:
            image = cv2.resize(image, (1920, int(1920 / aspect_ratio)))
        else:
            image = cv2.resize(image, (int(1080 * aspect_ratio), 1080))
    
    # Use OpenCV's Haar cascades to detect faces
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    face_locations = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    face_locations = face_recognition.face_locations(image)

    # for (left, top, width, height) in face_locations:
    for (top, right, bottom, left) in face_locations:
        # right = left + width
        # bottom = top + height
        start_time = time.time()
        person = recognize_face(image, (left, top, right, bottom))
        print(f"Recognized {person} in {time.time() - start_time} seconds")

        cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
        cv2.putText(image, person, (left, top), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    cv2.imshow("Faces found", image)
    cv2.waitKey(0)
