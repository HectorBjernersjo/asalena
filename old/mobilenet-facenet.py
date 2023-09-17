import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2

# Initialize MobileNetV2 for feature extraction
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(160, 160, 3), pooling='avg')

# Initialize face detector
detector = MTCNN()

def get_embedding(face_pixels):
    face_pixels = face_pixels.astype('float32')
    mean, std = face_pixels.mean(), face_pixels.std()
    face_pixels = (face_pixels - mean) / std
    samples = tf.expand_dims(face_pixels, axis=0)
    yhat = base_model.predict(samples)
    return yhat[0]

def extract_face_from_image(image, required_size=(160, 160)):
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    faces = detector.detect_faces(image_rgb)
    face_arrays = []
    face_positions = []
    for face in faces:
        x1, y1, width, height = face['box']
        x1, y1 = abs(x1), abs(y1)
        x2, y2 = x1 + width, y1 + height
        face_array = image_rgb[y1:y2, x1:x2]
        face_array = cv2.resize(face_array, required_size)
        face_arrays.append(face_array)
        face_positions.append((x1, y1, x2, y2))
    return face_arrays, face_positions

# Load saved embeddings
known_embeddings = np.load('face_embeddings.npy', allow_pickle=True).item()

# Load the image where you want to recognize faces
image_path = 'adam.jpg'
image = cv2.imread(image_path)

faces, positions = extract_face_from_image(image)

for face, (x1, y1, x2, y2) in zip(faces, positions):
    embedding = get_embedding(face)
    min_distance = float('inf')
    identity = None
    for name, known_embedding in known_embeddings.items():
        distance = np.linalg.norm(embedding - known_embedding)
        if distance < min_distance:
            min_distance = distance
            identity = name
    # if min_distance < 0.6:  # Adjust the threshold as needed
    label = f"{identity} ({min_distance:.2f})"
    cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
    cv2.putText(image, label, (x1, y1-10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 1)

# Save or display the result
output_path = 'recognized_image.jpg'
cv2.imshow('Recognized faces', image)
cv2.waitKey(0)