import os
import numpy as np
import cv2
from mtcnn.mtcnn import MTCNN
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from pathlib import Path
from PIL import Image
from tqdm import tqdm


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

def extract_face(filename, required_size=(160, 160)):
    pil_image = Image.open(filename)
    image = np.array(pil_image)

    faces = detector.detect_faces(image)
    if len(faces) == 0:
        return None
    x1, y1, width, height = faces[0]['box']
    x1, y1 = abs(x1), abs(y1)
    x2, y2 = x1 + width, y1 + height
    face = image[y1:y2, x1:x2]
    face = cv2.resize(face, required_size)
    return face


# Directory containing subdirectories for each person
directory = 'I:/.shortcut-targets-by-id/1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs/Åsalena/scaled_down'
# directory = os.path.realpath(directory)
embeddings = {}



for subdir in tqdm(os.listdir(directory)):
    path = f"{directory}/{subdir}"
    if not os.path.isdir(path):
        continue
    faces = []
    for filename in os.listdir(path):
        file_path = f"{path}/{filename}"
        # if not image continue
        if not file_path.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
            continue

        face = extract_face(file_path)
        if face is not None:
            embedding = get_embedding(face)
            faces.append(embedding)
    if faces:
        embeddings[subdir] = np.mean(faces, axis=0)

cv2.waitKey(0)

# Save embeddings to a file
np.save('face_embeddings.npy', embeddings)
