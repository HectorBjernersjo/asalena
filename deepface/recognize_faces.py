from deepface import DeepFace
import time
import cv2
# from PIL import Image0
import pickle
import os
# from scipy.spatial.distance import cosine
import faiss
import numpy as np
import get_embeddings

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_profileface.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

def get_face_locations(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def get_face_images(frame):
    faces = get_face_locations(frame)
    face_images = []
    for (x, y, w, h) in faces:
        face_images.append(frame[y:y+h, x:x+w])
    return face_images

TEST_IMAGES_PATH = "zotherimages"

model = "DeepID"

# embeddings = pickle.load(open("deepface/embeddings.pkl", "rb"))
index = faiss.read_index("deepface/face_index.faiss")
names = pickle.load(open("deepface/names.pkl", "rb"))

image_paths = [f"{TEST_IMAGES_PATH}/{name}" for name in os.listdir(TEST_IMAGES_PATH)] 

for path in image_paths:
    image = cv2.imread(path)
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        aspect_ratio = image.shape[0] / image.shape[1]
        if aspect_ratio > 1:
            image = cv2.resize(image, (int(1000 / aspect_ratio), 1000))
        else:
            image = cv2.resize(image, (1000, int(1000 * aspect_ratio)))

    face_locations = get_face_locations(image)
    
    for (x, y, w, h) in face_locations:
        face_image = image[y:y+h, x:x+w]
        
        start_time = time.time()
        embedding = get_embeddings.get_openface_embedding(face_image)
        print("Time to get embedding:", time.time() - start_time)
        start_time = time.time()
        embedding = np.array(embedding).astype('float32').reshape(1, -1)
        distances, indexes = index.search(embedding, 1)
        print("Time to search index:", time.time() - start_time)
        best_name = names[indexes[0][0]]
        distance = distances[0][0]

        if distance > 0.6:
            best_name = "Unknown"

        cv2.rectangle(image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.putText(image, f"{best_name} - {distance:.2f}", (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)
