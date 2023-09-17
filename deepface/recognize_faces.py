from deepface import DeepFace
import time
import cv2
from PIL import Image
import pickle
import os
from scipy.spatial.distance import cosine

TEST_IMAGES_PATH = "deepface/images/test_images"

embeddings = pickle.load(open("deepface/embeddings.pkl", "rb"))

image_paths = [f"{TEST_IMAGES_PATH}/{name}" for name in os.listdir(TEST_IMAGES_PATH)] 

for path in image_paths:
    image = cv2.imread(path)
    if image.shape[0] > 1000 or image.shape[1] > 1000:
        aspect_ratio = image.shape[0] / image.shape[1]
        if aspect_ratio > 1:
            image = cv2.resize(image, (int(1000 / aspect_ratio), 1000))
        else:
            image = cv2.resize(image, (1000, int(1000 * aspect_ratio)))

    # save the downscaled image to a temporary path
    temp_path = f"temp.jpg"
    cv2.imwrite(temp_path, image)
    

    current_embeddings = DeepFace.represent(temp_path, model_name="Facenet", enforce_detection=False)

    for embedding in current_embeddings:
        facial_area = embedding['facial_area']
        cv2.rectangle(image, (facial_area['x'], facial_area['y']), (facial_area['x'] + facial_area['w'], facial_area['y'] + facial_area['h']), (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)



    closest_distance = 100
    closest_name = None
    for name, embedding in embeddings.items():
        distance = cosine(current_embeddings[0]['embedding'], embedding[0]['embedding'])
        if distance < closest_distance:
            closest_distance = distance
            closest_name = name
    
    print(f"Closest match for {path} is {closest_name} with distance {closest_distance}")
    face_rect = current_embeddings[0]['facial_area']
    cv2.rectangle(image, (face_rect['x'], face_rect['y']), (face_rect['x'] + face_rect['w'], face_rect['y'] + face_rect['h']), (0, 255, 0), 2)
    cv2.imshow("image", image)
    cv2.waitKey(0)

    os.remove(temp_path)