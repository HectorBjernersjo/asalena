from deepface import DeepFace
import time
import cv2
# from PIL import Image
import pickle
import os
from scipy.spatial.distance import cosine
import faiss
import numpy as np

TEST_IMAGES_PATH = "deepface/images/only_faces"

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

    # save the downscaled image to a temporary path
    temp_path = f"temp.jpg"
    cv2.imwrite(temp_path, image)
    
    start_time = time.time()
    embeddingorg = DeepFace.represent(temp_path, model_name="Facenet", enforce_detection=False)
    embedding = np.array(embeddingorg[0]['embedding']).astype('float32').reshape(1, -1)  # reshape the embedding to 2D
    print("Time to get embedding:", time.time() - start_time)
    start_time = time.time()
    distances, indexes = index.search(embedding, 3)
    print("Time to search index:", time.time() - start_time)
    print("Distances:", distances)
    print("Indexes:", indexes)
    best_names = [names[i] for i in indexes[0]]
    print("Names:", best_names)

    # print(people)



    # closest_distance = 100
    # closest_name = None
    # for name, embedding in embeddings.items():
    #     distance = cosine(current_embeddings[0]['embedding'], embedding[0]['embedding'])
    #     if distance < closest_distance:
    #         closest_distance = distance
    #         closest_name = name
    
    # print(f"Closest match for {path} is {closest_name} with distance {closest_distance}")
    face_rect = embeddingorg[0]['facial_area']
    cv2.rectangle(image, (face_rect['x'], face_rect['y']), (face_rect['x'] + face_rect['w'], face_rect['y'] + face_rect['h']), (0, 255, 0), 2)
    cv2.putText(image, best_names[0], (face_rect['x'], face_rect['y'] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, best_names[1], (0, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
    cv2.putText(image, best_names[2], (0, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)

    cv2.imshow("image", image)
    cv2.waitKey(0)

    os.remove(temp_path)