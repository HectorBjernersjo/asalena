import os
import cv2
import numpy as np
import faiss
from deepface import DeepFace
from tqdm import tqdm
import pickle

IMG_FOLDER = "deepface/images/db_img"

# Step 1: Extract embeddings and store them in a list
embeddings = []
img_paths = []
names = []

for person_dir in os.listdir(IMG_FOLDER):
    for filename in tqdm(os.listdir(f"{IMG_FOLDER}/{person_dir}")):
        img_path = f"{IMG_FOLDER}/{person_dir}/{filename}"
        names.append(person_dir)
        embedding = DeepFace.represent(img_path, model_name="Facenet", enforce_detection=False)
        embeddings.append(embedding[0]['embedding'])
        img_paths.append(img_path)

# Convert the list of embeddings to a numpy array
embeddings = np.array(embeddings).astype('float32')  # faiss requires float32 type

with open("deepface/names.pkl", "wb") as f:
    pickle.dump(names, f)

# Step 2: Build the faiss index
d = embeddings.shape[1]  # dimension of the embeddings
index = faiss.IndexFlatL2(d)  # use a flat L2 index; you can choose other indexes depending on the requirements

# Add the embeddings to the index
index.add(embeddings)
# Save the index for later use if necessary
faiss.write_index(index, "deepface/face_index.faiss")

# From here on, you can use the index to perform similarity search on new face embeddings

# Optional: Displaying the images with rectangles
for i, img_path in enumerate(img_paths):
    img = cv2.imread(img_path)
    face_rect = embeddings[i]['facial_area']  # Assuming this is correct; adjust if needed
    cv2.rectangle(img, (face_rect['x'], face_rect['y']), (face_rect['x'] + face_rect['w'], face_rect['y'] + face_rect['h']), (0, 255, 0), 2)
    cv2.imshow("image", img)
    cv2.waitKey(0)
