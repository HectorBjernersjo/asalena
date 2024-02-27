import os
import cv2
import numpy as np
import faiss
from tqdm import tqdm
import pickle
from PIL import Image
import get_embeddings

IMG_FOLDER = "deepface/images/db_img"
IMG_FOLDER = "face_images"
model = "DeepID"
valid_img_formats = [".jpg", ".jpeg", ".png"]

# Step 1: Extract embeddings and store them in a list
openface_embeddings = []
facenet_embeddings = []
img_paths = []
names = []

for person_dir in os.listdir(IMG_FOLDER):
    if person_dir == "desktop.ini":
        continue
    for filename in tqdm(os.listdir(f"{IMG_FOLDER}/{person_dir}")):
        if not any(filename.endswith(ext) for ext in valid_img_formats):
            continue
        img_path = f"{IMG_FOLDER}/{person_dir}/{filename}"
        names.append(person_dir)
        pil_img = Image.open(img_path)
        cv2_img = cv2.cvtColor(np.array(pil_img), cv2.COLOR_RGB2BGR)
        # embedding = get_embeddings.get_deepface_embedding(cv2_img, model_name=model)
        # facenet_embedding = get_embeddings.get_facenet_embedding(cv2_img)
        # facenet_embeddings.append(facenet_embedding)
        embedding = get_embeddings.get_openface_embedding(cv2_img)
        openface_embeddings.append(embedding)
        img_paths.append(img_path)

# Convert the list of embeddings to a numpy array
openface_embeddings = np.array(openface_embeddings).astype('float32')  # faiss requires float32 type

with open("names.pkl", "wb") as f:
    pickle.dump(names, f)

# Step 2: Build the faiss index
d = openface_embeddings.shape[1]  # dimension of the embeddings
index = faiss.IndexFlatL2(d)  # use a flat L2 index; you can choose other indexes depending on the requirements

# Add the embeddings to the index
index.add(openface_embeddings)
# Save the index for later use if necessary
faiss.write_index(index, "face_index.faiss")
print("Index trained and saved")