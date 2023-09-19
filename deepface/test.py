# %%
import os
import get_embeddings
import time
from PIL import Image
import cv2
import numpy as np


IMG_FOLDER = "I:/.shortcut-targets-by-id/1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs/Åsalena/jpg_only_face_facerec"
start_image = "deepface/images/test_images/Anton1.jpg"
cv2_img = cv2.imread(start_image)

# embedding = get_embeddings.get_deepface_embedding(cv2_img, model_name="Facenet")
# embedding = get_embeddings.get_facenet_embedding(cv2_img)
get_embeddings.get_openface_embedding(cv2_img)

# %%
total_time = 0
img_number = 0
print("started")
for person_dir in os.listdir(IMG_FOLDER):
    for filename in os.listdir(f"{IMG_FOLDER}/{person_dir}"):
        if filename == "desktop.ini":
            continue
        img_path = f"{IMG_FOLDER}/{person_dir}/{filename}"
        pil_image = Image.open(img_path)
        cv2_image = cv2.cvtColor(np.array(pil_image), cv2.COLOR_RGB2BGR)
        start_time = time.time()
        # get_embeddings.get_deepface_embedding(cv2_image, model_name="Facenet")
        get_embeddings.get_facenet_embedding(cv2_image)
        get_embeddings.get_openface_embedding(cv2_image)
        total_time += time.time() - start_time
        img_number += 1
        if img_number % 50 == 0:
            break
    if img_number % 50 == 0:
        break
 
print(f"Average time: {total_time / img_number}")
# %%
