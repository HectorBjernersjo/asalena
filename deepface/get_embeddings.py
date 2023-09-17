from deepface import DeepFace
import time
import cv2
import io
from PIL import Image

WHOLE_IMAGES_PATH = "deepface/images/whole_images"
ONLY_FACES_PATH = "deepface/images/only_faces"

names = ["Hector", "Anton", "Filip"]

whole_image_paths = [f"{WHOLE_IMAGES_PATH}/{name}.jpg" for name in names]
only_face_paths = [f"{ONLY_FACES_PATH}/{name}" for name in names]

embeddings = {}
import os
for name, path in zip(names, whole_image_paths):
    # Load the image using cv2
    img = cv2.imread(path)

    # Downscale the image by 50%
    height, width = img.shape[:2]
    resized_img = cv2.resize(img, (int(width * 0.3), int(height * 0.3)))

    # Save the downscaled image to a temporary path
    temp_path = f"temp_{name}.jpg"
    cv2.imwrite(temp_path, resized_img)

    # Get the embeddings using the downscaled image
    start = time.time()
    embeddings[name] = DeepFace.represent(temp_path, model_name="Facenet", enforce_detection=False)
    print(f"Encoding {name} took {time.time() - start} seconds")
    
    rect = embeddings[name][0]['facial_area']
    cv2.rectangle(resized_img, (rect['x'], rect['y']), (rect['x'] + rect['w'], rect['y'] + rect['h']), (0, 255, 0), 2)
    cv2.imshow(name, resized_img)
    cv2.waitKey(0)

    # Remove the temporary image
    os.remove(temp_path)

# save the embeddings to a file
import pickle
with open("deepface/embeddings.pkl", "wb") as f:
    pickle.dump(embeddings, f)