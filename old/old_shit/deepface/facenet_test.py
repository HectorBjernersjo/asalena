# %%
import numpy as np
import cv2
from keras_facenet import FaceNet
import time

# Initialize FaceNet
embedder = FaceNet()

def preprocess_image(image_path):
    """Load the image using cv2 and resize it for FaceNet."""
    img = cv2.imread(image_path)
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img_resized = cv2.resize(img, (160, 160))
    img_array = np.expand_dims(img_resized, axis=0)
    return img_array

def get_embedding(image_path):
    """Get the embedding for a given image."""
    img_array = preprocess_image(image_path)
    # Directly get the embedding without detection
    embedding = embedder.embeddings(img_array)[0]
    return embedding
# %%
elon2 = "../onlyfaces/elon_musk/testelon2.jpg"
# Test
start_time = time.time()
# embedding = get_embedding('images/only_faces/Anton.jpg')
embedding = get_embedding(elon2)
print("Time to get embedding:", time.time() - start_time)

# %%
