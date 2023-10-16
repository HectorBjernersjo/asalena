# %%import os
import cv2
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV2
from tensorflow.keras.models import Model
from tqdm import tqdm
import os
# %%
# Load MobileNetV2 model without the top classification layer
base_model = MobileNetV2(weights='imagenet', include_top=False, input_shape=(96, 96, 3))
model = Model(inputs=base_model.input, outputs=base_model.layers[-1].output)

# Directory containing subdirectories of images
DATA_DIR = 'D:\\.shortcut-targets-by-id\\1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs\\Åsalena\\scaled_down'

# Use OpenCV's Haar cascades to detect faces
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

all_embeddings = []
all_names = []

for subdir in os.listdir(DATA_DIR):
    subdir_path = os.path.join(DATA_DIR, subdir)
    if os.path.isdir(subdir_path):
        for filename in tqdm(os.listdir(subdir_path)):
            filepath = r"D:\.shortcut-targets-by-id\1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs\Åsalena\scaled down\Hector\IMG_4990.JPG"
            image = cv2.imread(filepath)
            gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
            faces = face_cascade.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in faces:
                face = image[y:y+h, x:x+w]
                face = cv2.resize(face, (96, 96))
                face = face / 255.0
                face = np.expand_dims(face, axis=0)

                # Get the embedding of the detected face using MobileNet
                embedding = model.predict(face)
                all_embeddings.append(embedding)
                all_names.append(subdir)

# Convert lists to numpy arrays
all_embeddings = np.array(all_embeddings)
all_names = np.array(all_names)

# Save the embeddings and names
np.save('face_embeddings.npy', all_embeddings)
np.save('face_names.npy', all_names)
# %%

