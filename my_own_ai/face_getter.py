import cv2
import os
import numpy as np
from PIL import Image

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_profileface.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

TRAINING_DATA_PATH = "C:/Users/chris/OneDrive - Örebro kommun/Gymnasiearbete/training"
OUTPUT_ONLYFACES_PATH = "onlyfaces"

for persondir in os.listdir(TRAINING_DATA_PATH):
    persondir_path = f"{TRAINING_DATA_PATH}/{persondir}"
    if not os.path.isdir(f"{OUTPUT_ONLYFACES_PATH}/{persondir}"):
        os.mkdir(f"{OUTPUT_ONLYFACES_PATH}/{persondir}")

    for filename in os.listdir(persondir_path):
        filename_full = f"{persondir_path}/{filename}"
        if not filename_full.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
            continue
        pil_image = Image.open(filename_full)
        image = np.array(pil_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        cv2.imshow("gray", gray)
        faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        for (x, y, w, h) in faces:
            face_image = image[y:y+h, x:x+w]
            face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            # write image to file
            cv2.imwrite(f"{OUTPUT_ONLYFACES_PATH}/{persondir}/{filename}", face_image)