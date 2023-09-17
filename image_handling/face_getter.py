import cv2
import os
import numpy as np
from PIL import Image
import face_recognition

cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_profileface.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

TRAINING_DATA_PATH = "I:/.shortcut-targets-by-id/1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs/Åsalena/jpg"
OUTPUT_ONLYFACES_PATH = "I:/.shortcut-targets-by-id/1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs/Åsalena/jpg_only_face"

for persondir in os.listdir(TRAINING_DATA_PATH):
    persondir_path = f"{TRAINING_DATA_PATH}/{persondir}"
    if not os.path.isdir(f"{OUTPUT_ONLYFACES_PATH}/{persondir}"):
        if persondir == "desktop.ini":
            continue
        os.mkdir(f"{OUTPUT_ONLYFACES_PATH}/{persondir}")

    for filename in os.listdir(persondir_path):
        filename_full = f"{persondir_path}/{filename}"
        if not filename_full.endswith(('.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif')):
            continue
        pil_image = Image.open(filename_full)
        image = np.array(pil_image)
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
        # resize image to max 1280x720 without changing aspect ratio
        if image.shape[0] > 720 or image.shape[1] > 1280:
            if image.shape[0] > image.shape[1]:
                scale_factor = 720 / image.shape[0]
            else:
                scale_factor = 1280 / image.shape[1]
            image = cv2.resize(image, (0, 0), fx=scale_factor, fy=scale_factor)
        # cv2.imshow("gray", gray)
        # faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
        # for (x, y, w, h) in faces:
        #     face_image = image[y:y+h, x:x+w]
            # face_image = cv2.cvtColor(face_image, cv2.COLOR_BGR2RGB)
            # write image to file
        faces = face_recognition.face_locations(image)
        for face_location in faces:
            top, right, bottom, left = face_location
            face_image = image[top:bottom, left:right]
            pil_image = Image.fromarray(face_image)
            pil_image = pil_image.resize((128, 128))
            pil_image.save(f"{OUTPUT_ONLYFACES_PATH}/{persondir}/{filename}")
        print(f"Done with {filename_full}")