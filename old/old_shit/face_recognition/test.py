# detector.py
from pathlib import Path
import pickle
import face_recognition
from collections import Counter
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import time
import sys

# if __name__ == "__main__":
image_path = "face_recognition/onlyfaces/IMG_4525.jpg"
filepath = r"D:\.shortcut-targets-by-id\1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs\Åsalena\scaled down\Hector\IMG_4990.JPG"
face_location = (800, 220, 930, 350)

image = cv2.imread(image_path)
left, top, right, bottom = face_location
face_image = image[top:bottom, left:right]
cv2.imshow("Face", image)
cv2.waitKey(0)