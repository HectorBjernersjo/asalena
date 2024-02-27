import cv2
from mtcnn import MTCNN
import time
import os

print("[INFO] loading model...")
detector = MTCNN()

def resize_image(image, target_width=320):
    h, w = image.shape[:2]
    scale = target_width / w
    return cv2.resize(image, (target_width, int(h * scale)))

def detect_faces(frame):
    # Convert the image from BGR (OpenCV format) to RGB (MTCNN format)
    frame = resize_image(frame)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Use MTCNN to detect faces
    result = detector.detect_faces(rgb_frame)

    face_locations = []
    for face in result:
        confidence = face['confidence']
        if confidence > 0.45:
            x, y, width, height = face['box']
            endX = x + width
            endY = y + height
            face_locations.append((x, y, endX, endY))
    return face_locations
