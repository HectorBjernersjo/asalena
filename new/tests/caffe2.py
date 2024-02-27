import cv2
import numpy as np

# Load the model
model_file = "mobilenet_iter_73000.caffemodel"
config_file = "deploy.prototxt"
net = cv2.dnn.readNetFromCaffe(config_file, model_file)

def detect_faces(image):
    (h, w) = image.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(image, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    face_locations = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.1: # Threshold to filter weak detections
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            face_locations.append((startX, startY, endX, endY))
    
    return face_locations
