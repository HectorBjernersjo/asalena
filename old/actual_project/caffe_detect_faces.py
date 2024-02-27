import numpy as np
import os
import cv2
import time

print("[INFO] loading model...")
net = cv2.dnn.readNetFromCaffe("face_detection_caffe_deploy.txt", "face_detection_caffe.caffemodel")

def detect_faces(frame):
    (h, w) = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(cv2.resize(frame, (300, 300)), 1.0,
        (300, 300), (104.0, 177.0, 123.0))

    net.setInput(blob)
    detections = net.forward()

    face_locations = []
    for i in range(0, detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.45:
            box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
            (startX, startY, endX, endY) = box.astype("int")
            if startY < 0:
                startY = 0
            if startX < 0:
                startX = 0
            if endX > w:
                endX = w
            if endY > h:
                endY = h
            if endY < startY or endX < startX:
                continue
            face_location = (startX, startY, endX, endY)
            if max(face_location) > 640:
                print("WARNING: Face detection is not working as expected. Please check if the camera is working properly.")
            face_locations.append(face_location)

    if len(face_locations) > 0 and np.amax(face_locations) > 640:
        print("WARNING2: Face detection is not working as expected. Please check if the camera is working properly.")
    return face_locations

    
if __name__ == "__main__":
    for filename in os.listdir("zotherimages"):
        image = cv2.imread(f"zotherimages/{filename}")
        start_time = time.time()
        face_locations = detect_faces(image)
        for (left, top, right, bottom) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        print(f"Time to get face locations: {time.time() - start_time} seconds")
        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
