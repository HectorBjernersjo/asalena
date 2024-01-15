import cv2
import numpy as np
import random
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]


def find_body_positions(frame):
    h, w = frame.shape[:2]

    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()
    
    bodie_positions = []

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if classes[class_id] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
                if startY < 0:
                    startY = 0
                if startX < 0:
                    startX = 0
                if endX > w:
                    endX = w
                if endY > h:
                    endY = h
                bodie_positions.append((startX, startY, endX, endY))
    
    # for pos in body_positions:

    return bodie_positions

if __name__ == "__main__":
    import os
    import time
    test_images_path = "old_shit/zotherimages"
    for filename in os.listdir(test_images_path):
        input_image = cv2.imread(f"{test_images_path}/{filename}")
        start_time = time.time()
        body_positions = find_body_positions(input_image)
        print(f"Time to get face locations: {time.time() - start_time} seconds")
        for (x, y, w, h) in body_positions:
            cv2.rectangle(input_image, (x, y), (w, h), (0, 255, 0), 2)
        cv2.imshow("image", input_image)
        cv2.waitKey(0)