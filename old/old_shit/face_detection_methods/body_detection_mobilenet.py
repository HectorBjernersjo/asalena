import cv2
import numpy as np
import time

# Load MobileNetV2 + SSDLite
net = cv2.dnn.readNetFromCaffe("deploy.prototxt", "mobilenet_iter_73000.caffemodel")
classes = ["background", "aeroplane", "bicycle", "bird", "boat", "bottle", "bus", "car", "cat", "chair", "cow", "diningtable", "dog", "horse", "motorbike", "person", "pottedplant", "sheep", "sofa", "train", "tvmonitor"]

# Capture video from the default camera
cap = cv2.VideoCapture(0)

prev_frame_time = 0
frame_times = []

while True:
    ret, frame = cap.read()
    h, w = frame.shape[:2]
    blob = cv2.dnn.blobFromImage(frame, 0.007843, (300, 300), 127.5)
    net.setInput(blob)
    detections = net.forward()

    for i in range(detections.shape[2]):
        confidence = detections[0, 0, i, 2]
        if confidence > 0.2:
            class_id = int(detections[0, 0, i, 1])
            if classes[class_id] == "person":
                box = detections[0, 0, i, 3:7] * np.array([w, h, w, h])
                (startX, startY, endX, endY) = box.astype("int")
                cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)





    # Calculate and display average FPS over the last ten frames
    current_time = time.time()
    frame_times.append(current_time - prev_frame_time)
    if len(frame_times) > 10:
        frame_times.pop(0)
    avg_fps = 1 / (sum(frame_times) / len(frame_times))
    # print(avg_fps)
    prev_frame_time = current_time
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Avg FPS: {int(avg_fps)}", (10, 30), font, 1, (0, 255, 0), 2)

    cv2.imshow("MobileNetV2 + SSDLite Body Detection", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
