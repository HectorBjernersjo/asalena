import cv2
import numpy as np
import time

# Load YOLO
net = cv2.dnn.readNet("yolov3.weights", "yolov3.cfg")
layer_names = net.getLayerNames()
output_layers_indices = net.getUnconnectedOutLayers().flatten() - 1
output_layers = [layer_names[i] for i in output_layers_indices]
classes = [line.strip() for line in open("coco.names")]


# Capture video from the default camera
cap = cv2.VideoCapture(0)

times = []

while True:
    ret, frame = cap.read()
    height, width, channels = frame.shape

    # Detecting objects
    blob = cv2.dnn.blobFromImage(frame, 0.00392, (416, 416), (0, 0, 0), True, crop=False)
    net.setInput(blob)
    outs = net.forward(output_layers)

    # Information to show on the screen (class id, confidence, bounding box coordinates)
    class_ids = []
    confidences = []
    boxes = []

    # For each detection from each output layer, get the confidence, class id, bounding box params
    for out in outs:
        for detection in out:
            scores = detection[5:]
            class_id = np.argmax(scores)
            confidence = scores[class_id]
            if confidence > 0.5:
                # Object detected
                center_x = int(detection[0] * width)
                center_y = int(detection[1] * height)
                w = int(detection[2] * width)
                h = int(detection[3] * height)
                x = int(center_x - w / 2)
                y = int(center_y - h / 2)
                boxes.append([x, y, w, h])
                confidences.append(float(confidence))
                class_ids.append(class_id)

    indexes = cv2.dnn.NMSBoxes(boxes, confidences, 0.5, 0.4)

    # Draw bounding box for each detected object
    for i in range(len(boxes)):
        if i in indexes:
            label = str(classes[class_ids[i]])
            if label == "person":
                x, y, w, h = boxes[i]
                cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
                cv2.putText(frame, label, (x, y - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    cv2.imshow("YOLOv3 Body Detection", frame)

    times.append(time.time())
    if len(times) > 20:
        fps = len(times) / (times[-1] - times[0])
        times.pop(0)
        print("FPS: {}".format(fps))

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
