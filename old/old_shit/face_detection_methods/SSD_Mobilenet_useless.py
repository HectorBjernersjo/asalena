# %%
import tensorflow as tf
import cv2
import time
# %%
# Load the saved model
model_dir = 'ssdai/saved_model'
model = tf.saved_model.load(model_dir)

# Load the labels (if you have a label map, otherwise you can create a list manually)
# COCO dataset labels for instance would be like:
labels = ['background', 'person', 'bicycle', ...]  # and so on for all the 80 classes

def detect_faces(image):
    # Convert image to tensor
    input_tensor = tf.convert_to_tensor(image)
    input_tensor = input_tensor[tf.newaxis, ...]

    detections = model(input_tensor)

    # Extract detection boxes and scores
    detection_boxes = detections['detection_boxes'][0].numpy()
    detection_scores = detections['detection_scores'][0].numpy()
    detection_classes = detections['detection_classes'][0].numpy()

    height, width, _ = image.shape
    for box, score, cls in zip(detection_boxes, detection_scores, detection_classes):
        if score > 0.7:  # Adjust this threshold as necessary
            x1, y1, x2, y2 = box
            x1, y1, x2, y2 = int(x1 * width), int(y1 * height), int(x2 * width), int(y2 * height)
            cv2.rectangle(image, (x1, y1), (x2, y2), (0, 255, 0), 2)
            label = labels[int(cls)]
            cv2.putText(image, label, (x1, y1), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)
    return image

# Test the function
img_path = "zuck.jpg"
img = cv2.imread(img_path)

start_time = time.time()

detect_faces(img)

end_time = time.time()

print(f"Processed 30 frames in {end_time - start_time} seconds")

# Display the image with detected faces
cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()

# %%
