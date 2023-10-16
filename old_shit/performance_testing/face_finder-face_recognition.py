import face_recognition
import cv2
import time

IMG_PATH = "adam.jpg"
# Load the image and convert it from BGR to RGB
image = cv2.imread(IMG_PATH)

start_time = time.time()
# Find all the faces in the image
for i in range(10):
    face_locations = face_recognition.face_locations(image)
print(f"Time per image: {(time.time() - start_time)/10} seconds")

# find all the faces in the image
face_locations = face_recognition.face_locations(image)
for face_location in face_locations:
    top, right, bottom, left = face_location
    cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)
cv2.imshow("Faces found", image)
cv2.waitKey(0)