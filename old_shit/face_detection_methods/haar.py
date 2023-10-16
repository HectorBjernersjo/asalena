import cv2
import time

# Load the pre-trained model from OpenCV
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_faces(img):
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 4)
    for (x, y, w, h) in faces:
        cv2.rectangle(img, (x, y), (x+w, y+h), (255, 0, 0), 2)
    return img

# Test the function
img_path = "adam.jpg"
img = cv2.imread(img_path)

start_time = time.time()

for _ in range(30):
    detect_faces(img)

end_time = time.time()

print(f"Processed 30 frames in {end_time - start_time} seconds")
print(f"Average FPS: {30 / (end_time - start_time)}")
print("time per frame: ", (end_time - start_time)/30)

cv2.imshow('Detected Faces', img)
cv2.waitKey(0)
cv2.destroyAllWindows()