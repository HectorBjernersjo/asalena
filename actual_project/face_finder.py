import cv2
import os
    
cascPathface = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_frontalface_alt2.xml"
cascPatheyes = os.path.dirname(
    cv2.__file__) + "/data/haarcascade_profileface.xml"

faceCascade = cv2.CascadeClassifier(cascPathface)
eyeCascade = cv2.CascadeClassifier(cascPatheyes)

def get_face_locations(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = faceCascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30), flags=cv2.CASCADE_SCALE_IMAGE)
    return faces

def get_face_images(frame):
    faces = get_face_locations(frame)
    face_images = []
    for (x, y, w, h) in faces:
        face_images.append(frame[y:y+h, x:x+w])
    return face_images

if __name__ == "__main__":
    import os
    import time
    test_images_path = "zotherimages"
    for filename in os.listdir(test_images_path):
        input_image = cv2.imread(f"{test_images_path}/{filename}")
        start_time = time.time()
        face_locations = get_face_locations(input_image)
        print(f"Time to get face locations: {time.time() - start_time} seconds")
        for (x, y, w, h) in face_locations:
            cv2.rectangle(input_image, (x, y), (x+w, y+h), (0, 255, 0), 2)
        cv2.imshow("image", input_image)
        cv2.waitKey(0)