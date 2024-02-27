# from retinaface import RetinaFace
# resp = RetinaFace.detect_faces("test_images/adam.jpg")
# print(resp)

from retinaface import RetinaFace
import cv2
import numpy as np

def detect_faces_with_retinaface(image):
    # Convert the image from BGR (OpenCV format) to RGB (RetinaFace format)
    rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    
    # Detect faces in the image
    faces = RetinaFace.detect_faces(rgb_image)
    
    face_locations = []
    if faces is not None:
        for face_key in faces:
            face = faces[face_key]
            facial_area = face['facial_area']
            startX, startY, endX, endY = facial_area
            face_locations.append((startX, startY, endX, endY))
    
    return face_locations

# # Example usage
# # Load your image (replace 'path/to/your/image.jpg' with the actual file path)
# image_path = 'path/to/your/image.jpg'
# image = cv2.imread(image_path)
#
# # Detect faces in the image
# detected_faces = detect_faces_with_retinaface(image)
#
# # Draw rectangles around detected faces
# for (startX, startY, endX, endY) in detected_faces:
#     cv2.rectangle(image, (startX, startY), (endX, endY), (0, 255, 0), 2)
#
# # Display the output image with detected faces
# cv2.imshow('Detected Faces', image)
# cv2.waitKey(0)
# cv2.destroyAllWindows()
#
