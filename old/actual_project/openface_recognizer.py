import cv2
import torch
# from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision.transforms import transforms
import numpy as np
import pickle
import faiss
from PIL import Image, ImageDraw
import time

IMG_FOLDER = "face_images"

if __name__ != "__main__":
    openface_model = torch.load("openface_quantized.pt")
    openface_model.eval()
    index = faiss.read_index("face_index.faiss")
    names = pickle.load(open("names.pkl", "rb"))

def get_openface_embedding(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    pil_img = transforms.ToPILImage()(img)
    resized_img = transforms.Resize((160, 160))(pil_img)

    tensor_img = transforms.ToTensor()(resized_img)
    tensor_img = tensor_img.unsqueeze(0)  # Add batch dimension

    with torch.no_grad():
        embedding = openface_model(tensor_img)
    return np.array(embedding.data)[0]

def recognize_face_from_image(face_img):
    embedding = get_openface_embedding(face_img)
    
    embedding = np.array(embedding).astype('float32').reshape(1, -1)
    distances, indexes = index.search(embedding, 1)
    best_name = names[indexes[0][0]]
    distance = distances[0][0]
    if distance > 0.9:
        best_name = "Unknown"
    # distances round to two decimals
    # return f"{best_name}-{distances[0][0]:.2f}"
    return best_name, distance

def recognize_face_from_frame(frame, face_position):
    left, top, right, bottom = face_position
    face_img = frame[top:bottom, left:right]
    # save face image
    plt_image = Image.fromarray(cv2.cvtColor(face_img, cv2.COLOR_BGR2RGB))
    name , distance= recognize_face_from_image(face_img)
    # if distance > 0.5:
    #     plt_image.save(f"{IMG_FOLDER}/{name}-{distance}-{time.time()}.jpg")
    return f"{name} - {distance:.2f}"
