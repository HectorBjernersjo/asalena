from deepface import DeepFace
import cv2
import os
# from keras_facenet import FaceNet
import numpy as np
import torch
import torch.quantization
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision.transforms import transforms
import matplotlib.pyplot as plt

# facenet_model = FaceNet()

def get_deepface_embedding(cv2_img, model_name="Facenet"):
    """Get the embedding for a given image."""
    # Directly get the embedding without detection
    temp_path = "temp.jpg"
    cv2.imwrite(temp_path, cv2_img)
    embeddings = DeepFace.represent(temp_path, model_name=model_name, enforce_detection=False)
    os.remove(temp_path)
    return embeddings[0]

def get_facenet_embedding(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB
    img_resized = cv2.resize(img, (160, 160))
    img_array = np.expand_dims(img_resized, axis=0)
    embedding = facenet_model.embeddings(img_array)[0]
    return embedding

openface_preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

if __name__ != "__main__":
    openface_model = torch.load("deepface/openface_quantized.pt")
    openface_model.eval()


def get_openface_embedding(cv2_img):
    img = cv2.cvtColor(cv2_img, cv2.COLOR_BGR2RGB)  # Convert from BGR to RGB

    pil_img = transforms.ToPILImage()(img)
    # pil_img.show()  # Check after conversion to PIL

    resized_img = transforms.Resize((160, 160))(pil_img)
    # resized_img.show()  # Check after resizing

    tensor_img = transforms.ToTensor()(resized_img)
    # print(tensor_img.min(), tensor_img.max())  # Check tensor value range

    # normalized_img = fixed_image_standardization(tensor_img)
    # # print(normalized_img.min(), normalized_img.max())  # Check normalized tensor value range

    # img_tensor = normalized_img
    tensor_img = tensor_img.unsqueeze(0)  # Add batch dimension
    # plt.imshow(tensor_img[0].permute(1, 2, 0))
    # plt.show()
    with torch.no_grad():
        embedding = openface_model(tensor_img)
    return np.array(embedding.data)[0]

if __name__ == "__main__":
    openface_model = InceptionResnetV1(pretrained='vggface2').eval()

    # 1. Fuse Modules
    fuse_modules = [['conv2d_1a.conv', 'conv2d_1a.bn'],
                    ['conv2d_2a.conv', 'conv2d_2a.bn'],
                    # ... add other pairs as needed
                ]

    torch.quantization.fuse_modules(openface_model, fuse_modules, inplace=True)

    # 2. Quantize
    quantized_model = torch.quantization.quantize_dynamic(
        openface_model, {torch.nn.Linear, torch.nn.Conv2d}, dtype=torch.qint8
    )

    # Save and load as before
    torch.save(quantized_model, "deepface/openface_quantized.pt")
