import cv2
import torch
from facenet_pytorch import InceptionResnetV1, fixed_image_standardization
from torchvision.transforms import transforms

# Load the pre-trained model
model = InceptionResnetV1(pretrained='vggface2').eval()

# Load the image with cv2
image_path = 'deepface/images/only_faces/Anton.jpg'
img = cv2.imread(image_path)
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

# Preprocess the image
preprocess = transforms.Compose([
    transforms.ToPILImage(),
    transforms.Resize((160, 160)),
    transforms.ToTensor(),
    fixed_image_standardization
])

img_tensor = preprocess(img)
img_tensor = img_tensor.unsqueeze(0)  # Add batch dimension

# Check if CUDA is available and if so, use it
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
img_tensor = img_tensor.to(device)
model = model.to(device)

# Generate the embedding
with torch.no_grad():
    embedding = model(img_tensor)

print(embedding)
