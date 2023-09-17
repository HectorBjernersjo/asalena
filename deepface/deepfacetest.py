# %%
from deepface import DeepFace
import time

image1 = "reeves1.jpg"
image2 = "reeves2.jpg"
image3 = 'Jet_Li_2009_(cropped).jpg'
image4 = 'Denzel_Washington_2018.jpg'
image5 = 'Smiling_girl.jpg'
elon1 = "onlyfaces/elon_musk/testelon.jpg"
elon2 = "onlyfaces/elon_musk/testelon2.jpg"
zuck1 = "onlyfaces/mark_zuckerberg/testzuck.jpg"
zuck2 = "onlyfaces/mark_zuckerberg/testzuck2.jpg"

# %%
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from PIL import Image
# %%

image = image1

im = Image.open(image)

models = ['VGG-Face', 'Facenet', 'OpenFace', 'DeepFace', 'DeepID', 'ArcFace']

for model in models:
    start = time.time()
    embeddings = DeepFace.represent(image, model_name=model, enforce_detection=False)
    print(f"Encoding {image} with {model} took {time.time() - start} seconds")

    fig, ax = plt.subplots()
    ax.imshow(im)

    # draw a rectangle around the face
    face_coord = embeddings[0]['facial_area']
    rect = patches.Rectangle((face_coord['x'], face_coord['y']), 
                             face_coord['w'], face_coord['h'], 
                             linewidth=2, 
                             edgecolor='r', 
                             facecolor='none')
    ax.add_patch(rect)
    ax.set_title(f"Model: {model}")
    plt.show()

# %%
