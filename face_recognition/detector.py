# detector.py
from pathlib import Path
import pickle
import face_recognition
from collections import Counter
from PIL import Image, ImageDraw
from tqdm import tqdm
import cv2
import sys
import time

print(sys.version)


DEFAULT_ENCODINGS_PATH = Path("face_recognition/encodings.pkl")
TRAINING_PATH = Path("D:/.shortcut-targets-by-id/1-tD6MqrxaOmV-9_wxYe_CLwE2LvU_Vcs/Åsalena/scaled down")
TRAINING_PATH = Path("face_recognition/training")

Path("training").mkdir(exist_ok=True)
Path("output").mkdir(exist_ok=True)
Path("validation").mkdir(exist_ok=True)

def encode_known_faces(
    model: str = "hog", encodings_location: Path = DEFAULT_ENCODINGS_PATH
) -> None:
    names = []
    encodings = []  

    # List of valid image extensions
    valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.tiff', '.gif'}

    for filepath in tqdm(TRAINING_PATH.glob("*/*")):
        if filepath.suffix.lower() not in valid_exts:  # Filter for image files
            continue

        name = filepath.parent.name

        try:
            image = face_recognition.load_image_file(filepath)

            face_locations = face_recognition.face_locations(image, model=model)
            face_encodings = face_recognition.face_encodings(image, face_locations)

            for encoding in face_encodings:
                names.append(name)
                encodings.append(encoding)
        except Exception as e:
            print(f"Error processing {filepath}: {e}")

    name_encodings = {"names": names, "encodings": encodings}
    with encodings_location.open(mode="wb") as f:
        pickle.dump(name_encodings, f)

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def haar_face_locations(image):
    """
    Use OpenCV's Haar cascades to detect face locations.
    """
    # Convert the image from RGB to grayscale (Haar cascades need grayscale images)
    gray = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
    
    # Detect faces
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.1, minNeighbors=5, minSize=(30, 30))
    
    # Convert OpenCV face rectangles (x, y, w, h) to dlib style face locations [(top, right, bottom, left)]
    face_locations = [(y, x + w, y + h, x) for (x, y, w, h) in faces]
    return face_locations

def recognize_faces(
    image_location: str,
    model: str,
    encodings_location: Path = DEFAULT_ENCODINGS_PATH,
) -> None:
    start_load_encodings = time.time()
    with encodings_location.open(mode="rb") as f:
        loaded_encodings = pickle.load(f)
    print(f"Time to load encodings: {time.time() - start_load_encodings} seconds")

    start_load_image = time.time()
    input_image = face_recognition.load_image_file(image_location)
    print(f"Time to load input image: {time.time() - start_load_image} seconds")

    start_face_locations = time.time()
    if model == "hog" or model == "cnn":
        input_face_locations = face_recognition.face_locations(
            input_image, model=model
        )
    elif model == "haar":
        input_image_cv = cv2.imread(image_location)  # Load the image using OpenCV
        input_image = cv2.cvtColor(input_image_cv, cv2.COLOR_BGR2RGB)  # Convert it from BGR to RGB
        input_face_locations = haar_face_locations(input_image)

    print(f"Time to get face locations: {time.time() - start_face_locations} seconds")

    start_face_encodings = time.time()
    input_face_encodings = face_recognition.face_encodings(
        input_image, input_face_locations
    )
    print(f"Time to get face encodings: {time.time() - start_face_encodings} seconds")

    pillow_image = Image.fromarray(input_image)
    draw = ImageDraw.Draw(pillow_image)

    start_recognition = time.time()
    for bounding_box, unknown_encoding in zip(
        input_face_locations, input_face_encodings
    ):
        name = _recognize_face(unknown_encoding, loaded_encodings)
        if not name:
            name = "Unknown"
        _display_face(draw, bounding_box, name)
    print(f"Time for face recognition: {time.time() - start_recognition} seconds")

    del draw
    pillow_image.show()


def _recognize_face(unknown_encoding, loaded_encodings):
    boolean_matches = face_recognition.compare_faces(
        loaded_encodings["encodings"], unknown_encoding
    )
    votes = Counter(
        name
        for match, name in zip(boolean_matches, loaded_encodings["names"])
        if match
    )
    if votes:
        return votes.most_common(1)[0][0]


BOUNDING_BOX_COLOR = "blue"
TEXT_COLOR = "white"

# ...

def _display_face(draw, bounding_box, name):
    top, right, bottom, left = bounding_box
    draw.rectangle(((left, top), (right, bottom)), outline=BOUNDING_BOX_COLOR)
    text_left, text_top, text_right, text_bottom = draw.textbbox(
        (left, bottom), name
    )
    draw.rectangle(
        ((text_left, text_top), (text_right, text_bottom)),
        fill="blue",
        outline="blue",
    )
    draw.text(
        (text_left, text_top),
        name,
        fill="white",
    )

encode_known_faces()