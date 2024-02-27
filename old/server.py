import cv2
import threading
import numpy as np
import time
import os
import requests
import flask
from threading import Thread

from modules import body_finder
from modules import caffe_detect_faces
from modules import openface_recognizer

app = flask.Flask(__name__)

def fetch_latest_frame(cap, latest_frame):
    while True:
        ret, frame = cap.read()
        if ret:
            with threading.Lock():
                latest_frame[0] = frame
        else:
            break

def cutout_image(frame, corners):
    startX, startY, endX, endY = corners
    cutout = frame[startY:endY, startX:endX]
    return cutout

def get_middle_position(corners):
    startX, startY, endX, endY = corners
    middle_position = (startX + endX) / 2, (startY + endY) / 2
    return middle_position

def save_face(face_img, name, confidence):
    folder = "face_images"
    if confidence < 0.5:
        return
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(f"{folder}/{name}-{confidence}-{time.time()}.jpg", face_img)

# phone_url = "http://192.168.4.3:4747/video/1080p" # esp
# phone_url = "http://192.168.2.91:4747/video/1080p" # mange
# phone_url = "http://192.168.50.14:4747/video/1080p" # hemma
phone_url = "http://192.168.1.202:4747/video/1080p" # anton

cap = cv2.VideoCapture(phone_url)
people = {}
i = 0

latest_frame = [None]
thread = threading.Thread(target=fetch_latest_frame, args=(cap, latest_frame))
thread.start()


@app.route('/bodies', methods=['GET'])
def get_body_positions():
    return flask.jsonify(people)

@app.route('/faces', methods=['GET'])
def get_face_positions():
    with threading.Lock():
        frame = latest_frame[0]
    if frame is None:
        return flask.jsonify("No frame")

    face_positions = caffe_detect_faces.detect_faces(frame)
    locations_and_names = []
    for face_coords in face_positions:
        left, top, right, bottom = face_coords
        face = frame[top:bottom, left:right]
        name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))
        locations_and_names.append((get_middle_position(face), name))
    return flask.jsonify(locations_and_names)
@app.route('/', methods=['GET'])
def index():
    return flask.jsonify("Hello")

def run_app():
    app.run(host='0.0.0.0', port=5000, debug=True)

if __name__ == "__main__":
    flask_thread = Thread(target=run_app)
    flask_thread.start()

# while True:
#     i += 1
#     print("hej")
#     if latest_frame[0] is not None:
#         with threading.Lock():
#             frame = latest_frame[0]
#     else:
#         print("No frame")
#         continue
#
#     body_positions = body_finder.find_body_positions(frame)
#     people_in_screen = []
#
#     for bodypos in body_positions:
#         current_person_name = "No face"
#         startX, startY, endX, endY = bodypos
#         middle_position = get_middle_position(bodypos)
#         body = cutout_image(frame, bodypos)
#         
#         for name, middle_pos in people.items():
#             if abs(middle_pos[0] - middle_position[0]) < 50 and abs(middle_pos[1] - middle_position[1]) < 50:
#                 current_person_name = name
#                 people[name] = middle_position
#                 people_in_screen.append(name)
#                 break
#         
#         if current_person_name == "No face" or "Unknown" in current_person_name and i % 15 == 0 or i % 45 == 0:
#             if current_person_name in people:
#                 people.pop(current_person_name)
#             if current_person_name in people_in_screen:
#                 people_in_screen.remove(current_person_name)
#             bodyimg = cutout_image(frame, bodypos)
#             faces = caffe_detect_faces.detect_faces(bodyimg)
#             if len(faces) == 0:
#                 continue
#             print(f"Found {len(faces)} faces in body, should be 1")
#             face = faces[0]
#             
#             left, top, right, bottom = face
#             left += startX
#             top += startY
#             right += startX
#             bottom += startY
#             
#             current_person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))
#             confidence = float(current_person_name.split("-")[1])
#             name = current_person_name.split("-")[0]
#             save_face(frame[top:bottom, left:right], name, confidence)
#             cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
#             cv2.imshow("Face", frame[top:bottom, left:right])
#             
#             if current_person_name != "No face":
#                 people[current_person_name] = middle_position
#                 people_in_screen.append(current_person_name)
#
#
#         cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
#         cv2.putText(frame, current_person_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
#
#
#     for name, middle_pos in people.items():
#         if name not in people_in_screen:
#             people.pop(name)
#             break
