import cv2
import numpy as np
import time
import os
import pickle
import face_recognition
from collections import Counter
import requests

import body_finder
import face_finder
import face_recognizer

prev_frame_time = 0
frame_times = []


ENCODINGS_PATH = "face_recognition/encodingsold.pkl"

with open(ENCODINGS_PATH, "rb") as f:
        loaded_encodings = pickle.load(f)


def get_esp_frame():
    esp_url = "http://192.168.50.217/"
    stream = requests.get(esp_url, stream=True)
    byte_stream = bytes()
    for chunk in stream.iter_content(chunk_size=1024):
        byte_stream += chunk
        a = byte_stream.find(b'\xff\xd8')
        b = byte_stream.find(b'\xff\xd9')
        if a != -1 and b != -1:
            jpg = byte_stream[a:b+2]
            byte_stream = byte_stream[b+2:]
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            return frame
        
def cutout_image(frame, corners):
    startX, startY, endX, endY = corners
    cutout = frame[startY:endY, startX:endX]
    return cutout

def get_middle_position(corners):
    startX, startY, endX, endY = corners
    middle_position = (startX + endX) / 2, (startY + endY) / 2
    return middle_position

def update_frame_times(prev_frame_time, frame_times):
    current_time = time.time()
    frame_times.append(current_time - prev_frame_time)
    if len(frame_times) > 10:
        frame_times.pop(0)
    avg_fps = 1 / (sum(frame_times) / len(frame_times))
    # print(avg_fps)
    prev_frame_time = current_time
    return avg_fps, prev_frame_time

def draw_frame(frame, fps):
    font = cv2.FONT_HERSHEY_SIMPLEX
    cv2.putText(frame, f"Avg FPS: {int(fps)}", (10, 30), font, 1, (0, 255, 0), 2)
    cv2.imshow("Face and body detection", frame)

people = {}

while True:
    # ret, frame = cap.read()
    frame = get_esp_frame()
    
    body_positions = body_finder.find_body_positions(frame)
    
    people_in_screen = []
    for bodypos in body_positions:
        current_person_name = "No face"
        startX, startY, endX, endY = bodypos
        body = cutout_image(frame, bodypos)
        middle_position = get_middle_position(bodypos)

        # draw dot in middle of body
        cv2.circle(frame, (int(middle_position[0]), int(middle_position[1])), 5, (0, 0, 255), -1)

        for name, middle_pos in people.items():
            if abs(middle_pos[0] - middle_position[0]) < 50 and abs(middle_pos[1] - middle_position[1]) < 50:
                current_person_name = name
                people[name] = middle_position
                people_in_screen.append(name)
                break
        
        if current_person_name == "No face": 
            faces = face_finder.get_face_locations(body)
            if len(faces) == 0:
                continue
            face = faces[0]
            x, y, w, h = face
            left, top, right, bottom = startX + x, startY + y, startX + x + w, startY + y + h
            current_person_name = face_recognizer.recognize_face(frame, (left, top, right, bottom), loaded_encodings)
            if current_person_name != "No face":
                people[current_person_name] = middle_position
                people_in_screen.append(current_person_name)


        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        cv2.putText(frame, current_person_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)


        
    for name, middle_pos in people.items():
        if name not in people_in_screen:
            people.pop(name)
            break

    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)
    draw_frame(frame, avg_fps)	


    if cv2.waitKey(1) & 0xFF == ord('q'):
        break












     