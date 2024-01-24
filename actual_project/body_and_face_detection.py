import sys
sys.path.append('..')
import cv2
import threading
import numpy as np
import time
import os
import requests

import body_finder
import caffe_detect_faces
import openface_recognizer

prev_frame_time = 0
frame_times = []

time_for_face_detection_in_bodies = 0
times_for_face_detection_in_bodies = 0
time_for_face_recognition_general = 0
times_for_face_recognition_general = 0



def get_esp_frame():
    # url = "http://192.168.50.217/" # hemma
    # url = 'http://192.168.2.217' # mange
    url = 'http://192.168.1.202:4747/video' # telefon hos anton
    stream = requests.get(url, stream=True)
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

def save_face(face_img, name, confidence):
    folder = "face_images"
    if confidence < 0.5:
        return
    if not os.path.exists(folder):
        os.makedirs(folder)
    cv2.imwrite(f"{folder}/{name}-{confidence}-{time.time()}.jpg", face_img)


def fetch_latest_frame(cap, latest_frame):
    while True:
        ret, frame = cap.read()
        if ret:
            with threading.Lock():
                latest_frame[0] = frame
        else:
            break
        
use_esp = False
use_phone = True
use_webcam = False

phone_url = "http://192.168.4.3:4747/video/1080p" # esp
# phone_url = "http://192.168.2.91:4747/video/1080p" # mange

people = {}
if use_phone:
    cap = cv2.VideoCapture(phone_url)
elif use_webcam:
    cap = cv2.VideoCapture(0)
i = 0


latest_frame = [None]
thread = threading.Thread(target=fetch_latest_frame, args=(cap, latest_frame))
thread.start()

while True:
    start_time = time.time()
    if use_esp:
        frame = get_esp_frame()
    elif use_webcam:
        ret, frame = cap.read()
    else:
        if latest_frame[0] is not None:
            with threading.Lock():
                frame = latest_frame[0]
                # cv2.imshow('Camera Stream', latest_frame[0])
        else:
            continue

    i += 1
    # else:
    #     latest_frame = None
    #     start_time = time.time()
    #     while True:
    #         if cap.grab():
    #             ret, frame = cap.retrieve()
    #             if ret:
    #                 latest_frame = frame
    #         else:
    #             break
    #         # Break the loop if no new frame is available for 0.1 seconds
    #         if time.time() - start_time > 0.001:
    #             break
    #     frame = latest_frame
    start_time = time.time()
    body_positions = body_finder.find_body_positions(frame)

    people_in_screen = []
    
    for bodypos in body_positions:
        current_person_name = "No face"
        startX, startY, endX, endY = bodypos
        middle_position = get_middle_position(bodypos)
        body = cutout_image(frame, bodypos)

        
        for name, middle_pos in people.items():
            if abs(middle_pos[0] - middle_position[0]) < 50 and abs(middle_pos[1] - middle_position[1]) < 50:
                current_person_name = name
                people[name] = middle_position
                people_in_screen.append(name)
                break
        
        if i % 30 == 0:
            a = 1   
        if current_person_name == "No face" or "Unknown" in current_person_name and i % 15 == 0 or i % 45 == 0:
            if current_person_name in people:
                people.pop(current_person_name)
            if current_person_name in people_in_screen:
                people_in_screen.remove(current_person_name)
            bodyimg = cutout_image(frame, bodypos)
            faces = caffe_detect_faces.detect_faces(bodyimg)
            if len(faces) == 0:
                continue
            print(f"Found {len(faces)} faces in body, should be 1")
            face = faces[0]
            
            left, top, right, bottom = face
            left += startX
            top += startY
            right += startX
            bottom += startY
            
            current_person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))
            confidence = float(current_person_name.split("-")[1])
            name = current_person_name.split("-")[0]
            save_face(frame[top:bottom, left:right], name, confidence)
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            cv2.imshow("Face", frame[top:bottom, left:right])
            
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

    # print(f"Time to process frame: {time.time() - start_time} seconds")

    if cv2.waitKey(1) & 0xFF == ord('q'): 
        break
