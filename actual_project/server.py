from flask import Flask, request, jsonify, send_file
import cv2
import numpy as np
import pickle
import time
import face_recognition
import requests
from io import BytesIO

# Your existing imports...
import body_finder
import face_finder
import openface_recognizer

app = Flask(__name__)

ENCODINGS_PATH = "face_recognition/encodingsold.pkl"
with open(ENCODINGS_PATH, "rb") as f:
    loaded_encodings = pickle.load(f)

prev_frame_time = 0
frame_times = []

# Your existing functions...
# get_esp_frame, cutout_image, get_middle_position, update_frame_times, draw_frame

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

def cutout_image(frame, corners):
    startX, startY, endX, endY = corners
    cutout = frame[startY:endY, startX:endX]
    return cutout

def get_middle_position(corners):
    startX, startY, endX, endY = corners
    middle_position = (startX + endX) / 2, (startY + endY) / 2
    return middle_position

people = {}


@app.route('/process_frame', methods=['POST'])
def process_frame():
    start_time = time.time()
    global prev_frame_time, frame_times

    to_return = {}

    # Assuming the frame is sent as an image file
    file = request.files['frame']
    npimg = np.fromstring(file.read(), np.uint8)
    frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)

    print("time to get frame: ", time.time() - start_time)

    body_positions = body_finder.find_body_positions(frame)
    people_in_screen = []

    for bodypos in body_positions:
        current_person_name = "No face"
        startX, startY, endX, endY = bodypos
        body = cutout_image(frame, bodypos)
        middle_position = get_middle_position(bodypos)

        # cv2.circle(frame, (int(middle_position[0]), int(middle_position[1])), 5, (0, 0, 255), -1)

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
            current_person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))
            if current_person_name != "No face":
                people[current_person_name] = middle_position
                people_in_screen.append(current_person_name)
        to_return[current_person_name] = middle_position

        # cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        # cv2.putText(frame, current_person_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)

    for name, middle_pos in people.items():
        if name not in people_in_screen:
            people.pop(name)
            break

    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)
    draw_frame(frame, avg_fps)

    print("time at end of regular stuff: ", time.time() - start_time)
    # Convert the processed frame back to image format for sending
    # _, img_encoded = cv2.imencode('.jpg', frame)
    # response = send_file(BytesIO(img_encoded), mimetype='image/jpeg')
    print(f"Time to do everything: {time.time() - start_time} seconds")
    return to_return

if __name__ == '__main__':
    app.run(debug=True, host="0.0.0.0")
