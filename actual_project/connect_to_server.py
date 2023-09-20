import cv2
import requests
import time

cap = cv2.VideoCapture(0)
# VIDEO_PATH = "http://192.168.50.205:5000/process_frame"
# TEST_PATH = "http://192.168.50.205:5000/test"

VIDEO_PATH = "http://192.168.50.242:5000/process_frame"
TEST_PATH = "http://192.168.50.242:5000/test"

frame_times = []

while True:
    start_time = time.time()
    # ret, frame = cap.read()
    # _, img_encoded = cv2.imencode('.jpg', frame)
    # response = requests.post(VIDEO_PATH, files={'frame': img_encoded.tostring()})
    response = requests.get(TEST_PATH)
    # print(response.json())
    print(response)
    frame_times.append(time.time() - start_time)
    if len(frame_times) > 10:
        frame_times.pop(0)
    avg_fps = 1 / (sum(frame_times) / len(frame_times))
    print(avg_fps)