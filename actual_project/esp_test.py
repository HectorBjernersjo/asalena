import numpy as np
import requests
import time
import cv2

prev_frame_time = 0
frame_times = []

def get_esp_frame():
    # esp_url = "http://192.168.50.217/"
    esp_url = 'http://192.168.2.217'
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

while True:
    frame = get_esp_frame()
    start_time = time.time()


    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)
    draw_frame(frame, avg_fps)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

