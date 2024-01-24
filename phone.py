import cv2
import time
import threading

def fetch_latest_frame(cap, latest_frame):
    while True:
        ret, frame = cap.read()
        if ret:
            with threading.Lock():
                latest_frame[0] = frame
        else:
            break

def view_stream(stream_url):
    cap = cv2.VideoCapture(stream_url)

    if not cap.isOpened():
        print("Error: Unable to open video stream.")
        return

    latest_frame = [None]
    thread = threading.Thread(target=fetch_latest_frame, args=(cap, latest_frame))
    thread.start()

    while True:
        time.sleep(0.1)  # Simulate processing time

        if latest_frame[0] is not None:
            with threading.Lock():
                cv2.imshow('Camera Stream', latest_frame[0])

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    thread.join()

stream_url = 'http://192.168.4.2:4747/video/1080p'
view_stream(stream_url)

