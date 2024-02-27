import cv2
from threading import Thread
import queue
import flask

from modules import caffe_detect_faces
from modules import openface_recognizer

class VideoCaptureAsync:
    def __init__(self, uri):
        self.uri = uri
        self.cap = cv2.VideoCapture(uri)
        self.q = queue.Queue()
        self.running = True
        self.read_thread = Thread(target=self.read_frame, args=())
        self.read_thread.daemon = True

    def start(self):
        self.read_thread.start()
        
    def read_frame(self):
        while self.running:
            ret, frame = self.cap.read()
            if not ret:
                self.running = False
                break
            if not self.q.empty():
                try:
                    self.q.get_nowait()   # discard previous (unprocessed) frame
                except queue.Empty:
                    pass
            self.q.put(frame)

    def read(self):
        return self.q.get()

    def stop(self):
        self.running = False
        self.read_thread.join()
        self.cap.release()

# uri = "http://192.168.1.202:4747/video/1080p"
uri = "http://192.168.50.14:4747/video/1080p"
cap_async = VideoCaptureAsync(uri)
cap_async.start()

app = flask.Flask(__name__)

@app.route('/')
def index():
    return flask.jsonify({'status': 'ok'})


priolist = ['Anton', 'Filip', 'Hector']



@app.route('/faces', methods=['GET'])
def get_face_locations():
    frame = cap_async.read()
    faces = caffe_detect_faces.detect_faces(frame)
    names_and_locations = []
    for face_location in faces:
        name = openface_recognizer.recognize_face_from_frame(frame, face_location)
        x = (face_location[0] + face_location[2]) / 2
        y = (face_location[1] + face_location[3]) / 2
        max_x = frame.shape[1]
        max_y = frame.shape[0]
        middle_x = max_x / 2
        middle_y = max_y / 2
        x = (x - middle_x) / middle_x
        y = (middle_y - y) / middle_y
        size = (face_location[2] - face_location[0]) + (face_location[3] - face_location[1])
        size = int(size) / 2 / middle_x
        names_and_locations.append({"name": name, "x": x, "y": y, "size": size})
        cv2.rectangle(frame, (face_location[0], face_location[1]), (face_location[2], face_location[3]), (0, 255, 0), 2)

    for name in priolist:
        for i, person in enumerate(names_and_locations):
            if name in person['name']:
                return flask.jsonify(names_and_locations[i])

    cv2.imwrite("frame.jpg", frame)
    return flask.jsonify({'found': False})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True, use_reloader=False)
