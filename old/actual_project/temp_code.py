import time
lines_times = {}
lines_times["import cv2"] = { "total_time": 0, "times": 0, "linenumber": "0" }
lines_times["import numpy as np"] = { "total_time": 0, "times": 0, "linenumber": "1" }
lines_times["import time"] = { "total_time": 0, "times": 0, "linenumber": "2" }
lines_times["import os"] = { "total_time": 0, "times": 0, "linenumber": "3" }
lines_times["import requests"] = { "total_time": 0, "times": 0, "linenumber": "4" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "5" }
lines_times["import body_finder"] = { "total_time": 0, "times": 0, "linenumber": "6" }
lines_times["import caffe_detect_faces"] = { "total_time": 0, "times": 0, "linenumber": "7" }
lines_times["import openface_recognizer"] = { "total_time": 0, "times": 0, "linenumber": "8" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "9" }
lines_times["prev_frame_time = 0"] = { "total_time": 0, "times": 0, "linenumber": "10" }
lines_times["frame_times = []"] = { "total_time": 0, "times": 0, "linenumber": "11" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "12" }
lines_times["time_for_face_detection_in_bodies = 0"] = { "total_time": 0, "times": 0, "linenumber": "13" }
lines_times["times_for_face_detection_in_bodies = 0"] = { "total_time": 0, "times": 0, "linenumber": "14" }
lines_times["time_for_face_recognition_general = 0"] = { "total_time": 0, "times": 0, "linenumber": "15" }
lines_times["times_for_face_recognition_general = 0"] = { "total_time": 0, "times": 0, "linenumber": "16" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "17" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "18" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "19" }
lines_times["def get_esp_frame():"] = { "total_time": 0, "times": 0, "linenumber": "20" }
lines_times["    # url = \"http://192.168.50.217/\" # hemma"] = { "total_time": 0, "times": 0, "linenumber": "21" }
lines_times["    # url = 'http://192.168.2.217' # mange"] = { "total_time": 0, "times": 0, "linenumber": "22" }
lines_times["    url = 'http://192.168.1.202:4747/video' # telefon hos anton"] = { "total_time": 0, "times": 0, "linenumber": "23" }
lines_times["    stream = requests.get(url, stream=True)"] = { "total_time": 0, "times": 0, "linenumber": "24" }
lines_times["    byte_stream = bytes()"] = { "total_time": 0, "times": 0, "linenumber": "25" }
lines_times["    for chunk in stream.iter_content(chunk_size=1024):"] = { "total_time": 0, "times": 0, "linenumber": "26" }
lines_times["        byte_stream += chunk"] = { "total_time": 0, "times": 0, "linenumber": "27" }
lines_times["        a = byte_stream.find(b'\xff\xd8')"] = { "total_time": 0, "times": 0, "linenumber": "28" }
lines_times["        b = byte_stream.find(b'\xff\xd9')"] = { "total_time": 0, "times": 0, "linenumber": "29" }
lines_times["        if a != -1 and b != -1:"] = { "total_time": 0, "times": 0, "linenumber": "30" }
lines_times["            jpg = byte_stream[a:b+2]"] = { "total_time": 0, "times": 0, "linenumber": "31" }
lines_times["            byte_stream = byte_stream[b+2:]"] = { "total_time": 0, "times": 0, "linenumber": "32" }
lines_times["            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)"] = { "total_time": 0, "times": 0, "linenumber": "33" }
lines_times["            return frame"] = { "total_time": 0, "times": 0, "linenumber": "34" }
lines_times["        "] = { "total_time": 0, "times": 0, "linenumber": "35" }
lines_times["def cutout_image(frame, corners):"] = { "total_time": 0, "times": 0, "linenumber": "36" }
lines_times["    startX, startY, endX, endY = corners"] = { "total_time": 0, "times": 0, "linenumber": "37" }
lines_times["    cutout = frame[startY:endY, startX:endX]"] = { "total_time": 0, "times": 0, "linenumber": "38" }
lines_times["    return cutout"] = { "total_time": 0, "times": 0, "linenumber": "39" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "40" }
lines_times["def get_middle_position(corners):"] = { "total_time": 0, "times": 0, "linenumber": "41" }
lines_times["    startX, startY, endX, endY = corners"] = { "total_time": 0, "times": 0, "linenumber": "42" }
lines_times["    middle_position = (startX + endX) / 2, (startY + endY) / 2"] = { "total_time": 0, "times": 0, "linenumber": "43" }
lines_times["    return middle_position"] = { "total_time": 0, "times": 0, "linenumber": "44" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "45" }
lines_times["def update_frame_times(prev_frame_time, frame_times):"] = { "total_time": 0, "times": 0, "linenumber": "46" }
lines_times["    current_time = time.time()"] = { "total_time": 0, "times": 0, "linenumber": "47" }
lines_times["    frame_times.append(current_time - prev_frame_time)"] = { "total_time": 0, "times": 0, "linenumber": "48" }
lines_times["    if len(frame_times) > 10:"] = { "total_time": 0, "times": 0, "linenumber": "49" }
lines_times["        frame_times.pop(0)"] = { "total_time": 0, "times": 0, "linenumber": "50" }
lines_times["    avg_fps = 1 / (sum(frame_times) / len(frame_times))"] = { "total_time": 0, "times": 0, "linenumber": "51" }
lines_times["    # print(avg_fps)"] = { "total_time": 0, "times": 0, "linenumber": "52" }
lines_times["    prev_frame_time = current_time"] = { "total_time": 0, "times": 0, "linenumber": "53" }
lines_times["    return avg_fps, prev_frame_time"] = { "total_time": 0, "times": 0, "linenumber": "54" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "55" }
lines_times["def draw_frame(frame, fps):"] = { "total_time": 0, "times": 0, "linenumber": "56" }
lines_times["    font = cv2.FONT_HERSHEY_SIMPLEX"] = { "total_time": 0, "times": 0, "linenumber": "57" }
lines_times["    cv2.putText(frame, f\"Avg FPS: {int(fps)}\", (10, 30), font, 1, (0, 255, 0), 2)"] = { "total_time": 0, "times": 0, "linenumber": "58" }
lines_times["    cv2.imshow(\"Face and body detection\", frame)"] = { "total_time": 0, "times": 0, "linenumber": "59" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "60" }
lines_times["def save_face(face_img, name, confidence):"] = { "total_time": 0, "times": 0, "linenumber": "61" }
lines_times["    folder = \"face_images\""] = { "total_time": 0, "times": 0, "linenumber": "62" }
lines_times["    if confidence < 0.35:"] = { "total_time": 0, "times": 0, "linenumber": "63" }
lines_times["        folder += f\"/{name}\""] = { "total_time": 0, "times": 0, "linenumber": "64" }
lines_times["    if not os.path.exists(folder):"] = { "total_time": 0, "times": 0, "linenumber": "65" }
lines_times["        os.makedirs(folder)"] = { "total_time": 0, "times": 0, "linenumber": "66" }
lines_times["    cv2.imwrite(f\"{folder}/{name}-{time.time()}.jpg\", face_img)"] = { "total_time": 0, "times": 0, "linenumber": "67" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "68" }
lines_times["        "] = { "total_time": 0, "times": 0, "linenumber": "69" }
lines_times["use_esp = False"] = { "total_time": 0, "times": 0, "linenumber": "70" }
lines_times["use_phone = True"] = { "total_time": 0, "times": 0, "linenumber": "71" }
lines_times["use_webcam = False"] = { "total_time": 0, "times": 0, "linenumber": "72" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "73" }
lines_times["phone_url = \"http://192.168.1.202:4747/video\""] = { "total_time": 0, "times": 0, "linenumber": "74" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "75" }
lines_times["people = {}"] = { "total_time": 0, "times": 0, "linenumber": "76" }
lines_times["if use_phone:"] = { "total_time": 0, "times": 0, "linenumber": "77" }
lines_times["    cap = cv2.VideoCapture(phone_url)"] = { "total_time": 0, "times": 0, "linenumber": "78" }
lines_times["elif use_webcam:"] = { "total_time": 0, "times": 0, "linenumber": "79" }
lines_times["    cap = cv2.VideoCapture(0)"] = { "total_time": 0, "times": 0, "linenumber": "80" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "81" }
lines_times["i = 0"] = { "total_time": 0, "times": 0, "linenumber": "82" }
lines_times["while True:"] = { "total_time": 0, "times": 0, "linenumber": "83" }
lines_times["    i += 1"] = { "total_time": 0, "times": 0, "linenumber": "84" }
lines_times["    start_time = time.time()"] = { "total_time": 0, "times": 0, "linenumber": "85" }
lines_times["    if use_esp:"] = { "total_time": 0, "times": 0, "linenumber": "86" }
lines_times["        frame = get_esp_frame()"] = { "total_time": 0, "times": 0, "linenumber": "87" }
lines_times["    elif use_webcam:"] = { "total_time": 0, "times": 0, "linenumber": "88" }
lines_times["        ret, frame = cap.read()"] = { "total_time": 0, "times": 0, "linenumber": "89" }
lines_times["    else:"] = { "total_time": 0, "times": 0, "linenumber": "90" }
lines_times["        latest_frame = None"] = { "total_time": 0, "times": 0, "linenumber": "91" }
lines_times["        start_time = time.time()"] = { "total_time": 0, "times": 0, "linenumber": "92" }
lines_times["        while True:"] = { "total_time": 0, "times": 0, "linenumber": "93" }
lines_times["            if cap.grab():"] = { "total_time": 0, "times": 0, "linenumber": "94" }
lines_times["                ret, frame = cap.retrieve()"] = { "total_time": 0, "times": 0, "linenumber": "95" }
lines_times["                if ret:"] = { "total_time": 0, "times": 0, "linenumber": "96" }
lines_times["                    latest_frame = frame"] = { "total_time": 0, "times": 0, "linenumber": "97" }
lines_times["            else:"] = { "total_time": 0, "times": 0, "linenumber": "98" }
lines_times["                break"] = { "total_time": 0, "times": 0, "linenumber": "99" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "100" }
lines_times["            # Break the loop if no new frame is available for 0.1 seconds"] = { "total_time": 0, "times": 0, "linenumber": "101" }
lines_times["            if time.time() - start_time > 0.001:"] = { "total_time": 0, "times": 0, "linenumber": "102" }
lines_times["                break"] = { "total_time": 0, "times": 0, "linenumber": "103" }
lines_times["    start_time = time.time()"] = { "total_time": 0, "times": 0, "linenumber": "104" }
lines_times["    body_positions = body_finder.find_body_positions(frame)"] = { "total_time": 0, "times": 0, "linenumber": "105" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "106" }
lines_times["    people_in_screen = []"] = { "total_time": 0, "times": 0, "linenumber": "107" }
lines_times["    "] = { "total_time": 0, "times": 0, "linenumber": "108" }
lines_times["    for bodypos in body_positions:"] = { "total_time": 0, "times": 0, "linenumber": "109" }
lines_times["        current_person_name = \"No face\""] = { "total_time": 0, "times": 0, "linenumber": "110" }
lines_times["        startX, startY, endX, endY = bodypos"] = { "total_time": 0, "times": 0, "linenumber": "111" }
lines_times["        middle_position = get_middle_position(bodypos)"] = { "total_time": 0, "times": 0, "linenumber": "112" }
lines_times["        body = cutout_image(frame, bodypos)"] = { "total_time": 0, "times": 0, "linenumber": "113" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "114" }
lines_times["        "] = { "total_time": 0, "times": 0, "linenumber": "115" }
lines_times["        for name, middle_pos in people.items():"] = { "total_time": 0, "times": 0, "linenumber": "116" }
lines_times["            if abs(middle_pos[0] - middle_position[0]) < 50 and abs(middle_pos[1] - middle_position[1]) < 50:"] = { "total_time": 0, "times": 0, "linenumber": "117" }
lines_times["                current_person_name = name"] = { "total_time": 0, "times": 0, "linenumber": "118" }
lines_times["                people[name] = middle_position"] = { "total_time": 0, "times": 0, "linenumber": "119" }
lines_times["                people_in_screen.append(name)"] = { "total_time": 0, "times": 0, "linenumber": "120" }
lines_times["                break"] = { "total_time": 0, "times": 0, "linenumber": "121" }
lines_times["        "] = { "total_time": 0, "times": 0, "linenumber": "122" }
lines_times["        if i % 30 == 0:"] = { "total_time": 0, "times": 0, "linenumber": "123" }
lines_times["            a = 1   "] = { "total_time": 0, "times": 0, "linenumber": "124" }
lines_times["        if current_person_name == \"No face\" or \"Unknown\" in current_person_name and i % 15 == 0 or i % 45 == 0:"] = { "total_time": 0, "times": 0, "linenumber": "125" }
lines_times["            if current_person_name in people:"] = { "total_time": 0, "times": 0, "linenumber": "126" }
lines_times["                people.pop(current_person_name)"] = { "total_time": 0, "times": 0, "linenumber": "127" }
lines_times["            if current_person_name in people_in_screen:"] = { "total_time": 0, "times": 0, "linenumber": "128" }
lines_times["                people_in_screen.remove(current_person_name)"] = { "total_time": 0, "times": 0, "linenumber": "129" }
lines_times["            bodyimg = cutout_image(frame, bodypos)"] = { "total_time": 0, "times": 0, "linenumber": "130" }
lines_times["            faces = caffe_detect_faces.detect_faces(bodyimg)"] = { "total_time": 0, "times": 0, "linenumber": "131" }
lines_times["            if len(faces) == 0:"] = { "total_time": 0, "times": 0, "linenumber": "132" }
lines_times["                continue"] = { "total_time": 0, "times": 0, "linenumber": "133" }
lines_times["            print(f\"Found {len(faces)} faces in body, should be 1\")"] = { "total_time": 0, "times": 0, "linenumber": "134" }
lines_times["            face = faces[0]"] = { "total_time": 0, "times": 0, "linenumber": "135" }
lines_times["            "] = { "total_time": 0, "times": 0, "linenumber": "136" }
lines_times["            left, top, right, bottom = face"] = { "total_time": 0, "times": 0, "linenumber": "137" }
lines_times["            left += startX"] = { "total_time": 0, "times": 0, "linenumber": "138" }
lines_times["            top += startY"] = { "total_time": 0, "times": 0, "linenumber": "139" }
lines_times["            right += startX"] = { "total_time": 0, "times": 0, "linenumber": "140" }
lines_times["            bottom += startY"] = { "total_time": 0, "times": 0, "linenumber": "141" }
lines_times["            "] = { "total_time": 0, "times": 0, "linenumber": "142" }
lines_times["            current_person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))"] = { "total_time": 0, "times": 0, "linenumber": "143" }
lines_times["            confidence = float(current_person_name.split(\"-\")[1])"] = { "total_time": 0, "times": 0, "linenumber": "144" }
lines_times["            name = current_person_name.split(\"-\")[0]"] = { "total_time": 0, "times": 0, "linenumber": "145" }
lines_times["            save_face(frame[top:bottom, left:right], name, confidence)"] = { "total_time": 0, "times": 0, "linenumber": "146" }
lines_times["            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)"] = { "total_time": 0, "times": 0, "linenumber": "147" }
lines_times["            cv2.imshow(\"Face\", frame[top:bottom, left:right])"] = { "total_time": 0, "times": 0, "linenumber": "148" }
lines_times["            "] = { "total_time": 0, "times": 0, "linenumber": "149" }
lines_times["            if current_person_name != \"No face\":"] = { "total_time": 0, "times": 0, "linenumber": "150" }
lines_times["                people[current_person_name] = middle_position"] = { "total_time": 0, "times": 0, "linenumber": "151" }
lines_times["                people_in_screen.append(current_person_name)"] = { "total_time": 0, "times": 0, "linenumber": "152" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "153" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "154" }
lines_times["        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)"] = { "total_time": 0, "times": 0, "linenumber": "155" }
lines_times["        cv2.putText(frame, current_person_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)"] = { "total_time": 0, "times": 0, "linenumber": "156" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "157" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "158" }
lines_times["        "] = { "total_time": 0, "times": 0, "linenumber": "159" }
lines_times["    for name, middle_pos in people.items():"] = { "total_time": 0, "times": 0, "linenumber": "160" }
lines_times["        if name not in people_in_screen:"] = { "total_time": 0, "times": 0, "linenumber": "161" }
lines_times["            people.pop(name)"] = { "total_time": 0, "times": 0, "linenumber": "162" }
lines_times["            break"] = { "total_time": 0, "times": 0, "linenumber": "163" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "164" }
lines_times["    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)"] = { "total_time": 0, "times": 0, "linenumber": "165" }
lines_times["    draw_frame(frame, avg_fps)"] = { "total_time": 0, "times": 0, "linenumber": "166" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "167" }
lines_times["    # print(f\"Time to process frame: {time.time() - start_time} seconds\")"] = { "total_time": 0, "times": 0, "linenumber": "168" }
lines_times[""] = { "total_time": 0, "times": 0, "linenumber": "169" }
lines_times["    if cv2.waitKey(1) & 0xFF == ord('q'):"] = { "total_time": 0, "times": 0, "linenumber": "170" }
lines_times["        break"] = { "total_time": 0, "times": 0, "linenumber": "171" }
import cv2
import numpy as np
import time
import os
import requests
import body_finder
import caffe_detect_faces
import openface_recognizer
start_time_temp = time.time()
prev_frame_time = 0
time_for_line = time.time() - start_time_temp
lines_times["prev_frame_time = 0"]["total_time"] += time_for_line
lines_times["prev_frame_time = 0"]["times"] += 1
start_time_temp = time.time()
frame_times = []
time_for_line = time.time() - start_time_temp
lines_times["frame_times = []"]["total_time"] += time_for_line
lines_times["frame_times = []"]["times"] += 1
start_time_temp = time.time()
time_for_face_detection_in_bodies = 0
time_for_line = time.time() - start_time_temp
lines_times["time_for_face_detection_in_bodies = 0"]["total_time"] += time_for_line
lines_times["time_for_face_detection_in_bodies = 0"]["times"] += 1
start_time_temp = time.time()
times_for_face_detection_in_bodies = 0
time_for_line = time.time() - start_time_temp
lines_times["times_for_face_detection_in_bodies = 0"]["total_time"] += time_for_line
lines_times["times_for_face_detection_in_bodies = 0"]["times"] += 1
start_time_temp = time.time()
time_for_face_recognition_general = 0
time_for_line = time.time() - start_time_temp
lines_times["time_for_face_recognition_general = 0"]["total_time"] += time_for_line
lines_times["time_for_face_recognition_general = 0"]["times"] += 1
start_time_temp = time.time()
times_for_face_recognition_general = 0
time_for_line = time.time() - start_time_temp
lines_times["times_for_face_recognition_general = 0"]["total_time"] += time_for_line
lines_times["times_for_face_recognition_general = 0"]["times"] += 1
def get_esp_frame():
    # url = "http://192.168.50.217/" # hemma
    # url = 'http://192.168.2.217' # mange
    start_time_temp = time.time()
    url = 'http://192.168.1.202:4747/video' # telefon hos anton
    time_for_line = time.time() - start_time_temp
    lines_times["    url = 'http://192.168.1.202:4747/video' # telefon hos anton"]["total_time"] += time_for_line
    lines_times["    url = 'http://192.168.1.202:4747/video' # telefon hos anton"]["times"] += 1
    start_time_temp = time.time()
    stream = requests.get(url, stream=True)
    time_for_line = time.time() - start_time_temp
    lines_times["    stream = requests.get(url, stream=True)"]["total_time"] += time_for_line
    lines_times["    stream = requests.get(url, stream=True)"]["times"] += 1
    start_time_temp = time.time()
    byte_stream = bytes()
    time_for_line = time.time() - start_time_temp
    lines_times["    byte_stream = bytes()"]["total_time"] += time_for_line
    lines_times["    byte_stream = bytes()"]["times"] += 1
    for chunk in stream.iter_content(chunk_size=1024):
        start_time_temp = time.time()
        byte_stream += chunk
        time_for_line = time.time() - start_time_temp
        lines_times["        byte_stream += chunk"]["total_time"] += time_for_line
        lines_times["        byte_stream += chunk"]["times"] += 1
        start_time_temp = time.time()
        a = byte_stream.find(b'\xff\xd8')
        time_for_line = time.time() - start_time_temp
        lines_times["        a = byte_stream.find(b'\xff\xd8')"]["total_time"] += time_for_line
        lines_times["        a = byte_stream.find(b'\xff\xd8')"]["times"] += 1
        start_time_temp = time.time()
        b = byte_stream.find(b'\xff\xd9')
        time_for_line = time.time() - start_time_temp
        lines_times["        b = byte_stream.find(b'\xff\xd9')"]["total_time"] += time_for_line
        lines_times["        b = byte_stream.find(b'\xff\xd9')"]["times"] += 1
        if a != -1 and b != -1:
            start_time_temp = time.time()
            jpg = byte_stream[a:b+2]
            time_for_line = time.time() - start_time_temp
            lines_times["            jpg = byte_stream[a:b+2]"]["total_time"] += time_for_line
            lines_times["            jpg = byte_stream[a:b+2]"]["times"] += 1
            start_time_temp = time.time()
            byte_stream = byte_stream[b+2:]
            time_for_line = time.time() - start_time_temp
            lines_times["            byte_stream = byte_stream[b+2:]"]["total_time"] += time_for_line
            lines_times["            byte_stream = byte_stream[b+2:]"]["times"] += 1
            start_time_temp = time.time()
            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)
            time_for_line = time.time() - start_time_temp
            lines_times["            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)"]["total_time"] += time_for_line
            lines_times["            frame = cv2.imdecode(np.frombuffer(jpg, dtype=np.uint8), cv2.IMREAD_COLOR)"]["times"] += 1
            return frame
def cutout_image(frame, corners):
    start_time_temp = time.time()
    startX, startY, endX, endY = corners
    time_for_line = time.time() - start_time_temp
    lines_times["    startX, startY, endX, endY = corners"]["total_time"] += time_for_line
    lines_times["    startX, startY, endX, endY = corners"]["times"] += 1
    start_time_temp = time.time()
    cutout = frame[startY:endY, startX:endX]
    time_for_line = time.time() - start_time_temp
    lines_times["    cutout = frame[startY:endY, startX:endX]"]["total_time"] += time_for_line
    lines_times["    cutout = frame[startY:endY, startX:endX]"]["times"] += 1
    return cutout
def get_middle_position(corners):
    start_time_temp = time.time()
    startX, startY, endX, endY = corners
    time_for_line = time.time() - start_time_temp
    lines_times["    startX, startY, endX, endY = corners"]["total_time"] += time_for_line
    lines_times["    startX, startY, endX, endY = corners"]["times"] += 1
    start_time_temp = time.time()
    middle_position = (startX + endX) / 2, (startY + endY) / 2
    time_for_line = time.time() - start_time_temp
    lines_times["    middle_position = (startX + endX) / 2, (startY + endY) / 2"]["total_time"] += time_for_line
    lines_times["    middle_position = (startX + endX) / 2, (startY + endY) / 2"]["times"] += 1
    return middle_position
def update_frame_times(prev_frame_time, frame_times):
    start_time_temp = time.time()
    current_time = time.time()
    time_for_line = time.time() - start_time_temp
    lines_times["    current_time = time.time()"]["total_time"] += time_for_line
    lines_times["    current_time = time.time()"]["times"] += 1
    start_time_temp = time.time()
    frame_times.append(current_time - prev_frame_time)
    time_for_line = time.time() - start_time_temp
    lines_times["    frame_times.append(current_time - prev_frame_time)"]["total_time"] += time_for_line
    lines_times["    frame_times.append(current_time - prev_frame_time)"]["times"] += 1
    if len(frame_times) > 10:
        start_time_temp = time.time()
        frame_times.pop(0)
        time_for_line = time.time() - start_time_temp
        lines_times["        frame_times.pop(0)"]["total_time"] += time_for_line
        lines_times["        frame_times.pop(0)"]["times"] += 1
    start_time_temp = time.time()
    avg_fps = 1 / (sum(frame_times) / len(frame_times))
    time_for_line = time.time() - start_time_temp
    lines_times["    avg_fps = 1 / (sum(frame_times) / len(frame_times))"]["total_time"] += time_for_line
    lines_times["    avg_fps = 1 / (sum(frame_times) / len(frame_times))"]["times"] += 1
    # print(avg_fps)
    start_time_temp = time.time()
    prev_frame_time = current_time
    time_for_line = time.time() - start_time_temp
    lines_times["    prev_frame_time = current_time"]["total_time"] += time_for_line
    lines_times["    prev_frame_time = current_time"]["times"] += 1
    return avg_fps, prev_frame_time
def draw_frame(frame, fps):
    start_time_temp = time.time()
    font = cv2.FONT_HERSHEY_SIMPLEX
    time_for_line = time.time() - start_time_temp
    lines_times["    font = cv2.FONT_HERSHEY_SIMPLEX"]["total_time"] += time_for_line
    lines_times["    font = cv2.FONT_HERSHEY_SIMPLEX"]["times"] += 1
    start_time_temp = time.time()
    cv2.putText(frame, f"Avg FPS: {int(fps)}", (10, 30), font, 1, (0, 255, 0), 2)
    time_for_line = time.time() - start_time_temp
    lines_times["    cv2.putText(frame, f\"Avg FPS: {int(fps)}\", (10, 30), font, 1, (0, 255, 0), 2)"]["total_time"] += time_for_line
    lines_times["    cv2.putText(frame, f\"Avg FPS: {int(fps)}\", (10, 30), font, 1, (0, 255, 0), 2)"]["times"] += 1
    start_time_temp = time.time()
    cv2.imshow("Face and body detection", frame)
    time_for_line = time.time() - start_time_temp
    lines_times["    cv2.imshow(\"Face and body detection\", frame)"]["total_time"] += time_for_line
    lines_times["    cv2.imshow(\"Face and body detection\", frame)"]["times"] += 1
def save_face(face_img, name, confidence):
    start_time_temp = time.time()
    folder = "face_images"
    time_for_line = time.time() - start_time_temp
    lines_times["    folder = \"face_images\""]["total_time"] += time_for_line
    lines_times["    folder = \"face_images\""]["times"] += 1
    if confidence < 0.35:
        start_time_temp = time.time()
        folder += f"/{name}"
        time_for_line = time.time() - start_time_temp
        lines_times["        folder += f\"/{name}\""]["total_time"] += time_for_line
        lines_times["        folder += f\"/{name}\""]["times"] += 1
    if not os.path.exists(folder):
        start_time_temp = time.time()
        os.makedirs(folder)
        time_for_line = time.time() - start_time_temp
        lines_times["        os.makedirs(folder)"]["total_time"] += time_for_line
        lines_times["        os.makedirs(folder)"]["times"] += 1
    start_time_temp = time.time()
    cv2.imwrite(f"{folder}/{name}-{time.time()}.jpg", face_img)
    time_for_line = time.time() - start_time_temp
    lines_times["    cv2.imwrite(f\"{folder}/{name}-{time.time()}.jpg\", face_img)"]["total_time"] += time_for_line
    lines_times["    cv2.imwrite(f\"{folder}/{name}-{time.time()}.jpg\", face_img)"]["times"] += 1
start_time_temp = time.time()
use_esp = False
time_for_line = time.time() - start_time_temp
lines_times["use_esp = False"]["total_time"] += time_for_line
lines_times["use_esp = False"]["times"] += 1
start_time_temp = time.time()
use_phone = True
time_for_line = time.time() - start_time_temp
lines_times["use_phone = True"]["total_time"] += time_for_line
lines_times["use_phone = True"]["times"] += 1
start_time_temp = time.time()
use_webcam = False
time_for_line = time.time() - start_time_temp
lines_times["use_webcam = False"]["total_time"] += time_for_line
lines_times["use_webcam = False"]["times"] += 1
start_time_temp = time.time()
phone_url = "http://192.168.1.202:4747/video"
time_for_line = time.time() - start_time_temp
lines_times["phone_url = \"http://192.168.1.202:4747/video\""]["total_time"] += time_for_line
lines_times["phone_url = \"http://192.168.1.202:4747/video\""]["times"] += 1
start_time_temp = time.time()
people = {}
time_for_line = time.time() - start_time_temp
lines_times["people = {}"]["total_time"] += time_for_line
lines_times["people = {}"]["times"] += 1
if use_phone:
    start_time_temp = time.time()
    cap = cv2.VideoCapture(phone_url)
    time_for_line = time.time() - start_time_temp
    lines_times["    cap = cv2.VideoCapture(phone_url)"]["total_time"] += time_for_line
    lines_times["    cap = cv2.VideoCapture(phone_url)"]["times"] += 1
elif use_webcam:
    start_time_temp = time.time()
    cap = cv2.VideoCapture(0)
    time_for_line = time.time() - start_time_temp
    lines_times["    cap = cv2.VideoCapture(0)"]["total_time"] += time_for_line
    lines_times["    cap = cv2.VideoCapture(0)"]["times"] += 1
start_time_temp = time.time()
i = 0
time_for_line = time.time() - start_time_temp
lines_times["i = 0"]["total_time"] += time_for_line
lines_times["i = 0"]["times"] += 1
while True:
    start_time_temp = time.time()
    i += 1
    time_for_line = time.time() - start_time_temp
    lines_times["    i += 1"]["total_time"] += time_for_line
    lines_times["    i += 1"]["times"] += 1
    start_time_temp = time.time()
    start_time = time.time()
    time_for_line = time.time() - start_time_temp
    lines_times["    start_time = time.time()"]["total_time"] += time_for_line
    lines_times["    start_time = time.time()"]["times"] += 1
    if use_esp:
        start_time_temp = time.time()
        frame = get_esp_frame()
        time_for_line = time.time() - start_time_temp
        lines_times["        frame = get_esp_frame()"]["total_time"] += time_for_line
        lines_times["        frame = get_esp_frame()"]["times"] += 1
    elif use_webcam:
        start_time_temp = time.time()
        ret, frame = cap.read()
        time_for_line = time.time() - start_time_temp
        lines_times["        ret, frame = cap.read()"]["total_time"] += time_for_line
        lines_times["        ret, frame = cap.read()"]["times"] += 1
    else:
        start_time_temp = time.time()
        latest_frame = None
        time_for_line = time.time() - start_time_temp
        lines_times["        latest_frame = None"]["total_time"] += time_for_line
        lines_times["        latest_frame = None"]["times"] += 1
        start_time_temp = time.time()
        start_time = time.time()
        time_for_line = time.time() - start_time_temp
        lines_times["        start_time = time.time()"]["total_time"] += time_for_line
        lines_times["        start_time = time.time()"]["times"] += 1
        while True:
            if cap.grab():
                start_time_temp = time.time()
                ret, frame = cap.retrieve()
                time_for_line = time.time() - start_time_temp
                lines_times["                ret, frame = cap.retrieve()"]["total_time"] += time_for_line
                lines_times["                ret, frame = cap.retrieve()"]["times"] += 1
                if ret:
                    start_time_temp = time.time()
                    latest_frame = frame
                    time_for_line = time.time() - start_time_temp
                    lines_times["                    latest_frame = frame"]["total_time"] += time_for_line
                    lines_times["                    latest_frame = frame"]["times"] += 1
            else:
                break
            # Break the loop if no new frame is available for 0.1 seconds
            if time.time() - start_time > 0.001:
                break
    start_time_temp = time.time()
    start_time = time.time()
    time_for_line = time.time() - start_time_temp
    lines_times["    start_time = time.time()"]["total_time"] += time_for_line
    lines_times["    start_time = time.time()"]["times"] += 1
    start_time_temp = time.time()
    body_positions = body_finder.find_body_positions(frame)
    time_for_line = time.time() - start_time_temp
    lines_times["    body_positions = body_finder.find_body_positions(frame)"]["total_time"] += time_for_line
    lines_times["    body_positions = body_finder.find_body_positions(frame)"]["times"] += 1
    start_time_temp = time.time()
    people_in_screen = []
    time_for_line = time.time() - start_time_temp
    lines_times["    people_in_screen = []"]["total_time"] += time_for_line
    lines_times["    people_in_screen = []"]["times"] += 1
    for bodypos in body_positions:
        start_time_temp = time.time()
        current_person_name = "No face"
        time_for_line = time.time() - start_time_temp
        lines_times["        current_person_name = \"No face\""]["total_time"] += time_for_line
        lines_times["        current_person_name = \"No face\""]["times"] += 1
        start_time_temp = time.time()
        startX, startY, endX, endY = bodypos
        time_for_line = time.time() - start_time_temp
        lines_times["        startX, startY, endX, endY = bodypos"]["total_time"] += time_for_line
        lines_times["        startX, startY, endX, endY = bodypos"]["times"] += 1
        start_time_temp = time.time()
        middle_position = get_middle_position(bodypos)
        time_for_line = time.time() - start_time_temp
        lines_times["        middle_position = get_middle_position(bodypos)"]["total_time"] += time_for_line
        lines_times["        middle_position = get_middle_position(bodypos)"]["times"] += 1
        start_time_temp = time.time()
        body = cutout_image(frame, bodypos)
        time_for_line = time.time() - start_time_temp
        lines_times["        body = cutout_image(frame, bodypos)"]["total_time"] += time_for_line
        lines_times["        body = cutout_image(frame, bodypos)"]["times"] += 1
        for name, middle_pos in people.items():
            if abs(middle_pos[0] - middle_position[0]) < 50 and abs(middle_pos[1] - middle_position[1]) < 50:
                start_time_temp = time.time()
                current_person_name = name
                time_for_line = time.time() - start_time_temp
                lines_times["                current_person_name = name"]["total_time"] += time_for_line
                lines_times["                current_person_name = name"]["times"] += 1
                start_time_temp = time.time()
                people[name] = middle_position
                time_for_line = time.time() - start_time_temp
                lines_times["                people[name] = middle_position"]["total_time"] += time_for_line
                lines_times["                people[name] = middle_position"]["times"] += 1
                start_time_temp = time.time()
                people_in_screen.append(name)
                time_for_line = time.time() - start_time_temp
                lines_times["                people_in_screen.append(name)"]["total_time"] += time_for_line
                lines_times["                people_in_screen.append(name)"]["times"] += 1
                break
        if i % 30 == 0:
            start_time_temp = time.time()
            a = 1   
            time_for_line = time.time() - start_time_temp
            lines_times["            a = 1   "]["total_time"] += time_for_line
            lines_times["            a = 1   "]["times"] += 1
        if current_person_name == "No face" or "Unknown" in current_person_name and i % 15 == 0 or i % 45 == 0:
            if current_person_name in people:
                start_time_temp = time.time()
                people.pop(current_person_name)
                time_for_line = time.time() - start_time_temp
                lines_times["                people.pop(current_person_name)"]["total_time"] += time_for_line
                lines_times["                people.pop(current_person_name)"]["times"] += 1
            if current_person_name in people_in_screen:
                start_time_temp = time.time()
                people_in_screen.remove(current_person_name)
                time_for_line = time.time() - start_time_temp
                lines_times["                people_in_screen.remove(current_person_name)"]["total_time"] += time_for_line
                lines_times["                people_in_screen.remove(current_person_name)"]["times"] += 1
            start_time_temp = time.time()
            bodyimg = cutout_image(frame, bodypos)
            time_for_line = time.time() - start_time_temp
            lines_times["            bodyimg = cutout_image(frame, bodypos)"]["total_time"] += time_for_line
            lines_times["            bodyimg = cutout_image(frame, bodypos)"]["times"] += 1
            start_time_temp = time.time()
            faces = caffe_detect_faces.detect_faces(bodyimg)
            time_for_line = time.time() - start_time_temp
            lines_times["            faces = caffe_detect_faces.detect_faces(bodyimg)"]["total_time"] += time_for_line
            lines_times["            faces = caffe_detect_faces.detect_faces(bodyimg)"]["times"] += 1
            if len(faces) == 0:
                continue
            start_time_temp = time.time()
            print(f"Found {len(faces)} faces in body, should be 1")
            time_for_line = time.time() - start_time_temp
            lines_times["            print(f\"Found {len(faces)} faces in body, should be 1\")"]["total_time"] += time_for_line
            lines_times["            print(f\"Found {len(faces)} faces in body, should be 1\")"]["times"] += 1
            start_time_temp = time.time()
            face = faces[0]
            time_for_line = time.time() - start_time_temp
            lines_times["            face = faces[0]"]["total_time"] += time_for_line
            lines_times["            face = faces[0]"]["times"] += 1
            start_time_temp = time.time()
            left, top, right, bottom = face
            time_for_line = time.time() - start_time_temp
            lines_times["            left, top, right, bottom = face"]["total_time"] += time_for_line
            lines_times["            left, top, right, bottom = face"]["times"] += 1
            start_time_temp = time.time()
            left += startX
            time_for_line = time.time() - start_time_temp
            lines_times["            left += startX"]["total_time"] += time_for_line
            lines_times["            left += startX"]["times"] += 1
            start_time_temp = time.time()
            top += startY
            time_for_line = time.time() - start_time_temp
            lines_times["            top += startY"]["total_time"] += time_for_line
            lines_times["            top += startY"]["times"] += 1
            start_time_temp = time.time()
            right += startX
            time_for_line = time.time() - start_time_temp
            lines_times["            right += startX"]["total_time"] += time_for_line
            lines_times["            right += startX"]["times"] += 1
            start_time_temp = time.time()
            bottom += startY
            time_for_line = time.time() - start_time_temp
            lines_times["            bottom += startY"]["total_time"] += time_for_line
            lines_times["            bottom += startY"]["times"] += 1
            start_time_temp = time.time()
            current_person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))
            time_for_line = time.time() - start_time_temp
            lines_times["            current_person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))"]["total_time"] += time_for_line
            lines_times["            current_person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))"]["times"] += 1
            start_time_temp = time.time()
            confidence = float(current_person_name.split("-")[1])
            time_for_line = time.time() - start_time_temp
            lines_times["            confidence = float(current_person_name.split(\"-\")[1])"]["total_time"] += time_for_line
            lines_times["            confidence = float(current_person_name.split(\"-\")[1])"]["times"] += 1
            start_time_temp = time.time()
            name = current_person_name.split("-")[0]
            time_for_line = time.time() - start_time_temp
            lines_times["            name = current_person_name.split(\"-\")[0]"]["total_time"] += time_for_line
            lines_times["            name = current_person_name.split(\"-\")[0]"]["times"] += 1
            start_time_temp = time.time()
            save_face(frame[top:bottom, left:right], name, confidence)
            time_for_line = time.time() - start_time_temp
            lines_times["            save_face(frame[top:bottom, left:right], name, confidence)"]["total_time"] += time_for_line
            lines_times["            save_face(frame[top:bottom, left:right], name, confidence)"]["times"] += 1
            start_time_temp = time.time()
            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)
            time_for_line = time.time() - start_time_temp
            lines_times["            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)"]["total_time"] += time_for_line
            lines_times["            cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)"]["times"] += 1
            start_time_temp = time.time()
            cv2.imshow("Face", frame[top:bottom, left:right])
            time_for_line = time.time() - start_time_temp
            lines_times["            cv2.imshow(\"Face\", frame[top:bottom, left:right])"]["total_time"] += time_for_line
            lines_times["            cv2.imshow(\"Face\", frame[top:bottom, left:right])"]["times"] += 1
            if current_person_name != "No face":
                start_time_temp = time.time()
                people[current_person_name] = middle_position
                time_for_line = time.time() - start_time_temp
                lines_times["                people[current_person_name] = middle_position"]["total_time"] += time_for_line
                lines_times["                people[current_person_name] = middle_position"]["times"] += 1
                start_time_temp = time.time()
                people_in_screen.append(current_person_name)
                time_for_line = time.time() - start_time_temp
                lines_times["                people_in_screen.append(current_person_name)"]["total_time"] += time_for_line
                lines_times["                people_in_screen.append(current_person_name)"]["times"] += 1
        start_time_temp = time.time()
        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)
        time_for_line = time.time() - start_time_temp
        lines_times["        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)"]["total_time"] += time_for_line
        lines_times["        cv2.rectangle(frame, (startX, startY), (endX, endY), (0, 255, 0), 2)"]["times"] += 1
        start_time_temp = time.time()
        cv2.putText(frame, current_person_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)
        time_for_line = time.time() - start_time_temp
        lines_times["        cv2.putText(frame, current_person_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)"]["total_time"] += time_for_line
        lines_times["        cv2.putText(frame, current_person_name, (startX, startY - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 0), 2)"]["times"] += 1
    for name, middle_pos in people.items():
        if name not in people_in_screen:
            start_time_temp = time.time()
            people.pop(name)
            time_for_line = time.time() - start_time_temp
            lines_times["            people.pop(name)"]["total_time"] += time_for_line
            lines_times["            people.pop(name)"]["times"] += 1
            break
    start_time_temp = time.time()
    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)
    time_for_line = time.time() - start_time_temp
    lines_times["    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)"]["total_time"] += time_for_line
    lines_times["    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)"]["times"] += 1
    start_time_temp = time.time()
    draw_frame(frame, avg_fps)
    time_for_line = time.time() - start_time_temp
    lines_times["    draw_frame(frame, avg_fps)"]["total_time"] += time_for_line
    lines_times["    draw_frame(frame, avg_fps)"]["times"] += 1
    # print(f"Time to process frame: {time.time() - start_time} seconds")
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break
total_time = sum([data['total_time'] for data in lines_times.values()])
for line, data in lines_times.items():
    if data['times'] > 0:
        print("{:>4} {:<100} --- tottime: {:<5}, times: {:<4}, percent: {:<4}, avg: {:<4}".format(data['linenumber'], line, "{:.2f}".format(data['total_time']), data['times'], "{:.2f}".format(data['total_time'] / total_time * 100), "{:.2f}".format(data['total_time'] / data['times'])))
    else:
        print("{:>4} {:<100} --- tottime: {:<5}, times: {:<4}, percent: {:<4}".format(data['linenumber'], line, "{:.2f}".format(data['total_time']), data['times'], "{:.2f}".format(data['total_time'] / total_time * 100)))
print("\n\n\n\n")
sorted_lines = sorted(lines_times.items(), key=lambda x: x[1]['total_time'], reverse=True)
for line, data in sorted_lines:
    print("{:>4} {:<100} --- tottime: {:<5}, times: {:<4}, percent: {:<4}".format(data['linenumber'], line, "{:.2f}".format(data['total_time']), data['times'], "{:.2f}".format(data['total_time'] / total_time * 100)))
