import cv2
import numpy as np
import time
import os

import body_finder
import caffe_detect_faces
import openface_recognizer

# Global variables for frame time tracking
prev_frame_time = 0
frame_times = []

def get_frame_from_camera(cap):
    ret, frame = cap.read()
    return frame

def find_bodies_and_faces_in_frame(frame):
    body_positions = body_finder.find_body_positions(frame)
    return body_positions

def process_body_positions(frame, body_positions, people):
    people_in_screen = []
    for bodypos in body_positions:
        current_person_name, middle_position = process_single_body(frame, bodypos, people)
        if current_person_name != "No face":
            people[current_person_name] = middle_position
            people_in_screen.append(current_person_name)

        draw_body_rectangle(frame, bodypos, current_person_name)
    return people_in_screen

def process_single_body(frame, bodypos, people):
    current_person_name = "No face"
    middle_position = get_middle_position(bodypos)

    for name, middle_pos in people.items():
        if is_near(middle_pos, middle_position):
            current_person_name = name
            break

    if current_person_name == "No face":
        current_person_name = process_face_in_body(frame, bodypos)

    return current_person_name, middle_position

def process_face_in_body(frame, bodypos):
    body = cutout_image(frame, bodypos)
    faces = caffe_detect_faces.detect_faces(body)
    if faces:
        return identify_and_save_face(frame, faces[0], bodypos)
    return "No face"

def identify_and_save_face(frame, face, bodypos):
    startX, startY, endX, endY = bodypos
    left, top, right, bottom = adjust_face_coordinates(face, startX, startY)
    person_name = openface_recognizer.recognize_face_from_frame(frame, (left, top, right, bottom))
    save_face(frame[top:bottom, left:right], person_name)
    draw_face_rectangle(frame, left, top, right, bottom)
    return person_name

def remove_absent_people(people, people_in_screen):
    for name in list(people.keys()):
        if name not in people_in_screen:
            del people[name]

def update_and_display_frame(frame, prev_frame_time, frame_times):
    avg_fps, prev_frame_time = update_frame_times(prev_frame_time, frame_times)
    draw_frame(frame, avg_fps)
    return prev_frame_time

def main():
    people = {}
    cap = cv2.VideoCapture(0)

    global prev_frame_time
    while True:
        frame = get_frame_from_camera(cap)
        body_positions = find_bodies_and_faces_in_frame(frame)
        people_in_screen = process_body_positions(frame, body_positions, people)
        remove_absent_people(people, people_in_screen)
        prev_frame_time = update_and_display_frame(frame, prev_frame_time, frame_times)

        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
