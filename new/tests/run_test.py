import cv2
import os
import time
# import caffe_detect_faces
# import multilevel
import retina
# import caffe2


if __name__ == "__main__":
    # detection_function = caffe_detect_faces.detect_faces
    # detection_function = multilevel.detect_faces
    # detection_function = caffe2.detect_faces
    detection_function = retina.detect_faces_with_retinaface

    for filename in os.listdir("test_images"):
        image = cv2.imread(f"test_images/{filename}")
        start_time = time.time()
        face_locations = detection_function(image)
        for (left, top, right, bottom) in face_locations:
            cv2.rectangle(image, (left, top), (right, bottom), (0, 255, 0), 2)

        print(f"Time to get face locations: {time.time() - start_time} seconds")
        # show the output image
        cv2.imshow("Output", image)
        cv2.waitKey(0)
