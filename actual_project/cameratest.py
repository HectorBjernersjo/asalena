import cv2

cap = cv2.VideoCapture("http://<your_ip>:8080/video")  # Replace <your_ip> with your iPhone's IP address
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

