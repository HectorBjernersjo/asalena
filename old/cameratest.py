import cv2

cap = cv2.VideoCapture("http://192.168.1.202:8080")  # Replace <your_ip> with your iPhone's IP address
while True:
    ret, frame = cap.read()
    if not ret:
        break

    cv2.imshow("Video Stream", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
