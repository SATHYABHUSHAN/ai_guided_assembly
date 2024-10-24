import cv2
import numpy as np

def detect_washer_by_color(frame, lower_color, upper_color):
    hsv_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2HSV)
    color_mask = cv2.inRange(hsv_frame, lower_color, upper_color)
    color_result = cv2.bitwise_and(frame, frame, mask=color_mask)
    gray_image = cv2.cvtColor(color_result, cv2.COLOR_BGR2GRAY)
    blurred = cv2.GaussianBlur(gray_image, (9, 9), 2)
    circles = cv2.HoughCircles(
        blurred, 
        cv2.HOUGH_GRADIENT, 
        dp=1.2, 
        minDist=50, 
        param1=50, 
        param2=30, 
        minRadius=20, 
        maxRadius=150
    )
    return circles, color_result

color_ranges = {
    "blue": (np.array([100, 150, 50]), np.array([140, 255, 255])),
    "deep_teal": (np.array([60, 50, 50]), np.array([95, 255, 255])),
    "red": (np.array([0, 150, 50]), np.array([10, 255, 255]))
}

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not open webcam.")
    exit()

washer_status = "blue"

while True:
    ret, frame = cap.read()
    if not ret:
        print("Error: Could not read frame.")
        break

    lower_color, upper_color = color_ranges[washer_status]
    circles, color_result = detect_washer_by_color(frame, lower_color, upper_color)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
            cv2.circle(frame, (x, y), 2, (0, 255, 255), 3)

        if washer_status == "blue":
            detection_message = "Blue washer detected, go for deep teal washer"
            washer_status = "deep_teal"
        elif washer_status == "deep_teal":
            detection_message = "Deep teal washer detected, go for red"
            washer_status = "red"
        elif washer_status == "red":
            detection_message = "Red washer detected"

        cv2.putText(frame, detection_message, (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 255), 2)
        print(detection_message)

    cv2.imshow("Washer Detection", frame)
    cv2.imshow(f"{washer_status.capitalize()} Mask", color_result)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()

