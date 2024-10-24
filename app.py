import cv2
import numpy as np
import streamlit as st
import time

# Set Streamlit page config
st.set_page_config(page_title="Advitiix Technovate", page_icon="✔", layout="wide")

# Display logo and titles side by side
col1, col2 = st.columns([1, 4])  # Adjust the ratio as necessary

with col1:
    st.image("alogo.jpg", width=220)  # Set an appropriate width for the logo

with col2:
    st.markdown("<h1 style='text-align: left; color: blue; font-size: 24px;'>Advitiix Technovate</h1>", unsafe_allow_html=True)
    st.markdown("<h2 style='text-align: left; font-size: 18px;'>Ai guided assembly</h2>", unsafe_allow_html=True)

# Define CSS for radio buttons
st.markdown("""
    <style>
        .washer-status {
            font-size: 18px;  /* Reduced font size */
            padding: 10px;     /* Increased padding for spacing */
            margin-bottom: 15px; /* Increased margin for spacing */
            border: 2px solid #cccccc;
            border-radius: 5px; /* Reduced border radius */
            background-color: #ffffff; /* Added background color for contrast */
        }
        .washer-detected {
            background-color: #c3f7c3;
            color: green;
            font-weight: bold;
            border-color: green;
        }
        .washer-not-detected {
            background-color: #f7c3c3;
            color: red;
            font-weight: bold;
            border-color: red;
        }
    </style>
""", unsafe_allow_html=True)

# Washer detection logic
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
        minRadius=30,  # Reduced min radius for smaller washers
        maxRadius=50   # Reduced max radius for smaller washers
    )
    return circles, color_result

# Reinitialize camera after washer detection
def reinitialize_camera():
    global cap
    cap.release()  # Release the current capture object
    cap = cv2.VideoCapture(0)  # Reopen the camera
    if not cap.isOpened():
        st.error("Error: Could not access the webcam.")

# Initialize color ranges
color_ranges = {
    "blue": (np.array([100, 150, 50]), np.array([140, 255, 255])),
    "teal": (np.array([90, 50, 50]), np.array([105, 255, 150])),
    "red": (np.array([0, 150, 50]), np.array([10, 255, 255]))
}

# Open webcam
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    st.error("Error: Could not access the webcam.")
    cap = None

# Retry mechanism
retry_counter = 0
max_retries = 10

# Washer statuses to track detections
washer_statuses = {
    "blue": False,
    "teal": False,
    "red": False
}

# Only proceed if cap has been initialized
if cap is not None:  # Removed quit button logic
    washer_status = "blue"  # Start detection for blue washer first

    # Streamlit placeholder for the video feed
    # Using columns to place video and detection status side by side
    col1, col2 = st.columns([1, 1])  # Both columns take equal space (50% each)

    frame_placeholder = col1.empty()
    status_placeholder = col2.empty()

    # Processing loop for live feed
    while cap.isOpened():  # Removed quit button check
        ret, frame = cap.read()

        # If frame is not captured properly, retry or break after max_retries
        if not ret or frame is None:
            st.warning("Could not read frame. Retrying...")
            retry_counter += 1
            if retry_counter >= max_retries:
                st.error("Max retries reached, exiting...")
                break
            time.sleep(0.1)
            continue

        retry_counter = 0  # Reset retry counter on successful frame read

        # Washer detection logic
        lower_color, upper_color = color_ranges[washer_status]
        circles, color_result = detect_washer_by_color(frame, lower_color, upper_color)

        # If washer is detected, update washer statuses
        if circles is not None:
            circles = np.round(circles[0, :]).astype("int")
            for (x, y, r) in circles:
                cv2.circle(frame, (x, y), r, (0, 255, 0), 4)
                cv2.circle(frame, (x, y), 2, (0, 255, 255), 3)

            # Mark washer status as detected and switch to next washer
            washer_statuses[washer_status] = True
            if washer_status == "blue":
                washer_status = "teal"
                reinitialize_camera()
            elif washer_status == "teal":
                washer_status = "red"
            elif washer_status == "red":
                washer_status = None  # Stop detection after red washer

        # Convert frame for display in Streamlit
        frame_display = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        # Update the frame in the Streamlit app (live feed)
        frame_placeholder.image(frame_display, channels="RGB", use_column_width=True)

        # Update washer status display
        status_html = ""
        for washer, detected in washer_statuses.items():
            if detected:
                status_html += f"<div class='washer-status washer-detected'>{washer.capitalize()} washer detected</div>"
            else:
                status_html += f"<div class='washer-status washer-not-detected'>{washer.capitalize()} washer not detected</div>"
        status_placeholder.markdown(status_html, unsafe_allow_html=True)

        # Add a short delay to make the stream smoother
        time.sleep(0.03)

        # Stop loop after red washer is detected
        if washer_status is None:
            break

    # After the loop ends, ensure the final red washer message is displayed
    washer_statuses["red"] = True
    status_html = ""
    for washer, detected in washer_statuses.items():
        status_html += f"<div class='washer-status washer-detected'>{washer.capitalize()} washer detected</div>" if detected else \
                       f"<div class='washer-status washer-not-detected'>{washer.capitalize()} washer not detected</div>"
    status_placeholder.markdown(status_html, unsafe_allow_html=True)

    cap.release()

# Close all OpenCV windows
cv2.destroyAllWindows()
