import streamlit as st
import cv2
import pytesseract
import re
from PIL import Image
import numpy as np

# Streamlit app configuration
st.title("Live Camera Feed - Number Detection")
st.text("This app detects numbers from the camera feed and displays them in ascending order.")

# Initialize placeholders for Streamlit UI
frame_placeholder = st.empty()
numbers_placeholder = st.empty()

# Initialize camera
cap = cv2.VideoCapture(2)  # 0 for the default camera


if not cap.isOpened():
    st.error("Error: Could not access the camera.")
else:
    # Create a button to stop the camera feed (placed outside the loop)
    stop_button_pressed = st.button("Stop")

    # Set up a variable to store the detected numbers
    temp = []

    while not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame. Exiting...")
            break

        # Convert the frame to grayscale
        gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

        # Perform text detection using Tesseract
        text = pytesseract.image_to_string(gray_frame).strip()

        # Extract numbers from the text
        numbers = re.findall(r'\d+', text)  # Find all numeric substrings
        numbers = list(map(int, numbers))  # Convert to integers

        # Sort the numbers and update temp if changed
        sorted_numbers = sorted(numbers)
        if sorted_numbers != temp:
            temp = sorted_numbers
            numbers_placeholder.text(f"Detected Numbers (Ascending Order): {temp}")

        # Display the current frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_image, caption="Live Camera Feed", use_column_width=True)

        # Refresh the stop button's state
        stop_button_pressed = st.session_state.get("stop_button_pressed", False)

    cap.release()
    st.text("Camera feed stopped.")


################################################
################################################

# import streamlit as st
# import cv2
# import pytesseract
# from pytesseract import Output
# import numpy as np

# # Initialize app
# st.title("Number Detection from Camera Feed")

# # Input fields
# device_id = 2
# frame_width = st.sidebar.slider("Frame Width (w)", min_value=160, max_value=1280, value=640)
# frame_height = st.sidebar.slider("Frame Height (h)", min_value=120, max_value=960, value=480)

# # Placeholder for displaying the video and detected numbers
# frame_placeholder = st.empty()
# numbers_placeholder = st.empty()

# # Function to detect numbers in a frame
# def detect_numbers(frame):
#     # Convert to grayscale
#     gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
#     # Perform OCR
#     data = pytesseract.image_to_data(gray, output_type=Output.DICT)
#     numbers = []
#     for text in data['text']:
#         if text.isdigit():
#             numbers.append(int(text))
#     return sorted(set(numbers))

# # Start video capture
# cap = cv2.VideoCapture(device_id)
# cap.set(cv2.CAP_PROP_FRAME_WIDTH, frame_width)
# cap.set(cv2.CAP_PROP_FRAME_HEIGHT, frame_height)

# # Initialize variables
# detected_numbers = []

# # Add a stop button with a unique key
# stop = False
# stop_button = st.sidebar.button("Stop", key="stop_button")

# try:
#     while cap.isOpened() and not stop:
#         ret, frame = cap.read()
#         if not ret:
#             st.write("Failed to read from camera. Ensure device 2 is available.")
#             break

#         # Detect numbers
#         numbers_in_frame = detect_numbers(frame)
#         if numbers_in_frame:
#             detected_numbers = numbers_in_frame  # Overwrite with new numbers
        
#         # Display the video feed with cropped dimensions
#         # cropped_frame = frame[:frame_height, :frame_width]
#         # Calculate the center region coordinates
#         center_x, center_y = frame.shape[1] // 2, frame.shape[0] // 2
#         x1 = max(center_x - frame_width // 2, 0)
#         y1 = max(center_y - frame_height // 2, 0)
#         x2 = min(center_x + frame_width // 2, frame.shape[1])
#         y2 = min(center_y + frame_height // 2, frame.shape[0])

#         # Crop the central region
#         cropped_frame = frame[y1:y2, x1:x2]
#         cropped_frame_resized = cv2.resize(cropped_frame, (frame_width, frame_height))


#         cropped_frame_resized = cv2.resize(cropped_frame, (frame_width, frame_height))
#         frame_placeholder.image(
#             cropped_frame_resized,
#             channels="BGR",
#             caption="Cropped Camera Feed"
#         )

#         # Display detected numbers
#         numbers_placeholder.write(f"Detected Numbers (Ascending Order): {detected_numbers}")
        
#         # Check if the stop button was pressed
#         if stop_button:
#             stop = True

# except Exception as e:
#     st.error(f"An error occurred: {e}")

# finally:
#     cap.release()
#     cv2.destroyAllWindows()


# ################################################
# ################################################

