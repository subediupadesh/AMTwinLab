# import cv2
# import numpy as np
# import easyocr
# import time

# def preprocess_image(image):
#     """Preprocess the image to enhance digit detection"""
#     gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
#     _, threshold = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)
#     blurred = cv2.GaussianBlur(threshold, (3, 3), 0)
#     return blurred

# def detect_digits():
#     # Initialize camera
#     cap = cv2.VideoCapture(0)
#     cv2.ocl.setUseOpenCL(False)
    
#     # Initialize EasyOCR
#     reader = easyocr.Reader(['en'], gpu=False, model_storage_directory='.')
    
#     if not cap.isOpened():
#         print("Error: Could not open camera")
#         return
    
#     # Set lower resolution
#     cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
#     cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
#     last_process_time = time.time()
#     processed_frame = None
    
#     # Initialize the time_dt list
#     time_dt = []
    
#     print("Starting digit detection. Press 'q' to quit.")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Error: Could not read frame")
#             break
            
#         current_time = time.time()
        
#         # Process frame every 0.5 seconds
#         if current_time - last_process_time >= 0.5:
#             processed_frame = frame.copy()
#             preprocessed = preprocess_image(processed_frame)
            
#             try:
#                 # Detect text (focusing on digits)
#                 results = reader.readtext(
#                     preprocessed,
#                     allowlist='0123456789',
#                     # allowlist='0123456789thequickbrownfoxjumpsoverthelazydogTHEQUICKBROWNFOXJUMPSOVERTHELAZYDOG',
#                     paragraph=False,
#                     height_ths=2.0,
#                     width_ths=2.0,
#                     contrast_ths=0.1
#                 )
                
#                 # Clear the existing list if new digits are detected
#                 new_digits = []
                
#                 # Process detected digits
#                 for (bbox, text, prob) in results:
#                     if prob > 0.5:  # Confidence threshold
#                         # Convert bbox points to integers
#                         (tl, tr, br, bl) = bbox
#                         tl = tuple(map(int, tl))
#                         br = tuple(map(int, br))
                        
#                         # Add detected digit to new_digits list
#                         try:
#                             digit = int(text)
#                             new_digits.append(digit)
#                         except ValueError:
#                             continue
                        
#                         # Draw rectangle and text
#                         cv2.rectangle(processed_frame, tl, br, (0, 255, 0), 2)
#                         cv2.putText(processed_frame, 
#                                 #   f"Digit: {text} ({prob:.2f})", 
#                                 f"{text}", 
#                                   (tl[0], tl[1] - 10),
#                                   cv2.FONT_HERSHEY_SIMPLEX, 
#                                   0.7, 
#                                   (0, 255, 0), 
#                                   2)
                
#                 # Update time_dt list if new digits are detected
#                 if new_digits:
#                     time_dt = sorted(new_digits)
#                     print(f"Updated time_dt list: {time_dt}")
                
#                 # Display current time_dt list on frame
#                 # cv2.putText(processed_frame,
#                 #            f"Current List: {time_dt}",
#                 #            (10, 30),
#                 #            cv2.FONT_HERSHEY_SIMPLEX,
#                 #            0.7,
#                 #            (0, 255, 0),
#                 #            2)
                
#                 # # Draw processing time
#                 # process_time = time.time() - current_time
#                 # cv2.putText(processed_frame,
#                 #            f"Process Time: {process_time:.2f}s",
#                 #            (10, 60),
#                 #            cv2.FONT_HERSHEY_SIMPLEX,
#                 #            0.7,
#                 #            (0, 255, 0),
#                 #            2)
                
#             except Exception as e:
#                 print(f"Detection Error: {str(e)}")
            
#             last_process_time = current_time
        
#         # Display the result
#         if processed_frame is not None:
#             cv2.imshow('Digit Detection with List (CPU)', processed_frame)
#         else:
#             cv2.imshow('Digit Detection with List (CPU)', frame)
        
#         # Break the loop if 'q' is pressed
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Cleanup
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     detect_digits()


############################################
############################################


# import streamlit as st
# import cv2
# import easyocr
# import re
# from PIL import Image

# # Streamlit app configuration
# st.title("Live Camera Feed - Number Detection")
# st.text("This app detects numbers from the camera feed and displays them in ascending order.")

# # Initialize placeholders for Streamlit UI
# frame_placeholder = st.empty()
# numbers_placeholder = st.empty()

# # Initialize camera
# cap = cv2.VideoCapture(2)  # 0 for the default camera

# if not cap.isOpened():
#     st.error("Error: Could not access the camera.")
# else:
#     # Create a button to stop the camera feed (placed outside the loop)
#     stop_button_pressed = st.button("Stop")

#     # Set up a variable to store the detected numbers
#     temp = []

#     # Initialize EasyOCR Reader (only once)
#     reader = easyocr.Reader(['en'])  # Use English for OCR

#     # Frame counter for frequency optimization (process every 10th frame)
#     frame_count = 0

#     while not stop_button_pressed:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture frame. Exiting...")
#             break

#         frame_count += 1

#         # Resize the frame to reduce the processing time (e.g., 640x480)
#         frame_resized = cv2.resize(frame, (640, 480))

#         # Convert to grayscale for easier OCR processing
#         gray_frame = cv2.cvtColor(frame_resized, cv2.COLOR_BGR2GRAY)

#         # Process OCR only every 10th frame to reduce the workload
#         if frame_count % 30 == 0:
#             # Use EasyOCR to extract text from the resized and grayscaled frame
#             result = reader.readtext(gray_frame)

#             # Extract numbers from the detected text
#             numbers = []
#             for detection in result:
#                 text = detection[1]  # The detected text
#                 numbers.extend(re.findall(r'\d+', text))  # Find all numeric substrings

#             numbers = list(map(int, numbers))  # Convert to integers

#             # Sort the numbers and update temp if changed
#             sorted_numbers = sorted(numbers)
#             if sorted_numbers != temp:
#                 temp = sorted_numbers
#                 numbers_placeholder.text(f"Detected Numbers (Ascending Order): {temp}")

#         # Display the current frame
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_image = Image.fromarray(frame_rgb)
#         frame_placeholder.image(frame_image, caption="Live Camera Feed", use_column_width=True)

#         # Refresh the stop button's state
#         stop_button_pressed = st.session_state.get("stop_button_pressed", False)

#     cap.release()
#     st.text("Camera feed stopped.")


############################################
############################################


import streamlit as st
import cv2
import easyocr
import re
from PIL import Image

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

    # Initialize EasyOCR Reader with a smaller model (English only, lightweight model)
    reader = easyocr.Reader(['en'], model_storage_directory="models", gpu=False)  # Use smaller model by default

    # Frame counter for frequency optimization (process every 10th frame)
    frame_count = 0
    previous_text = None  # Track previous OCR output for comparison

    while not stop_button_pressed:
        ret, frame = cap.read()
        if not ret:
            st.error("Failed to capture frame. Exiting...")
            break

        frame_count += 1

        # Region of Interest (ROI) - Crop the frame to a smaller area (e.g., bottom-right corner)
        x1, y1, x2, y2 = 100, 100, 600, 400  # Adjust these values based on your area of interest
        roi = frame[y1:y2, x1:x2]

        # Compress the image (reduce quality) to speed up processing
        encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 30]  # Compress to 30% quality
        _, compressed_frame = cv2.imencode('.jpg', roi, encode_param)
        decompressed_frame = cv2.imdecode(compressed_frame, 1)

        # Convert to grayscale for easier OCR processing
        gray_frame = cv2.cvtColor(decompressed_frame, cv2.COLOR_BGR2GRAY)

        # Process OCR only every 10th frame to reduce the workload
        if frame_count % 30 == 0:
            # Use EasyOCR to extract text from the region of interest (ROI)
            result = reader.readtext(gray_frame)

            # Extract numbers from the detected text
            numbers = []
            for detection in result:
                text = detection[1]
                numbers.extend(re.findall(r'\d+', text))

            numbers = list(map(int, numbers))  # Convert to integers

            # Compare current text with previous text to skip unchanged frames
            current_text = " ".join(map(str, numbers))
            if current_text != previous_text:
                sorted_numbers = sorted(numbers)
                numbers_placeholder.text(f"Detected Numbers (Ascending Order): {sorted_numbers}")
                previous_text = current_text  # Update the stored text for comparison

        # Display the current frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        frame_image = Image.fromarray(frame_rgb)
        frame_placeholder.image(frame_image, caption="Live Camera Feed", use_column_width=True)

        # Refresh the stop button's state
        stop_button_pressed = st.session_state.get("stop_button_pressed", False)

    cap.release()
    st.text("Camera feed stopped.")


############################################
############################################


# import streamlit as st
# import cv2
# import easyocr
# import re
# from PIL import Image

# # Streamlit app configuration
# st.title("Live Camera Feed - Number Detection")
# st.text("This app detects numbers from the camera feed and displays them in ascending order.")

# # Initialize placeholders for Streamlit UI
# frame_placeholder = st.empty()
# numbers_placeholder = st.empty()

# # Initialize camera
# cap = cv2.VideoCapture(2)  # 0 for the default camera

# if not cap.isOpened():
#     st.error("Error: Could not access the camera.")
# else:
#     # Create a button to stop the camera feed (placed outside the loop)
#     stop_button_pressed = st.button("Stop")

#     # Set up a variable to store the detected numbers
#     temp = []

#     # Initialize EasyOCR Reader with a smaller model (English only, lightweight model)
#     reader = easyocr.Reader(['en'], model_storage_directory="models", gpu=False)  # Use smaller model by default

#     # Frame counter for frequency optimization (process every 10th frame)
#     frame_count = 0
#     previous_text = None  # Track previous OCR output for comparison

#     while not stop_button_pressed:
#         ret, frame = cap.read()
#         if not ret:
#             st.error("Failed to capture frame. Exiting...")
#             break

#         frame_count += 1

#         # Compress the image (reduce quality) to speed up processing
#         encode_param = [int(cv2.IMWRITE_JPEG_QUALITY), 50]  # Compress to 30% quality
#         _, compressed_frame = cv2.imencode('.jpg', frame, encode_param)
#         decompressed_frame = cv2.imdecode(compressed_frame, 1)

#         # Convert to grayscale for easier OCR processing
#         gray_frame = cv2.cvtColor(decompressed_frame, cv2.COLOR_BGR2GRAY)

#         # Process OCR only every 10th frame to reduce the workload
#         if frame_count % 30 == 0:
#             # Use EasyOCR to extract text from the entire frame
#             result = reader.readtext(gray_frame)

#             # Extract numbers from the detected text and draw bounding boxes
#             numbers = []
#             for detection in result:
#                 bbox, text, _ = detection
#                 # Check if the detected text contains a number
#                 digits = re.findall(r'\d+', text)
#                 for digit in digits:
#                     numbers.append(int(digit))

#                     # Draw a green bounding box around the detected digit
#                     (top_left, top_right, bottom_right, bottom_left) = bbox
#                     top_left = tuple(map(int, top_left))
#                     bottom_right = tuple(map(int, bottom_right))
#                     cv2.rectangle(frame, top_left, bottom_right, (0, 255, 0), 2)

#                     # Put the digit text just outside the green box
#                     x = top_left[0]
#                     y = top_left[1] - 10  # Slightly above the top-left corner
#                     cv2.putText(frame, digit, (x, y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2)

#             # Sort the numbers and update if changed
#             numbers = sorted(numbers)
#             current_text = " ".join(map(str, numbers))
#             if current_text != previous_text:
#                 numbers_placeholder.text(f"Detected Numbers (Ascending Order): {numbers}")
#                 previous_text = current_text

#         # Display the current frame with bounding boxes
#         frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#         frame_image = Image.fromarray(frame_rgb)
#         frame_placeholder.image(frame_image, caption="Live Camera Feed", use_column_width=True)

#         # Refresh the stop button's state
#         stop_button_pressed = st.session_state.get("stop_button_pressed", False)

#     cap.release()
#     st.text("Camera feed stopped.")
