# import cv2
# import pytesseract

# # If using Windows, specify the path to tesseract.exe
# # pytesseract.pytesseract.tesseract_cmd = r'C:\Program Files\Tesseract-OCR\tesseract.exe'

# def main():
#     # Open the camera
#     cap = cv2.VideoCapture(2)  # 0 for the default camera
    
#     if not cap.isOpened():
#         print("Error: Could not open the camera.")
#         return
    
#     print("Press 'q' to quit.")
    
#     while True:
#         ret, frame = cap.read()
#         if not ret:
#             print("Failed to capture frame")
#             break
        
#         # Convert to grayscale for better OCR performance
#         gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        
#         # Perform text detection using Tesseract
#         text = pytesseract.image_to_string(gray_frame)
        
#         # Display the text on the frame
#         cv2.putText(frame, "Text Detected" if text.strip() else "No Text", 
#                     (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
        
#         # Show the camera feed
#         cv2.imshow('Camera', frame)
        
#         # Print the text detected (optional)
#         if text.strip():
#             print("Detected text:", text.strip())
        
#         # Press 'q' to quit
#         if cv2.waitKey(1) & 0xFF == ord('q'):
#             break
    
#     # Release the camera and close the windows
#     cap.release()
#     cv2.destroyAllWindows()

# if __name__ == "__main__":
#     main()





import cv2
import pytesseract
import time
import re

def main():
    # Open the camera
    cap = cv2.VideoCapture(2)  # 0 for the default camera
    
    if not cap.isOpened():
        print("Error: Could not open the camera.")
        return
    
    print("Press 'q' to quit.")
    last_update_time = time.time()  # Track the last frame update time
    frame = None  # Initialize the frame variable
    temp = []  # Initialize the list to store detected numbers
    
    while True:
        # Capture the frame only if 1 second has passed
        current_time = time.time()
        if current_time - last_update_time >= 0.1:
            ret, frame = cap.read()
            if not ret:
                print("Failed to capture frame")
                break
            
            # Convert to grayscale for better OCR performance
            gray_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
            
            # Perform text detection using Tesseract
            text = pytesseract.image_to_string(gray_frame).strip()
            
            # Extract numbers from the text using regex
            numbers = re.findall(r'\d+', text)  # Find all numeric substrings
            numbers = list(map(int, numbers))  # Convert to integers
            
            # Sort numbers and update temp if there is a change
            sorted_numbers = sorted(numbers)
            if sorted_numbers != temp:  # Only update if the numbers have changed
                temp = sorted_numbers
                print(f"Updated list: {temp}")
            
            # Update the display time
            last_update_time = current_time
        
        # Show the last captured frame if available
        if frame is not None:
            display_text = "Numbers Detected" if temp else "No Numbers"
            cv2.putText(frame, display_text, (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)
            cv2.imshow('Camera', frame)
        
        # Press 'q' to quit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
    
    # Release the camera and close the windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
