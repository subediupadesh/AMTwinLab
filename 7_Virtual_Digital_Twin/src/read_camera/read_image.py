import cv2

def list_cameras():
    """Detect available cameras by trying to open them."""
    available_cameras = []
    for i in range(10):  # Test indices 0-9
        cap = cv2.VideoCapture(i)
        if cap.isOpened():
            available_cameras.append(i)
            cap.release()
    return available_cameras

def open_camera(camera_index):
    """Open the camera by index."""
    cap = cv2.VideoCapture(camera_index)
    if not cap.isOpened():
        print(f"Error: Unable to access camera {camera_index}.")
        return None
    return cap

def main():
    print("Detecting cameras...")
    cameras = list_cameras()
    
    if not cameras:
        print("No cameras detected!")
        return

    print(f"Available cameras: {cameras}")
    current_camera_index = cameras[0]
    cap = open_camera(current_camera_index)
    if not cap:
        return

    print("Press 'n' to switch to the next camera.")
    print("Press 'q' to quit the camera window.")

    while True:
        ret, frame = cap.read()
        if not ret:
            print(f"Error: Unable to read from camera {current_camera_index}.")
            break

        # Display the frame
        cv2.imshow('Camera Feed', frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord('q'):  # Quit
            break
        elif key == ord('n'):  # Switch to the next camera
            cap.release()
            current_camera_index = cameras[(cameras.index(current_camera_index) + 1) % len(cameras)]
            print(f"Switching to camera {current_camera_index}...")
            cap = open_camera(current_camera_index)
            if not cap:
                break

    # Release the camera and close the window
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()


##############################################
##############################################

### For Raspberry Pi

# import cv2

# # Function to set up the camera based on user choice
# def setup_camera(choice):
#     if choice == "0":
#         pipeline = "libcamerasrc ! videoconvert ! appsink"
#         cap = cv2.VideoCapture(pipeline, cv2.CAP_GSTREAMER)
#     elif choice == "1":
#         cap = cv2.VideoCapture(2)  # Use device 2 for the USB webcam
#     else:
#         print("Invalid choice. Exiting.")
#         exit()
#     return cap

# # Initial choice
# choice = input("Select camera:\n0 - Internal Camera (Raspberry Pi module)\n1 - USB Camera\n")

# cap = setup_camera(choice)

# while True:
#     _, frame = cap.read()
#     cv2.imshow("Camera", frame)

#     key = cv2.waitKey(1) & 0xFF
    
#     if key == 27:  # ESC key to exit
#         break
#     elif key == ord('n'):  # Press 'n' to switch camera
#         cap.release()  # Release the current camera
#         if choice == "0":
#             choice = "1"  # Switch to USB camera
#         else:
#             choice = "0"  # Switch to Raspberry Pi internal camera
#         cap = setup_camera(choice)  # Set up the new camera

# cap.release()
# cv2.destroyAllWindows()
