import cv2
from core.face_detector import detect_face
from core.gaze_tracker import track_gaze
from core.blink_detection import BlinkDetector

def main():
    print("Starting Face & Gaze Tracking with Blink Detection...")
    
    # Initialize modules
    face_detector = detect_face()
    gaze_tracker = track_gaze()
    blink_detector = BlinkDetector()  # Initialize the blink detector
    
    # Open webcam
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("Error: Could not open webcam.")
        return
    
    # Set capture properties
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
    
    # Run calibration
    print("Starting calibration process...")
    calibration_success = gaze_tracker.calibrate(cap)
    
    if not calibration_success:
        print("Calibration failed or was cancelled. Exiting.")
        cap.release()
        cv2.destroyAllWindows()
        return
    
    print("Calibration complete. Starting tracking...")
    print("Press 'q' to quit, 'c' to recalibrate")
    
    # Main loop
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            print("Failed to grab frame. Exiting...")
            break
        
        # Process frame with face detection
        processed_frame, face_results = face_detector.detect(frame)
        
        # Detect blinks
        blink_result = blink_detector.detect_blink(face_results, processed_frame)
        
        # Process with gaze tracker using processed_frame directly
        final_frame = gaze_tracker.process_frame(processed_frame)
        
       # In your main loop, add this line before displaying the frame:
        cv2.namedWindow("Eye & Gaze Tracking", cv2.WINDOW_NORMAL)
        cv2.setWindowProperty("Eye & Gaze Tracking", cv2.WND_PROP_TOPMOST, 1)

        # Then use your existing code to show the frame
        cv2.imshow("Eye & Gaze Tracking", final_frame) # Display the resulting frame
        
        # Handle key presses
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q'):
            break
        elif key == ord('c'):
            print("Recalibrating...")
            gaze_tracker.calibrate(cap)
    
    # Clean up
    cap.release()
    cv2.destroyAllWindows()
    print("Tracking stopped.")

if __name__ == "__main__":
    main()