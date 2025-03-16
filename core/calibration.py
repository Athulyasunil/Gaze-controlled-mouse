import cv2
import numpy as np
import time
import pyautogui

def calibrate_gaze(cap, face_mesh):
    """
    Performs a 9-point calibration for gaze tracking.
    Points are at center, four corners, and four middle edges of the screen.
    
    Args:
        cap: OpenCV video capture object
        face_mesh: MediaPipe face mesh object
    
    Returns:
        dict: Calibration data mapping eye positions to screen positions
    """
    screen_w, screen_h = pyautogui.size()
    
    # Calibration points (9-point calibration for better accuracy)
    calibration_points = [
        (screen_w // 2, screen_h // 2),            # Center
        (screen_w // 10, screen_h // 10),          # Top left
        (screen_w // 10, screen_h * 9 // 10),      # Bottom left
        (screen_w * 9 // 10, screen_h // 10),      # Top right
        (screen_w * 9 // 10, screen_h * 9 // 10),  # Bottom right
        (screen_w // 2, screen_h // 10),           # Top middle
        (screen_w // 2, screen_h * 9 // 10),       # Bottom middle
        (screen_w // 10, screen_h // 2),           # Left middle
        (screen_w * 9 // 10, screen_h // 2)        # Right middle
    ]
    
    # Lists to store calibration data
    eye_positions = []
    screen_positions = []
    
    # Create a full-screen window for calibration
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Instructions screen first
    instruction_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.putText(instruction_frame, "Gaze Calibration", (screen_w//2 - 150, screen_h//2 - 100),
                cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
    cv2.putText(instruction_frame, "Follow the dots with your eyes and focus on each one.", 
                (screen_w//2 - 300, screen_h//2 - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(instruction_frame, "Press SPACE when your eyes are focused on the dot.", 
                (screen_w//2 - 300, screen_h//2), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(instruction_frame, "Keep your head still during the calibration.", 
                (screen_w//2 - 300, screen_h//2 + 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    cv2.putText(instruction_frame, "Press SPACE to begin...", 
                (screen_w//2 - 150, screen_h//2 + 100), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
    
    cv2.imshow("Calibration", instruction_frame)
    cv2.waitKey(0)  # Wait for space key
    
    # Helper function to prepare for capturing eye position
    def prepare_for_capture(point, index, total):
        # Create a countdown effect
        for countdown in range(3, 0, -1):
            cal_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            # Draw target point 
            cv2.circle(cal_frame, point, 20, (0, 165, 255), -1)  # Outer circle
            cv2.circle(cal_frame, point, 10, (0, 0, 255), -1)    # Inner circle
            
            # Show countdown
            cv2.putText(cal_frame, f"Look at the circle ({index+1}/{total})", 
                       (screen_w//2 - 150, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(cal_frame, f"Ready in {countdown}...", 
                       (screen_w//2 - 100, 100), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", cal_frame)
            cv2.waitKey(1000)  # 1 second delay
        
        # Final frame before capturing
        cal_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(cal_frame, point, 20, (0, 255, 0), -1)  # Green when ready
        cv2.circle(cal_frame, point, 10, (255, 255, 255), -1)
        cv2.putText(cal_frame, "Focus on the circle!", (screen_w//2 - 150, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Calibration", cal_frame)
        time.sleep(0.5)  # Short pause before capture
    
    for i, point in enumerate(calibration_points):
        prepare_for_capture(point, i, len(calibration_points))
        
        # Capture multiple frames and use the average eye position
        eye_positions_for_point = []
        capture_frames = 15  # Capture more frames for better accuracy
        
        # Frame for capture indication
        capture_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        cv2.circle(capture_frame, point, 20, (0, 255, 0), -1)
        cv2.circle(capture_frame, point, 10, (255, 255, 255), -1)
        cv2.putText(capture_frame, "Capturing...", (screen_w//2 - 100, 50),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.imshow("Calibration", capture_frame)
        
        # Actually capture eye positions
        for _ in range(capture_frames):
            ret, frame = cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb_frame)
            
            if results.multi_face_landmarks:
                from core.utils import get_eye_landmarks
                face_landmarks = results.multi_face_landmarks[0]
                left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape)
                
                # Average position of both eyes
                avg_x = (left_eye[0] + right_eye[0]) / 2
                avg_y = (left_eye[1] + right_eye[1]) / 2
                
                eye_positions_for_point.append((avg_x, avg_y))
                time.sleep(0.1)  # Small delay between captures
        
        # Calculate average eye position for this calibration point
        if eye_positions_for_point:
            # Remove outliers (simple method - just discard highest and lowest values)
            if len(eye_positions_for_point) > 4:
                x_values = [p[0] for p in eye_positions_for_point]
                y_values = [p[1] for p in eye_positions_for_point]
                x_values.sort()
                y_values.sort()
                # Discard top and bottom 10%
                discard_count = max(1, len(eye_positions_for_point) // 10)
                x_values = x_values[discard_count:-discard_count]
                y_values = y_values[discard_count:-discard_count]
                avg_x = sum(x_values) / len(x_values)
                avg_y = sum(y_values) / len(y_values)
            else:
                avg_x = sum(p[0] for p in eye_positions_for_point) / len(eye_positions_for_point)
                avg_y = sum(p[1] for p in eye_positions_for_point) / len(eye_positions_for_point)
            
            eye_positions.append((avg_x, avg_y))
            screen_positions.append(point)
            
            # Show feedback
            feedback_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
            cv2.circle(feedback_frame, point, 20, (0, 255, 0), -1)  # Target point
            
            # Draw calibration progress
            progress_bar_width = int((i+1) / len(calibration_points) * (screen_w - 200))
            cv2.rectangle(feedback_frame, (100, screen_h - 50), 
                         (100 + progress_bar_width, screen_h - 30), (0, 255, 0), -1)
            cv2.rectangle(feedback_frame, (100, screen_h - 50), 
                         (screen_w - 100, screen_h - 30), (255, 255, 255), 2)
            
            cv2.putText(feedback_frame, f"Calibration Progress: {i+1}/{len(calibration_points)}", 
                       (100, screen_h - 70), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
            
            cv2.imshow("Calibration", feedback_frame)
            cv2.waitKey(500)  # Show feedback for 0.5 seconds
    
    # Final calibration complete message
    final_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
    cv2.putText(final_frame, "Calibration Complete!", (screen_w//2 - 200, screen_h//2 - 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
    cv2.putText(final_frame, "Press any key to continue...", (screen_w//2 - 180, screen_h//2 + 50),
               cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.imshow("Calibration", final_frame)
    cv2.waitKey(1000)
    
    cv2.destroyWindow("Calibration")
    
    # Create calibration data
    calibration_data = {
        "screen_points": screen_positions,
        "eye_points": eye_positions
    }
    
    # Debug output for calibration points
    print(f"Calibration completed with {len(eye_positions)} points")
    
    return calibration_data