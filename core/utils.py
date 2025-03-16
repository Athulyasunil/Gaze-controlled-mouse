import numpy as np

def get_eye_landmarks(face_landmarks, frame_shape):
    """
    Extract normalized eye position coordinates from face landmarks.
    
    Args:
        face_landmarks: MediaPipe face landmarks
        frame_shape: Shape of the video frame
    
    Returns:
        tuple: (left_eye_pos, right_eye_pos) as normalized coordinates
    """
    h, w = frame_shape[:2]
    
    # Left iris landmarks (468, 469, 470, 471)
    left_iris_landmarks = [face_landmarks.landmark[idx] for idx in [468, 469, 470, 471]]
    left_iris_x = sum(lm.x for lm in left_iris_landmarks) / len(left_iris_landmarks)
    left_iris_y = sum(lm.y for lm in left_iris_landmarks) / len(left_iris_landmarks)
    
    # Right iris landmarks (473, 474, 475, 476)
    right_iris_landmarks = [face_landmarks.landmark[idx] for idx in [473, 474, 475, 476]]
    right_iris_x = sum(lm.x for lm in right_iris_landmarks) / len(right_iris_landmarks)
    right_iris_y = sum(lm.y for lm in right_iris_landmarks) / len(right_iris_landmarks)
    
    # Return normalized coordinates
    left_eye_pos = (left_iris_x, left_iris_y)
    right_eye_pos = (right_iris_x, right_iris_y)
    
    return left_eye_pos, right_eye_pos

def map_gaze_to_screen(gaze_pos, calibration_data, smoothing_factor=0.3):
    """
    Maps the gaze position to screen coordinates using calibration data.
    Implements polynomial mapping for more accurate cursor control.
    
    Args:
        gaze_pos: Tuple of (x, y) normalized gaze position
        calibration_data: Dict containing calibration data
        smoothing_factor: Factor for smoothing cursor movement (0-1)
    
    Returns:
        tuple: (x, y) screen coordinates
    """
    # Check if calibration data is valid
    if not calibration_data or "eye_points" not in calibration_data or "screen_points" not in calibration_data:
        # Return a default position (screen center) if no valid calibration data
        import pyautogui
        screen_w, screen_h = pyautogui.size()
        return (screen_w // 2, screen_h // 2)
    
    # Check if calibration points exist
    if len(calibration_data["eye_points"]) == 0 or len(calibration_data["screen_points"]) == 0:
        import pyautogui
        screen_w, screen_h = pyautogui.size()
        return (screen_w // 2, screen_h // 2)
    
    # Import screen size for scaling
    import pyautogui
    screen_w, screen_h = pyautogui.size()
    
    # Convert gaze_pos to numpy array for calculations
    gaze_pos = np.array(gaze_pos)
    eye_points = np.array(calibration_data["eye_points"])
    screen_points = np.array(calibration_data["screen_points"])
    
    # Find closest calibration points
    try:
        distances = np.sqrt(np.sum((eye_points - gaze_pos.reshape(1, 2))**2, axis=1))
        
        # Get indices of nearest points (at least 1, up to 3)
        num_points = min(3, len(eye_points))
        if num_points == 0:
            return (screen_w // 2, screen_h // 2)
            
        nearest_indices = np.argsort(distances)[:num_points]
        
        # Calculate weights based on distances (inverse distance weighting)
        weights = 1.0 / (distances[nearest_indices] + 1e-6)
        weights = weights / np.sum(weights)
        
        # Weighted average of corresponding screen points
        screen_x = np.sum(weights * np.array([screen_points[i][0] for i in nearest_indices]))
        screen_y = np.sum(weights * np.array([screen_points[i][1] for i in nearest_indices]))
        
        # Apply minimal center bias - less pull toward center
        center_pull = 0.05  # Reduced from 0.15 to allow more edge movement
        screen_x = screen_x * (1 - center_pull) + (screen_w / 2) * center_pull
        screen_y = screen_y * (1 - center_pull) + (screen_h / 2) * center_pull
        
        # Apply amplification to reach screen edges
        # This expands the range of movement to reach corners
        amplification = 1.2  # Values > 1 expand the range
        screen_x = ((screen_x / screen_w) - 0.5) * amplification * screen_w + (screen_w / 2)
        screen_y = ((screen_y / screen_h) - 0.5) * amplification * screen_h + (screen_h / 2)
        
        # Make sure we don't go beyond screen boundaries
        screen_x = max(0, min(screen_w - 1, screen_x))
        screen_y = max(0, min(screen_h - 1, screen_y))
        
        return (int(screen_x), int(screen_y))
    
    except Exception as e:
        print(f"Error in mapping gaze to screen: {e}")
        return (screen_w // 2, screen_h // 2)