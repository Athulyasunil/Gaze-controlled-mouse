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
    
    # Extract just the (x,y) coordinates from eye_points, ignoring head pose data
    eye_points = np.array([p[:2] for p in calibration_data["eye_points"]])
    screen_points = np.array(calibration_data["screen_points"])
    
    # Cache for last calculated positions - used for temporal smoothing
    if not hasattr(map_gaze_to_screen, 'last_positions'):
        map_gaze_to_screen.last_positions = []
    
    # Find closest calibration points
    try:
        # Calculate all distances from current gaze to calibration points
        distances = np.sqrt(np.sum((eye_points - gaze_pos.reshape(1, 2))**2, axis=1))
        
        # Get indices of nearest points (adaptive number based on distance)
        # Closer points will use fewer neighbors, distant points will use more
        max_neighbors = min(4, len(eye_points))
        
        # Distance threshold - if we're very close to a calibration point, use fewer neighbors
        dist_threshold = 0.05  # normalized distance
        min_neighbors = 1
        
        if np.min(distances) < dist_threshold:
            num_points = min_neighbors
        else:
            num_points = max_neighbors
        
        if num_points == 0:
            return (screen_w // 2, screen_h // 2)
            
        nearest_indices = np.argsort(distances)[:num_points]
        
        # Calculate weights based on distances with adaptive weighting
        # Points that are very close get higher weights than those farther away
        # Use squared inverse distance for more aggressive weighting
        weights = 1.0 / (distances[nearest_indices]**2 + 1e-6)
        weights = weights / np.sum(weights)
        
        # Weighted average of corresponding screen points
        screen_x = np.sum(weights * np.array([screen_points[i][0] for i in nearest_indices]))
        screen_y = np.sum(weights * np.array([screen_points[i][1] for i in nearest_indices]))
        
        # Apply adaptive center bias - less pull when confident
        # More center pull when fewer calibration points or uncertain
        confidence = 1.0 - min(1.0, np.min(distances) * 5)  # Higher confidence when closer to calibration points
        center_pull = 0.03 + (0.1 * (1 - confidence))  # Adaptive center pull based on confidence
        
        screen_x = screen_x * (1 - center_pull) + (screen_w / 2) * center_pull
        screen_y = screen_y * (1 - center_pull) + (screen_h / 2) * center_pull
        
        # Apply non-linear amplification - more sensitive in center, less at edges
        # This helps with fine control in the center while still reaching edges
        cx, cy = screen_w / 2, screen_h / 2
        dx, dy = screen_x - cx, screen_y - cy
        
        # Non-linear amplification function (cubic)
        # Adjust these parameters for sensitivity
        amplification = 1.15  # Base amplification
        non_linearity = 0.2   # Higher values make center more sensitive than edges
        
        # Non-linear amplification that's more sensitive in center
        dx_norm = dx / cx  # Normalize to -1 to 1
        dy_norm = dy / cy  # Normalize to -1 to 1
        
        # Apply non-linear curve (cubic function)
        dx_nl = dx_norm * (1 - non_linearity * abs(dx_norm)**2)
        dy_nl = dy_norm * (1 - non_linearity * abs(dy_norm)**2)
        
        # Convert back and apply amplification
        screen_x = cx + dx_nl * cx * amplification
        screen_y = cy + dy_nl * cy * amplification
        
        # Add temporal smoothing (across multiple calls)
        # Store current raw position
        current_pos = (screen_x, screen_y)
        map_gaze_to_screen.last_positions.append(current_pos)
        
        # Keep a limited history
        max_history = 5
        if len(map_gaze_to_screen.last_positions) > max_history:
            map_gaze_to_screen.last_positions.pop(0)
        
        # Apply temporal smoothing with exponential decay
        # More weight to recent positions
        alpha = smoothing_factor  # Smoothing factor (0-1)
        
        # Calculate exponentially weighted positions
        if len(map_gaze_to_screen.last_positions) > 1:
            # Create weights with exponential decay
            weights = np.array([alpha * (1 - alpha)**(len(map_gaze_to_screen.last_positions) - i - 1) 
                               for i in range(len(map_gaze_to_screen.last_positions))])
            weights = weights / np.sum(weights)  # Normalize weights
            
            # Apply weighted average
            smoothed_x = sum(pos[0] * w for pos, w in zip(map_gaze_to_screen.last_positions, weights))
            smoothed_y = sum(pos[1] * w for pos, w in zip(map_gaze_to_screen.last_positions, weights))
            
            screen_x, screen_y = smoothed_x, smoothed_y
        
        # Make sure we don't go beyond screen boundaries
        screen_x = max(0, min(screen_w - 1, screen_x))
        screen_y = max(0, min(screen_h - 1, screen_y))
        
        return (int(screen_x), int(screen_y))
    
    except Exception as e:
        print(f"Error in mapping gaze to screen: {e}")
        return (screen_w // 2, screen_h // 2)