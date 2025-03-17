import cv2
import numpy as np
import time
from core.utils import get_eye_landmarks  # Assuming this is part of your codebase
import mediapipe as mp

mp_face_mesh = mp.solutions.face_mesh

def estimate_head_pose(image, face_landmarks, image_w, image_h):
    """
    Estimates head pose using OpenCV's solvePnP function.
    
    Args:
        image: Input frame (BGR format)
        face_landmarks: Face mesh landmarks
        image_w: Image width
        image_h: Image height
    
    Returns:
        rotation_vector, translation_vector: Head pose parameters
    """
    # Define 3D model points of key facial landmarks
    model_points = np.array([
        (0.0, 0.0, 0.0),        # Nose tip
        (0.0, -63.6, -12.5),     # Chin
        (-43.3, 32.7, -26.0),    # Left eye corner
        (43.3, 32.7, -26.0),     # Right eye corner
        (-28.9, -28.9, -24.1),   # Left mouth corner
        (28.9, -28.9, -24.1)     # Right mouth corner
    ], dtype=np.float32)

    # Get corresponding 2D points from face mesh landmarks
    image_points = np.array([
        (face_landmarks.landmark[1].x * image_w, face_landmarks.landmark[1].y * image_h),   # Nose tip
        (face_landmarks.landmark[152].x * image_w, face_landmarks.landmark[152].y * image_h), # Chin
        (face_landmarks.landmark[33].x * image_w, face_landmarks.landmark[33].y * image_h),  # Left eye corner
        (face_landmarks.landmark[263].x * image_w, face_landmarks.landmark[263].y * image_h), # Right eye corner
        (face_landmarks.landmark[61].x * image_w, face_landmarks.landmark[61].y * image_h),  # Left mouth corner
        (face_landmarks.landmark[291].x * image_w, face_landmarks.landmark[291].y * image_h) # Right mouth corner
    ], dtype=np.float32)

    # Camera matrix (assuming focal length is roughly equal to width of the frame)
    focal_length = image_w
    camera_matrix = np.array([
        [focal_length, 0, image_w / 2],
        [0, focal_length, image_h / 2],
        [0, 0, 1]
    ], dtype=np.float32)

    # Assume no lens distortion
    dist_coeffs = np.zeros((4, 1), dtype=np.float32)

    # Solve PnP to get rotation and translation vectors
    success, rotation_vector, translation_vector = cv2.solvePnP(
        model_points, image_points, camera_matrix, dist_coeffs, flags=cv2.SOLVEPNP_ITERATIVE)

    if success:
        return rotation_vector, translation_vector
    else:
        return None, None

def perform_calibration(cap, face_mesh, blink_detector, screen_w, screen_h):
    """
    Performs a 9-point calibration for gaze tracking with blink-based selection.
    
    Args:
        cap: OpenCV video capture object
        face_mesh: MediaPipe face mesh object
        blink_detector: BlinkDetector instance for blink-based selection
        screen_w: Screen width in pixels
        screen_h: Screen height in pixels
    
    Returns:
        dict: Calibration data mapping eye positions to screen positions
    """
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
    
    # Window for calibration
    cv2.namedWindow("Calibration", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Calibration", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    # Process each calibration point
    for i, point in enumerate(calibration_points):
        # Capture eye position for current calibration point
        eye_pos = capture_eye_position_with_blink(cap, face_mesh, blink_detector, point, i, len(calibration_points), screen_w, screen_h)
        
        if eye_pos:
            eye_positions.append(eye_pos)
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
            cv2.waitKey(1)
            time.sleep(0.5)  # Short pause between points
    
    cv2.destroyWindow("Calibration")
    
    # Create and return calibration data
    calibration_data = {
        "screen_points": screen_positions,
        "eye_points": eye_positions
    }
    
    print(f"Calibration completed with {len(eye_positions)} points")
    
    return calibration_data

def capture_eye_position_with_blink(cap, face_mesh, blink_detector, point, index, total, screen_w, screen_h):
    eye_positions_for_point = []
    head_poses_for_point = []
    blink_detected = False
    start_time = time.time()
    
    while not blink_detected and time.time() - start_time < 10:  # 10 sec timeout
        ret, frame = cap.read()
        if not ret:
            continue
        
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(rgb_frame)

        # Create calibration display frame
        calib_frame = np.zeros((screen_h, screen_w, 3), dtype=np.uint8)
        
        # Draw the target point
        cv2.circle(calib_frame, point, 20, (0, 0, 255), -1)  # Red dot to look at
        cv2.circle(calib_frame, point, 25, (255, 255, 255), 2)  # White outline
        
        # Show small webcam feed in corner
        webcam_display = cv2.resize(frame, (320, 240))
        calib_frame[20:20+240, 20:20+320] = webcam_display
        
        # Add instructions
        cv2.putText(calib_frame, f"Look at the red dot ({index+1}/{total})", 
                  (screen_w//2 - 200, 50), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        cv2.putText(calib_frame, "Double-blink to select", 
                  (screen_w//2 - 150, 90), 
                  cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)

        if results.multi_face_landmarks:
            face_landmarks = results.multi_face_landmarks[0]

            # Get eye landmarks
            left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape)
            avg_x = (left_eye[0] + right_eye[0]) / 2
            avg_y = (left_eye[1] + right_eye[1]) / 2
            eye_positions_for_point.append((avg_x, avg_y))

            # Get head pose
            rotation_vector, translation_vector = estimate_head_pose(frame, face_landmarks, frame.shape[1], frame.shape[0])
            if rotation_vector is not None:
                head_poses_for_point.append(rotation_vector.flatten())  # Store head rotation

            # Detect blink
            blink_result = blink_detector.detect_blink(results, frame)
            if blink_result['action_performed']:
                blink_detected = True
                cv2.putText(calib_frame, "Blink detected!", 
                          (screen_w//2 - 100, screen_h - 100), 
                          cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

            cv2.imshow("Calibration", calib_frame)
            cv2.waitKey(1)

    if eye_positions_for_point:
        avg_x = sum(p[0] for p in eye_positions_for_point) / len(eye_positions_for_point)
        avg_y = sum(p[1] for p in eye_positions_for_point) / len(eye_positions_for_point)

        # Average head pose if available
        if head_poses_for_point:
            avg_head_pose = np.mean(head_poses_for_point, axis=0).tolist()
        else:
            avg_head_pose = [0, 0, 0]  # Default to no rotation

        return (avg_x, avg_y, avg_head_pose)
    
    return None

def create_calibration_mapping(calibration_data):
    from sklearn.preprocessing import PolynomialFeatures
    from sklearn.linear_model import LinearRegression
    from sklearn.pipeline import make_pipeline

    eye_points = np.array([p[:2] for p in calibration_data["eye_points"]])  # Extract eye positions
    head_poses = np.array([p[2] for p in calibration_data["eye_points"]])  # Extract head poses
    screen_points = np.array(calibration_data["screen_points"])

    # Combine eye position and head pose as input features
    X = np.hstack((eye_points, head_poses))

    # Train separate models for X and Y coordinates
    model_x = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())
    model_y = make_pipeline(PolynomialFeatures(degree=2), LinearRegression())

    model_x.fit(X, screen_points[:, 0])
    model_y.fit(X, screen_points[:, 1])

    def map_gaze_to_screen(eye_pos, head_pose):
        """Maps eye and head pose to screen coordinates."""
        if not eye_pos or not head_pose:
            return None

        input_features = np.array([eye_pos + head_pose])
        screen_x = model_x.predict(input_features)[0]
        screen_y = model_y.predict(input_features)[0]

        return (int(screen_x), int(screen_y))

    return map_gaze_to_screen
