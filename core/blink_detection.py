import time
import pyautogui
import numpy as np
import cv2

class BlinkDetector:
    def __init__(self):
        # MediaPipe landmark indices for eyes
        # Left eye landmarks
        self.LEFT_EYE = [362, 382, 381, 380, 374, 373, 390, 249, 263, 466, 388, 387, 386, 385]
        # Right eye landmarks
        self.RIGHT_EYE = [33, 7, 163, 144, 145, 153, 154, 155, 133, 173, 157, 158, 159, 160, 161, 246]
        
        # MediaPipe landmark indices specifically for calculating EAR
        # Left eye points for EAR calculation (upper & lower points, left & right corner points)
        self.LEFT_EYE_EAR_POINTS = [386, 374, 362, 263]
        # Right eye points for EAR calculation
        self.RIGHT_EYE_EAR_POINTS = [159, 145, 33, 133]
        
        # Blink detection parameters - adjusted for better detection
        self.BLINK_THRESHOLD = 0.18  # Lowered threshold for more sensitive detection
        self.BLINK_TIME = 0.2  # Increased to allow for slower blinks
        self.BLINK_SEQUENCE_TIME = 0.9  # Increased time window for capturing sequences
        
        # State tracking
        self.blink_timestamps = []
        self.last_blink_state = False
        self.current_blink_state = False
        self.last_action_time = 0
        
        print("MediaPipe Blink detector initialized with improved parameters")
        
    def get_landmark_coords(self, landmarks, img_shape):
        """Convert normalized MediaPipe landmarks to pixel coordinates."""
        h, w = img_shape[:2]
        return [(int(lm.x * w), int(lm.y * h)) for lm in landmarks.landmark]
    
    def eye_aspect_ratio(self, landmarks, eye_points):
        """Calculate EAR using MediaPipe landmarks."""
        # Extract points (vertical and horizontal points for EAR measurement)
        top = landmarks[eye_points[0]]    # Upper eyelid
        bottom = landmarks[eye_points[1]]  # Lower eyelid
        left = landmarks[eye_points[2]]    # Left corner
        right = landmarks[eye_points[3]]   # Right corner
        
        # Calculate vertical distance (height of eye)
        vertical = np.sqrt((top[0] - bottom[0])**2 + (top[1] - bottom[1])**2)
        
        # Calculate horizontal distance (width of eye)
        horizontal = np.sqrt((left[0] - right[0])**2 + (left[1] - right[1])**2)
        
        # Calculate EAR
        if horizontal == 0:  # Avoid division by zero
            return 0
        
        return vertical / horizontal
    
    def detect_blink(self, results, frame):
        """Detect blinks with MediaPipe face landmarks with improved detection logic."""
        if not results.multi_face_landmarks:
            return {
                'is_blink': False,
                'blink_count': 0,
                'action_performed': False
            }
        
        # Get the first face landmarks
        face_landmarks = results.multi_face_landmarks[0]
        
        # Convert landmarks to pixel coordinates
        h, w = frame.shape[:2]
        landmark_coords = []
        for lm in face_landmarks.landmark:
            x, y = int(lm.x * w), int(lm.y * h)
            landmark_coords.append((x, y))
        
        # Calculate EAR for both eyes
        left_ear = self.eye_aspect_ratio(landmark_coords, self.LEFT_EYE_EAR_POINTS)
        right_ear = self.eye_aspect_ratio(landmark_coords, self.RIGHT_EYE_EAR_POINTS)
        avg_ear = (left_ear + right_ear) / 2.0
        
        # Update blink state
        self.last_blink_state = self.current_blink_state
        self.current_blink_state = avg_ear < self.BLINK_THRESHOLD
        
        # Detect blink edges (when eyes close - transition from open to closed)
        if self.current_blink_state and not self.last_blink_state:
            current_time = time.time()
            
            # Only register if enough time passed since last blink
            if not self.blink_timestamps or (current_time - self.blink_timestamps[-1]) > self.BLINK_TIME:
                self.blink_timestamps.append(current_time)
                print(f"Blink detected at {current_time}, total blinks: {len(self.blink_timestamps)}")
        
        # Remove old timestamps beyond BLINK_SEQUENCE_TIME
        current_time = time.time()
        self.blink_timestamps = [t for t in self.blink_timestamps 
                               if (current_time - t) <= self.BLINK_SEQUENCE_TIME]
        
        # Process blink sequences
        blink_count = len(self.blink_timestamps)
        action_performed = False
        
        # Wait for 300ms after last blink before triggering action to allow for triple blinks
        if blink_count >= 2 and (current_time - self.last_action_time) > 0.5:
            if current_time - self.blink_timestamps[-1] > 0.3:  # Wait to ensure sequence is complete
                if blink_count == 3:
                    pyautogui.doubleClick()  # Double click
                    action_performed = True
                    print("Triple blink detected: Double click")
                    self.last_action_time = current_time
                    self.blink_timestamps.clear()
                elif blink_count == 2:
                    pyautogui.click()  # Single click
                    action_performed = True
                    print("Double blink detected: Single click")
                    self.last_action_time = current_time
                    self.blink_timestamps.clear()
        
        # Visualize blink state on frame
        blink_state = "BLINK" if self.current_blink_state else "OPEN"
        ear_text = f"EAR: {avg_ear:.2f}, State: {blink_state}"
        cv2.putText(frame, ear_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        blink_text = f"Blinks: {blink_count}, Action: {'Yes' if action_performed else 'No'}"
        cv2.putText(frame, blink_text, (10, 60), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
        
        # Draw blinking visualization
        if self.current_blink_state:
            cv2.putText(frame, "BLINKING!", (frame.shape[1]//2 - 70, frame.shape[0]//2), 
                      cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 0, 255), 3)
        
        return {
            'avg_EAR': avg_ear,
            'is_blink': self.current_blink_state,
            'blink_count': blink_count,
            'action_performed': action_performed
        }