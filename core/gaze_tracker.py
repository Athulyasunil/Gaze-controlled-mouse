import cv2
import numpy as np
import pyautogui
from core.utils import map_gaze_to_screen, get_eye_landmarks
import time


# Constants
Gaze_Update_Interval = 0.02  # Gaze updates every 20ms
Mouse_Idle_Time = 2.0  # Resume gaze after 2 seconds of mouse inactivity

# Tracking variables
last_mouse_pos = pyautogui.position()
mouse_active = False
last_mouse_time = time.time()

def is_mouse_moving():
    """Check if the mouse has moved manually."""
    global last_mouse_pos, last_mouse_time, mouse_active
    current_pos = pyautogui.position()
    
    if current_pos != last_mouse_pos:
        last_mouse_time = time.time()  # Update last active time
        mouse_active = True
    elif time.time() - last_mouse_time > Mouse_Idle_Time:
        mouse_active = False  # Resume gaze control after idle time

    last_mouse_pos = current_pos  # Update last position
    return mouse_active
class GazeTracker:
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.screen_w, self.screen_h = pyautogui.size()
        self.calibration_data = None
        self.prev_positions = []  # Store multiple previous positions for better smoothing
        self.max_history = 5      # Reduced from 10 for more responsive cursor
        self.smoothing_factor = 0.5  # Reduced for more responsive movement (0-1)
        self.last_move_time = time.time()
        self.move_delay = 0.02    # Reduced minimum seconds between cursor moves
        
        # Disable PyAutoGUI's failsafe
        pyautogui.FAILSAFE = False
    
    def calibrate(self, cap):
        """Run the calibration process."""
        from core.calibration import calibrate_gaze
        self.calibration_data = calibrate_gaze(cap, self.face_detector.face_mesh)
        # Reset position history after calibration
        self.prev_positions = []
        return self.calibration_data is not None
    
    def process_frame(self, frame):
        """Process a single frame and update cursor position."""
        frame, results = self.face_detector.detect(frame)
        
        if not results.multi_face_landmarks:
            return frame
        
        face_landmarks = results.multi_face_landmarks[0]
        left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape)
        
        # Get average gaze position from both eyes
        gaze_x = (left_eye[0] + right_eye[0]) / 2
        gaze_y = (left_eye[1] + right_eye[1]) / 2
        gaze_pos = (gaze_x, gaze_y)
        
        # Map gaze to screen coordinates
        raw_screen_pos = map_gaze_to_screen(gaze_pos, self.calibration_data, self.smoothing_factor)
        
        # Apply multi-point smoothing
        if raw_screen_pos:
            # Add new position to history
            self.prev_positions.append(raw_screen_pos)
            # Keep history at fixed length
            if len(self.prev_positions) > self.max_history:
                self.prev_positions.pop(0)
            
            # Apply weighted average smoothing (more weight to recent positions)
            if len(self.prev_positions) > 1:
                weights = np.linspace(0.5, 1.0, len(self.prev_positions))
                weights = weights / np.sum(weights)  # Normalize weights
                
                x_avg = sum(p[0] * w for p, w in zip(self.prev_positions, weights))
                y_avg = sum(p[1] * w for p, w in zip(self.prev_positions, weights))
                
                screen_pos = (int(x_avg), int(y_avg))
            else:
                screen_pos = raw_screen_pos
            
            # Move cursor with rate limiting
            current_time = time.time()
            if current_time - self.last_move_time > self.move_delay:
                try:
                    # Reduce duration for more responsive movement
                    if not is_mouse_moving():
                        pyautogui.moveTo(screen_pos[0], screen_pos[1], duration=0.03)
                        self.last_move_time = current_time
                except Exception as e:
                    print(f"Error moving cursor: {e}")
        
        # Visualize gaze point and mapped screen position on frame
        h, w = frame.shape[:2]
        # Draw normalized gaze position
        gaze_frame_x, gaze_frame_y = int(gaze_x * w), int(gaze_y * h)
        cv2.circle(frame, (gaze_frame_x, gaze_frame_y), 5, (0, 0, 255), -1)
        
        # Show screen position mapping (scaled to frame)
        if raw_screen_pos:
            screen_frame_x = int((raw_screen_pos[0] / self.screen_w) * w)
            screen_frame_y = int((raw_screen_pos[1] / self.screen_h) * h)
            cv2.circle(frame, (screen_frame_x, screen_frame_y), 8, (255, 0, 0), 2)
            
            # Draw line connecting the two points
            cv2.line(frame, (gaze_frame_x, gaze_frame_y), 
                    (screen_frame_x, screen_frame_y), (255, 255, 0), 2)
        
        # Add debug info - show calibration and screen bounds info
        cv2.putText(frame, f"Gaze: ({gaze_x:.2f}, {gaze_y:.2f})", (10, 30),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 0, 255), 2)
        
        if self.prev_positions and len(self.prev_positions) > 0:
            latest = self.prev_positions[-1]
            cv2.putText(frame, f"Cursor: ({latest[0]}/{self.screen_w}, {latest[1]}/{self.screen_h})", (10, 60),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 0, 0), 2)
        
        return frame

def track_gaze():
    """
    Initialize gaze tracking module.
    This function is called from main.py.
    """
    print("Gaze tracking module initialized")
    from core.face_detector import detect_face
    return GazeTracker(detect_face())