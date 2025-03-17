import cv2
import numpy as np
import pyautogui
import mediapipe as mp
from core.utils import get_eye_landmarks, map_gaze_to_screen
import time
from core.kalman_filter import GazeKalmanFilter

# Constants
Gaze_Update_Interval = 0.02
Mouse_Idle_Time = 2.0

# Tracking variables
last_mouse_pos = pyautogui.position()
mouse_active = False
last_mouse_time = time.time()

mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

def is_mouse_moving():
    """Check if the mouse has moved manually."""
    global last_mouse_pos, last_mouse_time, mouse_active
    current_pos = pyautogui.position()

    if current_pos != last_mouse_pos:
        last_mouse_time = time.time()
        mouse_active = True
    elif time.time() - last_mouse_time > Mouse_Idle_Time:
        mouse_active = False

    last_mouse_pos = current_pos
    return mouse_active

class GazeTracker:
    def __init__(self, face_detector):
        self.face_detector = face_detector
        self.screen_w, self.screen_h = pyautogui.size()
        self.calibration_data = None
        self.prev_positions = []
        self.max_history = 3
        self.smoothing_factor = 0.3
        self.last_move_time = time.time()
        self.move_delay = 0.08
        self.dead_zone = 3
        self.dampening_factor = 0.4
        self.head_position_history = []
        self.head_history_size = 5
        self.reference_head_pos = None  # Set during calibration
        self.head_weight = 0.3  # Reduced head influence
        self.kalman_filter = GazeKalmanFilter(self.screen_w, self.screen_h)
        pyautogui.FAILSAFE = False

    def get_head_position(self, face_landmarks):
        """Extract head position from face landmarks."""
        nose_tip = face_landmarks.landmark[4]
        return (nose_tip.x, nose_tip.y, nose_tip.z)

    def calibrate(self, cap):
        """Run the calibration process."""
        from ui.calibration_ui import GazeTrackingTutorial
        from core.blink_detection import BlinkDetector

        blink_detector = BlinkDetector()
        tutorial = GazeTrackingTutorial(cap, face_mesh, blink_detector)

        try:
            self.calibration_data = tutorial.run_tutorial()
            if self.calibration_data is None:
                print("Calibration not completed. Using default mapping.")
                self.calibration_data = self.default_calibration()
        except Exception as e:
            print(f"Error during calibration: {e}")
            self.calibration_data = self.default_calibration()

        # Capture head position over multiple frames
        head_positions = []
        for _ in range(10):  # Take 10 samples
            ret, frame = cap.read()
            if ret:
                _, results = self.face_detector.detect(frame)
                if results.multi_face_landmarks:
                    head_pos = self.get_head_position(results.multi_face_landmarks[0])
                    head_positions.append(head_pos)
            time.sleep(0.05)  # Small delay to get different samples

        if head_positions:
            self.reference_head_pos = np.mean(head_positions, axis=0)
            print(f"Reference head position captured: {self.reference_head_pos}")

        return True

    def default_calibration(self):
        """Returns a default calibration mapping if calibration fails."""
        return {
            "screen_points": [(0, 0), (self.screen_w//2, 0), (self.screen_w, 0),
                            (0, self.screen_h//2), (self.screen_w//2, self.screen_h//2), (self.screen_w, self.screen_h//2),
                            (0, self.screen_h), (self.screen_w//2, self.screen_h), (self.screen_w, self.screen_h)],
            "eye_points": [(0.3, 0.3), (0.5, 0.3), (0.7, 0.3),
                            (0.3, 0.5), (0.5, 0.5), (0.7, 0.5),
                            (0.3, 0.7), (0.5, 0.7), (0.7, 0.7)]
        }
    def get_head_movement_compensation(self, head_pos):
        if self.reference_head_pos is None or head_pos is None:
            return (0, 0)  # Default no compensation
        
        # Ensure head_pos is a NumPy array
        if isinstance(head_pos, list):
            head_pos = np.array(head_pos)
        
        # Compute difference in head pose
        head_offset = head_pos - self.reference_head_pos

        return (head_offset[0], head_offset[1])

  
    def process_frame(self, frame):
        """Process a frame and update cursor position."""
        frame, results = self.face_detector.detect(frame)

        if not results.multi_face_landmarks:
            return frame

        face_landmarks = results.multi_face_landmarks[0]
        head_pos = self.get_head_position(face_landmarks)
        head_offset_x, head_offset_y = self.get_head_movement_compensation(head_pos)

        left_eye, right_eye = get_eye_landmarks(face_landmarks, frame.shape)
        gaze_x = (left_eye[0] + right_eye[0]) / 2
        gaze_y = (left_eye[1] + right_eye[1]) / 2

        adjusted_gaze_x = gaze_x - (head_offset_x / (self.screen_w * 2))
        adjusted_gaze_y = gaze_y - (head_offset_y / (self.screen_h * 2))

        adjusted_gaze_x = max(0, min(1, adjusted_gaze_x))
        adjusted_gaze_y = max(0, min(1, adjusted_gaze_y))

        raw_screen_pos = map_gaze_to_screen((adjusted_gaze_x, adjusted_gaze_y), self.calibration_data, self.smoothing_factor)

        if raw_screen_pos:
            screen_pos = self.kalman_filter.update(raw_screen_pos)
            current_time = time.time()
            if current_time - self.last_move_time > self.move_delay:
                if not is_mouse_moving():
                    pyautogui.moveTo(screen_pos[0], screen_pos[1], duration=0.03)
                    self.last_move_time = current_time

        return frame
def track_gaze():
    """
    Initialize gaze tracking module.
    This function is called from main.py.
    """
    print("Gaze tracking module initialized")
    from core.face_detector import detect_face
    return GazeTracker(detect_face())