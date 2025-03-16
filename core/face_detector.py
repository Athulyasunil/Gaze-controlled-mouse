import cv2
import mediapipe as mp

class FaceDetector:
    def __init__(self):
        self.mp_face_mesh = mp.solutions.face_mesh
        self.face_mesh = self.mp_face_mesh.FaceMesh(
            max_num_faces=1,
            refine_landmarks=True,  # Refined for iris detection
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5
        )
    
    def detect(self, frame):
        """Detect face and iris landmarks for both eyes."""
        rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = self.face_mesh.process(rgb_frame)
        
        if results.multi_face_landmarks:
            for face_landmarks in results.multi_face_landmarks:
                for idx, landmark in enumerate(face_landmarks.landmark):
                    h, w, _ = frame.shape
                    x, y = int(landmark.x * w), int(landmark.y * h)
                    
                    # ✅ Detect BOTH eyes:
                    # Left iris: 468, 469, 470, 471
                    # Right iris: 473, 474, 475, 476
                    if idx in [468, 469, 470, 471, 473, 474, 475, 476]:
                        cv2.circle(frame, (x, y), 2, (0, 255, 0), -1)  # Green dots for both irises
                    
                    # Optional: Draw face landmarks
                    if idx % 10 == 0:
                        cv2.circle(frame, (x, y), 1, (255, 0, 0), -1)  # Blue dots for face
        
        return frame, results

def detect_face():
    """
    Initialize face detection module.
    This function is called from main.py.
    """
    print("Face detection module initialized")
    return FaceDetector()