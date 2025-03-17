import cv2
import numpy as np
import time
import pyautogui
from core.calibration import perform_calibration

class GazeTrackingTutorial:
    def __init__(self, cap, face_mesh, blink_detector):
        """
        Initialize the gaze tracking tutorial UI.
        
        Args:
            cap: OpenCV video capture object
            face_mesh: MediaPipe face mesh object
            blink_detector: BlinkDetector instance for blink-based selection
        """
        self.cap = cap
        self.face_mesh = face_mesh
        self.blink_detector = blink_detector
        self.screen_w, self.screen_h = pyautogui.size()
        
        # Create a full-screen window for tutorial
        cv2.namedWindow("Gaze Tracking Tutorial", cv2.WND_PROP_FULLSCREEN)
        cv2.setWindowProperty("Gaze Tracking Tutorial", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    
    def run_tutorial(self):
        
        """
        Run the complete tutorial and calibration process.
        
        Returns:
            dict: Calibration data mapping eye positions to screen positions
        """
        if not self._show_welcome_screens():
            return None
            
        if not self._run_blink_practice():
            return None
            
        # Run the actual calibration
        calibration_data = perform_calibration(
            self.cap, 
            self.face_mesh, 
            self.blink_detector,
            self.screen_w,
            self.screen_h
        )
        
        self._show_completion_screen(calibration_data)
        return calibration_data
    
    def _show_welcome_screens(self):
        """Show the tutorial welcome screens with instructions."""
        # Tutorial sequence content
        tutorial_screens = [
            {
                "title": "Welcome to Gaze Tracking",
                "text": [
                    "This system allows you to control your computer with your eyes.",
                    "We'll guide you through a simple calibration and tutorial.",
                    "Keep your head steady during the entire process.",
                    "Double-blink to continue..."
                ]
            },
            {
                "title": "How It Works",
                "text": [
                    "1. The system tracks the position of your eyes",
                    "2. You'll look at 9 different points on the screen",
                    "3. For each point, focus your gaze and blink twice to select",
                    "4. This helps the system learn how your eye positions",
                    "   map to screen coordinates.",
                    "Double-blink to continue..."
                ]
            },
            {
                "title": "Blink Controls",
                "text": [
                    "You can control the cursor using blinks:",
                    "• Double-blink = Single Click",
                    "• Triple-blink = Double Click",
                    "Let's practice blinking before we start.",
                    "Try to double-blink now...",
                    "Double-blink when you're ready to begin calibration"
                ]
            }
        ]
        
        # Display tutorial screens with blink navigation
        for screen in tutorial_screens:
            # Wait for a double-blink to proceed
            continue_tutorial = False
            blink_practice_count = 0
            last_blink_time = time.time()
            
            while not continue_tutorial:
                ret, frame = self.cap.read()
                if not ret:
                    continue
                
                # Convert frame to RGB for MediaPipe
                rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                
                # Process frame with MediaPipe
                results = self.face_mesh.process(rgb_frame)
                
                # Create the tutorial frame
                tutorial_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
                cv2.putText(tutorial_frame, screen["title"], (self.screen_w//2 - 200, 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
                
                for i, line in enumerate(screen["text"]):
                    cv2.putText(tutorial_frame, line, (self.screen_w//2 - 300, 200 + i*40),
                                cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                
                # Detect blinks
                blink_result = self.blink_detector.detect_blink(results, frame)
                
                # Show webcam feed in corner for feedback
                webcam_display = cv2.resize(frame, (320, 240))
                tutorial_frame[20:20+240, 20:20+320] = webcam_display
                
                # Check for blink action
                if blink_result['action_performed']:
                    # Show confirmation of detected blink
                    cv2.putText(tutorial_frame, "Blink Detected!", (self.screen_w//2 - 100, self.screen_h - 100),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    
                    # Add a visual progress indicator
                    current_time = time.time()
                    if current_time - last_blink_time < 1.5:  # Within 1.5 seconds of last blink
                        blink_practice_count += 1
                        # Show success after 2 successful blinks
                        if blink_practice_count >= 1:
                            cv2.putText(tutorial_frame, "Great! Moving to next step...", 
                                        (self.screen_w//2 - 200, self.screen_h - 50),
                                        cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                            continue_tutorial = True
                    
                    last_blink_time = current_time
                
                # Display the tutorial frame
                cv2.imshow("Gaze Tracking Tutorial", tutorial_frame)
                key = cv2.waitKey(1) & 0xFF
                if key == 27:  # ESC key as emergency exit
                    return False
                
                # Add a short delay after blink detection
                if continue_tutorial:
                    time.sleep(1.5)
                    break
        
        return True
    
    def _run_blink_practice(self):
        """Run a blink practice session to ensure user understands controls."""
        practice_complete = False
        practice_start_time = time.time()
        successful_blinks = 0
        
        while not practice_complete:
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            practice_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
            cv2.putText(practice_frame, "Blink Practice", (self.screen_w//2 - 150, 100),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
            
            cv2.putText(practice_frame, "Try to double-blink 3 times", (self.screen_w//2 - 200, 200),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Detect blinks
            blink_result = self.blink_detector.detect_blink(results, frame)
            
            # Show webcam feed in corner
            webcam_display = cv2.resize(frame, (320, 240))
            practice_frame[20:20+240, 20:20+320] = webcam_display
            
            # Progress indicator
            cv2.putText(practice_frame, f"Successful blinks: {successful_blinks}/3", 
                        (self.screen_w//2 - 150, 300),
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
            
            # Check for blink action
            if blink_result['action_performed']:
                successful_blinks += 1
                cv2.putText(practice_frame, "Blink Detected!", (self.screen_w//2 - 100, self.screen_h - 100),
                            cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                
                if successful_blinks >= 3:
                    cv2.putText(practice_frame, "Great! Ready to start calibration.", 
                                (self.screen_w//2 - 250, self.screen_h - 50),
                                cv2.FONT_HERSHEY_SIMPLEX, 1.2, (0, 255, 0), 2)
                    practice_complete = True
                
                # Display the confirmation
                cv2.imshow("Gaze Tracking Tutorial", practice_frame)
                cv2.waitKey(1)
                time.sleep(1)  # Short pause after each successful blink
            
            # Display the practice frame
            cv2.imshow("Gaze Tracking Tutorial", practice_frame)
            key = cv2.waitKey(1) & 0xFF
            if key == 27:  # ESC key as emergency exit
                return False
            
            # Safety timeout (1 minute)
            if time.time() - practice_start_time > 60 and not practice_complete:
                practice_complete = True
        
        # Final preparation screen
        prep_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        cv2.putText(prep_frame, "Calibration Starting", (self.screen_w//2 - 180, self.screen_h//2 - 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 255, 255), 2)
        cv2.putText(prep_frame, "Look at each dot and double-blink to select it",
                    (self.screen_w//2 - 300, self.screen_h//2 + 50),
                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (255, 255, 255), 2)
        cv2.imshow("Gaze Tracking Tutorial", prep_frame)
        cv2.waitKey(1)
        time.sleep(2)  # Give time to read instructions
        
        return True
    
    def _show_completion_screen(self, calibration_data):
        """Show the calibration completion screen with results."""
        # Final calibration complete message
        final_frame = np.zeros((self.screen_h, self.screen_w, 3), dtype=np.uint8)
        
        if calibration_data:
            points_collected = len(calibration_data.get("screen_points", []))
            
            cv2.putText(final_frame, "Calibration Complete!", (self.screen_w//2 - 200, self.screen_h//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 2)
            cv2.putText(final_frame, f"Successfully mapped {points_collected} points", 
                       (self.screen_w//2 - 230, self.screen_h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(final_frame, "You can now control the cursor with your eyes!", 
                       (self.screen_w//2 - 300, self.screen_h//2 + 70),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
            cv2.putText(final_frame, "Double-blink to click, triple-blink to double-click", 
                       (self.screen_w//2 - 300, self.screen_h//2 + 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
        else:
            cv2.putText(final_frame, "Calibration Incomplete", (self.screen_w//2 - 200, self.screen_h//2 - 50),
                       cv2.FONT_HERSHEY_SIMPLEX, 1.5, (255, 0, 0), 2)
            cv2.putText(final_frame, "Please try again to improve accuracy", 
                       (self.screen_w//2 - 250, self.screen_h//2 + 20),
                       cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        # Display exit instructions at bottom
        cv2.putText(final_frame, "Double-blink to exit tutorial...", 
                   (self.screen_w//2 - 200, self.screen_h - 100),
                   cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
        
        cv2.imshow("Gaze Tracking Tutorial", final_frame)
        
        # Wait for double-blink to exit
        exit_tutorial = False
        start_time = time.time()
        
        while not exit_tutorial and time.time() - start_time < 15:  # 15 second timeout
            ret, frame = self.cap.read()
            if not ret:
                continue
            
            rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = self.face_mesh.process(rgb_frame)
            
            # Detect blinks
            blink_result = self.blink_detector.detect_blink(results, frame)
            
            if blink_result['action_performed']:
                exit_tutorial = True
                break
                
            cv2.waitKey(1)
        
        cv2.destroyWindow("Gaze Tracking Tutorial")