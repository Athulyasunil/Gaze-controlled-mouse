import numpy as np
from filterpy.kalman import KalmanFilter

class GazeKalmanFilter:
    def __init__(self, screen_w, screen_h):
        # State: [x, y, dx, dy]
        # x, y: cursor position
        # dx, dy: cursor velocity
        self.kf = KalmanFilter(dim_x=4, dim_z=2)
        
        # State transition matrix (position + velocity model)
        dt = 1.0/30.0  # Assuming 30fps
        self.kf.F = np.array([
            [1, 0, dt, 0],
            [0, 1, 0, dt],
            [0, 0, 1, 0],
            [0, 0, 0, 1]
        ])
        
        # Measurement matrix (we only measure position)
        self.kf.H = np.array([
            [1, 0, 0, 0],
            [0, 1, 0, 0]
        ])
        
        # Measurement noise (adjust based on your camera/system)
        self.kf.R = np.eye(2) * 10
        
        # Process noise (how much we expect the state to change)
        self.kf.Q = np.eye(4)
        self.kf.Q[0:2, 0:2] *= 0.01  # position uncertainty
        self.kf.Q[2:4, 2:4] *= 0.1   # velocity uncertainty
        
        # Initial state
        self.kf.x = np.array([screen_w/2, screen_h/2, 0, 0])
        
        # Initial uncertainty
        self.kf.P = np.eye(4) * 500
        
        # Bounds
        self.screen_w = screen_w
        self.screen_h = screen_h
        
        # Saccade detection
        self.saccade_threshold = 100  # pixels
        self.last_measurement = None
    
    def update(self, measurement):
        """
        Update filter with new measurement (x, y)
        Returns filtered position (x, y)
        """
        if self.last_measurement is not None:
            # Detect saccades (rapid eye movements)
            distance = np.sqrt((measurement[0] - self.last_measurement[0])**2 + 
                              (measurement[1] - self.last_measurement[1])**2)
            
            if distance > self.saccade_threshold:
                # Saccade detected, increase uncertainty
                self.kf.P[0:2, 0:2] *= 10
        
        self.last_measurement = measurement
        
        # Predict
        self.kf.predict()
        
        # Update
        self.kf.update(measurement)
        
        # Ensure the position stays within screen bounds
        x = max(0, min(self.screen_w, self.kf.x[0]))
        y = max(0, min(self.screen_h, self.kf.x[1]))
        
        return (int(x), int(y))
    
    def reset(self, position):
        """Reset the filter state"""
        self.kf.x = np.array([position[0], position[1], 0, 0])
        self.kf.P = np.eye(4) * 500
        self.last_measurement = None