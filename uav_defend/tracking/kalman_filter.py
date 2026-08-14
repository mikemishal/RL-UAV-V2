"""
Kalman Filter for enemy state estimation.

Implements a constant-velocity Kalman filter for tracking enemy UAVs
using noisy position measurements.
"""

import numpy as np


class EnemyKalmanFilter:
    """
    Constant-velocity Kalman filter for 2D enemy tracking.
    
    State vector: x = [px, py, vx, vy]^T
        - px, py: position in 2D
        - vx, vy: velocity in 2D
    
    Measurement vector: z = [px, py]^T
        - Only position is observed (e.g., from radar/sensor detection)
    
    The filter assumes constant velocity motion with process noise
    to account for acceleration/maneuvers.
    
    Attributes:
        dt (float): Time step between predictions.
        x (np.ndarray): State estimate [px, py, vx, vy].
        P (np.ndarray): State covariance matrix (4x4).
        F (np.ndarray): State transition matrix (4x4).
        H (np.ndarray): Measurement matrix (2x4).
        Q (np.ndarray): Process noise covariance (4x4).
        R (np.ndarray): Measurement noise covariance (2x2).
        initialized (bool): Whether the filter has been initialized.
    
    Example:
        >>> kf = EnemyKalmanFilter(dt=0.1, process_var=1.0, measurement_var=0.5)
        >>> kf.initialize(np.array([10.0, 20.0]))
        >>> kf.predict()
        >>> kf.update(np.array([10.5, 20.3]))
        >>> pos = kf.get_position()
        >>> vel = kf.get_velocity()
    """
    
    def __init__(self, dt: float, process_var: float, measurement_var: float):
        """
        Initialize the Kalman filter with system parameters.
        
        Args:
            dt: Time step between filter updates (seconds).
            process_var: Process noise variance. Higher values allow the filter
                         to track more aggressive maneuvers but increase noise.
            measurement_var: Measurement noise variance. Should reflect the
                             accuracy of the position sensor.
        """
        self.dt = dt
        self.process_var = process_var
        self.measurement_var = measurement_var
        
        # State vector: [px, py, vx, vy]
        self.x = np.zeros(4)
        
        # State covariance matrix (initialized with high uncertainty)
        self.P = np.eye(4) * 1000.0
        
        # State transition matrix (constant velocity model)
        # x_new = F @ x_old
        # px_new = px + vx * dt
        # py_new = py + vy * dt
        # vx_new = vx
        # vy_new = vy
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0]
        ])
        
        # Measurement matrix (we only observe position)
        # z = H @ x => [px_measured, py_measured] = H @ [px, py, vx, vy]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0]
        ])
        
        # Process noise covariance
        # Uses discrete white noise model for constant velocity
        # Noise enters through acceleration, integrated to velocity and position
        self.Q = self._compute_process_noise(dt, process_var)
        
        # Measurement noise covariance
        self.R = np.eye(2) * measurement_var
        
        # Track initialization state
        self.initialized = False
    
    def _compute_process_noise(self, dt: float, var: float) -> np.ndarray:
        """
        Compute process noise covariance matrix for constant-velocity model.
        
        Uses the discrete white noise acceleration model where noise
        enters as random acceleration.
        
        Args:
            dt: Time step.
            var: Process noise variance (acceleration variance).
        
        Returns:
            4x4 process noise covariance matrix.
        """
        # Noise covariance for position-velocity pairs
        # Derived from integrating white noise acceleration
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        # Block diagonal for x and y (independent)
        q = np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ]
        ]) * var
        
        return q
    
    def initialize(self, position: np.ndarray) -> None:
        """
        Initialize the filter with a first position measurement.
        
        Sets the initial state to the measured position with zero velocity.
        Resets covariance to reflect high uncertainty in velocity.
        
        Args:
            position: Initial position measurement [px, py].
        """
        position = np.asarray(position).flatten()
        if position.shape[0] != 2:
            raise ValueError(f"Position must be 2D, got shape {position.shape}")
        
        # Set initial position, velocity assumed zero
        self.x[0] = position[0]  # px
        self.x[1] = position[1]  # py
        self.x[2] = 0.0          # vx (unknown)
        self.x[3] = 0.0          # vy (unknown)
        
        # Reset covariance
        # Low uncertainty in position (we just measured it)
        # High uncertainty in velocity (we don't know it yet)
        self.P = np.diag([
            self.measurement_var,      # px uncertainty
            self.measurement_var,      # py uncertainty
            100.0,                      # vx uncertainty (high)
            100.0                       # vy uncertainty (high)
        ])
        
        self.initialized = True
    
    def predict(self) -> np.ndarray:
        """
        Predict the next state based on constant-velocity motion model.
        
        Propagates state forward by dt using the transition matrix F
        and increases uncertainty according to process noise Q.
        
        Returns:
            Predicted state vector [px, py, vx, vy].
        
        Raises:
            RuntimeError: If filter has not been initialized.
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before predict()")
        
        # State prediction: x = F @ x
        self.x = self.F @ self.x
        
        # Covariance prediction: P = F @ P @ F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        # Ensure symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)
        
        return self.x.copy()
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update state estimate with a new position measurement.
        
        Incorporates the measurement z using the Kalman gain to
        optimally blend prediction with observation.
        
        Args:
            z: Position measurement [px, py].
        
        Returns:
            Updated state vector [px, py, vx, vy].
        
        Raises:
            RuntimeError: If filter has not been initialized.
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before update()")
        
        z = np.asarray(z).flatten()
        if z.shape[0] != 2:
            raise ValueError(f"Measurement must be 2D, got shape {z.shape}")
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P @ H^T @ S^-1
        # For 2x2 S, direct inverse is efficient and numerically stable
        try:
            S_inv = np.linalg.inv(S)
        except np.linalg.LinAlgError:
            # Fallback to pseudo-inverse if singular
            S_inv = np.linalg.pinv(S)
        
        K = self.P @ self.H.T @ S_inv
        
        # State update
        self.x = self.x + K @ y
        
        # Covariance update (Joseph form for numerical stability)
        # P = (I - K @ H) @ P @ (I - K @ H)^T + K @ R @ K^T
        I_KH = np.eye(4) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Ensure symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)
        
        return self.x.copy()
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state estimate.
        
        Returns:
            State vector [px, py, vx, vy].
        """
        return self.x.copy()
    
    def get_position(self) -> np.ndarray:
        """
        Get the current estimated position.
        
        Returns:
            Position vector [px, py].
        """
        return self.x[:2].copy()
    
    def get_velocity(self) -> np.ndarray:
        """
        Get the current estimated velocity.
        
        Returns:
            Velocity vector [vx, vy].
        """
        return self.x[2:4].copy()
    
    def get_covariance(self) -> np.ndarray:
        """
        Get the current state covariance matrix.
        
        Returns:
            4x4 covariance matrix P.
        """
        return self.P.copy()
    
    def get_position_uncertainty(self) -> np.ndarray:
        """
        Get the position uncertainty (standard deviation).
        
        Returns:
            Position std [sigma_px, sigma_py].
        """
        return np.sqrt(np.diag(self.P)[:2])
    
    def get_velocity_uncertainty(self) -> np.ndarray:
        """
        Get the velocity uncertainty (standard deviation).
        
        Returns:
            Velocity std [sigma_vx, sigma_vy].
        """
        return np.sqrt(np.diag(self.P)[2:4])
    
    def reset(self) -> None:
        """
        Reset the filter to uninitialized state.
        
        Clears state and covariance, requiring re-initialization.
        """
        self.x = np.zeros(4)
        self.P = np.eye(4) * 1000.0
        self.initialized = False
