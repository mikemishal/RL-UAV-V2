"""
2D Constant-Velocity Kalman Filter for enemy drone tracking.

This module implements a linear Kalman filter for tracking a moving target
in 2D space, assuming constant velocity motion model between observations.

The Kalman filter provides optimal state estimation under the assumptions:
1. Linear dynamics and measurement models
2. Gaussian process and measurement noise
3. Known noise covariances

Math Overview:
--------------
State vector: x = [px, py, vx, vy]^T
    - px, py: position in 2D
    - vx, vy: velocity in 2D

Dynamics model (constant velocity):
    x_{k+1} = F * x_k + w_k
    where w_k ~ N(0, Q) is process noise

Measurement model (position only):
    z_k = H * x_k + v_k
    where v_k ~ N(0, R) is measurement noise

Predict step:
    x_pred = F * x
    P_pred = F * P * F^T + Q

Update step (when measurement available):
    y = z - H * x_pred          (innovation)
    S = H * P_pred * H^T + R    (innovation covariance)
    K = P_pred * H^T * S^{-1}   (Kalman gain)
    x = x_pred + K * y          (state update)
    P = (I - K * H) * P_pred    (covariance update)
"""

import numpy as np


class KalmanTracker:
    """
    2D Constant-Velocity Kalman Filter for tracking a moving target.
    
    This filter maintains an estimate of position and velocity in 2D space,
    using position-only measurements to update the state estimate.
    
    Attributes:
        dt: Time step between predictions
        x: State vector [px, py, vx, vy]
        P: State covariance matrix (4x4)
        F: State transition matrix (4x4)
        H: Measurement matrix (2x4)
        Q: Process noise covariance (4x4)
        R: Measurement noise covariance (2x2)
    """
    
    def __init__(
        self,
        dt: float = 0.5,
        process_var: float = 1.0,
        measurement_var: float = 1.0,
    ):
        """
        Initialize the Kalman filter with specified parameters.
        
        Args:
            dt: Time step duration between predictions (default: 0.5)
            process_var: Process noise variance (acceleration uncertainty).
                         Higher values = more trust in measurements.
            measurement_var: Measurement noise variance.
                            Higher values = more trust in predictions.
        """
        self.dt = dt
        self.process_var = process_var
        self.measurement_var = measurement_var
        
        # State vector: [px, py, vx, vy]
        # Initialized to zeros until initialize() is called
        self.x = np.zeros(4, dtype=np.float64)
        
        # State transition matrix F (constant velocity model)
        # x_{k+1} = F * x_k
        # [px']   [1  0  dt  0 ] [px]
        # [py'] = [0  1  0   dt] [py]
        # [vx']   [0  0  1   0 ] [vx]
        # [vy']   [0  0  0   1 ] [vy]
        self.F = np.array([
            [1.0, 0.0, dt,  0.0],
            [0.0, 1.0, 0.0, dt ],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ], dtype=np.float64)
        
        # Measurement matrix H (observe position only)
        # z = H * x
        # [zx]   [1  0  0  0] [px]
        # [zy] = [0  1  0  0] [py]
        #                     [vx]
        #                     [vy]
        self.H = np.array([
            [1.0, 0.0, 0.0, 0.0],
            [0.0, 1.0, 0.0, 0.0],
        ], dtype=np.float64)
        
        # Process noise covariance Q
        # Models uncertainty in the constant-velocity assumption
        # Derived from discrete white noise acceleration model:
        # Q = G * G^T * process_var, where G = [dt^2/2, dt^2/2, dt, dt]^T
        # Simplified diagonal approximation for computational efficiency
        self.Q = self._build_process_noise(process_var)
        
        # Measurement noise covariance R
        # Models sensor noise in position measurements
        self.R = np.array([
            [measurement_var, 0.0],
            [0.0, measurement_var],
        ], dtype=np.float64)
        
        # State covariance matrix P
        # Initialized with high uncertainty until initialize() is called
        self.P = np.eye(4, dtype=np.float64) * 1000.0
        
        # Track initialization state
        self._initialized = False
    
    def _build_process_noise(self, process_var: float) -> np.ndarray:
        """
        Build process noise covariance matrix Q.
        
        Uses the discrete white noise acceleration model where acceleration
        is modeled as white noise with variance process_var.
        
        The exact Q matrix is derived from:
            Q = integral(F * G * process_var * G^T * F^T, dt)
        where G is the noise input matrix.
        
        For constant velocity with acceleration noise:
            G = [dt^2/2, dt^2/2, dt, dt]^T
        
        Args:
            process_var: Variance of acceleration noise
            
        Returns:
            4x4 process noise covariance matrix
        """
        dt = self.dt
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        # Full process noise covariance (from integration)
        # This models random acceleration affecting position and velocity
        q = process_var
        Q = np.array([
            [dt4/4, 0,     dt3/2, 0    ],
            [0,     dt4/4, 0,     dt3/2],
            [dt3/2, 0,     dt2,   0    ],
            [0,     dt3/2, 0,     dt2  ],
        ], dtype=np.float64) * q
        
        return Q
    
    def initialize(self, position: np.ndarray) -> None:
        """
        Initialize the filter with an initial position measurement.
        
        Sets the state to the given position with zero velocity,
        and resets the covariance to reflect high velocity uncertainty.
        
        Args:
            position: Initial position [px, py] as numpy array
        """
        position = np.asarray(position, dtype=np.float64).flatten()
        if position.shape[0] != 2:
            raise ValueError(f"Position must be 2D, got shape {position.shape}")
        
        # Initialize state: known position, unknown velocity (zero initial)
        self.x = np.array([
            position[0],  # px
            position[1],  # py
            0.0,          # vx (unknown, assume zero)
            0.0,          # vy (unknown, assume zero)
        ], dtype=np.float64)
        
        # Initialize covariance: low position uncertainty, high velocity uncertainty
        # Position variance matches measurement noise
        # Velocity variance is high since we don't know initial velocity
        self.P = np.diag([
            self.measurement_var,      # px variance
            self.measurement_var,      # py variance
            100.0,                      # vx variance (high uncertainty)
            100.0,                      # vy variance (high uncertainty)
        ]).astype(np.float64)
        
        self._initialized = True
    
    def predict(self) -> np.ndarray:
        """
        Predict the next state using the constant-velocity motion model.
        
        Updates both state estimate and covariance:
            x_pred = F * x
            P_pred = F * P * F^T + Q
        
        Returns:
            Predicted state vector [px, py, vx, vy]
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized. Call initialize() first.")
        
        # State prediction: x_pred = F * x
        self.x = self.F @ self.x
        
        # Covariance prediction: P_pred = F * P * F^T + Q
        self.P = self.F @ self.P @ self.F.T + self.Q
        
        return self.x.copy()
    
    def update(self, z: np.ndarray) -> np.ndarray:
        """
        Update the state estimate with a new position measurement.
        
        Implements the Kalman filter update equations:
            y = z - H * x           (innovation/residual)
            S = H * P * H^T + R     (innovation covariance)
            K = P * H^T * S^{-1}    (Kalman gain)
            x = x + K * y           (state update)
            P = (I - K * H) * P     (covariance update)
        
        Args:
            z: Position measurement [zx, zy] as numpy array
            
        Returns:
            Updated state vector [px, py, vx, vy]
        """
        if not self._initialized:
            raise RuntimeError("Filter not initialized. Call initialize() first.")
        
        z = np.asarray(z, dtype=np.float64).flatten()
        if z.shape[0] != 2:
            raise ValueError(f"Measurement must be 2D, got shape {z.shape}")
        
        # Innovation (measurement residual): y = z - H * x
        y = z - self.H @ self.x
        
        # Innovation covariance: S = H * P * H^T + R
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P * H^T * S^{-1}
        # Using solve instead of inverse for numerical stability
        K = self.P @ self.H.T @ np.linalg.inv(S)
        
        # State update: x = x + K * y
        self.x = self.x + K @ y
        
        # Covariance update: P = (I - K * H) * P
        # Using Joseph form for numerical stability:
        # P = (I - K*H) * P * (I - K*H)^T + K * R * K^T
        I_KH = np.eye(4, dtype=np.float64) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        return self.x.copy()
    
    def get_position(self) -> np.ndarray:
        """
        Get the current position estimate.
        
        Returns:
            Position [px, py] as numpy array
        """
        return self.x[:2].copy()
    
    def get_velocity(self) -> np.ndarray:
        """
        Get the current velocity estimate.
        
        Returns:
            Velocity [vx, vy] as numpy array
        """
        return self.x[2:].copy()
    
    def get_state(self) -> np.ndarray:
        """
        Get the full state vector.
        
        Returns:
            State [px, py, vx, vy] as numpy array
        """
        return self.x.copy()
    
    def get_covariance(self) -> np.ndarray:
        """
        Get the state covariance matrix.
        
        Returns:
            4x4 covariance matrix P
        """
        return self.P.copy()
    
    def get_position_uncertainty(self) -> np.ndarray:
        """
        Get the position uncertainty (standard deviation).
        
        Returns:
            Position std [std_px, std_py] as numpy array
        """
        return np.sqrt(np.diag(self.P)[:2])
    
    @property
    def is_initialized(self) -> bool:
        """Check if the filter has been initialized."""
        return self._initialized


# Example usage and testing
if __name__ == "__main__":
    print("=" * 60)
    print("Kalman Tracker Test")
    print("=" * 60)
    
    # Create tracker with dt=0.5 (matching env timestep)
    tracker = KalmanTracker(dt=0.5, process_var=1.0, measurement_var=0.5)
    
    # Simulate a target moving in a straight line with noise
    np.random.seed(42)
    true_velocity = np.array([5.0, 3.0])  # True velocity
    true_position = np.array([0.0, 0.0])   # Start position
    
    print(f"\nTrue velocity: {true_velocity}")
    print(f"Initial position: {true_position}")
    print()
    
    # Initialize tracker with first noisy measurement
    noise = np.random.randn(2) * 0.5
    z0 = true_position + noise
    tracker.initialize(z0)
    print(f"Step 0: Initialized at z={z0}")
    print(f"        State: pos={tracker.get_position()}, vel={tracker.get_velocity()}")
    
    # Run for 10 steps
    for step in range(1, 11):
        # Move target
        true_position = true_position + true_velocity * tracker.dt
        
        # Generate noisy measurement
        noise = np.random.randn(2) * 0.5
        z = true_position + noise
        
        # Predict
        tracker.predict()
        
        # Update with measurement
        tracker.update(z)
        
        # Report
        pos_est = tracker.get_position()
        vel_est = tracker.get_velocity()
        pos_err = np.linalg.norm(pos_est - true_position)
        vel_err = np.linalg.norm(vel_est - true_velocity)
        
        print(f"Step {step:2d}: true_pos={true_position}, z={z}")
        print(f"         est_pos={pos_est}, est_vel={vel_est}")
        print(f"         pos_err={pos_err:.3f}, vel_err={vel_err:.3f}")
    
    print()
    print("=" * 60)
    print("Test complete!")
    print("=" * 60)
