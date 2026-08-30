"""
Kalman Filter for enemy state estimation.

Implements a constant-velocity Kalman filter for tracking enemy UAVs
in 3D using noisy position measurements.
"""

import numpy as np


class EnemyKalmanFilter:
    """
    Constant-velocity Kalman filter for 3D enemy tracking.
    
    State vector: x = [px, py, pz, vx, vy, vz]^T
        - px, py, pz: position in 3D
        - vx, vy, vz: velocity in 3D
    
    Measurement vector: z = [px, py, pz]^T
        - Only position is observed (e.g., from radar/sensor detection)
    
    The filter assumes constant velocity motion with process noise
    to account for acceleration/maneuvers.
    
    Attributes:
        dt (float): Time step between predictions.
        x (np.ndarray): State estimate [px, py, pz, vx, vy, vz].
        P (np.ndarray): State covariance matrix (6x6).
        F (np.ndarray): State transition matrix (6x6).
        H (np.ndarray): Measurement matrix (3x6).
        Q (np.ndarray): Process noise covariance (6x6).
        R (np.ndarray): Measurement noise covariance (3x3).
        initialized (bool): Whether the filter has been initialized.
    
    Example:
        >>> kf = EnemyKalmanFilter(dt=0.1, process_var=1.0, measurement_var=0.5)
        >>> kf.initialize(np.array([10.0, 20.0, 15.0]))
        >>> kf.predict()
        >>> kf.update(np.array([10.5, 20.3, 14.8]))
        >>> pos = kf.get_position()
        >>> vel = kf.get_velocity()
    """
    
    STATE_DIM = 6
    MEAS_DIM = 3
    
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
        
        # State vector: [px, py, pz, vx, vy, vz]
        self.x = np.zeros(self.STATE_DIM)
        
        # State covariance matrix (initialized with high uncertainty)
        self.P = np.eye(self.STATE_DIM) * 1000.0
        
        # State transition matrix (constant velocity model)
        # x_new = F @ x_old ; p_new = p + v*dt (per axis), v_new = v (per axis)
        # F = [[I_3, dt*I_3], [0, I_3]]
        self.F = np.eye(self.STATE_DIM)
        self.F[0:3, 3:6] = np.eye(3) * dt
        
        # Measurement matrix (we only observe position)
        # z = H @ x => [px, py, pz]_measured = H @ [px, py, pz, vx, vy, vz]
        # H = [I_3, 0]
        self.H = np.zeros((self.MEAS_DIM, self.STATE_DIM))
        self.H[0:3, 0:3] = np.eye(3)
        
        # Process noise covariance
        # Uses discrete white noise model for constant velocity, generalized to 3 axes
        self.Q = self._compute_process_noise(dt, process_var)
        
        # Measurement noise covariance
        self.R = np.eye(self.MEAS_DIM) * measurement_var
        
        # Track initialization state
        self.initialized = False
    
    def _compute_process_noise(self, dt: float, var: float) -> np.ndarray:
        """
        Compute process noise covariance matrix for constant-velocity model.
        
        Uses the discrete white noise acceleration model where noise
        enters as random acceleration, generalized to three independent axes:
        
            Q = q * [[dt^4/4 * I_3, dt^3/2 * I_3],
                     [dt^3/2 * I_3, dt^2   * I_3]]
        
        Args:
            dt: Time step.
            var: Process noise variance (acceleration variance), i.e. q.
        
        Returns:
            6x6 process noise covariance matrix, position-first ordering.
        """
        dt2 = dt * dt
        dt3 = dt2 * dt
        dt4 = dt3 * dt
        
        I3 = np.eye(3)
        Q = np.zeros((self.STATE_DIM, self.STATE_DIM))
        Q[0:3, 0:3] = (dt4 / 4.0) * I3
        Q[0:3, 3:6] = (dt3 / 2.0) * I3
        Q[3:6, 0:3] = (dt3 / 2.0) * I3
        Q[3:6, 3:6] = dt2 * I3
        Q *= var
        
        return Q
    
    def initialize(self, position: np.ndarray) -> None:
        """
        Initialize the filter with a first position measurement.
        
        Sets the initial state to the measured position with zero velocity.
        Resets covariance to reflect high uncertainty in velocity.
        
        Args:
            position: Initial position measurement [px, py, pz].
        """
        position = np.asarray(position).flatten()
        if position.shape[0] != 3:
            raise ValueError(f"Position must be 3D, got shape {position.shape}")
        
        # Set initial position, velocity assumed zero
        self.x[0:3] = position
        self.x[3:6] = 0.0
        
        # Reset covariance
        # Low uncertainty in position (we just measured it)
        # High uncertainty in velocity (we don't know it yet)
        self.P = np.diag([
            self.measurement_var,  # px uncertainty
            self.measurement_var,  # py uncertainty
            self.measurement_var,  # pz uncertainty
            100.0,                  # vx uncertainty (high)
            100.0,                  # vy uncertainty (high)
            100.0,                  # vz uncertainty (high)
        ])
        
        self.initialized = True
    
    def predict(self) -> np.ndarray:
        """
        Predict the next state based on constant-velocity motion model.
        
        Propagates state forward by dt using the transition matrix F
        and increases uncertainty according to process noise Q.
        
        Returns:
            Predicted state vector [px, py, pz, vx, vy, vz].
        
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
            z: Position measurement [px, py, pz].
        
        Returns:
            Updated state vector [px, py, pz, vx, vy, vz].
        
        Raises:
            RuntimeError: If filter has not been initialized.
        """
        if not self.initialized:
            raise RuntimeError("Filter must be initialized before update()")
        
        z = np.asarray(z).flatten()
        if z.shape[0] != 3:
            raise ValueError(f"Measurement must be 3D, got shape {z.shape}")
        
        # Innovation (measurement residual)
        y = z - self.H @ self.x
        
        # Innovation covariance
        S = self.H @ self.P @ self.H.T + self.R
        
        # Kalman gain: K = P @ H^T @ S^-1
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
        I_KH = np.eye(self.STATE_DIM) - K @ self.H
        self.P = I_KH @ self.P @ I_KH.T + K @ self.R @ K.T
        
        # Ensure symmetry (numerical stability)
        self.P = 0.5 * (self.P + self.P.T)
        
        return self.x.copy()
    
    def get_state(self) -> np.ndarray:
        """
        Get the current state estimate.
        
        Returns:
            State vector [px, py, pz, vx, vy, vz], shape (6,).
        """
        return self.x.copy()
    
    def get_position(self) -> np.ndarray:
        """
        Get the current estimated position.
        
        Returns:
            Position vector [px, py, pz], shape (3,).
        """
        return self.x[0:3].copy()
    
    def get_velocity(self) -> np.ndarray:
        """
        Get the current estimated velocity.
        
        Returns:
            Velocity vector [vx, vy, vz], shape (3,).
        """
        return self.x[3:6].copy()
    
    def get_covariance(self) -> np.ndarray:
        """
        Get the current state covariance matrix.
        
        Returns:
            6x6 covariance matrix P.
        """
        return self.P.copy()
    
    def get_position_uncertainty(self) -> np.ndarray:
        """
        Get the position uncertainty (standard deviation).
        
        Returns:
            Position std [sigma_px, sigma_py, sigma_pz], shape (3,).
        """
        return np.sqrt(np.diag(self.P)[0:3])
    
    def get_velocity_uncertainty(self) -> np.ndarray:
        """
        Get the velocity uncertainty (standard deviation).
        
        Returns:
            Velocity std [sigma_vx, sigma_vy, sigma_vz], shape (3,).
        """
        return np.sqrt(np.diag(self.P)[3:6])
    
    def reset(self) -> None:
        """
        Reset the filter to uninitialized state.
        
        Clears state and covariance, requiring re-initialization.
        """
        self.x = np.zeros(self.STATE_DIM)
        self.P = np.eye(self.STATE_DIM) * 1000.0
        self.initialized = False
