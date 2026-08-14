"""Environment configuration parameters."""

from dataclasses import dataclass


@dataclass
class EnvConfig:
    """Configuration for the UAV Defend environment."""
    
    # Domain: 3D engagement volume.
    #   x, y in [-L, L] (horizontal extent)
    #   z in [0, max_altitude] (altitude)
    L: float = 50.0  # Half-size of the horizontal square domain
    max_altitude: float = 30.0  # H: Upper vertical bound of the engagement volume

    # Enemy spawn altitude range (must lie within [0, max_altitude])
    enemy_spawn_altitude_min: float = 10.0
    enemy_spawn_altitude_max: float = 30.0
    
    # Time parameters
    max_steps: int = 2000  # Tmax: Maximum steps before episode ends
    dt: float = 0.5  # Time step duration (∆t)
    
    # Numerical stability
    eps: float = 1e-8  # ε: Small value for numerical stability

    def __post_init__(self) -> None:
        """Validate altitude configuration."""
        if self.enemy_spawn_altitude_min < 0:
            raise ValueError(
                f"enemy_spawn_altitude_min must be >= 0, got {self.enemy_spawn_altitude_min}"
            )
        if self.enemy_spawn_altitude_min > self.enemy_spawn_altitude_max:
            raise ValueError(
                "enemy_spawn_altitude_min must be <= enemy_spawn_altitude_max, got "
                f"{self.enemy_spawn_altitude_min} > {self.enemy_spawn_altitude_max}"
            )
        if self.enemy_spawn_altitude_max > self.max_altitude:
            raise ValueError(
                "enemy_spawn_altitude_max must be <= max_altitude, got "
                f"{self.enemy_spawn_altitude_max} > {self.max_altitude}"
            )
    
    # =========================================================================
    # Speed Configuration (realistic but training-friendly defaults)
    # - Soldier is intentionally much slower than drones (human on foot)
    # - Defender is faster than enemy so RL training remains feasible
    # - These values work well with dt=0.5 timestep for numerical stability
    # =========================================================================
    
    # Soldier movement parameters
    v_s: float = 1.5  # Soldier speed (slow human movement, much slower than drones)
    
    # Enemy drone parameters (weaving pursuit)
    v_e: float = 12.0  # Enemy drone speed (realistic and threatening)
    rho: float = 0.85  # AR(1) coefficient for weave bias persistence (lower = faster direction changes)
    sigma_a: float = 0.5  # Weave bias noise standard deviation (higher = more lateral movement)
    sigma_e: float = 0.15  # Heading noise standard deviation
    weave_amplitude: float = 1.5  # Multiplier for lateral weave component
    
    # Defender drone parameters
    v_d: float = 18.0  # Defender speed (faster than enemy, makes interception feasible for RL training)
    
    # RL reward shaping parameters
    reward_intercept: float = 100.0  # Reward for intercepting enemy (WIN)
    reward_soldier_caught: float = -100.0  # Penalty for enemy catching soldier (LOSS)
    reward_unsafe_intercept: float = -150.0  # Severe penalty for unsafe intercept (discourage risky behavior)
    reward_timeout: float = -100.0  # Penalty for timeout (failed to intercept)
    reward_progress_scale: float = 5.0  # Scale for distance progress reward (closing on enemy)
    reward_time_penalty: float = -0.05  # Small time penalty per step
    reward_tracking_scale: float = 1.0  # Scale for tracking error improvement reward (after detection)
    reward_proximity_warning: float = -0.5  # Per-step penalty when enemy is close to soldier (increased)
    
    # =========================================================================
    # Distance Thresholds (geometry parameters)
    # - detection_radius >> intercept_radius: early warning for reaction time
    # - threat_radius: immediate danger zone around soldier
    # - intercept_radius: neutralization distance (defender catches enemy)
    # - unsafe_intercept_radius: collateral risk zone around soldier
    # =========================================================================
    detection_radius: float = 15.0  # Maximum sensing range for detecting the enemy drone
    intercept_radius: float = 2.5   # Distance at which defender neutralizes/intercepts the enemy
    threat_radius: float = 2.0      # Enemy reaches soldier and causes mission failure
    unsafe_intercept_radius: float = 3.5  # If intercept occurs this close to soldier, it's a failure
    
    # =========================================================================
    # Kalman Tracking and Sensing Noise Configuration
    # - measurement_var is the single official sensor-noise parameter.
    # - After detection, a noisy enemy-position measurement is generated each step
    #   with covariance measurement_var * I for all controller modes.
    # - If use_kalman_tracking=True, Kalman filter smooths these measurements.
    # - If use_kalman_tracking=False, policies receive the raw noisy measurement.
    # =========================================================================
    use_kalman_tracking: bool = True  # If True, use Kalman filter for enemy state estimation
    process_var: float = 1.0           # Process noise variance (higher = trust measurements more)
    measurement_var: float = 0.5       # Enemy position measurement noise variance (per axis)
    lead_time: float = 0.0             # Prediction lead time for extrapolating enemy position (seconds)
