"""
Gymnasium-compatible environment for UAV defense RL training.

This is the UNIFIED environment used for BOTH:
  1. Baseline evaluation with hand-designed policies (e.g., GreedyInterceptPolicy)
  2. RL training with algorithms like PPO, SAC, TD3

The environment itself contains NO policy logic. All defender control is
provided externally via the `step(action)` interface.

Design Philosophy:
  - Environment provides dynamics and observations
  - Policy provides actions (whether scripted or learned)
  - Same environment instance works for evaluation and training
"""

from __future__ import annotations

import gymnasium as gym
import numpy as np
from gymnasium import spaces

from uav_defend.config.env_config import EnvConfig
from uav_defend.tracking import EnemyKalmanFilter


class SoldierEnv(gym.Env):
    """
    A 2D discrete-time environment for RL training with partial observability.
    
    ============================================================================
    UNIFIED ENVIRONMENT FOR BASELINE AND RL
    ============================================================================
    This environment is designed to work identically for:
      - Scripted/baseline policies (e.g., GreedyInterceptPolicy)
      - RL algorithms (e.g., PPO from Stable-Baselines3)
    
    The environment contains NO hardcoded policy logic. The defender drone
    is controlled entirely by external actions passed to step().
    
    Usage with baseline policy:
        env = SoldierEnv()
        policy = GreedyInterceptPolicy()
        obs, info = env.reset(seed=42)
        while True:
            action = policy.act(obs, info)  # Policy provides action
            obs, reward, done, _, info = env.step(action)
            if done:
                break
    
    Usage with RL (Stable-Baselines3):
        from stable_baselines3 import PPO
        env = SoldierEnv()
        model = PPO("MlpPolicy", env, verbose=1)
        model.learn(total_timesteps=100000)
    ============================================================================
    
    Domain: Ω = [-L, L]² (square centered at origin).
    
    Entities:
        - Soldier (s ∈ ℝ²): Gaussian random walk (uncontrolled)
        - Defender (d ∈ ℝ²): Controlled by external policy via 2D action
        - Enemy (e ∈ ℝ²): Weaving pursuit toward soldier (uncontrolled)
    
    Partial Observability with Shared Sensor Noise:
        - Before detection: Enemy state is MASKED (zeros)
        - After detection: a noisy enemy-position measurement is generated every step
        - If Kalman is enabled, e_hat/v_hat come from filtered estimates of that measurement
        - If Kalman is disabled, e_hat is the raw noisy measurement and v_hat is zero
        - Detection occurs when defender is within detection_radius of enemy
        - Policies never receive the true enemy state through observation channels
        - Defender motion is controlled entirely by external policy (action-driven)
    
    Observation Space (spaces.Box, shape=9, dtype=float32):
        All values normalized to [-1, 1]:
                [soldier_x, soldier_y, defender_x, defender_y, detected_flag, x1, x2, x3, x4]
        - detected_flag: 0.0 (not detected) or 1.0 (detected)
                - If use_kalman_tracking=True:
                    x1, x2 = e_hat_x, e_hat_y (Kalman estimate), x3, x4 = v_hat_x, v_hat_y
                - If use_kalman_tracking=False:
                    x1, x2 = measured enemy position, x3, x4 = defender-to-measurement vector
        - All enemy-related values are 0.0 before detection
    
    Action Space (spaces.Box, shape=2, dtype=float32):
        2D continuous action vector in [-1, 1]².
        - Normalized to unit vector if norm > eps
        - Defender displacement = v_d * dt * normalized_action
        - If action norm < eps, defender does not move
        - Action controls defender heading at ALL times
    
    Reward (dense shaping for RL):
        +100 for intercepting enemy safely (WIN)
        -100 for soldier caught (LOSS)
        -100 for unsafe intercept (intercept too close to soldier)
        -100 for timeout
        +5.0 * (prev_dist - curr_dist) for closing distance to enemy
        -0.05 per step (encourages efficiency)
    
    Termination:
        - Intercepted safely: dist_de < intercept_radius AND dist_es > unsafe_intercept_radius (WIN)
        - Soldier caught: dist_es < threat_radius (LOSS)
        - Unsafe intercept: dist_de < intercept_radius AND dist_es <= unsafe_intercept_radius (LOSS)
        - Timeout: step_count >= max_steps
    
    Gymnasium/SB3 Compatibility:
        - Passes gymnasium.utils.env_checker.check_env()
        - Compatible with Stable-Baselines3 algorithms
        - Fixed-size observation and action spaces
        - Proper reset() and step() signatures
    """
    
    metadata = {"render_modes": ["human", "rgb_array"], "render_fps": 30}
    
    def __init__(
        self,
        config: EnvConfig | None = None,
        render_mode: str | None = None,
    ):
        """
        Initialize the SoldierEnv.
        
        This environment is policy-agnostic: it accepts any 2D action vector
        for defender control, whether from a scripted baseline or RL algorithm.
        
        Args:
            config: Environment configuration. Uses defaults if None.
                    - If config.use_kalman_tracking=True: observation contains
                      Kalman estimates (e_hat, v_hat) - for RL-Kalman track
                                        - If config.use_kalman_tracking=False: observation contains
                                            noisy enemy measurements - for direct RL track
            render_mode: One of "human", "rgb_array", or None.
        
        Example:
            # Direct RL track (noisy measurements without Kalman filtering)
            config = EnvConfig(use_kalman_tracking=False)
            env = SoldierEnv(config=config)
            
            # RL-Kalman track (Kalman estimates in observations)
            config = EnvConfig(use_kalman_tracking=True)
            env = SoldierEnv(config=config)
        """
        super().__init__()
        
        self.config = config if config is not None else EnvConfig()
        self.render_mode = render_mode
        
        # =====================================================================
        # OBSERVATION SPACE (fixed-size, normalized for RL)
        # =====================================================================
        # Shape: (9,), all values in [-1, 1]
        # Layout: [soldier(2), defender(2), detected_flag(1), enemy_info(4)]
        #
        # If config.use_kalman_tracking=True (RL-Kalman track):
        #   - e_hat: Kalman estimated enemy position (zeros before detection)
        #   - v_hat: Kalman estimated enemy velocity (zeros before detection)
        #   Policy acts on ESTIMATED state, not ground truth.
        #
        # If config.use_kalman_tracking=False (direct RL track):
        #   - enemy_meas: noisy enemy position measurement (zeros before detection)
        #   - rel_to_enemy: defender→measurement vector (zeros before detection)
        #   Policy acts on measured state without filtering.
        self.observation_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(9,),
            dtype=np.float32,
        )
        
        # Track previous distance for reward shaping
        self._prev_defender_enemy_dist: float = 0.0
        
        # =====================================================================
        # ACTION SPACE (2D continuous heading vector)
        # =====================================================================
        # Shape: (2,), values in [-1, 1]
        # Interpretation: 2D direction vector for defender movement
        #   - Normalized to unit vector if norm > eps
        #   - Defender moves: displacement = v_d * dt * normalized_action
        #   - If norm < eps, defender stays stationary
        #
        # This is the same interface for both:
        #   - Scripted policies: policy.act(obs, info) -> action
        #   - RL algorithms: model.predict(obs) -> action
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(2,),
            dtype=np.float32,
        )
        
        # Internal state
        self._soldier_pos: np.ndarray | None = None
        self._defender_pos: np.ndarray | None = None
        self._enemy_pos: np.ndarray | None = None
        self._weave_bias: float = 0.0  # AR(1) lateral weave bias 'a'
        self._step_count: int = 0
        self._enemy_detected: bool = False  # Tracking state: enemy detected?
        self._np_random: np.random.Generator | None = None
        
        # Kalman filter for enemy tracking (initialized on first detection)
        self._kf: EnemyKalmanFilter | None = None
        self._e_hat: np.ndarray | None = None  # Estimated enemy position
        self._v_hat: np.ndarray | None = None  # Estimated enemy velocity
        self._enemy_measurement: np.ndarray | None = None  # Latest noisy position measurement
        self._prev_tracking_error: float | None = None  # For tracking error improvement reward
        
        # Measurement noise standard deviation for enemy position sensing
        # Derived from config measurement_var (variance = std^2)
        self._measurement_noise_std: float = np.sqrt(self.config.measurement_var)
    
    def reset(
        self,
        *,
        seed: int | None = None,
        options: dict | None = None,
    ) -> tuple[np.ndarray, dict]:
        """
        Reset the environment to an initial state.
        
        The soldier is initialized at the origin (0, 0).
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
        
        Returns:
            observation: Soldier position with shape (2,).
            info: Additional information dict.
        """
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)
        
        # Initialize soldier at the origin
        self._soldier_pos = np.array([0.0, 0.0], dtype=np.float32)
        
        # Initialize defender co-located with soldier: d0 = s0
        self._defender_pos = self._soldier_pos.copy()
        
        # Initialize enemy at random position on boundary edge
        self._enemy_pos = self._spawn_enemy_at_edge()
        
        # Initialize weave bias to zero: a0 = 0
        self._weave_bias = 0.0
        
        self._step_count = 0
        
        # Initialize detection state: enemy not detected at start
        self._enemy_detected = False
        
        # Reset Kalman filter state
        self._kf = None
        self._e_hat = None
        self._v_hat = None
        self._enemy_measurement = None
        self._prev_tracking_error = None  # For tracking error improvement reward
        
        # Calculate initial distances
        initial_enemy_soldier_dist = np.linalg.norm(self._enemy_pos - self._soldier_pos)
        initial_defender_enemy_dist = np.linalg.norm(self._defender_pos - self._enemy_pos)
        
        # Store for reward shaping
        self._prev_defender_enemy_dist = initial_defender_enemy_dist
        
        return self._get_obs(), self._get_info("ongoing", initial_enemy_soldier_dist, initial_defender_enemy_dist)
    
    def _spawn_enemy_at_edge(self) -> np.ndarray:
        """
        Spawn enemy at a random location on the boundary of [-L, L]².
        
        Uniformly choose one of four edges, then sample a coordinate
        uniformly along that edge.
        
        Returns:
            Enemy position array of shape (2,).
        """
        L = self.config.L
        edge = self._np_random.integers(0, 4)  # 0=top, 1=bottom, 2=left, 3=right
        coord = self._np_random.uniform(-L, L)  # Position along the edge
        
        if edge == 0:  # Top edge: y = L
            return np.array([coord, L], dtype=np.float32)
        elif edge == 1:  # Bottom edge: y = -L
            return np.array([coord, -L], dtype=np.float32)
        elif edge == 2:  # Left edge: x = -L
            return np.array([-L, coord], dtype=np.float32)
        else:  # Right edge: x = L
            return np.array([L, coord], dtype=np.float32)
    
    def step(
        self,
        action: np.ndarray,
    ) -> tuple[np.ndarray, float, bool, bool, dict]:
        """
        Execute one time step in the environment.
        
        This is the unified interface for both baseline and RL control.
        The environment applies the action to move the defender, then
        updates all other entities (soldier, enemy) and computes reward.
        
        Args:
            action: 2D continuous action vector in [-1, 1]², shape=(2,).
                   This can come from:
                     - Scripted policy: policy.act(obs, info) -> np.ndarray
                     - RL algorithm: model.predict(obs) -> (action, state)
                   The action is normalized to a unit vector if norm > eps.
                   If norm < eps, defender does not move.
        
        Returns:
            observation: Shape (9,) normalized state vector.
            reward: Scalar reward (dense shaping for RL training).
            terminated: True if episode ended (win/loss/timeout).
            truncated: Always False (no external truncation).
            info: Dict with additional state for debugging/visualization.
        
        Note:
            The environment does NOT contain any policy logic.
            All defender control comes from the external action.
        """
        assert self._soldier_pos is not None, "Call reset() before step()"
        
        # Move soldier stochastically (uncontrolled)
        self._move_soldier()
        
        # Move enemy with weaving pursuit toward soldier (uncontrolled)
        self._move_enemy()
        
        # Update enemy detection state and Kalman tracking estimates.
        # Called here, in step(), so that it is a first-class environment
        # behavior available to ANY controller — not tied to RL logic.
        # Uses the positions after this step's entity movements but before
        # the defender displacement, matching pre-refactor semantics.
        self._update_detection()
        
        # Move defender based on external action (controlled via step())
        self._move_defender(action)
        
        self._step_count += 1
        
        # Calculate distances
        enemy_soldier_dist = np.linalg.norm(self._enemy_pos - self._soldier_pos)
        defender_enemy_dist = np.linalg.norm(self._defender_pos - self._enemy_pos)
        defender_soldier_dist = np.linalg.norm(self._defender_pos - self._soldier_pos)
        
        # Check termination conditions using dist_de and dist_es
        dist_de = defender_enemy_dist  # defender to enemy
        dist_es = enemy_soldier_dist   # enemy to soldier
        
        # Check if defender is close enough to intercept
        can_intercept = dist_de <= self.config.intercept_radius
        
        # Check if intercept would be unsafe (too close to soldier)
        unsafe_zone = dist_es <= self.config.unsafe_intercept_radius
        
        # WIN: Safe intercept (defender catches enemy, far enough from soldier)
        intercepted = can_intercept and not unsafe_zone
        
        # LOSS: Enemy reaches soldier (threat zone)
        soldier_caught = dist_es <= self.config.threat_radius
        
        # LOSS: Unsafe intercept (caught enemy but too close to soldier - collateral risk)
        unsafe_intercept = can_intercept and unsafe_zone and not soldier_caught
        
        # Episode terminates on any terminal condition or timeout
        terminated = intercepted or soldier_caught or unsafe_intercept or (self._step_count >= self.config.max_steps)
        truncated = False
        
        # Determine outcome and reward
        if soldier_caught:
            # Check soldier_caught FIRST (highest priority failure)
            outcome = "soldier_caught"
            reward = self.config.reward_soldier_caught  # LOSS
        elif unsafe_intercept:
            # Unsafe intercept: caught enemy but too close to soldier
            outcome = "unsafe_intercept"
            reward = self.config.reward_unsafe_intercept  # LOSS
        elif intercepted:
            # Safe intercept (WIN)
            outcome = "intercepted"
            reward = self.config.reward_intercept  # WIN
        elif self._step_count >= self.config.max_steps:
            outcome = "timeout"
            reward = self.config.reward_timeout  # LOSS (failed to intercept)
        else:
            outcome = "ongoing"
            reward = 0.0
            
            # 1. Progress reward: positive for closing distance to enemy (using TRUE position)
            progress = self._prev_defender_enemy_dist - dist_de
            reward += self.config.reward_progress_scale * progress
            
            # 2. Small time penalty (encourages efficiency)
            reward += self.config.reward_time_penalty
            
            # 3. Tracking error improvement reward (only when Kalman is active and
            #    enemy is detected — meaningless otherwise since e_hat == true pos)
            if (self.config.use_kalman_tracking
                    and self._enemy_detected
                    and self._kf is not None
                    and self._e_hat is not None):
                tracking_error = float(np.linalg.norm(self._enemy_pos - self._e_hat))
                if self._prev_tracking_error is not None:
                    tracking_improvement = self._prev_tracking_error - tracking_error
                    reward += self.config.reward_tracking_scale * tracking_improvement
                self._prev_tracking_error = tracking_error
            
            # 4. Proximity warning: penalty when enemy gets close to soldier
            # Scaled by how close enemy is (inverse distance)
            proximity_threshold = self.config.unsafe_intercept_radius * 3.0  # ~10.5 units
            if dist_es < proximity_threshold:
                # Stronger penalty as enemy gets closer
                proximity_factor = 1.0 - (dist_es / proximity_threshold)
                reward += self.config.reward_proximity_warning * proximity_factor
            
            # Clip reward for numerical stability
            reward = np.clip(reward, -50.0, 50.0)
        
        # Update previous distance for next step
        self._prev_defender_enemy_dist = defender_enemy_dist
        
        return self._get_obs(), reward, terminated, truncated, self._get_info(
            outcome, enemy_soldier_dist, defender_enemy_dist
        )
    
    def _move_defender(self, action: np.ndarray) -> None:
        """
        Move the defender drone based on externally provided action.
        
        Args:
            action: 2D continuous action vector in [-1, 1]².
                   Normalized to unit vector if norm > eps.
                   Defender displacement = v_d * dt * normalized_action.
                   If action norm < eps, defender does not move.
        
        Note:
            Detection and Kalman tracking are managed independently by
            _update_detection(), called from step() before this method.
        """
        eps = self.config.eps
        
        # Action-driven defender motion
        action = np.asarray(action, dtype=np.float32)
        action_norm = np.linalg.norm(action)
        
        # If action is too small, defender doesn't move
        if action_norm < eps:
            return
        
        # Normalize action to unit vector
        direction = action / action_norm
        
        # Move defender: displacement = v_d * dt * normalized_action
        displacement = self.config.v_d * self.config.dt * direction
        new_pos = self._defender_pos + displacement
        
        # Apply reflecting boundary conditions
        new_pos = self._reflect_boundary(new_pos)
        self._defender_pos = new_pos.astype(np.float32)
    
    def _update_detection(self) -> None:
        """
        Update enemy detection state and Kalman tracking estimates.

        Called once per step() BEFORE defender displacement, after all entity
        positions have been updated. This is a standalone environment behavior:
        the results are available to any controller through info['e_hat'],
        info['v_hat'], and info['tracking_error'] — no RL wrapper required.

        Detection occurs when the defender is within detection_radius of the
        enemy. Once detected, tracking persists for the rest of the episode.

        If use_kalman_tracking=True (recommended for KalmanGreedyInterceptPolicy
        and PPO RL-Kalman):
            - EnemyKalmanFilter is initialized on first detection.
            - predict() + update() called every subsequent step.
            - e_hat and v_hat carry filtered estimates with proper uncertainty.
            - tracking_error in info is the Euclidean filter error.

        If use_kalman_tracking=False (GreedyInterceptPolicy, Direct RL PPO):
            - e_hat is set to the noisy enemy measurement (no filtering).
            - v_hat is zeros (velocity not observable without a filter).
            - tracking_error in info is the measurement error magnitude.
        """
        # Check for detection (only if not already detected)
        if not self._enemy_detected:
            defender_enemy_dist = np.linalg.norm(self._enemy_pos - self._defender_pos)
            if defender_enemy_dist <= self.config.detection_radius:
                self._enemy_detected = True  # Start tracking!
                self._enemy_measurement = (
                    self._enemy_pos
                    + self._np_random.normal(0.0, self._measurement_noise_std, size=(2,))
                ).astype(np.float32)
                
                if self.config.use_kalman_tracking:
                    # Initialize Kalman filter with first noisy measurement
                    self._kf = EnemyKalmanFilter(
                        dt=self.config.dt,
                        process_var=self.config.process_var,
                        measurement_var=self.config.measurement_var,
                    )
                    self._kf.initialize(self._enemy_measurement.astype(np.float64))
                    self._e_hat = self._kf.get_position().astype(np.float32)
                    self._v_hat = self._kf.get_velocity().astype(np.float32)
                else:
                    # No Kalman: use raw noisy measurement directly.
                    self._e_hat = self._enemy_measurement.copy()
                    self._v_hat = np.zeros(2, dtype=np.float32)  # Velocity unknown without filter
        else:
            # Already detected: update tracking
            self._enemy_measurement = (
                self._enemy_pos
                + self._np_random.normal(0.0, self._measurement_noise_std, size=(2,))
            ).astype(np.float32)
            if self.config.use_kalman_tracking and self._kf is not None:
                # Kalman predict + update with noisy measurement
                self._kf.predict()
                self._kf.update(self._enemy_measurement.astype(np.float64))
                
                # Apply lead time prediction if configured
                if self.config.lead_time > 0.0:
                    # Extrapolate position forward by lead_time
                    pos = self._kf.get_position()
                    vel = self._kf.get_velocity()
                    self._e_hat = (pos + vel * self.config.lead_time).astype(np.float32)
                else:
                    self._e_hat = self._kf.get_position().astype(np.float32)
                self._v_hat = self._kf.get_velocity().astype(np.float32)
            else:
                # No Kalman: expose raw noisy measurement directly.
                # v_hat is not available without a filter; keep as zeros.
                self._e_hat = self._enemy_measurement.copy()
                self._v_hat = np.zeros(2, dtype=np.float32)
    
    def _move_soldier(self) -> None:
        """
        Move the soldier with a true Gaussian random walk.
        
        - Sample 2D Gaussian displacement with scale σ = v_s * dt
        - Variable step magnitude (Gaussian-distributed)
        - Reflecting boundary conditions at [-L, L]²
        """
        # True Gaussian random walk: sample displacement directly
        # Scale (standard deviation) determines typical step size
        sigma = self.config.v_s * self.config.dt
        displacement = self._np_random.normal(loc=0.0, scale=sigma, size=(2,))
        
        # Update position
        new_pos = self._soldier_pos + displacement
        
        # Reflecting boundary conditions at [-L, L]²
        new_pos = self._reflect_boundary(new_pos)
        
        self._soldier_pos = new_pos.astype(np.float32)
    
    def _move_enemy(self) -> None:
        """
        Move the enemy with a stochastic weaving pursuit policy.
        
        Weaving pursuit dynamics:
        1. Compute vector to soldier: r = s - e
        2. Unit vector toward soldier: r_hat = r / (||r|| + eps)
        3. Perpendicular unit vector: r_perp = [-r_hat[1], r_hat[0]]
        4. Update weave bias (AR(1)): a <- rho * a + sigma_a * eta, eta ~ N(0,1)
        5. Heading noise: z ~ N(0, I_2)
        6. Unnormalized direction: u_raw = r_hat + a * r_perp + sigma_e * z
        7. Normalize: u = u_raw / (||u_raw|| + eps)
        8. Move: e_next = e + v_e * dt * u
        9. Apply reflecting boundary conditions
        """
        eps = self.config.eps
        
        # Vector from enemy to soldier
        r = self._soldier_pos - self._enemy_pos
        r_norm = np.linalg.norm(r)
        
        # Unit vector toward soldier
        if r_norm > eps:
            r_hat = r / r_norm
        else:
            # Enemy is on top of soldier, pick random direction
            angle = self._np_random.uniform(0, 2 * np.pi)
            r_hat = np.array([np.cos(angle), np.sin(angle)])
        
        # Perpendicular unit vector (90° CCW rotation)
        r_perp = np.array([-r_hat[1], r_hat[0]])
        
        # Update weave bias with AR(1) process: a <- rho * a + sigma_a * eta
        eta = self._np_random.normal(0.0, 1.0)
        self._weave_bias = self.config.rho * self._weave_bias + self.config.sigma_a * eta
        
        # Heading noise
        z = self._np_random.normal(0.0, 1.0, size=(2,))
        
        # Unnormalized direction with weave amplitude multiplier
        weave_component = self._weave_bias * self.config.weave_amplitude * r_perp
        u_raw = r_hat + weave_component + self.config.sigma_e * z
        
        # Normalize direction
        u_norm = np.linalg.norm(u_raw)
        if u_norm > eps:
            u = u_raw / u_norm
        else:
            u = r_hat  # Fallback to direct pursuit
        
        # Move enemy
        displacement = self.config.v_e * self.config.dt * u
        new_pos = self._enemy_pos + displacement
        
        # Apply reflecting boundary conditions
        new_pos = self._reflect_boundary(new_pos)
        
        self._enemy_pos = new_pos.astype(np.float32)
    
    def _reflect_boundary(self, pos: np.ndarray) -> np.ndarray:
        """
        Apply reflecting boundary conditions at [-L, L]².
        
        If position exceeds boundary, reflect it back into domain.
        
        Args:
            pos: Position array of shape (2,).
        
        Returns:
            Reflected position within [-L, L]².
        """
        L = self.config.L
        
        for i in range(2):
            # Reflect until within bounds (handles multiple reflections)
            while pos[i] < -L or pos[i] > L:
                if pos[i] < -L:
                    pos[i] = -2 * L - pos[i]  # Reflect off lower boundary
                if pos[i] > L:
                    pos[i] = 2 * L - pos[i]  # Reflect off upper boundary
        
        return pos
    
    def _get_obs(self) -> np.ndarray:
        """
        Get normalized observation for RL with partial observability.
        
        The observation format depends on config.use_kalman_tracking:
        
        Returns:
            obs: Array of shape (9,), all values normalized to [-1, 1].
            
            If config.use_kalman_tracking=True (RL-Kalman track):
                [soldier_x, soldier_y, defender_x, defender_y, detected_flag,
                 e_hat_x, e_hat_y, v_hat_x, v_hat_y]
                - e_hat: Kalman estimated enemy position (normalized by L)
                - v_hat: Kalman estimated enemy velocity (normalized by v_e)
                - All enemy info is zeros before detection
                - Policy acts on ESTIMATED state, not ground truth
                
            If config.use_kalman_tracking=False (direct RL track):
                [soldier_x, soldier_y, defender_x, defender_y, detected_flag,
                 meas_x, meas_y, rel_x, rel_y]
                - meas: noisy enemy position measurement (normalized by L)
                - rel: relative position from defender to measurement (normalized by L)
                - All enemy info is zeros before detection
                - Policy acts on measured state without filtering
        """
        L = self.config.L
        v_e = self.config.v_e
        
        # Normalize positions to [-1, 1]
        soldier_norm = self._soldier_pos / L
        defender_norm = self._defender_pos / L
        
        # Detection flag
        detected_flag = np.array([1.0 if self._enemy_detected else 0.0])
        
        if self.config.use_kalman_tracking:
            # RL-KALMAN TRACK: Use Kalman filter estimates
            # Policy acts on estimated state, not ground truth
            if self._enemy_detected and self._e_hat is not None:
                e_hat_norm = np.clip(self._e_hat / L, -1.0, 1.0)
                v_hat_norm = np.clip(self._v_hat / v_e, -1.0, 1.0)
            else:
                e_hat_norm = np.array([0.0, 0.0])
                v_hat_norm = np.array([0.0, 0.0])
            
            return np.concatenate([
                soldier_norm, 
                defender_norm,
                detected_flag,
                e_hat_norm,
                v_hat_norm
            ]).astype(np.float32)
        else:
            # DIRECT RL TRACK: Use raw noisy measurement
            if self._enemy_detected and self._enemy_measurement is not None:
                enemy_norm = self._enemy_measurement / L
                defender_to_enemy = (self._enemy_measurement - self._defender_pos) / L
                defender_to_enemy = np.clip(defender_to_enemy, -1.0, 1.0)
            else:
                enemy_norm = np.array([0.0, 0.0])
                defender_to_enemy = np.array([0.0, 0.0])
            
            return np.concatenate([
                soldier_norm, 
                defender_norm,
                detected_flag,
                enemy_norm,
                defender_to_enemy
            ]).astype(np.float32)
    
    def _get_info(self, outcome: str = "ongoing", enemy_soldier_dist: float = 0.0, 
                  defender_enemy_dist: float = 0.0) -> dict:
        """Get additional info dict.
        
        Args:
            outcome: Episode outcome - "ongoing", "intercepted", "soldier_caught", 
                    "unsafe_intercept", or "timeout".
            enemy_soldier_dist: Distance between enemy and soldier.
            defender_enemy_dist: Distance between defender and enemy.
        """
        # Check if current state is in unsafe intercept zone
        unsafe_zone = enemy_soldier_dist <= self.config.unsafe_intercept_radius
        
        # Compute tracking error whenever an estimated/measured enemy position is available.
        if self._enemy_detected and self._e_hat is not None:
            tracking_error = float(np.linalg.norm(self._enemy_pos - self._e_hat))
        else:
            tracking_error = None
        
        return {
            "step_count": self._step_count,
            "soldier_pos": self._soldier_pos.copy(),
            "defender_pos": self._defender_pos.copy(),
            "enemy_pos": self._enemy_pos.copy(),
            "weave_bias": self._weave_bias,
            "enemy_soldier_dist": enemy_soldier_dist,
            "defender_enemy_dist": defender_enemy_dist,
            "enemy_detected": self._enemy_detected,  # Use stored tracking state
            "unsafe_intercept": unsafe_zone,  # True if enemy is in unsafe zone around soldier
            "outcome": outcome,
            # Kalman filter tracking info
            "detected": self._enemy_detected,
            "enemy_measurement": (
                self._enemy_measurement.copy() if self._enemy_measurement is not None else None
            ),
            "e_hat": self._e_hat.copy() if self._e_hat is not None else None,
            "v_hat": self._v_hat.copy() if self._v_hat is not None else None,
            "tracking_error": tracking_error,
        }
    
    def render(self) -> np.ndarray | None:
        """Render the environment (placeholder for future implementation)."""
        if self.render_mode == "rgb_array":
            # Return a simple representation (to be expanded later)
            return self._render_frame()
        return None
    
    def _render_frame(self) -> np.ndarray:
        """Render a frame as an RGB array."""
        # Simple placeholder: 100x100 black image with entities
        size = 100
        frame = np.zeros((size, size, 3), dtype=np.uint8)
        L = self.config.L
        
        # Draw soldier as blue square
        if self._soldier_pos is not None:
            x = int((self._soldier_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._soldier_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [0, 0, 255]  # Blue
        
        # Draw defender as green square
        if self._defender_pos is not None:
            x = int((self._defender_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._defender_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-1, 2):
                for dy in range(-1, 2):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [0, 255, 0]  # Green
        
        # Draw enemy as red square
        if self._enemy_pos is not None:
            x = int((self._enemy_pos[0] + L) / (2 * L) * (size - 1))
            y = int((self._enemy_pos[1] + L) / (2 * L) * (size - 1))
            x = np.clip(x, 0, size - 1)
            y = np.clip(y, 0, size - 1)
            for dx in range(-2, 3):
                for dy in range(-2, 3):
                    px, py = x + dx, y + dy
                    if 0 <= px < size and 0 <= py < size:
                        frame[py, px] = [255, 0, 0]  # Red
        
        return frame
    
    def close(self) -> None:
        """Clean up resources."""
        pass
