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
    A three-dimensional constrained point-mass engagement environment for RL
    training with partial observability.

    NOTE: This is a 3D constrained point-mass UAV dynamics simulator. Both
    aircraft (defender and hostile UAV) maintain a persistent 3D velocity
    that approaches a controller/guidance-commanded desired velocity subject
    to bounded acceleration, a bounded horizontal turn rate, a maximum total
    speed, and bounded climb/descent rates. This is NOT six-DOF or
    aerodynamic flight dynamics, and the parameter defaults are provisional
    simulation values rather than validated hardware specifications.
    
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
    
    Domain: 3D engagement volume, x, y ∈ [-L, L] (horizontal) and z ∈ [0, H]
    (altitude), where H = max_altitude.
    
    Entities:
        - Soldier (s ∈ ℝ³): Planar Gaussian random walk (uncontrolled); always
          remains a ground entity with s_z = 0 at all times.
        - Defender (d ∈ ℝ³): Persistent 3D velocity that tracks a
          controller-commanded desired velocity subject to dynamic limits
          (acceleration, turn rate, climb/descent rate, max speed).
        - Enemy (e ∈ ℝ³): Persistent 3D velocity that tracks a weaving
          pursuit-commanded desired velocity, subject to the same class of
          dynamic limits; naturally descends toward the ground target while
          pursuing.
    
    Partial Observability with Shared Sensor Noise:
        - Before detection: Enemy state is MASKED (zeros)
        - After detection: a noisy enemy-position measurement is generated every step
        - If Kalman is enabled, e_hat/v_hat come from filtered estimates of that measurement
        - If Kalman is disabled, e_hat is the raw noisy measurement and v_hat is zero
        - Detection occurs when defender is within detection_radius of enemy (3D distance)
        - Policies never receive the true enemy state through observation channels
        - Defender motion is controlled entirely by external policy (action-driven)
    
    Observation Space (spaces.Box, shape=16, dtype=float32):
        All values normalized to [-1, 1]:
            [soldier_x, soldier_y, soldier_z,
             defender_x, defender_y, defender_z,
             defender_vx, defender_vy, defender_vz,
             detected_flag,
             enemy_info_1..3, enemy_info_4..6]
        - defender velocity is normalized by v_d (component-wise)
        - detected_flag: 0.0 (not detected) or 1.0 (detected)
                - If use_kalman_tracking=True:
                    enemy_info_1..3 = e_hat (Kalman estimated position),
                    enemy_info_4..6 = v_hat (Kalman estimated velocity)
                - If use_kalman_tracking=False:
                    enemy_info_1..3 = raw noisy enemy position measurement,
                    enemy_info_4..6 = defender-to-measurement relative vector
        - All enemy-related values are 0.0 before detection
    
    Action Space (spaces.Box, shape=3, dtype=float32):
        3D continuous action vector in [-1, 1]³, interpreted as a DESIRED
        velocity direction (not an instantaneous displacement command):
        - Normalized to unit vector if norm > eps -> desired_velocity = v_d * direction
        - If action norm <= eps, desired_velocity = [0,0,0] (command to stop)
        - The environment's constrained-velocity dynamics determine how
          quickly the defender's actual persistent velocity can turn,
          accelerate, or decelerate toward that desired velocity.
    
    Reward (dense shaping for RL):
        +100 for intercepting enemy safely (WIN)
        -100 for soldier caught (LOSS)
        -100 for unsafe intercept (intercept too close to soldier)
        -100 for timeout
        +5.0 * (prev_dist - curr_dist) for closing distance to enemy
        -0.05 per step (encourages efficiency)
    
    Termination (all distances are true 3D Euclidean distances):
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
        
        This environment is policy-agnostic: it accepts any 3D action vector
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
        # Shape: (16,), all values in [-1, 1]
        # Layout: [soldier(3), defender(3), defender_vel(3), detected_flag(1), enemy_info(6)]
        #
        # defender_vel is normalized by v_d (component-wise); since total
        # defender speed is dynamically constrained to <= v_d, each component
        # remains within [-1, 1].
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
            shape=(16,),
            dtype=np.float32,
        )
        
        # Track previous distance for reward shaping
        self._prev_defender_enemy_dist: float = 0.0
        
        # =====================================================================
        # ACTION SPACE (3D desired-velocity direction)
        # =====================================================================
        # Shape: (3,), values in [-1, 1]
        # Interpretation: 3D DESIRED velocity direction for the defender.
        #   - Normalized to unit vector if norm > eps -> desired_velocity = v_d * direction
        #   - If norm <= eps, desired_velocity = [0,0,0] (command to stop)
        #   - The defender's actual persistent velocity approaches this
        #     desired velocity subject to bounded acceleration, bounded
        #     horizontal turn rate, and bounded climb/descent rate. The
        #     action does NOT instantaneously set the defender's velocity.
        #
        # This is the same interface for both:
        #   - Scripted policies: policy.act(obs, info) -> action
        #   - RL algorithms: model.predict(obs) -> action
        self.action_space = spaces.Box(
            low=-1.0,
            high=1.0,
            shape=(3,),
            dtype=np.float32,
        )
        
        # Internal state
        self._soldier_pos: np.ndarray | None = None
        self._defender_pos: np.ndarray | None = None
        self._enemy_pos: np.ndarray | None = None
        self._defender_vel: np.ndarray | None = None  # Persistent 3D velocity [vx,vy,vz], m/s
        self._enemy_vel: np.ndarray | None = None      # Persistent 3D velocity [vx,vy,vz], m/s
        self._defender_dynamics_info: dict = {}  # Per-step dynamics diagnostics (see _advance_velocity)
        self._enemy_dynamics_info: dict = {}
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
        
        The soldier is initialized at the origin (0, 0, 0) and remains a
        ground entity (z = 0) for the entire episode.
        
        Args:
            seed: Random seed for reproducibility.
            options: Additional options (unused).
        
        Returns:
            observation: Shape (16,) normalized state vector.
            info: Additional information dict.
        """
        super().reset(seed=seed)
        self._np_random = np.random.default_rng(seed)
        
        # Initialize soldier at the origin, on the ground (z = 0)
        self._soldier_pos = np.array([0.0, 0.0, 0.0], dtype=np.float32)
        
        # Initialize defender co-located with soldier: d0 = s0 = [x, y, 0]
        self._defender_pos = self._soldier_pos.copy()
        
        # Defender begins stationary: it must accelerate from rest rather
        # than instantly travel at v_d.
        self._defender_vel = np.zeros(3, dtype=np.float32)
        
        # Initialize enemy at random position on boundary edge
        self._enemy_pos = self._spawn_enemy_at_edge()
        
        # The hostile UAV represents an aircraft already entering the
        # engagement volume: initialize its velocity approximately toward
        # the soldier at its maximum speed, rather than from rest, to avoid
        # an artificial launch-from-rest transient.
        pursuit_dir = self._soldier_pos - self._enemy_pos
        pursuit_norm = np.linalg.norm(pursuit_dir)
        if pursuit_norm > self.config.eps:
            pursuit_dir = pursuit_dir / pursuit_norm
        else:
            pursuit_dir = np.array([1.0, 0.0, 0.0])  # Safe fallback (numerically degenerate case)
        initial_enemy_vel = self.config.v_e * pursuit_dir
        # Respect the vertical-rate invariant maintained everywhere else:
        # clip the initial vertical component to [-max_descent, max_climb]
        # (this can only reduce the total initial speed slightly below v_e,
        # never increase it, so the max-speed bound also remains satisfied).
        initial_enemy_vel[2] = np.clip(
            initial_enemy_vel[2],
            -self.config.enemy_max_descent_rate,
            self.config.enemy_max_climb_rate,
        )
        self._enemy_vel = initial_enemy_vel.astype(np.float32)
        
        # Reset per-step dynamics diagnostics (see _advance_velocity)
        self._defender_dynamics_info = {
            "accel_used": 0.0,
            "turn_rate_used_deg": 0.0,
            "accel_saturated": False,
            "turn_saturated": False,
            "climb_saturated": False,
        }
        self._enemy_dynamics_info = dict(self._defender_dynamics_info)
        
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
        Spawn enemy at a random location on the vertical lateral boundary of
        the 3D engagement volume, i.e. on the perimeter of the horizontal
        square [-L, L]² at a randomly sampled altitude.
        
        Uniformly choose one of four horizontal edges (x = ±L or y = ±L), then
        sample the other horizontal coordinate uniformly along that edge, and
        independently sample altitude z ~ Uniform(enemy_spawn_altitude_min,
        enemy_spawn_altitude_max).
        
        Returns:
            Enemy position array of shape (3,).
        """
        L = self.config.L
        edge = self._np_random.integers(0, 4)  # 0=top, 1=bottom, 2=left, 3=right
        coord = self._np_random.uniform(-L, L)  # Position along the edge
        z = self._np_random.uniform(
            self.config.enemy_spawn_altitude_min,
            self.config.enemy_spawn_altitude_max,
        )
        
        if edge == 0:  # Top edge: y = L
            return np.array([coord, L, z], dtype=np.float32)
        elif edge == 1:  # Bottom edge: y = -L
            return np.array([coord, -L, z], dtype=np.float32)
        elif edge == 2:  # Left edge: x = -L
            return np.array([-L, coord, z], dtype=np.float32)
        else:  # Right edge: x = L
            return np.array([L, coord, z], dtype=np.float32)
    
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
            action: 3D continuous action vector in [-1, 1]³, shape=(3,),
                   interpreted as a DESIRED velocity direction (see class
                   docstring). This can come from:
                     - Scripted policy: policy.act(obs, info) -> np.ndarray
                     - RL algorithm: model.predict(obs) -> (action, state)
                   The action is normalized to a unit vector if norm > eps.
                   If norm <= eps, the desired velocity is [0,0,0] (a command
                   to decelerate toward a stop, not an instantaneous stop).
        
        Returns:
            observation: Shape (16,) normalized state vector.
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
    
    def _advance_velocity(
        self,
        current_velocity: np.ndarray,
        desired_velocity: np.ndarray,
        max_speed: float,
        max_accel: float,
        max_turn_rate_rad: float,
        max_climb_rate: float,
        max_descent_rate: float,
    ) -> tuple[np.ndarray, dict]:
        """
        Advance a persistent 3D point-mass velocity toward a desired velocity,
        subject to a maximum horizontal turn rate, maximum acceleration,
        maximum total speed, and maximum climb/descent rate.

        This is the single reusable constrained-velocity update shared by
        both the defender and the hostile UAV (a 3D constrained point-mass
        dynamics model — NOT six-DOF or aerodynamic flight dynamics).

        Order of operations:
          1. Vertical-rate command limiting: clip desired_velocity[2] into
             [-max_descent_rate, max_climb_rate].
          2. Horizontal turn-rate limiting: rotate the current horizontal
             heading toward the desired horizontal heading by at most
             max_turn_rate_rad * dt (shortest signed angular path, wrapped
             to [-pi, pi]), adopting the desired horizontal speed magnitude.
             Special cases (documented explicitly, not merely defaulted):
               - If current horizontal speed <= eps, the previous heading is
                 undefined; heading initializes directly toward the desired
                 heading with NO rate limit applied (there is nothing to
                 rate-limit against). Acceleration limiting (step 3) still
                 governs how quickly speed itself can rise.
               - If desired horizontal speed <= eps (e.g. a "stop" command),
                 heading has no defined target; the current heading is held
                 while horizontal speed is commanded toward zero.
             The result is the "dynamically permissible target velocity".
          3. Acceleration limiting: clip the change from current_velocity to
             the target velocity to at most max_accel * dt in magnitude,
             preserving the direction of the change. Because this is
             equivalent to next = current + t*(target - current) for some
             t in [0, 1], the result is a convex combination of current and
             target.
          4. Total-speed / vertical-rate safety enforcement: since both
             current_velocity and the target velocity already satisfy the
             max_speed ball and the climb/descent bounds (both are convex
             constraints), and step 3 produces a convex combination of the
             two, the result satisfies both bounds automatically. A
             defensive clip is applied regardless, to guard against
             floating-point overshoot (never used to silently paper over a
             design error).

        Args:
            current_velocity: Current persistent velocity [vx, vy, vz], shape (3,).
            desired_velocity: Controller/guidance-commanded desired velocity
                [vx, vy, vz], shape (3,); ||desired_velocity|| is expected to
                be <= max_speed (e.g. max_speed * unit_direction).
            max_speed: Maximum total speed (m/s).
            max_accel: Maximum acceleration magnitude (m/s^2).
            max_turn_rate_rad: Maximum horizontal heading-rate magnitude (rad/s).
            max_climb_rate: Maximum upward vertical speed (m/s), positive.
            max_descent_rate: Maximum downward vertical speed magnitude (m/s), positive.

        Returns:
            (next_velocity, diagnostics):
                next_velocity: Constrained velocity, shape (3,), float32.
                diagnostics: dict with keys "accel_used" (m/s^2, actually
                    applied this step), "turn_rate_used_deg" (deg/s, actually
                    applied this step), "accel_saturated" (bool),
                    "turn_saturated" (bool), "climb_saturated" (bool, True if
                    the vertical-rate command limit in step 1 was active).
        """
        dt = self.config.dt
        eps = self.config.eps
        
        current_velocity = np.asarray(current_velocity, dtype=np.float64)
        desired_velocity = np.asarray(desired_velocity, dtype=np.float64)
        
        # 1. Vertical-rate command limiting
        desired_vz = desired_velocity[2]
        desired_vz_clipped = float(np.clip(desired_vz, -max_descent_rate, max_climb_rate))
        climb_saturated = abs(desired_vz_clipped - desired_vz) > 1e-9
        
        # 2. Horizontal turn-rate limiting
        cur_h = current_velocity[0:2]
        des_h = desired_velocity[0:2]
        ch = float(np.linalg.norm(cur_h))
        dh = float(np.linalg.norm(des_h))
        
        turn_rate_used_rad = 0.0
        turn_saturated = False
        
        if dh <= eps:
            # Desired horizontal command is (near) zero: hold current
            # heading, command zero horizontal speed (bounded braking).
            target_heading = np.arctan2(cur_h[1], cur_h[0]) if ch > eps else 0.0
            target_h_mag = 0.0
        elif ch <= eps:
            # Current horizontal velocity undefined (at/near rest):
            # initialize heading directly toward the desired heading; no
            # rate limit applies to an undefined previous heading.
            target_heading = np.arctan2(des_h[1], des_h[0])
            target_h_mag = dh
        else:
            psi_current = np.arctan2(cur_h[1], cur_h[0])
            psi_desired = np.arctan2(des_h[1], des_h[0])
            delta_psi = psi_desired - psi_current
            delta_psi = (delta_psi + np.pi) % (2 * np.pi) - np.pi  # wrap to [-pi, pi]
            max_delta = max_turn_rate_rad * dt
            if abs(delta_psi) > max_delta:
                delta_psi = np.sign(delta_psi) * max_delta
                turn_saturated = True
            target_heading = psi_current + delta_psi
            target_h_mag = dh
            turn_rate_used_rad = abs(delta_psi) / dt if dt > 0 else 0.0
        
        target_velocity = np.array([
            target_h_mag * np.cos(target_heading),
            target_h_mag * np.sin(target_heading),
            desired_vz_clipped,
        ])
        
        # 3. Acceleration limiting (bounded convex-combination step toward target)
        delta_v = target_velocity - current_velocity
        delta_v_norm = float(np.linalg.norm(delta_v))
        max_delta_v = max_accel * dt
        accel_saturated = delta_v_norm > max_delta_v
        if accel_saturated and delta_v_norm > eps:
            delta_v = delta_v / delta_v_norm * max_delta_v
        next_velocity = current_velocity + delta_v
        accel_used = float(np.linalg.norm(delta_v)) / dt if dt > 0 else 0.0
        
        # 4. Total-speed / vertical-rate safety enforcement (defensive clip)
        speed = float(np.linalg.norm(next_velocity))
        if speed > max_speed and speed > eps:
            next_velocity = next_velocity / speed * max_speed
        next_velocity[2] = np.clip(next_velocity[2], -max_descent_rate, max_climb_rate)
        
        diagnostics = {
            "accel_used": accel_used,
            "turn_rate_used_deg": float(np.degrees(turn_rate_used_rad)),
            "accel_saturated": bool(accel_saturated),
            "turn_saturated": bool(turn_saturated),
            "climb_saturated": bool(climb_saturated),
        }
        
        return next_velocity.astype(np.float32), diagnostics
    
    def _apply_boundary(self, pos: np.ndarray, vel: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """
        Clip position to the 3D engagement volume (x, y \u2208 [-L, L], z \u2208 [0, H])
        and zero ONLY the outward-normal velocity component at any boundary
        the position has reached. Tangential velocity components are
        preserved unchanged (no bounce/reflection for persistent-velocity
        vehicles).
        
        For example, at z = 0 with vz < 0 (descending into the ground),
        vz is set to 0 while vx, vy are untouched. This prevents outward
        velocity from continuing to accumulate against a boundary while the
        vehicle is resting against it.
        
        Args:
            pos: Position array of shape (3,).
            vel: Persistent velocity array of shape (3,).
        
        Returns:
            (clipped_pos, clipped_vel), each of shape (3,).
        """
        L = self.config.L
        H = self.config.max_altitude
        pos = pos.copy()
        vel = vel.copy()
        
        if pos[0] > L:
            pos[0] = L
            if vel[0] > 0:
                vel[0] = 0.0
        elif pos[0] < -L:
            pos[0] = -L
            if vel[0] < 0:
                vel[0] = 0.0
        
        if pos[1] > L:
            pos[1] = L
            if vel[1] > 0:
                vel[1] = 0.0
        elif pos[1] < -L:
            pos[1] = -L
            if vel[1] < 0:
                vel[1] = 0.0
        
        if pos[2] > H:
            pos[2] = H
            if vel[2] > 0:
                vel[2] = 0.0
        elif pos[2] < 0.0:
            pos[2] = 0.0
            if vel[2] < 0:
                vel[2] = 0.0
        
        return pos, vel
    
    def _move_defender(self, action: np.ndarray) -> None:
        """
        Move the defender drone using constrained 3D point-mass dynamics.
        
        Args:
            action: 3D continuous action vector in [-1, 1]³, interpreted as
                   a DESIRED velocity direction. Normalized to a unit vector
                   if norm > eps -> desired_velocity = v_d * direction. If
                   action norm <= eps, desired_velocity = [0,0,0] (command
                   to decelerate toward a stop).
        
        The defender's persistent velocity (self._defender_vel) approaches
        this desired velocity subject to defender_max_accel,
        defender_max_turn_rate_deg, defender_max_climb_rate, and
        defender_max_descent_rate (see _advance_velocity). Position is then
        integrated as p_{t+1} = p_t + v_{t+1} * dt, followed by physically
        consistent boundary handling (_apply_boundary): outward velocity is
        zeroed only if the position actually reached a boundary.
        
        Note:
            Detection and Kalman tracking are managed independently by
            _update_detection(), called from step() before this method.
        """
        eps = self.config.eps
        
        action = np.asarray(action, dtype=np.float32)
        action_norm = np.linalg.norm(action)
        
        if action_norm > eps:
            desired_direction = action / action_norm
            desired_velocity = self.config.v_d * desired_direction
        else:
            desired_velocity = np.zeros(3, dtype=np.float32)
        
        next_vel, diag = self._advance_velocity(
            current_velocity=self._defender_vel,
            desired_velocity=desired_velocity,
            max_speed=self.config.v_d,
            max_accel=self.config.defender_max_accel,
            max_turn_rate_rad=np.radians(self.config.defender_max_turn_rate_deg),
            max_climb_rate=self.config.defender_max_climb_rate,
            max_descent_rate=self.config.defender_max_descent_rate,
        )
        self._defender_dynamics_info = diag
        
        new_pos = self._defender_pos + next_vel * self.config.dt
        new_pos, next_vel = self._apply_boundary(new_pos, next_vel)
        
        self._defender_pos = new_pos.astype(np.float32)
        self._defender_vel = next_vel.astype(np.float32)
    
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
                    + self._np_random.normal(0.0, self._measurement_noise_std, size=(3,))
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
                    self._v_hat = np.zeros(3, dtype=np.float32)  # Velocity unknown without filter
        else:
            # Already detected: update tracking
            self._enemy_measurement = (
                self._enemy_pos
                + self._np_random.normal(0.0, self._measurement_noise_std, size=(3,))
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
                self._v_hat = np.zeros(3, dtype=np.float32)
    
    def _move_soldier(self) -> None:
        """
        Move the soldier with a true Gaussian random walk.
        
        The soldier remains a ground entity: only x, y are perturbed and
        z is held fixed at 0.0 at all times (no vertical motion).
        
        - Sample 2D (horizontal) Gaussian displacement with scale σ = v_s * dt
        - Variable step magnitude (Gaussian-distributed)
        - Reflecting boundary conditions at [-L, L]² (horizontal only)
        """
        # True Gaussian random walk: sample horizontal displacement directly
        # Scale (standard deviation) determines typical step size
        sigma = self.config.v_s * self.config.dt
        displacement_xy = self._np_random.normal(loc=0.0, scale=sigma, size=(2,))
        displacement = np.array(
            [displacement_xy[0], displacement_xy[1], 0.0], dtype=np.float32
        )
        
        # Update position
        new_pos = self._soldier_pos + displacement
        
        # Reflecting boundary conditions at [-L, L]² × [0, H]
        new_pos = self._reflect_boundary(new_pos)
        
        # Soldier is a ground entity: enforce z = 0 explicitly
        new_pos[2] = 0.0
        
        self._soldier_pos = new_pos.astype(np.float32)
    
    def _move_enemy(self) -> None:
        """
        Move the enemy with a stochastic weaving pursuit policy in 3D,
        using constrained point-mass dynamics.
        
        Weaving pursuit GUIDANCE (unchanged from the 3D-foundation task):
        1. Compute vector to soldier: r = s - e (full 3D; since the soldier
           is always at z=0, this naturally drives the enemy to descend
           toward the ground target as it pursues)
        2. Unit vector toward soldier: r_hat = r / (||r|| + eps)
        3. Horizontal lateral unit vector: lateral = [-r_hat_y, r_hat_x, 0],
           normalized; falls back to a fixed horizontal direction if the
           pursuit direction is (near-)vertical and the horizontal
           component is degenerate (norm <= eps)
        4. Update weave bias (AR(1)): a <- rho * a + sigma_a * eta, eta ~ N(0,1)
        5. Heading noise: z ~ N(0, I_3) (generalized to 3 dimensions)
        6. Unnormalized direction: u_raw = r_hat + a * lateral + sigma_e * z
        7. Normalize: u = u_raw / (||u_raw|| + eps)
        
        DYNAMICS (new in this task): guidance now produces a DESIRED
        velocity direction, not instantaneous motion. desired_velocity =
        v_e * u is fed through the same reusable constrained-velocity update
        used by the defender (_advance_velocity), subject to
        enemy_max_accel, enemy_max_turn_rate_deg, enemy_max_climb_rate, and
        enemy_max_descent_rate. Position is then integrated as
        p_{t+1} = p_t + v_{t+1} * dt, followed by physically consistent
        boundary handling (_apply_boundary).
        
        Note: vertical weaving is NOT modeled here; altitude change results
        only from 3D pursuit of the ground-level soldier target. Hostile
        evasion, proportional navigation, and lead pursuit are NOT modeled
        here either (deferred to a later task).
        """
        eps = self.config.eps
        
        # Vector from enemy to soldier (full 3D)
        r = self._soldier_pos - self._enemy_pos
        r_norm = np.linalg.norm(r)
        
        # Unit vector toward soldier
        if r_norm > eps:
            r_hat = r / r_norm
        else:
            # Enemy is on top of soldier, pick a random 3D unit direction
            rand_vec = self._np_random.normal(0.0, 1.0, size=(3,))
            rand_norm = np.linalg.norm(rand_vec)
            if rand_norm > eps:
                r_hat = rand_vec / rand_norm
            else:
                r_hat = np.array([1.0, 0.0, 0.0])
        
        # Horizontal lateral vector (perpendicular to the horizontal
        # projection of the pursuit direction, 90° CCW rotation in XY)
        lateral_raw = np.array([-r_hat[1], r_hat[0], 0.0])
        lateral_norm = np.linalg.norm(lateral_raw)
        if lateral_norm > eps:
            lateral = lateral_raw / lateral_norm
        else:
            # Pursuit direction is (near-)vertical: horizontal weave axis is
            # degenerate. Fall back to a fixed horizontal direction rather
            # than dividing by zero.
            lateral = np.array([1.0, 0.0, 0.0])
        
        # Update weave bias with AR(1) process: a <- rho * a + sigma_a * eta
        eta = self._np_random.normal(0.0, 1.0)
        self._weave_bias = self.config.rho * self._weave_bias + self.config.sigma_a * eta
        
        # Heading noise, generalized to 3 dimensions
        z = self._np_random.normal(0.0, 1.0, size=(3,))
        
        # Unnormalized direction with weave amplitude multiplier (horizontal weave only)
        weave_component = self._weave_bias * self.config.weave_amplitude * lateral
        u_raw = r_hat + weave_component + self.config.sigma_e * z
        
        # Normalize direction
        u_norm = np.linalg.norm(u_raw)
        if u_norm > eps:
            u = u_raw / u_norm
        else:
            u = r_hat  # Fallback to direct pursuit
        
        # Guidance produces a DESIRED velocity; dynamics determine the
        # actual persistent velocity subject to acceleration/turn/vertical limits.
        desired_velocity = self.config.v_e * u
        
        next_vel, diag = self._advance_velocity(
            current_velocity=self._enemy_vel,
            desired_velocity=desired_velocity,
            max_speed=self.config.v_e,
            max_accel=self.config.enemy_max_accel,
            max_turn_rate_rad=np.radians(self.config.enemy_max_turn_rate_deg),
            max_climb_rate=self.config.enemy_max_climb_rate,
            max_descent_rate=self.config.enemy_max_descent_rate,
        )
        self._enemy_dynamics_info = diag
        
        new_pos = self._enemy_pos + next_vel * self.config.dt
        new_pos, next_vel = self._apply_boundary(new_pos, next_vel)
        
        self._enemy_pos = new_pos.astype(np.float32)
        self._enemy_vel = next_vel.astype(np.float32)
    
    def _reflect_boundary(self, pos: np.ndarray) -> np.ndarray:
        """
        Apply reflecting boundary conditions within the 3D engagement volume:
        x, y ∈ [-L, L] and z ∈ [0, H] where H = max_altitude.
        
        If position exceeds a boundary, reflect it back into the domain.
        This is applied independently per axis, with the ground (z=0) and
        ceiling (z=H) acting as reflecting boundaries for altitude, exactly
        as the horizontal walls do for x and y.
        
        Args:
            pos: Position array of shape (3,).
        
        Returns:
            Reflected position within [-L, L]² × [0, H].
        """
        L = self.config.L
        H = self.config.max_altitude
        lo = (-L, -L, 0.0)
        hi = (L, L, H)
        
        for i in range(3):
            # Reflect until within bounds (handles multiple reflections)
            while pos[i] < lo[i] or pos[i] > hi[i]:
                if pos[i] < lo[i]:
                    pos[i] = 2 * lo[i] - pos[i]  # Reflect off lower boundary
                if pos[i] > hi[i]:
                    pos[i] = 2 * hi[i] - pos[i]  # Reflect off upper boundary
        
        return pos
    
    def _normalize_pos(self, pos: np.ndarray) -> np.ndarray:
        """
        Normalize a 3D position vector to a dimensionally-appropriate scale.
        
        Horizontal coordinates (x, y) are normalized by L (placing them in
        [-1, 1]); altitude (z) is normalized by max_altitude (placing it in
        [0, 1], a valid subset of the overall [-1, 1] Box).
        
        Args:
            pos: Position array of shape (3,).
        
        Returns:
            Normalized position array of shape (3,).
        """
        L = self.config.L
        H = self.config.max_altitude
        return np.array([pos[0] / L, pos[1] / L, pos[2] / H], dtype=np.float32)
    
    def _get_obs(self) -> np.ndarray:
        """
        Get normalized observation for RL with partial observability.
        
        The observation format depends on config.use_kalman_tracking:
        
        Returns:
            obs: Array of shape (16,), all values normalized to [-1, 1].
            
            [soldier(3), defender(3), defender_vel(3), detected_flag(1), enemy_info(6)]
            
            - defender_vel: defender's persistent 3D velocity normalized by
              v_d (component-wise); always within [-1, 1] since total
              defender speed is dynamically constrained to <= v_d.
            
            If config.use_kalman_tracking=True (RL-Kalman track):
                enemy_info = [e_hat(3), v_hat(3)]
                - e_hat: Kalman estimated enemy position (x,y normalized by L,
                  z normalized by max_altitude)
                - v_hat: Kalman estimated enemy velocity (normalized by v_e)
                - All enemy info is zeros before detection
                - Policy acts on ESTIMATED state, not ground truth
                
            If config.use_kalman_tracking=False (direct RL track):
                enemy_info = [meas(3), rel(3)]
                - meas: noisy enemy position measurement (x,y normalized by L,
                  z normalized by max_altitude)
                - rel: relative vector from defender to measurement (x,y
                  normalized by L, z normalized by max_altitude)
                - All enemy info is zeros before detection
                - Policy acts on measured state without filtering
        """
        L = self.config.L
        H = self.config.max_altitude
        v_e = self.config.v_e
        v_d = self.config.v_d
        
        # Normalize positions
        soldier_norm = self._normalize_pos(self._soldier_pos)
        defender_norm = self._normalize_pos(self._defender_pos)
        
        # Normalize defender's own persistent velocity by its max speed.
        # Guard against a misconfigured (non-positive) v_d.
        if v_d > 0:
            defender_vel_norm = np.clip(self._defender_vel / v_d, -1.0, 1.0).astype(np.float32)
        else:
            defender_vel_norm = np.zeros(3, dtype=np.float32)
        
        # Detection flag
        detected_flag = np.array([1.0 if self._enemy_detected else 0.0])
        
        if self.config.use_kalman_tracking:
            # RL-KALMAN TRACK: Use Kalman filter estimates
            # Policy acts on estimated state, not ground truth
            if self._enemy_detected and self._e_hat is not None:
                e_hat_norm = np.clip(self._normalize_pos(self._e_hat), -1.0, 1.0)
                v_hat_norm = np.clip(self._v_hat / v_e, -1.0, 1.0)
            else:
                e_hat_norm = np.zeros(3, dtype=np.float32)
                v_hat_norm = np.zeros(3, dtype=np.float32)
            
            return np.concatenate([
                soldier_norm, 
                defender_norm,
                defender_vel_norm,
                detected_flag,
                e_hat_norm,
                v_hat_norm
            ]).astype(np.float32)
        else:
            # DIRECT RL TRACK: Use raw noisy measurement
            if self._enemy_detected and self._enemy_measurement is not None:
                enemy_norm = np.clip(self._normalize_pos(self._enemy_measurement), -1.0, 1.0)
                rel = self._enemy_measurement - self._defender_pos
                defender_to_enemy = np.clip(
                    np.array([rel[0] / L, rel[1] / L, rel[2] / H], dtype=np.float32),
                    -1.0, 1.0,
                )
            else:
                enemy_norm = np.zeros(3, dtype=np.float32)
                defender_to_enemy = np.zeros(3, dtype=np.float32)
            
            return np.concatenate([
                soldier_norm, 
                defender_norm,
                defender_vel_norm,
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
        
        Dynamics diagnostic fields (units: velocity in m/s, acceleration in
        m/s^2, turn rate in deg/s) are computed by _advance_velocity() during
        _move_defender() / _move_enemy() and are NOT part of the RL
        observation (only defender_vel is observable, per the observation
        layout above).
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
            "defender_vel": self._defender_vel.copy(),  # m/s, shape (3,)
            "enemy_vel": self._enemy_vel.copy(),        # m/s, shape (3,)
            "defender_speed": float(np.linalg.norm(self._defender_vel)),  # m/s
            "enemy_speed": float(np.linalg.norm(self._enemy_vel)),        # m/s
            "defender_accel": self._defender_dynamics_info["accel_used"],  # m/s^2
            "enemy_accel": self._enemy_dynamics_info["accel_used"],        # m/s^2
            "defender_turn_rate": self._defender_dynamics_info["turn_rate_used_deg"],  # deg/s
            "enemy_turn_rate": self._enemy_dynamics_info["turn_rate_used_deg"],        # deg/s
            "defender_accel_saturated": self._defender_dynamics_info["accel_saturated"],
            "enemy_accel_saturated": self._enemy_dynamics_info["accel_saturated"],
            "defender_turn_saturated": self._defender_dynamics_info["turn_saturated"],
            "enemy_turn_saturated": self._enemy_dynamics_info["turn_saturated"],
            "defender_climb_saturated": self._defender_dynamics_info["climb_saturated"],
            "enemy_climb_saturated": self._enemy_dynamics_info["climb_saturated"],
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
