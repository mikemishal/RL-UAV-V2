"""
Proportional Navigation Policy - 3-D vector PN guidance baseline.

=============================================================================
WHAT THIS IS
=============================================================================
A classical 3-D vector proportional-navigation (PN) guidance law, adapted to
the environment's common desired-velocity-DIRECTION control interface (the
same 3-D action interface used by GreedyInterceptPolicy, KalmanGreedyInterceptPolicy,
RandomPolicy, and PPO). This is NOT direct acceleration control: the PN
acceleration command is used only to construct a one-step desired velocity,
which is then normalized to a direction and handed to the environment. The
environment's existing constrained 3-D point-mass dynamics
(_advance_velocity: acceleration limit, turn-rate limit, speed limit,
climb/descent limits) remain the sole source of actual vehicle motion; this
policy never bypasses them and never writes to the environment's internal
state directly.

This is NOT:
    - lead pursuit / predicted-intercept-point guidance (a separate,
      later baseline);
    - constant-bearing / collision-cone guidance;
    - augmented PN;
    - optimal control;
    - globally-optimal or guaranteed interception.

=============================================================================
GROUND-TRUTH INDEPENDENCE (hard requirement)
=============================================================================
This controller NEVER reads `info["enemy_pos"]` or `info["enemy_vel"]`
(the environment's ground-truth hostile state, exposed only for
evaluation/debugging). Guidance uses only information legitimately available
to the defender:

    defender_pos, defender_vel   (the defender's own true state -- always legitimate)
    soldier_pos                  (escort fallback target)
    enemy_detected                (detection flag)
    enemy_measurement             (raw noisy hostile position -- "measurement" mode)
    e_hat, v_hat                  (Kalman hostile position/velocity estimate -- "kalman" mode)

=============================================================================
TWO SENSING MODES, ONE GUIDANCE LAW
=============================================================================
Both modes use the identical PN guidance law (relative geometry, LOS rate,
closing speed, PN acceleration, velocity-direction adaptation). They differ
ONLY in how the target position/velocity is estimated:

    state_source="measurement" (canonical name "pn"):
        Hostile position is the raw noisy measurement `enemy_measurement`.
        Hostile velocity is the environment's standardized finite-difference
        measurement velocity `info["enemy_measurement_velocity"]` (valid only
        when `info["enemy_measurement_velocity_valid"]` is True, i.e. from
        the second measurement onward): v_hat_e(t) = (z_t - z_{t-1}) / dt.
        This estimator is computed ONCE by the environment (see
        SoldierEnv._update_detection) and shared by the Direct observation
        and every measurement-mode controller (Greedy, PN, Lead) -- this
        policy does NOT maintain its own independent finite-difference
        history.

    state_source="kalman" (canonical name "pn_kalman"):
        Hostile position/velocity are taken directly from the environment's
        existing 3-D constant-velocity Kalman estimator (e_hat, v_hat). This
        policy does NOT run an independent Kalman filter of its own.
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig


class ProportionalNavigationPolicy:
    """
    3-D vector proportional-navigation guidance, adapted to the environment's
    common desired-velocity-direction action interface.

    Guidance law (see module docstring for the full picture):
        r         = p_hat_e - p_d                          (relative position)
        R         = ||r||                                  (range)
        r_hat     = r / R                                  (unit LOS)
        v_rel     = v_hat_e - v_d                           (relative velocity;
                                                              v_d is the
                                                              defender's own
                                                              TRUE velocity --
                                                              legitimate,
                                                              since it is the
                                                              defender's own state)
        V_c       = -(r . v_rel) / R                       (closing speed)
        omega_LOS = (r x v_rel) / R^2                       (3-D LOS angular rate)
        a_PN      = N * V_c * (omega_LOS x r_hat)           (PN acceleration,
                                                              only when V_c > 0)
        v_cmd     = v_d + a_PN * dt                         (one-step velocity command)
        u_cmd     = v_cmd / ||v_cmd||                        (returned action)

    When PN is not sufficiently well-defined (see `act()` / class docstring
    for the exact fallback conditions), the policy falls back to pure pursuit
    of the current legitimate target-position estimate/measurement -- never
    to ground truth.

    Attributes (diagnostics, NOT part of the RL observation):
        last_range: float | None
        last_los_unit: np.ndarray | None, shape (3,) -- unit LOS vector r_hat
        last_closing_speed: float | None
        last_los_rate: np.ndarray | None, shape (3,)
        last_target_velocity_estimate: np.ndarray | None, shape (3,)
        last_pn_acceleration: np.ndarray | None, shape (3,)
        last_guidance_mode: str | None, one of:
            "escort", "first_measurement_fallback", "low_speed_fallback",
            "nonclosing_fallback", "pn", "degenerate_fallback"
    """

    VALID_STATE_SOURCES = ("measurement", "kalman")

    def __init__(
        self,
        state_source: str = "measurement",
        navigation_constant: float = 3.0,
        config: EnvConfig | None = None,
        dt: float | None = None,
        v_d: float | None = None,
        eps: float | None = None,
        min_pn_speed_fraction: float = 0.10,
    ):
        """
        Initialize the proportional navigation policy.

        Args:
            state_source: "measurement" (raw noisy measurement + finite-
                difference velocity) or "kalman" (environment's e_hat/v_hat).
            navigation_constant: PN gain N. Provisional simulation default 3.0.
            config: Optional EnvConfig used ONLY to read dt/v_d/eps defaults;
                never mutated. If None, a default EnvConfig() is constructed
                solely to source these three values.
            dt: Override for the guidance time step (seconds). Defaults to
                config.dt.
            v_d: Override for the defender's maximum speed (m/s), used only
                to derive the low-speed startup threshold. Defaults to
                config.v_d.
            eps: Override for the numerical-stability epsilon. Defaults to
                config.eps.
            min_pn_speed_fraction: PN activates only once the defender's own
                speed exceeds min_pn_speed_fraction * v_d (a controller
                startup guard, NOT an environment parameter). Must be in
                [0, 1).
        """
        if state_source not in self.VALID_STATE_SOURCES:
            raise ValueError(
                f"state_source must be one of {self.VALID_STATE_SOURCES}, got '{state_source}'"
            )
        if navigation_constant <= 0:
            raise ValueError(f"navigation_constant must be > 0, got {navigation_constant}")
        if not (0.0 <= min_pn_speed_fraction < 1.0):
            raise ValueError(
                f"min_pn_speed_fraction must be in [0, 1), got {min_pn_speed_fraction}"
            )

        cfg = config if config is not None else EnvConfig()
        self.state_source = state_source
        self.navigation_constant = float(navigation_constant)
        self.min_pn_speed_fraction = float(min_pn_speed_fraction)
        self.dt = float(dt) if dt is not None else float(cfg.dt)
        self.v_d = float(v_d) if v_d is not None else float(cfg.v_d)
        self.eps = float(eps) if eps is not None else float(cfg.eps)

        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")

        # Diagnostics (NOT part of the RL observation; for testing/debugging).
        self.last_range: float | None = None
        self.last_los_unit: np.ndarray | None = None
        self.last_closing_speed: float | None = None
        self.last_los_rate: np.ndarray | None = None
        self.last_target_velocity_estimate: np.ndarray | None = None
        self.last_pn_acceleration: np.ndarray | None = None
        self.last_guidance_mode: str | None = None

    def reset(self) -> None:
        """
        Reset all per-episode diagnostic state.

        The measurement-mode finite-difference velocity is owned by the
        environment (see SoldierEnv._update_detection), so there is no
        internal finite-difference history to clear here.
        """
        self.last_range = None
        self.last_los_unit = None
        self.last_closing_speed = None
        self.last_los_rate = None
        self.last_target_velocity_estimate = None
        self.last_pn_acceleration = None
        self.last_guidance_mode = None

    def _get_target_state(self, info: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return (target_position, target_velocity) using ONLY legitimate
        environment fields for the configured state_source. NEVER reads
        info["enemy_pos"] or info["enemy_vel"].

        Returns:
            target_pos: shape (3,) or None if no measurement/estimate is
                available this step.
            target_vel: shape (3,) or None if velocity is not yet
                established (measurement mode: fewer than two measurements
                so far, per info["enemy_measurement_velocity_valid"]; kalman
                mode: v_hat unavailable).
        """
        if self.state_source == "kalman":
            e_hat = info.get("e_hat")
            v_hat = info.get("v_hat")
            if e_hat is None:
                return None, None
            target_pos = np.asarray(e_hat, dtype=np.float64)
            target_vel = np.asarray(v_hat, dtype=np.float64) if v_hat is not None else None
            return target_pos, target_vel

        # state_source == "measurement": consume the environment's
        # standardized finite-difference measurement velocity -- the same
        # estimator used by the Direct observation and shared across all
        # measurement-mode controllers. No independent per-policy finite
        # differencing.
        meas = info.get("enemy_measurement")
        if meas is None:
            return None, None
        target_pos = np.asarray(meas, dtype=np.float64)
        if info.get("enemy_measurement_velocity_valid", False):
            target_vel = np.asarray(info.get("enemy_measurement_velocity"), dtype=np.float64)
        else:
            target_vel = None
        return target_pos, target_vel

    def _pursue(self, target: np.ndarray | None, defender_pos: np.ndarray, eps: float) -> np.ndarray:
        """Pure-pursuit fallback direction toward `target` (never ground truth)."""
        if target is None:
            return np.zeros(3, dtype=np.float32)
        target = np.asarray(target, dtype=np.float64)
        direction = target - defender_pos
        dist = float(np.linalg.norm(direction))
        if dist < eps:
            return np.zeros(3, dtype=np.float32)
        return (direction / dist).astype(np.float32)

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Compute the PN-guided (or fallback pure-pursuit) desired-direction action.

        Args:
            obs: Environment observation (unused directly; all state is read
                from `info` for precision, matching the existing Greedy /
                KalmanGreedy baseline convention).
            info: Environment info dict. Only these keys are read:
                defender_pos, defender_vel, soldier_pos, enemy_detected, and
                (depending on state_source) enemy_measurement OR e_hat/v_hat.
                info["enemy_pos"] and info["enemy_vel"] (ground truth) are
                NEVER read.

        Returns:
            action: 3-D desired-velocity-direction vector in [-1, 1]^3,
                shape (3,). Fed through the environment's constrained
                dynamics exactly like any other policy's action.

        Fallback to pure pursuit (never PN) occurs, in this order, when:
            1. Enemy not yet detected -> escort (pursue soldier_pos).
            2. No measurement/estimate available this step -> escort.
            3. Target velocity not yet established -> first_measurement_fallback.
            4. Range degenerate (R <= eps) -> degenerate_fallback.
            5. Defender speed <= min_pn_speed_fraction * v_d -> low_speed_fallback.
            6. Closing speed V_c <= 0 -> nonclosing_fallback.
            7. PN acceleration or velocity command non-finite / near-zero norm
               -> degenerate_fallback.
        Otherwise guidance mode is "pn".
        """
        eps = self.eps
        defender_pos = np.asarray(info.get("defender_pos"), dtype=np.float64)
        defender_vel_raw = info.get("defender_vel")
        defender_vel = (
            np.asarray(defender_vel_raw, dtype=np.float64)
            if defender_vel_raw is not None
            else np.zeros(3, dtype=np.float64)
        )
        soldier_pos = info.get("soldier_pos")
        enemy_detected = bool(info.get("enemy_detected", False))

        # Reset per-step diagnostics; filled in as guidance proceeds.
        self.last_range = None
        self.last_los_unit = None
        self.last_closing_speed = None
        self.last_los_rate = None
        self.last_target_velocity_estimate = None
        self.last_pn_acceleration = None

        if not enemy_detected:
            self.last_guidance_mode = "escort"
            return self._pursue(soldier_pos, defender_pos, eps)

        target_pos, target_vel = self._get_target_state(info)

        if target_pos is None:
            self.last_guidance_mode = "escort"
            return self._pursue(soldier_pos, defender_pos, eps)

        if target_vel is None:
            self.last_guidance_mode = "first_measurement_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        self.last_target_velocity_estimate = target_vel.astype(np.float32)

        # Relative geometry (section 9-10)
        r = target_pos - defender_pos
        R = float(np.linalg.norm(r))
        self.last_range = R

        if R <= eps:
            self.last_guidance_mode = "degenerate_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        r_hat = r / R
        self.last_los_unit = r_hat.astype(np.float32)
        v_rel = target_vel - defender_vel

        # Closing speed (section 11)
        Vc = -float(np.dot(r, v_rel)) / R
        self.last_closing_speed = Vc

        # Startup guard: PN requires meaningful own-vehicle velocity (section 15)
        defender_speed = float(np.linalg.norm(defender_vel))
        min_speed = self.min_pn_speed_fraction * self.v_d
        if defender_speed <= min_speed:
            self.last_guidance_mode = "low_speed_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        if Vc <= 0:
            self.last_guidance_mode = "nonclosing_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        # 3-D LOS angular-rate vector (section 12)
        omega_los = np.cross(r, v_rel) / (R * R)

        # PN acceleration command (section 13)
        a_pn = self.navigation_constant * Vc * np.cross(omega_los, r_hat)

        if not np.all(np.isfinite(a_pn)):
            self.last_guidance_mode = "degenerate_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        self.last_los_rate = omega_los.astype(np.float32)
        self.last_pn_acceleration = a_pn.astype(np.float32)

        # Adapt to the common desired-velocity-direction interface (section 14).
        # NOTE: this constructs a DESIRED direction only; the environment's
        # existing _advance_velocity() constrained dynamics (acceleration,
        # turn-rate, speed, climb/descent limits) determine the actual
        # vehicle motion. This policy never writes to _defender_vel/_defender_pos.
        v_cmd = defender_vel + a_pn * self.dt
        v_cmd_norm = float(np.linalg.norm(v_cmd))

        if v_cmd_norm <= eps or not np.all(np.isfinite(v_cmd)):
            self.last_guidance_mode = "degenerate_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        self.last_guidance_mode = "pn"
        return (v_cmd / v_cmd_norm).astype(np.float32)

    def __repr__(self) -> str:
        return (
            f"ProportionalNavigationPolicy(state_source='{self.state_source}', "
            f"navigation_constant={self.navigation_constant}, "
            f"min_pn_speed_fraction={self.min_pn_speed_fraction})"
        )
