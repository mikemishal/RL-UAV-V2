"""
Lead Intercept Policy - 3-D constant-velocity predictive interception baseline.

=============================================================================
WHAT THIS IS
=============================================================================
A classical 3-D constant-velocity lead-intercept guidance law: the hostile
UAV's current position/velocity are estimated, its velocity is assumed
constant over a short prediction horizon, and an idealized interceptor
traveling at the defender's nominal maximum speed (config.v_d) is used to
analytically solve for the earliest future interception time and the
corresponding predicted intercept point. The controller returns only the
3-D DIRECTION from the current defender position toward that point -- the
same common desired-velocity-direction action interface used by every other
policy (GreedyInterceptPolicy, KalmanGreedyInterceptPolicy,
ProportionalNavigationPolicy, RandomPolicy, PPO).

This is "constant-velocity predictive interception" / "3-D constant-velocity
lead-intercept guidance". It is NOT:
    - globally optimal control;
    - minimum-time optimal guidance;
    - a perfect / ground-truth lead solution;
    - proportional navigation (a separate, already-implemented baseline).

The intercept-speed assumption `s = config.v_d` is an IDEALIZED kinematic
constant used only inside the analytical quadratic. The actual defender
still starts from rest, has bounded acceleration, bounded turn rate, and
bounded climb/descent rate -- it will generally NOT arrive at the predicted
point exactly at t_intercept. That mismatch is intentional: the environment
alone determines physical realizability (see _advance_velocity()). Because
the hostile UAV may maneuver/evade after the prediction is made, the
constant-velocity prediction becomes stale between steps -- the intercept
point is therefore recomputed every step from the latest legitimate target
estimate, which is the meaningful behavior this baseline is intended to
exhibit.

=============================================================================
GROUND-TRUTH INDEPENDENCE (hard requirement)
=============================================================================
This controller NEVER reads `info["enemy_pos"]` or `info["enemy_vel"]`
(ground truth, exposed only for evaluation/debugging). Guidance uses only:

    defender_pos, defender_vel   (defender's own true state -- always legitimate)
    soldier_pos                  (escort fallback target)
    enemy_detected                (detection flag)
    enemy_measurement             (raw noisy hostile position -- "measurement" mode)
    e_hat, v_hat                  (Kalman hostile position/velocity estimate -- "kalman" mode)

=============================================================================
TWO SENSING MODES, ONE INTERCEPT GEOMETRY
=============================================================================
    state_source="measurement" (canonical name "lead"):
        Hostile position is the raw noisy measurement `enemy_measurement`.
        Hostile velocity is the environment's standardized finite-difference
        measurement velocity `info["enemy_measurement_velocity"]` (valid only
        when `info["enemy_measurement_velocity_valid"]` is True). This
        estimator is computed ONCE by the environment (see
        SoldierEnv._update_detection) and shared by the Direct observation
        and every measurement-mode controller (Greedy, PN, Lead) -- no
        independent per-policy finite differencing.

    state_source="kalman" (canonical name "lead_kalman"):
        Hostile position/velocity come directly from the environment's
        existing 3-D Kalman estimator (e_hat, v_hat). No independent Kalman
        filter is instantiated here, and `enemy_measurement` is never used
        for this variant's guidance.
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig


class LeadInterceptPolicy:
    """
    3-D constant-velocity lead-intercept guidance, adapted to the common
    desired-velocity-direction action interface.

    Intercept geometry:
        r = p_hat_e - p_d                          (relative position)
        Assumed constant target motion: p_e(t) = p_hat_e + v_hat_e * t
        Ideal interceptor at speed s = config.v_d: ||r + v_hat_e t|| = s t
        => quadratic  a t^2 + b t + c = 0  where
               a = v_hat_e . v_hat_e - s^2
               b = 2 * (r . v_hat_e)
               c = r . r
        Earliest positive real root t_intercept is selected (see
        _solve_intercept_time for the numerically stable solver).
        p_intercept = p_hat_e + v_hat_e * t_intercept
        lead_vector = p_intercept - p_d
        action      = lead_vector / ||lead_vector||

    When no valid intercept solution exists (see `act()` for the exact
    fallback conditions), the policy falls back to pure pursuit of the
    current legitimate target-position estimate/measurement -- never to
    ground truth.

    Attributes (diagnostics, NOT part of the RL observation):
        last_guidance_mode: str | None, one of:
            "escort", "first_measurement_fallback", "no_velocity_fallback",
            "no_real_solution_fallback", "no_positive_root_fallback",
            "degenerate_fallback", "lead"
        last_target_position: np.ndarray | None, shape (3,)
        last_target_velocity_estimate: np.ndarray | None, shape (3,)
        last_relative_position: np.ndarray | None, shape (3,)
        last_quadratic_a: float | None
        last_quadratic_b: float | None
        last_quadratic_c: float | None
        last_discriminant: float | None
        last_intercept_time: float | None
        last_intercept_point: np.ndarray | None, shape (3,)
        last_action: np.ndarray | None, shape (3,)
    """

    VALID_STATE_SOURCES = ("measurement", "kalman")

    def __init__(
        self,
        state_source: str = "measurement",
        config: EnvConfig | None = None,
        dt: float | None = None,
        v_d: float | None = None,
        eps: float | None = None,
    ):
        """
        Initialize the lead-intercept policy.

        Args:
            state_source: "measurement" (raw noisy measurement + finite-
                difference velocity) or "kalman" (environment's e_hat/v_hat).
            config: Optional EnvConfig used ONLY to read dt/v_d/eps defaults;
                never mutated. If None, a default EnvConfig() is constructed
                solely to source these three values.
            dt: Override for the finite-difference time step (seconds).
                Defaults to config.dt.
            v_d: Override for the assumed idealized interceptor speed `s`
                used in the intercept-time quadratic. Defaults to config.v_d.
                (Exposed as an override primarily to test intercept geometry
                in isolation from the environment's default v_d=18.0.)
            eps: Override for the numerical-stability epsilon. Defaults to
                config.eps.
        """
        if state_source not in self.VALID_STATE_SOURCES:
            raise ValueError(
                f"state_source must be one of {self.VALID_STATE_SOURCES}, got '{state_source}'"
            )

        cfg = config if config is not None else EnvConfig()
        self.state_source = state_source
        self.dt = float(dt) if dt is not None else float(cfg.dt)
        self.v_d = float(v_d) if v_d is not None else float(cfg.v_d)
        self.eps = float(eps) if eps is not None else float(cfg.eps)

        if self.dt <= 0:
            raise ValueError(f"dt must be > 0, got {self.dt}")
        if self.v_d <= 0:
            raise ValueError(f"v_d must be > 0, got {self.v_d}")
        if self.eps <= 0:
            raise ValueError(f"eps must be > 0, got {self.eps}")

        self._init_diagnostics()

    def _init_diagnostics(self) -> None:
        """(Re)initialize all diagnostic attributes to their empty state."""
        self.last_guidance_mode: str | None = None
        self.last_target_position: np.ndarray | None = None
        self.last_target_velocity_estimate: np.ndarray | None = None
        self.last_relative_position: np.ndarray | None = None
        self.last_quadratic_a: float | None = None
        self.last_quadratic_b: float | None = None
        self.last_quadratic_c: float | None = None
        self.last_discriminant: float | None = None
        self.last_intercept_time: float | None = None
        self.last_intercept_point: np.ndarray | None = None
        self.last_action: np.ndarray | None = None

    def reset(self) -> None:
        """
        Reset all per-episode diagnostic state.

        The measurement-mode finite-difference velocity is owned by the
        environment (see SoldierEnv._update_detection), so there is no
        internal finite-difference history to clear here.
        """
        self._init_diagnostics()

    def _get_target_state(self, info: dict) -> tuple[np.ndarray | None, np.ndarray | None]:
        """
        Return (target_position, target_velocity) using ONLY legitimate
        environment fields for the configured state_source. NEVER reads
        info["enemy_pos"] or info["enemy_vel"].
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
            action = np.zeros(3, dtype=np.float32)
        else:
            target = np.asarray(target, dtype=np.float64)
            direction = target - defender_pos
            dist = float(np.linalg.norm(direction))
            if dist < eps:
                action = np.zeros(3, dtype=np.float32)
            else:
                action = (direction / dist).astype(np.float32)
        self.last_action = action.copy()
        return action

    @staticmethod
    def _solve_intercept_time(
        a: float, b: float, c: float, eps: float
    ) -> tuple[float | None, float | None, str]:
        """
        Solve a t^2 + b t + c = 0 for the earliest positive real root,
        robustly handling the normal-quadratic, near-linear, and
        floating-point-roundoff-near-zero-discriminant cases.

        Returns:
            (t_intercept, discriminant, reason) where:
                t_intercept: earliest positive real root, or None if none exists.
                discriminant: b^2-4ac (normal-quadratic branch only), else None.
                reason: "ok" | "no_real_solution" | "no_positive_root"
        """
        if abs(a) > eps:
            # CASE A: normal quadratic.
            discriminant = b * b - 4.0 * a * c
            if discriminant < -eps:
                return None, discriminant, "no_real_solution"
            if discriminant < 0.0:
                discriminant = 0.0  # floating-point roundoff clamp

            sqrt_D = float(np.sqrt(discriminant))
            sign_b = 1.0 if b >= 0.0 else -1.0
            q = -0.5 * (b + sign_b * sqrt_D)

            roots: list[float] = []
            if abs(q) > eps:
                # Numerically stable formulation (avoids catastrophic
                # cancellation in (-b +/- sqrt(D))/(2a) when b and sqrt_D
                # are close in magnitude).
                roots.append(q / a)
                roots.append(c / q)
            else:
                # q ~ 0: fall back to the direct quadratic formula, which is
                # safe here precisely because q's cancellation risk is what
                # the stable form exists to avoid, and q~0 means that risk
                # is not present in this branch.
                roots.append((-b + sqrt_D) / (2.0 * a))
                roots.append((-b - sqrt_D) / (2.0 * a))

            positive_roots = [t for t in roots if np.isfinite(t) and t > eps]
            if not positive_roots:
                return None, discriminant, "no_positive_root"
            return min(positive_roots), discriminant, "ok"

        # CASE B: near-linear geometry (a ~ 0).
        if abs(b) > eps:
            t = -c / b
            if np.isfinite(t) and t > eps:
                return t, None, "ok"
            return None, None, "no_positive_root"

        # abs(a) <= eps and abs(b) <= eps: no useful predicted solution.
        return None, None, "no_real_solution"

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Compute the lead-intercept (or fallback pure-pursuit) desired-
        direction action.

        Args:
            obs: Environment observation (unused directly; all state is read
                from `info`, matching the existing baseline-policy convention).
            info: Environment info dict. Only these keys are read:
                defender_pos, soldier_pos, enemy_detected, and (depending on
                state_source) enemy_measurement OR e_hat/v_hat.
                info["enemy_pos"] and info["enemy_vel"] (ground truth) are
                NEVER read.

        Returns:
            action: 3-D desired-velocity-direction vector in [-1, 1]^3,
                shape (3,). The environment's existing constrained dynamics
                (_advance_velocity) determine actual defender motion; this
                policy never bypasses them and never writes to
                _defender_pos / _defender_vel.
        """
        eps = self.eps
        defender_pos = np.asarray(info.get("defender_pos"), dtype=np.float64)
        soldier_pos = info.get("soldier_pos")
        enemy_detected = bool(info.get("enemy_detected", False))

        self._init_diagnostics()

        if not enemy_detected:
            self.last_guidance_mode = "escort"
            return self._pursue(soldier_pos, defender_pos, eps)

        target_pos, target_vel = self._get_target_state(info)

        if target_pos is None:
            self.last_guidance_mode = "escort"
            return self._pursue(soldier_pos, defender_pos, eps)

        self.last_target_position = target_pos.astype(np.float32)

        if target_vel is None:
            self.last_guidance_mode = (
                "first_measurement_fallback" if self.state_source == "measurement" else "no_velocity_fallback"
            )
            return self._pursue(target_pos, defender_pos, eps)

        self.last_target_velocity_estimate = target_vel.astype(np.float32)

        # Relative geometry (section 7)
        r = target_pos - defender_pos
        self.last_relative_position = r.astype(np.float32)
        R = float(np.linalg.norm(r))

        if R <= eps:
            self.last_guidance_mode = "degenerate_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        s = self.v_d
        a = float(np.dot(target_vel, target_vel) - s * s)
        b = float(2.0 * np.dot(r, target_vel))
        c = float(np.dot(r, r))
        self.last_quadratic_a = a
        self.last_quadratic_b = b
        self.last_quadratic_c = c

        t_intercept, discriminant, reason = self._solve_intercept_time(a, b, c, eps)
        self.last_discriminant = discriminant

        if reason != "ok":
            self.last_guidance_mode = (
                "no_real_solution_fallback" if reason == "no_real_solution" else "no_positive_root_fallback"
            )
            return self._pursue(target_pos, defender_pos, eps)

        self.last_intercept_time = t_intercept

        # Predicted intercept point (section 10). NOT clipped to the
        # engagement volume -- this is a mathematical aim point; only its
        # direction is returned, and the environment alone handles bounds.
        p_intercept = target_pos + target_vel * t_intercept
        if not np.all(np.isfinite(p_intercept)):
            self.last_guidance_mode = "degenerate_fallback"
            return self._pursue(target_pos, defender_pos, eps)
        self.last_intercept_point = p_intercept.astype(np.float32)

        lead_vector = p_intercept - defender_pos
        lead_norm = float(np.linalg.norm(lead_vector))
        if lead_norm <= eps or not np.isfinite(lead_norm):
            self.last_guidance_mode = "degenerate_fallback"
            return self._pursue(target_pos, defender_pos, eps)

        action = (lead_vector / lead_norm).astype(np.float32)
        self.last_guidance_mode = "lead"
        self.last_action = action.copy()
        return action

    def __repr__(self) -> str:
        return f"LeadInterceptPolicy(state_source='{self.state_source}', v_d={self.v_d})"
