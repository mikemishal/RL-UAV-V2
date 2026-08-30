"""
Lead+residual composition -- the SINGLE mathematical function shared by
BOTH the Lead-Residual PPO training environment
(experiments/lead_residual_training_env.py) and the evaluation-time policy
wrapper (uav_defend/policies/residual/lead_residual_ppo_policy_wrapper.py).
There is exactly one implementation of this math in the entire repository.

=============================================================================
MATH (see docs/lead_residual_ppo_design.md for the full derivation)
=============================================================================
Given the canonical unit Lead direction u_L and a raw PPO residual r_raw in
R^3 (from a Box(-1, 1, shape=(3,)) action space):

    r = r_raw / ||r_raw||   if ||r_raw|| > 1  else  r_raw     (norm-clip)

    r_perp = r - (r . u_L) u_L                                (drop parallel
                                                                 component --
                                                                 only angular
                                                                 corrections
                                                                 are meaningful,
                                                                 see item 5)

    theta_max = radians(residual_max_angle_deg)

    u_raw = u_L + tan(theta_max) * r_perp

    u_hybrid = u_raw / ||u_raw||

Property (proved in the design doc): since r_perp is orthogonal to u_L and
||r_perp|| <= 1, u_raw's component along u_L has magnitude exactly 1 and its
orthogonal component has magnitude at most tan(theta_max), so:

    angle(u_hybrid, u_L) <= theta_max

Zero-residual invariance: r = 0 => r_perp = 0 => u_hybrid == u_L.
Parallel-residual invariance: r parallel to u_L (either sign) => r_perp = 0
    => u_hybrid == u_L.

`residual_max_angle_deg` is a PROVISIONAL development-only hyperparameter
(see DEFAULT_RESIDUAL_MAX_ANGLE_DEG below) -- it is NOT publication-locked and
must never be selected/tuned using the already-opened final-nominal seeds
(41000..45999).
"""

from __future__ import annotations

import numpy as np

# Provisional development default -- NOT a publication-locked hyperparameter.
# May only be changed later using NEW development/validation seeds; must
# NEVER be selected using the already-opened 41000..45999 final baseline
# results (see docs/lead_residual_ppo_design.md).
DEFAULT_RESIDUAL_MAX_ANGLE_DEG = 30.0

_EPS = 1e-8


def compose_lead_residual(
    lead_direction: np.ndarray,
    residual_action: np.ndarray,
    residual_max_angle_deg: float,
    eps: float = _EPS,
) -> np.ndarray:
    """
    Compose the analytical Lead direction with a bounded directional PPO
    residual correction.

    Args:
        lead_direction: shape (3,). The canonical Lead controller's action
            (expected unit norm, but re-normalized defensively here). May be
            the zero vector (Lead's own degenerate/no-target fallback).
        residual_action: shape (3,). Raw PPO action, expected to already lie
            in [-1, 1]^3 (the residual PPO's Box action space) but norm-clipped
            defensively regardless.
        residual_max_angle_deg: Maximum angular authority theta_max (degrees)
            the residual may rotate u_L by. See DEFAULT_RESIDUAL_MAX_ANGLE_DEG.
            Must satisfy 0 <= theta_max < 90 (see "Angle domain" below).
        eps: Numerical-stability threshold for near-zero-norm degenerate cases.

    Returns:
        u_hybrid: shape (3,), float32, unit norm (or exactly zero iff
            lead_direction itself is the zero vector -- see "Degenerate
            safety" below). Never NaN/Inf.

    Angle domain: the tangent-based construction (u_raw = u_L + tan(theta_max)
    r_perp) is mathematically valid only for 0 <= theta_max < 90 degrees --
    tan(theta_max) diverges to +inf as theta_max -> 90 and is negative (an
    inverted/wrapped-around correction, not a bounded one) beyond 90. Raises
    ValueError for theta_max < 0, theta_max >= 90, NaN, or Inf. Values close
    to (but below) 90 are mathematically admissible but not necessarily
    sensible hyperparameters (an almost-unconstrained rotation). At
    theta_max=0, tan(0)=0 so u_hybrid == u_L for every residual action
    (verified by test_theta_max_zero_ignores_residual in test_lead_residual_ppo.py).

    Degenerate safety (item 10): this function never emits NaN/Inf and never
    divides by zero.
        - If ||lead_direction|| < eps (Lead had no legitimate target: escort
          fallback with no soldier_pos, or a genuinely degenerate geometry),
          there is no direction to compose with -- ground truth is NEVER
          substituted; the function returns the zero vector, matching Lead's
          own existing zero-fallback convention.
        - If ||lead_direction|| >= eps, u_raw = u_L_unit + tan(theta_max)*r_perp
          has parallel-component magnitude exactly 1 and an orthogonal
          component, so ||u_raw|| >= 1 always (Pythagorean) -- the
          renormalization below is therefore never a division by
          near-zero in this branch; the defensive eps check is
          included regardless as defense-in-depth.

    Bit-identical zero-residual/parallel-residual fast path: whenever r_perp
    is EXACTLY the zero vector (guaranteed for residual_action == [0,0,0],
    since 0 - (0 . u)u == 0 exactly under IEEE754; may also occur for an
    exactly-parallel residual), this function returns the ORIGINAL
    lead_direction array cast to float32 UNCHANGED -- no renormalize
    round-trip -- so that "zero learned correction" reproduces Lead
    bit-for-bit, not merely within a numerical tolerance.
    """
    theta_max_deg = float(residual_max_angle_deg)
    if not np.isfinite(theta_max_deg) or theta_max_deg < 0.0 or theta_max_deg >= 90.0:
        raise ValueError(
            f"residual_max_angle_deg must satisfy 0 <= theta_max < 90 (tangent-based "
            f"construction is undefined/unbounded at 90deg), got {residual_max_angle_deg}"
        )

    u_L = np.asarray(lead_direction, dtype=np.float64).reshape(3)
    r_raw = np.asarray(residual_action, dtype=np.float64).reshape(3)

    if not np.all(np.isfinite(u_L)):
        raise ValueError(f"lead_direction must be finite, got {lead_direction}")
    if not np.all(np.isfinite(r_raw)):
        raise ValueError(f"residual_action must be finite, got {residual_action}")

    norm_uL = float(np.linalg.norm(u_L))
    if norm_uL < eps:
        # Lead itself has no legitimate direction (e.g. no soldier_pos target
        # available for the escort fallback) -- never substitute ground
        # truth; propagate the same "no direction" result Lead would.
        return np.zeros(3, dtype=np.float32)
    u_L_unit = u_L / norm_uL

    r_norm = float(np.linalg.norm(r_raw))
    r = (r_raw / r_norm) if r_norm > 1.0 else r_raw

    r_perp = r - float(np.dot(r, u_L_unit)) * u_L_unit

    if np.array_equal(r_perp, np.zeros(3)):
        # Exact fast path: the residual contributes NOTHING (r_perp is
        # bit-exact zero -- guaranteed for residual_action == [0,0,0], since
        # 0 - (0 . u)*u == 0 exactly under IEEE754). Skip the renormalize
        # round-trip entirely and return the ORIGINAL lead_direction bits
        # unchanged, so zero-residual reproduces Lead bit-for-bit rather than
        # merely "within tolerance".
        return np.asarray(lead_direction, dtype=np.float32).reshape(3).copy()

    theta_max = np.deg2rad(theta_max_deg)
    u_raw = u_L_unit + np.tan(theta_max) * r_perp

    norm_u_raw = float(np.linalg.norm(u_raw))
    if norm_u_raw < eps:
        # Unreachable in practice (see docstring), retained as defense-in-depth.
        return u_L_unit.astype(np.float32)

    u_hybrid = (u_raw / norm_u_raw).astype(np.float32)
    if not np.all(np.isfinite(u_hybrid)):
        raise RuntimeError(f"compose_lead_residual produced a non-finite result: {u_hybrid}")
    return u_hybrid


def angle_between_deg(a: np.ndarray, b: np.ndarray, eps: float = _EPS) -> float:
    """Angle in degrees between two vectors (test/diagnostic helper only)."""
    a = np.asarray(a, dtype=np.float64).reshape(3)
    b = np.asarray(b, dtype=np.float64).reshape(3)
    na, nb = float(np.linalg.norm(a)), float(np.linalg.norm(b))
    if na < eps or nb < eps:
        return 0.0
    cos_theta = float(np.dot(a, b) / (na * nb))
    cos_theta = min(1.0, max(-1.0, cos_theta))
    return float(np.degrees(np.arccos(cos_theta)))
