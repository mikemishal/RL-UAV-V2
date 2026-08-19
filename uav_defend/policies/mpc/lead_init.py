"""
Pure, measurement-track-only constant-velocity lead-intercept DIRECTION
helper, used to deterministically initialize RMPC's CEM search around the
strong analytical Lead guidance direction (`docs/rmpc_baseline_design.md`
Section 11's nominal-initialization note).

This module reuses `LeadInterceptPolicy._solve_intercept_time` -- the
existing, already-tested, numerically stable quadratic-root solver -- by
direct call, rather than re-deriving a second copy of the intercept-time
quadratic. `LeadInterceptPolicy` itself is NEVER instantiated, mutated, or
otherwise modified: `_solve_intercept_time` is a pure `@staticmethod` with
no dependence on instance state, so it can be called directly via the
class.

GROUND-TRUTH INDEPENDENCE: `solve_lead_direction` takes ONLY `target_pos`,
`target_vel` (may be `None`), `defender_pos`, `v_d`, `eps` as arguments --
it never reads `info`, `enemy_pos`, `enemy_vel`, `e_hat`, `v_hat`, and never
performs Kalman filtering.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy

VALID_LEAD_DIRECTION_MODES = (
    "lead",
    "no_velocity_fallback",
    "no_real_solution_fallback",
    "no_positive_root_fallback",
    "degenerate_fallback",
)


@dataclass
class LeadDirectionResult:
    """Result of `solve_lead_direction`."""

    direction: np.ndarray  # shape (3,), dtype float32, unit vector or zero
    guidance_mode: str  # one of VALID_LEAD_DIRECTION_MODES


def _pursue(target_pos: np.ndarray, defender_pos: np.ndarray, eps: float) -> np.ndarray:
    """Pure-pursuit direction toward `target_pos` (mirrors the project's
    existing eps-guarded zero-vector fallback convention)."""
    direction = target_pos - defender_pos
    dist = float(np.linalg.norm(direction))
    if dist < eps:
        return np.zeros(3, dtype=np.float32)
    return (direction / dist).astype(np.float32)


def solve_lead_direction(target_pos: np.ndarray, target_vel: np.ndarray | None,
                          defender_pos: np.ndarray, v_d: float, eps: float) -> LeadDirectionResult:
    """
    Pure constant-velocity lead-intercept direction, mirroring
    `LeadInterceptPolicy.act()`'s core geometry exactly (identical
    fallback conditions, in the identical order), given a NON-None
    `target_pos` (the "no target detected yet" escort case is the
    caller's responsibility, matching every other policy's convention in
    this repository).

    Args:
        target_pos: legitimate target position estimate, shape (3,).
        target_vel: legitimate target velocity estimate, shape (3,), or
            `None` if unavailable (e.g. first-measurement step).
        defender_pos: defender's own true position, shape (3,).
        v_d: assumed idealized interceptor speed (typically `config.v_d`).
        eps: numerical-stability epsilon.

    Returns:
        LeadDirectionResult with a unit-norm (or zero, for degenerate
        geometry) direction and the guidance mode that produced it. Every
        non-"lead" mode returns a pure-pursuit direction toward
        `target_pos`.
    """
    target_pos = np.asarray(target_pos, dtype=np.float64)
    defender_pos = np.asarray(defender_pos, dtype=np.float64)

    if target_vel is None:
        return LeadDirectionResult(direction=_pursue(target_pos, defender_pos, eps),
                                    guidance_mode="no_velocity_fallback")

    target_vel = np.asarray(target_vel, dtype=np.float64)
    r = target_pos - defender_pos
    R = float(np.linalg.norm(r))
    if R <= eps:
        return LeadDirectionResult(direction=_pursue(target_pos, defender_pos, eps),
                                    guidance_mode="degenerate_fallback")

    s = float(v_d)
    a = float(np.dot(target_vel, target_vel) - s * s)
    b = float(2.0 * np.dot(r, target_vel))
    c = float(np.dot(r, r))

    t_intercept, discriminant, reason = LeadInterceptPolicy._solve_intercept_time(a, b, c, eps)

    if reason != "ok":
        mode = "no_real_solution_fallback" if reason == "no_real_solution" else "no_positive_root_fallback"
        return LeadDirectionResult(direction=_pursue(target_pos, defender_pos, eps), guidance_mode=mode)

    p_intercept = target_pos + target_vel * t_intercept
    if not np.all(np.isfinite(p_intercept)):
        return LeadDirectionResult(direction=_pursue(target_pos, defender_pos, eps),
                                    guidance_mode="degenerate_fallback")

    lead_vector = p_intercept - defender_pos
    lead_norm = float(np.linalg.norm(lead_vector))
    if lead_norm <= eps or not np.isfinite(lead_norm):
        return LeadDirectionResult(direction=_pursue(target_pos, defender_pos, eps),
                                    guidance_mode="degenerate_fallback")

    direction = (lead_vector / lead_norm).astype(np.float32)
    return LeadDirectionResult(direction=direction, guidance_mode="lead")
