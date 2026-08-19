"""
Six-scenario hostile-motion uncertainty model for the RMPC baseline.

Per `docs/rmpc_baseline_design.md` Section 8.2: a bounded, physical-limits-
only scenario set, NOT a reproduction of the simulator's true hostile
policy. Every scenario is propagated using ONLY the hostile's PUBLIC
physical limits (`config.v_e`, `config.enemy_max_accel`,
`config.enemy_max_turn_rate_deg`, `config.enemy_max_climb_rate`,
`config.enemy_max_descent_rate`) through the same pure
`advance_velocity` helper used everywhere else in this repository.

CRITICAL INFORMATION-CONTRACT CONSTRAINT: this module MUST NEVER read or
depend on `enemy_evasion_gain`, `enemy_evasion_radius`, or the simulator's
`_compute_enemy_evasion` formula/weave/heading-noise state. These describe
the simulated hostile's DECISION POLICY, not its airframe physics, and are
excluded entirely (not merely their realized values) -- see Section 8.1 of
the design document.

Five scenarios (S1-S5) are OPEN-LOOP: their desired-velocity command is
fixed for the entire planning horizon and does not depend on the candidate
defender trajectory being evaluated. The sixth scenario (S6, MAX_AWAY) is
the only CLOSED-LOOP scenario: at each predicted step it commands a
direction away from the candidate defender's predicted position at that
same step.
"""

from __future__ import annotations

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.dynamics.constrained_point_mass import advance_velocity

SCENARIO_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6")

_OPEN_LOOP_SCENARIO_IDS: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5")


def _base_direction(hostile_pos0: np.ndarray, hostile_vel0: np.ndarray | None,
                     soldier_pos: np.ndarray, eps: float) -> np.ndarray:
    """
    Nominal-continuation direction (S1's basis direction, also used to build
    S2-S5): the current velocity estimate's direction if available and
    non-degenerate, else a pursuit-only fallback direction toward the
    (fixed) soldier position, else the zero vector if both are degenerate.
    """
    if hostile_vel0 is not None:
        hostile_vel0 = np.asarray(hostile_vel0, dtype=np.float64)
        vel_norm = float(np.linalg.norm(hostile_vel0))
        if vel_norm > eps:
            return hostile_vel0 / vel_norm

    diff = np.asarray(soldier_pos, dtype=np.float64) - np.asarray(hostile_pos0, dtype=np.float64)
    diff_norm = float(np.linalg.norm(diff))
    if diff_norm > eps:
        return diff / diff_norm

    return np.zeros(3, dtype=np.float64)


def _rotate_horizontal(vec: np.ndarray, angle_rad: float) -> np.ndarray:
    """Rotate the (x, y) components of `vec` by `angle_rad` about +z (right-
    hand convention: positive angle is a counter-clockwise/left turn), the z
    component is left unchanged."""
    c, s = np.cos(angle_rad), np.sin(angle_rad)
    x, y, z = float(vec[0]), float(vec[1]), float(vec[2])
    return np.array([c * x - s * y, s * x + c * y, z], dtype=np.float64)


def _scenario_desired_velocity(scenario_id: str, base_dir: np.ndarray, config: EnvConfig,
                                eps: float) -> np.ndarray:
    """Build the (constant, open-loop) desired-velocity 3-vector for
    scenarios S1-S5 given the shared base direction."""
    v_e = config.v_e

    if scenario_id == "S1":
        norm = float(np.linalg.norm(base_dir))
        return v_e * (base_dir / norm) if norm > eps else np.zeros(3, dtype=np.float64)

    if scenario_id in ("S2", "S3"):
        angle = np.pi / 2.0 if scenario_id == "S2" else -np.pi / 2.0
        rotated = _rotate_horizontal(base_dir, angle)
        norm = float(np.linalg.norm(rotated))
        return v_e * (rotated / norm) if norm > eps else np.zeros(3, dtype=np.float64)

    if scenario_id in ("S4", "S5"):
        horiz = base_dir[:2]
        horiz_norm = float(np.linalg.norm(horiz))
        horiz_unit = horiz / horiz_norm if horiz_norm > eps else np.zeros(2, dtype=np.float64)
        vz = config.enemy_max_climb_rate if scenario_id == "S4" else -config.enemy_max_descent_rate
        return np.array([v_e * horiz_unit[0], v_e * horiz_unit[1], vz], dtype=np.float64)

    raise ValueError(f"'{scenario_id}' is not an open-loop scenario id")


def _hostile_dynamics_kwargs(config: EnvConfig) -> dict:
    """Hostile's PUBLIC physical limits only -- never evasion gain/radius."""
    return dict(
        max_speed=config.v_e,
        max_accel=config.enemy_max_accel,
        max_turn_rate_rad=np.radians(config.enemy_max_turn_rate_deg),
        max_climb_rate=config.enemy_max_climb_rate,
        max_descent_rate=config.enemy_max_descent_rate,
        dt=config.dt,
        eps=config.eps,
    )


def simulate_open_loop_scenario(scenario_id: str, hostile_pos0: np.ndarray,
                                 hostile_vel0: np.ndarray | None, soldier_pos: np.ndarray,
                                 horizon: int, config: EnvConfig) -> np.ndarray:
    """
    Propagate one of the open-loop scenarios (S1-S5) for `horizon` steps.

    Returns:
        positions: shape (horizon + 1, 3) array, `positions[0]` is
            `hostile_pos0` and `positions[k]` is the predicted position
            after `k` steps.
    """
    if scenario_id not in _OPEN_LOOP_SCENARIO_IDS:
        raise ValueError(f"'{scenario_id}' is not an open-loop scenario id; use simulate_max_away_scenario for S6")

    eps = config.eps
    hostile_pos0 = np.asarray(hostile_pos0, dtype=np.float64)
    base_dir = _base_direction(hostile_pos0, hostile_vel0, soldier_pos, eps)
    desired_velocity = _scenario_desired_velocity(scenario_id, base_dir, config, eps)
    dyn_kwargs = _hostile_dynamics_kwargs(config)

    positions = np.empty((horizon + 1, 3), dtype=np.float64)
    positions[0] = hostile_pos0
    vel = np.zeros(3, dtype=np.float64) if hostile_vel0 is None else np.asarray(hostile_vel0, dtype=np.float64)

    for k in range(1, horizon + 1):
        vel, _ = advance_velocity(current_velocity=vel, desired_velocity=desired_velocity, **dyn_kwargs)
        vel = np.asarray(vel, dtype=np.float64)
        positions[k] = positions[k - 1] + vel * config.dt

    return positions


def simulate_max_away_scenario(hostile_pos0: np.ndarray, hostile_vel0: np.ndarray | None,
                                defender_positions: np.ndarray, config: EnvConfig) -> np.ndarray:
    """
    Propagate the S6 (MAX_AWAY) closed-loop scenario: at each predicted
    step, the hostile's desired direction is AWAY from the candidate
    defender's predicted position at that same step -- a purely geometric,
    physical-limits-only construction. Does NOT use `enemy_evasion_gain`,
    `enemy_evasion_radius`, or the simulator's `_compute_enemy_evasion`
    formula/weave/heading-noise state.

    Args:
        defender_positions: shape (horizon + 1, 3), the ALREADY-ROLLED-OUT
            candidate defender trajectory (index 0 is the current true
            defender position, matching `hostile_pos0`'s reference step).

    Returns:
        positions: shape (horizon + 1, 3), same convention as
            `simulate_open_loop_scenario`.
    """
    horizon = defender_positions.shape[0] - 1
    eps = config.eps
    dyn_kwargs = _hostile_dynamics_kwargs(config)

    positions = np.empty((horizon + 1, 3), dtype=np.float64)
    positions[0] = np.asarray(hostile_pos0, dtype=np.float64)
    vel = np.zeros(3, dtype=np.float64) if hostile_vel0 is None else np.asarray(hostile_vel0, dtype=np.float64)

    for k in range(1, horizon + 1):
        away = positions[k - 1] - defender_positions[k - 1]
        away_norm = float(np.linalg.norm(away))
        if away_norm > eps:
            desired_dir = away / away_norm
        else:
            # Degenerate coincident-position geometry: no well-defined
            # "away" direction -- hold current heading via a zero command
            # (mirrors the project's zero-vector fallback convention).
            desired_dir = np.zeros(3, dtype=np.float64)
        desired_velocity = config.v_e * desired_dir

        vel, _ = advance_velocity(current_velocity=vel, desired_velocity=desired_velocity, **dyn_kwargs)
        vel = np.asarray(vel, dtype=np.float64)
        positions[k] = positions[k - 1] + vel * config.dt

    return positions


def simulate_all_scenarios(hostile_pos0: np.ndarray, hostile_vel0: np.ndarray | None,
                            soldier_pos: np.ndarray, defender_positions: np.ndarray,
                            config: EnvConfig) -> dict[str, np.ndarray]:
    """
    Convenience helper: simulate all six scenarios and return
    `{scenario_id: positions}`. `defender_positions` (shape (H+1, 3)) is
    required for S6 only but accepted here for a single convenient call
    site; S1-S5 ignore it entirely.
    """
    horizon = defender_positions.shape[0] - 1
    result: dict[str, np.ndarray] = {}
    for scenario_id in _OPEN_LOOP_SCENARIO_IDS:
        result[scenario_id] = simulate_open_loop_scenario(
            scenario_id, hostile_pos0, hostile_vel0, soldier_pos, horizon, config
        )
    result["S6"] = simulate_max_away_scenario(hostile_pos0, hostile_vel0, defender_positions, config)
    return result
