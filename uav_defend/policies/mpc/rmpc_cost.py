"""
RMPC robust objective: defender rollout, per-scenario terminal-priority
scoring (mirroring `SoldierEnv.step()`'s own terminal-check order and
reward magnitudes), and robust (min-over-scenarios) aggregation.

Per `docs/rmpc_baseline_design.md` Section 10: introduces ZERO new tunable
weight constants beyond the four already-fixed `EnvConfig` reward
magnitudes (`reward_intercept`, `reward_soldier_caught`,
`reward_unsafe_intercept`, `reward_progress_scale`).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.dynamics.constrained_point_mass import advance_velocity
from uav_defend.policies.mpc.hostile_scenarios import SCENARIO_IDS, simulate_all_scenarios


@dataclass
class ScenarioOutcome:
    """Per-scenario predicted-terminal-event outcome for one candidate `U`."""

    terminal_step: int | None  # predicted step k in [1, H] at which a terminal event fired, else None
    terminal_type: str | None  # "soldier_caught" | "unsafe_intercept" | "intercepted" | None
    score: float


@dataclass
class CandidateEvaluation:
    """Full robust-evaluation result for one candidate control sequence `U`."""

    robust_score: float
    scenario_scores: dict[str, float]
    scenario_outcomes: dict[str, ScenarioOutcome]
    defender_positions: np.ndarray  # shape (H + 1, 3)


def rollout_defender(directions: np.ndarray, defender_pos0: np.ndarray, defender_vel0: np.ndarray,
                      config: EnvConfig) -> np.ndarray:
    """
    Propagate the candidate defender trajectory for control sequence
    `directions` (shape (H, 3), already-normalized unit directions, or the
    zero vector for a "hold/decelerate" step) using the defender's own
    physical limits.

    Returns:
        positions: shape (H + 1, 3), `positions[0] == defender_pos0`.
    """
    horizon = directions.shape[0]
    dyn_kwargs = dict(
        max_speed=config.v_d,
        max_accel=config.defender_max_accel,
        max_turn_rate_rad=np.radians(config.defender_max_turn_rate_deg),
        max_climb_rate=config.defender_max_climb_rate,
        max_descent_rate=config.defender_max_descent_rate,
        dt=config.dt,
        eps=config.eps,
    )

    positions = np.empty((horizon + 1, 3), dtype=np.float64)
    positions[0] = np.asarray(defender_pos0, dtype=np.float64)
    vel = np.asarray(defender_vel0, dtype=np.float64).copy()

    for k in range(1, horizon + 1):
        desired_velocity = config.v_d * directions[k - 1]
        vel, _ = advance_velocity(current_velocity=vel, desired_velocity=desired_velocity, **dyn_kwargs)
        vel = np.asarray(vel, dtype=np.float64)
        positions[k] = positions[k - 1] + vel * config.dt

    return positions


def score_trajectory(defender_positions: np.ndarray, hostile_positions: np.ndarray,
                      soldier_pos_fixed: np.ndarray, config: EnvConfig) -> ScenarioOutcome:
    """
    Score one (defender, hostile) predicted trajectory pair against a fixed
    soldier reference position, mirroring `SoldierEnv.step()`'s own
    per-step terminal-check PRIORITY ORDER exactly: `soldier_caught` is
    checked before `unsafe_intercept`, which is checked before
    `intercepted`, at every predicted step `k = 1..H`.
    """
    horizon = defender_positions.shape[0] - 1
    soldier_pos_fixed = np.asarray(soldier_pos_fixed, dtype=np.float64)

    dist_de_0 = float(np.linalg.norm(defender_positions[0] - hostile_positions[0]))

    for k in range(1, horizon + 1):
        dist_es_k = float(np.linalg.norm(hostile_positions[k] - soldier_pos_fixed))
        if dist_es_k <= config.threat_radius:
            return ScenarioOutcome(terminal_step=k, terminal_type="soldier_caught",
                                    score=config.reward_soldier_caught)

        dist_de_k = float(np.linalg.norm(defender_positions[k] - hostile_positions[k]))
        if dist_de_k <= config.intercept_radius:
            if dist_es_k <= config.unsafe_intercept_radius:
                return ScenarioOutcome(terminal_step=k, terminal_type="unsafe_intercept",
                                        score=config.reward_unsafe_intercept)
            return ScenarioOutcome(terminal_step=k, terminal_type="intercepted",
                                    score=config.reward_intercept)

    dist_de_h = float(np.linalg.norm(defender_positions[horizon] - hostile_positions[horizon]))
    # Progress-only fallback: dist_de_0 is fixed across all candidates at a
    # given decision, so maximizing (dist_de_0 - dist_de_h) is equivalent to
    # minimizing terminal defender-hostile distance -- no additional
    # `-dist_de_h` term is needed or included (that term would be an
    # undeclared extra weight beyond `reward_progress_scale`, which is the
    # ONLY tunable weight this objective introduces).
    fallback_score = config.reward_progress_scale * (dist_de_0 - dist_de_h)
    return ScenarioOutcome(terminal_step=None, terminal_type=None, score=fallback_score)


def evaluate_candidate(directions: np.ndarray, defender_pos0: np.ndarray, defender_vel0: np.ndarray,
                        hostile_pos0: np.ndarray, hostile_vel0: np.ndarray | None,
                        soldier_pos_fixed: np.ndarray, config: EnvConfig) -> CandidateEvaluation:
    """
    Full robust-objective evaluation of one candidate control sequence: roll
    out the defender once, simulate all six hostile scenarios (S1-S5
    open-loop, S6 closed-loop against the defender rollout), score each
    scenario pair, and aggregate via `min` (Section 10.2).
    """
    defender_positions = rollout_defender(directions, defender_pos0, defender_vel0, config)

    hostile_trajectories = simulate_all_scenarios(
        hostile_pos0, hostile_vel0, soldier_pos_fixed, defender_positions, config
    )

    scenario_scores: dict[str, float] = {}
    scenario_outcomes: dict[str, ScenarioOutcome] = {}
    for scenario_id in SCENARIO_IDS:
        outcome = score_trajectory(defender_positions, hostile_trajectories[scenario_id],
                                    soldier_pos_fixed, config)
        scenario_scores[scenario_id] = outcome.score
        scenario_outcomes[scenario_id] = outcome

    robust_score = min(scenario_scores.values())

    return CandidateEvaluation(
        robust_score=robust_score,
        scenario_scores=scenario_scores,
        scenario_outcomes=scenario_outcomes,
        defender_positions=defender_positions,
    )
