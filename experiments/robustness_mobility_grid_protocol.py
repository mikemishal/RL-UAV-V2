"""
Locked defender-hostile speed interaction grid protocol -- SINGLE SOURCE OF TRUTH.

A NEW, separate, confirmatory 5x5 Cartesian (v_d, v_e) grid -- NOT an
extension of the opened 1-D mobility experiment (results/robustness/mobility/,
seeds 32000-32999). See experiments/mobility_dynamics_audit.py and
results/audits/mobility_dynamics/ for the audit motivating this grid:

  - No software/configuration bug was found in the 1-D mobility experiment.
  - v_d is BOTH the defender's maximum speed AND the commanded/desired
    cruise-speed magnitude for every nonzero action (any nonzero action
    normalizes away its own magnitude -- see GreedyInterceptPolicy usage and
    SoldierEnv._move_defender()).
  - v_e sets the hostile's desired velocity magnitude, but hostile
    acceleration/turn-rate/climb/descent limits stay FIXED regardless of v_e,
    so higher v_e does not proportionally increase vertical maneuverability
    -- this decoupling interacts with the environment's full-3D pursuit
    geometry and full-3D threat/intercept distance definitions.
  - Consequently, performance is NOT expected to depend monotonically on the
    simple ratio mu = v_d / v_e; this grid explicitly tests that.

This module deliberately does NOT call itself a "mobility-ratio sweep" --
it is the "defender-hostile speed interaction grid" (a two-parameter
operating-envelope study). mu is recorded as a descriptive field only.

Seed ranges (disjoint from all prior protocol ranges -- see
test_robustness_mobility_grid.py):

    Training seeds                                : 42, 43, 44
    Validation seeds                               : 10000..10099
    Diagnostic seeds                               : 12000..12199
    Final nominal seeds (OPENED, locked)            : 20000..24999
    Acquisition ablation seeds (OPENED, locked)     : 25000..25999
    Detection-radius sweep seeds (OPENED, locked)   : 30000..30999
    Measurement-noise sweep seeds (OPENED, locked)  : 31000..31999
    1-D mobility sweep seeds (OPENED, locked)       : 32000..32999
    Mobility-dynamics audit diagnostic seeds        : 18000..18099 (non-publication)
    Reserved (untouched by this task):
        hostile-evasion robustness                  : 33000..33999
        maneuverability/dynamics robustness          : 34000..34999
    MOBILITY_GRID_SEEDS (OPENED here)               : 35000..35499
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from itertools import product
from pathlib import Path

from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.final_experiment_protocol import PPO_TRAINING_SEEDS, frozen_model_path

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# NOMINAL MOBILITY (must appear exactly once in the grid)
# =============================================================================
NOMINAL_DEFENDER_SPEED = 18.0
NOMINAL_HOSTILE_SPEED = 12.0

# =============================================================================
# FULL CARTESIAN GRID -- prespecified BEFORE opening seed 35000
# =============================================================================
DEFENDER_SPEED_VALUES: tuple[float, ...] = (12.0, 15.0, 18.0, 21.0, 24.0)
HOSTILE_SPEED_VALUES: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0, 16.0)
assert NOMINAL_DEFENDER_SPEED in DEFENDER_SPEED_VALUES
assert NOMINAL_HOSTILE_SPEED in HOSTILE_SPEED_VALUES

# The four EXACT mu=1.5 cells -- a prespecified key equal-ratio comparison
# (item 42/43): same relative mobility ratio, four different absolute speeds.
EQUAL_RATIO_MU = 1.5
EQUAL_RATIO_CELLS: tuple[tuple[float, float], ...] = ((15.0, 10.0), (18.0, 12.0), (21.0, 14.0), (24.0, 16.0))
for _v_d, _v_e in EQUAL_RATIO_CELLS:
    assert abs(_v_d / _v_e - EQUAL_RATIO_MU) < 1e-9
    assert _v_d in DEFENDER_SPEED_VALUES and _v_e in HOSTILE_SPEED_VALUES

# Fixed maneuverability limits -- NEVER scaled with speed (item 14).
FIXED_DEFENDER_MAX_ACCEL = 8.0
FIXED_DEFENDER_MAX_TURN_RATE_DEG = 90.0
FIXED_DEFENDER_MAX_CLIMB_RATE = 6.0
FIXED_DEFENDER_MAX_DESCENT_RATE = 6.0
FIXED_ENEMY_MAX_ACCEL = 6.0
FIXED_ENEMY_MAX_TURN_RATE_DEG = 75.0
FIXED_ENEMY_MAX_CLIMB_RATE = 5.0
FIXED_ENEMY_MAX_DESCENT_RATE = 5.0

# Fixed sensing -- NEVER changed by this grid (item 15).
FIXED_DETECTION_RADIUS = 15.0
FIXED_MEASUREMENT_VAR = 0.5
FIXED_PROCESS_VAR = 1.0
FIXED_LEAD_TIME = 0.0

# Fixed hostile guidance/evasion -- NEVER changed by this grid (item 16).
FIXED_ENEMY_EVASION_ENABLED = True
FIXED_ENEMY_EVASION_RADIUS = 20.0
FIXED_ENEMY_EVASION_GAIN = 0.75


@dataclass(frozen=True)
class GridCell:
    """One of the 25 Cartesian (v_d, v_e) grid cells."""

    defender_speed: float
    hostile_speed: float

    @property
    def mobility_ratio(self) -> float:
        return self.defender_speed / self.hostile_speed

    @property
    def is_nominal(self) -> bool:
        return self.defender_speed == NOMINAL_DEFENDER_SPEED and self.hostile_speed == NOMINAL_HOSTILE_SPEED


def _build_grid_cells() -> tuple[GridCell, ...]:
    return tuple(GridCell(v_d, v_e) for v_d, v_e in product(DEFENDER_SPEED_VALUES, HOSTILE_SPEED_VALUES))


GRID_CELLS: tuple[GridCell, ...] = _build_grid_cells()
EXPECTED_GRID_CELLS = 25
assert len(GRID_CELLS) == EXPECTED_GRID_CELLS
assert len({(c.defender_speed, c.hostile_speed) for c in GRID_CELLS}) == EXPECTED_GRID_CELLS
assert sum(1 for c in GRID_CELLS if c.is_nominal) == 1

# =============================================================================
# MOBILITY-GRID SEEDS -- a previously untouched range
# =============================================================================
MOBILITY_GRID_SEED_OFFSET = 35_000
MOBILITY_GRID_EPISODES = 500
MOBILITY_GRID_SEEDS: range = range(MOBILITY_GRID_SEED_OFFSET, MOBILITY_GRID_SEED_OFFSET + MOBILITY_GRID_EPISODES)

# Reserved for LATER robustness sweeps -- do NOT evaluate in this task.
HOSTILE_EVASION_SWEEP_SEED_OFFSET = 33_000
DYNAMICS_SWEEP_SEED_OFFSET = 34_000
RESERVED_UNTOUCHED_SEEDS: range = range(HOSTILE_EVASION_SWEEP_SEED_OFFSET, MOBILITY_GRID_SEED_OFFSET)
MAX_ALLOWED_SEED = MOBILITY_GRID_SEED_OFFSET + MOBILITY_GRID_EPISODES - 1  # 35499

# =============================================================================
# STATISTICS PROTOCOL (unchanged across every locked experiment)
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def assert_grid_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked mobility-grid seed range."""
    if seed not in MOBILITY_GRID_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in MOBILITY_GRID_SEEDS "
            f"({MOBILITY_GRID_SEED_OFFSET}..{MOBILITY_GRID_SEED_OFFSET + MOBILITY_GRID_EPISODES - 1})."
        )


# =============================================================================
# SIX PRIMARY PAPER METHODS ONLY (no PN-Kalman / Lead-Kalman)
# =============================================================================
PRIMARY_PAPER_METHODS: tuple[str, ...] = ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
SCRIPTED_METHODS: tuple[str, ...] = ("greedy", "kalman_greedy", "pn", "lead")
PPO_METHODS: tuple[str, ...] = ("ppo", "ppo_kalman")


def _estimator_for_method(method: str) -> str:
    return "kalman" if method in ("kalman_greedy", "ppo_kalman") else "measurement"


def _track_for_ppo_method(method: str) -> str:
    return "kalman" if method == "ppo_kalman" else "direct"


@dataclasses.dataclass(frozen=True)
class GridPolicyInstance:
    """One concrete, runnable policy instance evaluated across the 25-cell grid."""

    paper_method: str
    policy_instance: str
    estimator: str
    training_seed: int | None
    model_path: Path | None


def build_grid_policy_instances() -> tuple[GridPolicyInstance, ...]:
    instances: list[GridPolicyInstance] = []
    for method in SCRIPTED_METHODS:
        instances.append(GridPolicyInstance(
            paper_method=method, policy_instance=method,
            estimator=_estimator_for_method(method), training_seed=None, model_path=None,
        ))
    for method in PPO_METHODS:
        track = _track_for_ppo_method(method)
        for seed in PPO_TRAINING_SEEDS:
            instances.append(GridPolicyInstance(
                paper_method=method, policy_instance=f"{method}_seed_{seed}",
                estimator=_estimator_for_method(method), training_seed=seed,
                model_path=frozen_model_path(track, seed),
            ))
    return tuple(instances)


GRID_POLICY_INSTANCES: tuple[GridPolicyInstance, ...] = build_grid_policy_instances()
EXPECTED_POLICY_INSTANCES = 10
EXPECTED_RAW_ROWS = EXPECTED_GRID_CELLS * EXPECTED_POLICY_INSTANCES * MOBILITY_GRID_EPISODES  # 125,000


def build_policy(instance: GridPolicyInstance):
    """Instantiate the concrete policy for a GridPolicyInstance. 'greedy' is
    ALWAYS the corrected GreedyInterceptPolicy from commit 0dbdadd -- never
    reconstruct the old escort-only behavior."""
    if instance.model_path is None:
        return get_policy(instance.paper_method)
    return get_policy(instance.paper_method, model_path=str(instance.model_path), deterministic=True)


def build_grid_env_config(estimator: str, defender_speed: float, hostile_speed: float) -> EnvConfig:
    """
    Nominal EnvConfig with ONLY v_d, v_e (and, for the Kalman track,
    use_kalman_tracking) changed. All maneuverability, sensing, and evasion
    parameters are explicitly passed at their FIXED nominal values (not
    swept) -- see assert_only_grid_speed_differs().
    """
    return EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        v_d=defender_speed,
        v_e=hostile_speed,
        defender_max_accel=FIXED_DEFENDER_MAX_ACCEL,
        defender_max_turn_rate_deg=FIXED_DEFENDER_MAX_TURN_RATE_DEG,
        defender_max_climb_rate=FIXED_DEFENDER_MAX_CLIMB_RATE,
        defender_max_descent_rate=FIXED_DEFENDER_MAX_DESCENT_RATE,
        enemy_max_accel=FIXED_ENEMY_MAX_ACCEL,
        enemy_max_turn_rate_deg=FIXED_ENEMY_MAX_TURN_RATE_DEG,
        enemy_max_climb_rate=FIXED_ENEMY_MAX_CLIMB_RATE,
        enemy_max_descent_rate=FIXED_ENEMY_MAX_DESCENT_RATE,
        detection_radius=FIXED_DETECTION_RADIUS,
        measurement_var=FIXED_MEASUREMENT_VAR,
        process_var=FIXED_PROCESS_VAR,
        lead_time=FIXED_LEAD_TIME,
        enemy_evasion_enabled=FIXED_ENEMY_EVASION_ENABLED,
        enemy_evasion_radius=FIXED_ENEMY_EVASION_RADIUS,
        enemy_evasion_gain=FIXED_ENEMY_EVASION_GAIN,
    )


def assert_only_grid_speed_differs(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """
    Programmatically verify the ONLY behavioral EnvConfig fields differing
    between a swept config and the nominal config are v_d/v_e (and
    use_kalman_tracking, expected between tracks). Raises AssertionError
    otherwise -- catches any accidental maneuverability, sensing, evasion,
    or reward drift.
    """
    swept_dict = dataclasses.asdict(swept_config)
    nominal_dict = dataclasses.asdict(nominal_config)
    allowed_diff_keys = {"v_d", "v_e", "use_kalman_tracking"}
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    unexpected = diffs - allowed_diff_keys
    assert not unexpected, f"unexpected EnvConfig fields differ from nominal: {unexpected}"


# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE opening seed 35000)
# =============================================================================
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_lead", "ppo", "lead"),                             # items 21.Q5-Q7, 34, 38
)
ESTIMATOR_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("kalman_greedy_vs_greedy", "kalman_greedy", "greedy"),     # item 35
    ("ppo_kalman_vs_ppo", "ppo_kalman", "ppo"),                  # item 36
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_greedy", "ppo", "greedy"),
    ("ppo_vs_pn", "ppo", "pn"),
    ("lead_vs_greedy", "lead", "greedy"),
    ("pn_vs_greedy", "pn", "greedy"),
)
ALL_COMPARISONS: tuple[tuple[str, str, str], ...] = PRIMARY_COMPARISONS + ESTIMATOR_COMPARISONS + SECONDARY_COMPARISONS

# =============================================================================
# OUTPUT LOCATION (gitignored; never committed)
# =============================================================================
ROBUSTNESS_MOBILITY_GRID_OUTPUT_ROOT = PROJECT_ROOT / "results" / "robustness" / "mobility_grid"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
GRID_SUCCESS_SUMMARY_FILENAME = "grid_success_summary.csv"
PPO_TRAINING_SEED_SUMMARY_FILENAME = "ppo_training_seed_summary.csv"
PAIRWISE_COMPARISONS_FILENAME = "pairwise_comparisons.csv"
FEASIBILITY_REGION_SUMMARY_FILENAME = "feasibility_region_summary.csv"
PPO_VS_LEAD_ENVELOPE_FILENAME = "ppo_vs_lead_envelope.csv"
EQUAL_RATIO_SUMMARY_FILENAME = "equal_ratio_summary.csv"
SWEET_SPOT_SUMMARY_FILENAME = "sweet_spot_summary.csv"
INTERACTION_SUMMARY_FILENAME = "interaction_summary.csv"
FAILURE_MODE_SUMMARY_FILENAME = "failure_mode_summary.csv"
OPERATIONAL_SUMMARY_FILENAME = "operational_summary.csv"
ESTIMATOR_SUMMARY_FILENAME = "estimator_summary.csv"
MOBILITY_GRID_SUMMARY_FILENAME = "mobility_grid_summary.json"
