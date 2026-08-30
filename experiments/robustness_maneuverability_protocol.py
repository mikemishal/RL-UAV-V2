"""
Locked defender-maneuverability (accel x turn-rate) interaction grid protocol
-- SINGLE SOURCE OF TRUTH.

A 3x3 Cartesian grid over (defender_max_accel, defender_max_turn_rate_deg).
This experiment does NOT introduce new dynamics -- the environment already
has bounded acceleration/turn-rate/climb/descent authority for the defender
(see SoldierEnv._move_defender / _apply_bounded_dynamics). It evaluates
robustness to the VALUES of those already-existing constraints, directly
responding to the reviewer criticism that the original defender could
instantaneously change heading with no acceleration/turn-rate limits.

Defender speed (v_d=18), vertical-rate limits, hostile dynamics, sensing,
evasion, reward, and every other parameter are held fixed. Action semantics
are unchanged: desired_velocity = v_d * normalize(action); only how rapidly
the vehicle's actual velocity can approach that desired vector changes.

Seed ranges (disjoint from all prior protocol ranges -- see
test_robustness_maneuverability.py):

    Training seeds                                : 42, 43, 44
    Validation seeds                               : 10000..10099
    Diagnostic seeds                               : 12000..12199
    Final nominal seeds (OPENED, locked)            : 20000..24999
    Acquisition ablation seeds (OPENED, locked)     : 25000..25999
    Detection-radius sweep seeds (OPENED, locked)   : 30000..30999
    Measurement-noise sweep seeds (OPENED, locked)  : 31000..31999
    1-D mobility sweep seeds (OPENED, locked)       : 32000..32999
    Hostile-evasion sweep seeds (OPENED, locked)    : 33000..33999
    MANEUVERABILITY_SEEDS (OPENED here)             : 34000..34999
    Mobility-grid sweep seeds (OPENED, locked)      : 35000..35499 (never rerun)
    Untouched (this task must not evaluate)         : >=35500
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
# MANEUVERABILITY GRID -- prespecified BEFORE opening seed 34000
# =============================================================================
DEFENDER_MAX_ACCEL_VALUES: tuple[float, ...] = (4.0, 8.0, 12.0)     # m/s^2: restrictive / nominal (PPO training) / permissive
DEFENDER_MAX_TURN_RATE_VALUES: tuple[float, ...] = (45.0, 90.0, 135.0)  # deg/s: restrictive / nominal (PPO training) / permissive
NOMINAL_DEFENDER_MAX_ACCEL = 8.0
NOMINAL_DEFENDER_MAX_TURN_RATE_DEG = 90.0
assert NOMINAL_DEFENDER_MAX_ACCEL in DEFENDER_MAX_ACCEL_VALUES
assert NOMINAL_DEFENDER_MAX_TURN_RATE_DEG in DEFENDER_MAX_TURN_RATE_VALUES
# Note: these are simulation parameters for a robustness study, NOT
# hardware-validated or claimed-realistic airframe specifications.

# Frozen nominal mobility -- never swept in this study (item 12/15).
NOMINAL_DEFENDER_SPEED = 18.0
NOMINAL_HOSTILE_SPEED = 12.0

# Vertical maneuverability is FIXED -- this study isolates horizontal
# heading authority + velocity-vector acceleration authority only (item 14).
FIXED_DEFENDER_MAX_CLIMB_RATE = 6.0
FIXED_DEFENDER_MAX_DESCENT_RATE = 6.0

# Hostile dynamics -- FIXED, never swept (item 12).
FIXED_ENEMY_MAX_ACCEL = 6.0
FIXED_ENEMY_MAX_TURN_RATE_DEG = 75.0
FIXED_ENEMY_MAX_CLIMB_RATE = 5.0
FIXED_ENEMY_MAX_DESCENT_RATE = 5.0

# Sensing -- FIXED, never swept (item 12).
FIXED_DETECTION_RADIUS = 15.0
FIXED_MEASUREMENT_VAR = 0.5
FIXED_PROCESS_VAR = 1.0
FIXED_LEAD_TIME = 0.0

# Hostile evasion -- FIXED at the locked nominal condition (item 12).
FIXED_ENEMY_EVASION_ENABLED = True
FIXED_ENEMY_EVASION_RADIUS = 20.0
FIXED_ENEMY_EVASION_GAIN = 0.75


@dataclass(frozen=True)
class ManeuverabilityCell:
    """One of the 9 Cartesian (accel, turn_rate) grid cells."""

    defender_max_accel: float
    defender_max_turn_rate_deg: float

    @property
    def is_nominal(self) -> bool:
        return (
            self.defender_max_accel == NOMINAL_DEFENDER_MAX_ACCEL
            and self.defender_max_turn_rate_deg == NOMINAL_DEFENDER_MAX_TURN_RATE_DEG
        )

    @property
    def cell_label(self) -> str:
        return f"accel_{self.defender_max_accel:g}_turn_{self.defender_max_turn_rate_deg:g}"


def _build_cells() -> tuple[ManeuverabilityCell, ...]:
    return tuple(
        ManeuverabilityCell(accel, turn)
        for accel, turn in product(DEFENDER_MAX_ACCEL_VALUES, DEFENDER_MAX_TURN_RATE_VALUES)
    )


MANEUVERABILITY_CELLS: tuple[ManeuverabilityCell, ...] = _build_cells()
EXPECTED_MANEUVERABILITY_CELLS = 9
assert len(MANEUVERABILITY_CELLS) == EXPECTED_MANEUVERABILITY_CELLS
assert len({(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS}) == EXPECTED_MANEUVERABILITY_CELLS
assert sum(1 for c in MANEUVERABILITY_CELLS if c.is_nominal) == 1

RESTRICTIVE_CELL = ManeuverabilityCell(4.0, 45.0)
PERMISSIVE_CELL = ManeuverabilityCell(12.0, 135.0)
NOMINAL_CELL = ManeuverabilityCell(NOMINAL_DEFENDER_MAX_ACCEL, NOMINAL_DEFENDER_MAX_TURN_RATE_DEG)
assert RESTRICTIVE_CELL in MANEUVERABILITY_CELLS
assert PERMISSIVE_CELL in MANEUVERABILITY_CELLS
assert NOMINAL_CELL in MANEUVERABILITY_CELLS

# =============================================================================
# MANEUVERABILITY SWEEP SEEDS -- the previously reserved range, now opened
# =============================================================================
MANEUVERABILITY_SEED_OFFSET = 34_000
MANEUVERABILITY_EPISODES = 1_000
MANEUVERABILITY_SEEDS: range = range(
    MANEUVERABILITY_SEED_OFFSET, MANEUVERABILITY_SEED_OFFSET + MANEUVERABILITY_EPISODES
)

# The previously-opened mobility-grid range -- locked data, never rerun here.
MOBILITY_GRID_SEED_OFFSET = 35_000
MOBILITY_GRID_EPISODES = 500
MOBILITY_GRID_SEEDS: range = range(MOBILITY_GRID_SEED_OFFSET, MOBILITY_GRID_SEED_OFFSET + MOBILITY_GRID_EPISODES)
MAX_ALLOWED_SEED = MANEUVERABILITY_SEED_OFFSET + MANEUVERABILITY_EPISODES - 1  # 34999

# =============================================================================
# STATISTICS PROTOCOL (unchanged across every locked experiment)
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def assert_maneuverability_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked maneuverability sweep seed range."""
    if seed not in MANEUVERABILITY_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in MANEUVERABILITY_SEEDS "
            f"({MANEUVERABILITY_SEED_OFFSET}..{MANEUVERABILITY_SEED_OFFSET + MANEUVERABILITY_EPISODES - 1})."
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
class ManeuverabilityPolicyInstance:
    """One concrete, runnable policy instance evaluated across the 9-cell grid."""

    paper_method: str
    policy_instance: str
    estimator: str
    training_seed: int | None
    model_path: Path | None


def build_maneuverability_policy_instances() -> tuple[ManeuverabilityPolicyInstance, ...]:
    instances: list[ManeuverabilityPolicyInstance] = []
    for method in SCRIPTED_METHODS:
        instances.append(ManeuverabilityPolicyInstance(
            paper_method=method, policy_instance=method,
            estimator=_estimator_for_method(method), training_seed=None, model_path=None,
        ))
    for method in PPO_METHODS:
        track = _track_for_ppo_method(method)
        for seed in PPO_TRAINING_SEEDS:
            instances.append(ManeuverabilityPolicyInstance(
                paper_method=method, policy_instance=f"{method}_seed_{seed}",
                estimator=_estimator_for_method(method), training_seed=seed,
                model_path=frozen_model_path(track, seed),
            ))
    return tuple(instances)


MANEUVERABILITY_POLICY_INSTANCES: tuple[ManeuverabilityPolicyInstance, ...] = build_maneuverability_policy_instances()
EXPECTED_N_POLICY_INSTANCES = 10
EXPECTED_RAW_ROWS = EXPECTED_MANEUVERABILITY_CELLS * EXPECTED_N_POLICY_INSTANCES * MANEUVERABILITY_EPISODES  # 90,000


def build_policy(instance: ManeuverabilityPolicyInstance):
    """Instantiate the concrete policy for a ManeuverabilityPolicyInstance. 'greedy' is
    ALWAYS the corrected GreedyInterceptPolicy from commit 0dbdadd -- never
    reconstruct the old escort-only behavior."""
    if instance.model_path is None:
        return get_policy(instance.paper_method)
    return get_policy(instance.paper_method, model_path=str(instance.model_path), deterministic=True)


def build_maneuverability_env_config(estimator: str, defender_max_accel: float, defender_max_turn_rate_deg: float) -> EnvConfig:
    """
    Nominal EnvConfig with ONLY defender_max_accel, defender_max_turn_rate_deg
    (and, for the Kalman track, use_kalman_tracking) changed. Defender speed,
    vertical-rate limits, hostile dynamics, sensing, and evasion are all
    explicitly passed at their FIXED nominal values (not swept) -- see
    assert_only_maneuverability_differs().
    """
    return EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        v_d=NOMINAL_DEFENDER_SPEED,
        v_e=NOMINAL_HOSTILE_SPEED,
        defender_max_accel=defender_max_accel,
        defender_max_turn_rate_deg=defender_max_turn_rate_deg,
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


def assert_only_maneuverability_differs(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """
    Programmatically verify the ONLY behavioral EnvConfig fields differing
    between a swept config and the nominal config are defender_max_accel,
    defender_max_turn_rate_deg (and use_kalman_tracking, expected between
    tracks). Raises AssertionError otherwise -- catches any accidental
    mobility, vertical-rate, sensing, evasion, or reward drift.
    """
    swept_dict = dataclasses.asdict(swept_config)
    nominal_dict = dataclasses.asdict(nominal_config)
    allowed_diff_keys = {"defender_max_accel", "defender_max_turn_rate_deg", "use_kalman_tracking"}
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    unexpected = diffs - allowed_diff_keys
    assert not unexpected, f"unexpected EnvConfig fields differ from nominal: {unexpected}"


# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE opening seed 34000)
# =============================================================================
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_lead", "ppo", "lead"),
)
ESTIMATOR_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("kalman_greedy_vs_greedy", "kalman_greedy", "greedy"),
    ("ppo_kalman_vs_ppo", "ppo_kalman", "ppo"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_greedy", "ppo", "greedy"),
    ("ppo_vs_pn", "ppo", "pn"),
    ("lead_vs_greedy", "lead", "greedy"),
    ("pn_vs_greedy", "pn", "greedy"),
)
ALL_COMPARISONS: tuple[tuple[str, str, str], ...] = PRIMARY_COMPARISONS + ESTIMATOR_COMPARISONS + SECONDARY_COMPARISONS

# =============================================================================
# PRESPECIFIED RESEARCH QUESTIONS (item 17) -- descriptive labels only, no
# assumed direction; answered post-hoc from the measured cells.
# =============================================================================
RESEARCH_QUESTIONS: tuple[str, ...] = (
    "Q1. How does interception success depend jointly on defender acceleration and turn rate?",
    "Q2. Which constraint is more influential over the tested range?",
    "Q3. Is there a strong acceleration x turn-rate interaction?",
    "Q4. Does PPO degrade more slowly than Lead as defender maneuverability becomes restrictive?",
    "Q5. Does Lead outperform PPO in any constrained-dynamics region?",
    "Q6. Does PPO outperform Lead in any constrained-dynamics region?",
    "Q7. Does Kalman filtering help under any defender-maneuverability condition?",
    "Q8. How much does performance improve when maneuverability is relaxed beyond the training condition?",
)

# Feasibility coverage thresholds (item 36) -- discrete measured-cell coverage only.
FEASIBILITY_SUCCESS_THRESHOLDS: tuple[float, ...] = (0.70, 0.80, 0.90)

# =============================================================================
# OUTPUT LOCATION (gitignored; never committed)
# =============================================================================
ROBUSTNESS_MANEUVERABILITY_OUTPUT_ROOT = PROJECT_ROOT / "results" / "robustness" / "maneuverability"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
MANEUVERABILITY_SUCCESS_SUMMARY_FILENAME = "maneuverability_success_summary.csv"
PPO_TRAINING_SEED_SUMMARY_FILENAME = "ppo_training_seed_summary.csv"
PPO_VS_LEAD_COMPARISONS_FILENAME = "ppo_vs_lead_comparisons.csv"
ESTIMATOR_COMPARISONS_FILENAME = "estimator_comparisons.csv"
ACCELERATION_EFFECT_SUMMARY_FILENAME = "acceleration_effect_summary.csv"
TURN_RATE_EFFECT_SUMMARY_FILENAME = "turn_rate_effect_summary.csv"
INTERACTION_SUMMARY_FILENAME = "interaction_summary.csv"
FEASIBILITY_SUMMARY_FILENAME = "feasibility_summary.csv"
ACQUISITION_SUMMARY_FILENAME = "acquisition_summary.csv"
POST_DETECTION_SUMMARY_FILENAME = "post_detection_summary.csv"
FAILURE_MODE_SUMMARY_FILENAME = "failure_mode_summary.csv"
OPERATIONAL_SUMMARY_FILENAME = "operational_summary.csv"
ESTIMATOR_SUMMARY_FILENAME = "estimator_summary.csv"
MANEUVERABILITY_SUMMARY_FILENAME = "maneuverability_summary.json"
