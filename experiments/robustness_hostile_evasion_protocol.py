"""
Locked hostile-evasion-strength robustness sweep protocol -- SINGLE SOURCE OF TRUTH.

Varies ONLY the reactive-evasion gain lambda_e = enemy_evasion_gain, holding
enemy_evasion_radius=20.0 and enemy_evasion_enabled=True fixed at every point
(gain=0 is mathematically equivalent to no reactive-evasion contribution --
see SoldierEnv._compute_enemy_evasion: activation is still computed from true
3-D defender-enemy distance, but effective_weight = gain * activation = 0
when gain=0, so evasion_component is the zero vector by construction, NOT by
disabling the mechanism).

    alpha(d)              = clip(1 - d/R_e, 0, 1)          (activation)
    effective_weight       = lambda_e * alpha(d)             (info["enemy_evasion_weight"])
    evasion_component     = effective_weight * u_away

This addresses the reviewer concern that the hostile UAV was insufficiently
evasive. gain=0 is NOT "no maneuvering" -- the hostile still pursues, weaves,
and has stochastic heading noise; only the reactive-away component is zero.
gain=1.0 is the maximum TESTED gain under the current parameterization, not
an optimal/adversarial evasion strategy.

PPO/PPO-Kalman were trained only at gain=0.75 (nominal); gains 0/0.25/0.5/1.0
are zero-shot evasion-strength generalization conditions.

Seed ranges (disjoint from all prior protocol ranges -- see
test_robustness_hostile_evasion.py):

    Training seeds                                : 42, 43, 44
    Validation seeds                               : 10000..10099
    Diagnostic seeds                               : 12000..12199
    Final nominal seeds (OPENED, locked)            : 20000..24999
    Acquisition ablation seeds (OPENED, locked)     : 25000..25999
    Detection-radius sweep seeds (OPENED, locked)   : 30000..30999
    Measurement-noise sweep seeds (OPENED, locked)  : 31000..31999
    1-D mobility sweep seeds (OPENED, locked)       : 32000..32999
    HOSTILE_EVASION_SEEDS (OPENED here)             : 33000..33999
    Reserved (untouched by this task):
        maneuverability/dynamics robustness          : 34000..34999
    Mobility-grid sweep seeds (OPENED, locked)      : 35000..35499
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.final_experiment_protocol import PPO_TRAINING_SEEDS, frozen_model_path

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# EVASION-GAIN GRID -- prespecified BEFORE opening seed 33000
# =============================================================================
EVASION_GAIN_VALUES: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00)
NOMINAL_EVASION_GAIN = 0.75  # PPO/PPO-Kalman training condition
EVASION_RADIUS = 20.0        # FIXED at every gain -- only strength is swept, not onset distance
for _g in EVASION_GAIN_VALUES:
    assert 0.0 <= _g <= 1.0
assert NOMINAL_EVASION_GAIN in EVASION_GAIN_VALUES

# Nominal mobility/dynamics -- frozen at every gain (item 8).
NOMINAL_DEFENDER_SPEED = 18.0
NOMINAL_HOSTILE_SPEED = 12.0
FIXED_DEFENDER_MAX_ACCEL = 8.0
FIXED_DEFENDER_MAX_TURN_RATE_DEG = 90.0
FIXED_DEFENDER_MAX_CLIMB_RATE = 6.0
FIXED_DEFENDER_MAX_DESCENT_RATE = 6.0
FIXED_ENEMY_MAX_ACCEL = 6.0
FIXED_ENEMY_MAX_TURN_RATE_DEG = 75.0
FIXED_ENEMY_MAX_CLIMB_RATE = 5.0
FIXED_ENEMY_MAX_DESCENT_RATE = 5.0
FIXED_DETECTION_RADIUS = 15.0
FIXED_MEASUREMENT_VAR = 0.5
FIXED_PROCESS_VAR = 1.0
FIXED_LEAD_TIME = 0.0
FIXED_ENEMY_EVASION_ENABLED = True  # ALWAYS True, even at gain=0 (item 6) -- never disable the mechanism

# =============================================================================
# HOSTILE-EVASION SWEEP SEEDS -- the previously reserved range, now opened
# =============================================================================
HOSTILE_EVASION_SEED_OFFSET = 33_000
HOSTILE_EVASION_EPISODES = 1_000
HOSTILE_EVASION_SEEDS: range = range(
    HOSTILE_EVASION_SEED_OFFSET, HOSTILE_EVASION_SEED_OFFSET + HOSTILE_EVASION_EPISODES
)

# Reserved for the LATER maneuverability/dynamics robustness study -- do NOT evaluate here.
DYNAMICS_SWEEP_SEED_OFFSET = 34_000
DYNAMICS_SWEEP_EPISODES = 1_000
RESERVED_UNTOUCHED_SEEDS: range = range(DYNAMICS_SWEEP_SEED_OFFSET, DYNAMICS_SWEEP_SEED_OFFSET + DYNAMICS_SWEEP_EPISODES)  # 34000..34999
MAX_ALLOWED_SEED = HOSTILE_EVASION_SEED_OFFSET + HOSTILE_EVASION_EPISODES - 1  # 33999

# =============================================================================
# STATISTICS PROTOCOL (unchanged across every locked experiment)
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def assert_evasion_sweep_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked hostile-evasion sweep seed range."""
    if seed not in HOSTILE_EVASION_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in HOSTILE_EVASION_SEEDS "
            f"({HOSTILE_EVASION_SEED_OFFSET}..{HOSTILE_EVASION_SEED_OFFSET + HOSTILE_EVASION_EPISODES - 1})."
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
class EvasionPolicyInstance:
    """One concrete, runnable policy instance evaluated across the evasion-gain grid."""

    paper_method: str
    policy_instance: str
    estimator: str
    training_seed: int | None
    model_path: Path | None


def build_evasion_policy_instances() -> tuple[EvasionPolicyInstance, ...]:
    instances: list[EvasionPolicyInstance] = []
    for method in SCRIPTED_METHODS:
        instances.append(EvasionPolicyInstance(
            paper_method=method, policy_instance=method,
            estimator=_estimator_for_method(method), training_seed=None, model_path=None,
        ))
    for method in PPO_METHODS:
        track = _track_for_ppo_method(method)
        for seed in PPO_TRAINING_SEEDS:
            instances.append(EvasionPolicyInstance(
                paper_method=method, policy_instance=f"{method}_seed_{seed}",
                estimator=_estimator_for_method(method), training_seed=seed,
                model_path=frozen_model_path(track, seed),
            ))
    return tuple(instances)


EVASION_POLICY_INSTANCES: tuple[EvasionPolicyInstance, ...] = build_evasion_policy_instances()
EXPECTED_N_POLICY_INSTANCES = 10
EXPECTED_ROWS = len(EVASION_GAIN_VALUES) * EXPECTED_N_POLICY_INSTANCES * HOSTILE_EVASION_EPISODES  # 50,000


def build_policy(instance: EvasionPolicyInstance):
    """Instantiate the concrete policy for an EvasionPolicyInstance. 'greedy' is
    ALWAYS the corrected GreedyInterceptPolicy from commit 0dbdadd -- never
    reconstruct the old escort-only behavior."""
    if instance.model_path is None:
        return get_policy(instance.paper_method)
    return get_policy(instance.paper_method, model_path=str(instance.model_path), deterministic=True)


def build_evasion_env_config(estimator: str, evasion_gain: float) -> EnvConfig:
    """
    Nominal EnvConfig with ONLY enemy_evasion_gain (and, for the Kalman
    track, use_kalman_tracking) changed. enemy_evasion_enabled stays True at
    EVERY gain (including 0.0) -- see module docstring. All other
    maneuverability/sensing/mobility parameters are explicitly passed at
    their FIXED nominal values (not swept).
    """
    return EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        v_d=NOMINAL_DEFENDER_SPEED,
        v_e=NOMINAL_HOSTILE_SPEED,
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
        enemy_evasion_radius=EVASION_RADIUS,
        enemy_evasion_gain=evasion_gain,
    )


def assert_only_evasion_gain_differs(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """
    Programmatically verify the ONLY behavioral EnvConfig fields differing
    between a swept config and the nominal config are enemy_evasion_gain
    (and use_kalman_tracking, expected between tracks). Raises
    AssertionError otherwise -- catches any accidental mobility,
    maneuverability, sensing, or reward drift.
    """
    swept_dict = dataclasses.asdict(swept_config)
    nominal_dict = dataclasses.asdict(nominal_config)
    allowed_diff_keys = {"enemy_evasion_gain", "use_kalman_tracking"}
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    unexpected = diffs - allowed_diff_keys
    assert not unexpected, f"unexpected EnvConfig fields differ from nominal: {unexpected}"


# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE opening seed 33000)
# =============================================================================
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_lead", "ppo", "lead"),                              # items 16, Q3/Q4
)
ESTIMATOR_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("kalman_greedy_vs_greedy", "kalman_greedy", "greedy"),      # item 17
    ("ppo_kalman_vs_ppo", "ppo_kalman", "ppo"),                   # item 18
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
ROBUSTNESS_HOSTILE_EVASION_OUTPUT_ROOT = PROJECT_ROOT / "results" / "robustness" / "hostile_evasion"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
EVASION_SUMMARY_FILENAME = "evasion_summary.csv"
PPO_TRAINING_SEED_SUMMARY_FILENAME = "ppo_training_seed_summary.csv"
PPO_VS_LEAD_COMPARISONS_FILENAME = "ppo_vs_lead_comparisons.csv"
ESTIMATOR_SUCCESS_COMPARISONS_FILENAME = "estimator_success_comparisons.csv"
WITHIN_METHOD_EVASION_EFFECTS_FILENAME = "within_method_evasion_effects.csv"
ACQUISITION_EVASION_SUMMARY_FILENAME = "acquisition_evasion_summary.csv"
FAILURE_MODE_SUMMARY_FILENAME = "failure_mode_summary.csv"
OPERATIONAL_SUMMARY_FILENAME = "operational_summary.csv"
ESTIMATOR_SUMMARY_FILENAME = "estimator_summary.csv"
HOSTILE_EVASION_SUMMARY_FILENAME = "hostile_evasion_summary.json"
