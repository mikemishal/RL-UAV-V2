"""
Locked detection-radius robustness sweep protocol -- SINGLE SOURCE OF TRUTH.

The FIRST robustness sweep for the paper. Varies ONLY `detection_radius`
(everything else -- PPO weights, Kalman filter, reward, dynamics, hostile
evasion -- remains exactly as in the final nominal experiment) across a
previously untouched seed range.

Seed ranges (disjoint from all prior protocol ranges -- see
test_robustness_detection_radius.py):

    Training seeds                              : 42, 43, 44
    Validation seeds                             : 10000..10099
    Diagnostic seeds                             : 12000..12199
    Final nominal seeds (OPENED, locked)          : 20000..24999
    Acquisition ablation seeds (OPENED, locked)   : 25000..25999
    DETECTION_SWEEP_SEEDS (OPENED here)           : 30000..30999
    Reserved for later robustness sweeps:
        measurement-noise sweep                   : 31000..31999
        mobility/speed sweep                      : 32000..32999
        evasion sweep                              : 33000..33999
        dynamics/maneuverability sweep             : 34000..34999
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from uav_defend.config import EnvConfig

from experiments.final_experiment_protocol import (
    PPO_TRAINING_SEEDS,
    POLICY_INSTANCES,
    PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
)

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# DETECTION-RADIUS GRID -- prespecified BEFORE opening seed 30000
# =============================================================================
DETECTION_RADIUS_VALUES: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
NOMINAL_DETECTION_RADIUS = 15.0
assert NOMINAL_DETECTION_RADIUS in DETECTION_RADIUS_VALUES

# =============================================================================
# ROBUSTNESS SWEEP SEEDS -- a previously untouched range
# =============================================================================
DETECTION_SWEEP_SEED_OFFSET = 30_000
DETECTION_SWEEP_EPISODES = 1_000
DETECTION_SWEEP_SEEDS: range = range(
    DETECTION_SWEEP_SEED_OFFSET, DETECTION_SWEEP_SEED_OFFSET + DETECTION_SWEEP_EPISODES
)

# Reserved for LATER robustness sweeps -- do NOT evaluate in this task.
MEASUREMENT_NOISE_SWEEP_SEED_OFFSET = 31_000
MOBILITY_SPEED_SWEEP_SEED_OFFSET = 32_000
EVASION_SWEEP_SEED_OFFSET = 33_000
DYNAMICS_SWEEP_SEED_OFFSET = 34_000
RESERVED_FUTURE_ROBUSTNESS_SEEDS: range = range(MEASUREMENT_NOISE_SWEEP_SEED_OFFSET, 35_000)

# =============================================================================
# STATISTICS PROTOCOL
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def assert_detection_sweep_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked detection-radius sweep seed range."""
    if seed not in DETECTION_SWEEP_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in DETECTION_SWEEP_SEEDS "
            f"({DETECTION_SWEEP_SEED_OFFSET}..{DETECTION_SWEEP_SEED_OFFSET + DETECTION_SWEEP_EPISODES - 1})."
        )


# =============================================================================
# 12 policy instances -- IDENTICAL to the final nominal experiment's instances
# (same 6 scripted + same 6 frozen 400k PPO models); only detection_radius is
# swept, applied per-instance at env-construction time in the runner.
# =============================================================================
SWEEP_POLICY_INSTANCES = POLICY_INSTANCES
EXPECTED_N_POLICY_INSTANCES = 12
EXPECTED_RAW_ROW_COUNT = EXPECTED_N_POLICY_INSTANCES * len(DETECTION_RADIUS_VALUES) * DETECTION_SWEEP_EPISODES  # 72,000


def build_sweep_env_config(estimator: str, detection_radius: float) -> EnvConfig:
    """
    Nominal EnvConfig with ONLY detection_radius (and, for the Kalman track,
    use_kalman_tracking) changed -- see assert_only_detection_radius_differs().
    """
    return EnvConfig(use_kalman_tracking=(estimator == "kalman"), detection_radius=detection_radius)


def assert_only_detection_radius_differs(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """
    Programmatically verify the ONLY behavioral EnvConfig fields differing
    between a swept config and the nominal config are detection_radius (and
    use_kalman_tracking, which is expected to differ between tracks but not
    because of the sweep itself). Raises AssertionError otherwise.
    """
    swept_dict = dataclasses.asdict(swept_config)
    nominal_dict = dataclasses.asdict(nominal_config)
    allowed_diff_keys = {"detection_radius", "use_kalman_tracking"}
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    unexpected = diffs - allowed_diff_keys
    assert not unexpected, f"unexpected EnvConfig fields differ from nominal: {unexpected}"


# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE opening seed 30000)
# =============================================================================
# (comparison_name, method_a, method_b) -- reported as A minus B (success-rate
# percentage-point difference) AT EVERY detection radius.
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_lead", "ppo", "lead"),                       # Q2 (Direct)
    ("ppo_kalman_vs_lead_kalman", "ppo_kalman", "lead_kalman"),  # Q2 (Kalman)
    ("ppo_kalman_vs_ppo", "ppo_kalman", "ppo"),           # Q3 (estimator effect)
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_greedy", "ppo", "greedy"),
    ("ppo_vs_pn", "ppo", "pn"),
    ("ppo_kalman_vs_kalman_greedy", "ppo_kalman", "kalman_greedy"),
    ("ppo_kalman_vs_pn_kalman", "ppo_kalman", "pn_kalman"),
    ("kalman_greedy_vs_greedy", "kalman_greedy", "greedy"),
    ("pn_kalman_vs_pn", "pn_kalman", "pn"),
    ("lead_kalman_vs_lead", "lead_kalman", "lead"),
)
ALL_COMPARISONS: tuple[tuple[str, str, str], ...] = PRIMARY_COMPARISONS + SECONDARY_COMPARISONS

# Scripted controllers sharing the identical pre-detection escort law --
# their pre-detection trajectory must match at every fixed (radius, seed).
SCRIPTED_PRE_DETECTION_IDENTITY_GROUP: tuple[str, ...] = (
    "greedy", "pn", "lead", "kalman_greedy", "pn_kalman", "lead_kalman",
)

# =============================================================================
# OUTPUT LOCATION (gitignored; never committed)
# =============================================================================
ROBUSTNESS_DETECTION_OUTPUT_ROOT = PROJECT_ROOT / "results" / "robustness" / "detection_radius"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
RADIUS_SUMMARY_FILENAME = "radius_summary.csv"
PPO_TRAINING_SEED_SUMMARY_FILENAME = "ppo_training_seed_summary.csv"
PAIRWISE_COMPARISONS_FILENAME = "pairwise_comparisons.csv"
ACQUISITION_SUMMARY_FILENAME = "acquisition_summary.csv"
ESTIMATOR_SUMMARY_FILENAME = "estimator_summary.csv"
DETECTION_RADIUS_SUMMARY_FILENAME = "detection_radius_summary.json"
