"""
Locked measurement-noise robustness sweep protocol -- SINGLE SOURCE OF TRUTH.

The SECOND robustness sweep for the paper. Varies ONLY `measurement_var`
(everything else -- PPO weights, Kalman process_var/equations, reward,
dynamics, hostile evasion, detection_radius -- remains exactly as in the
final nominal experiment) across a previously untouched seed range.

From this task forward, robustness sweeps use ONLY the six primary paper
methods (PN-Kalman and Lead-Kalman are no longer propagated through
robustness experiments, though their implementations and historical nominal
data remain untouched/preserved).

Seed ranges (disjoint from all prior protocol ranges -- see
test_robustness_measurement_noise.py):

    Training seeds                              : 42, 43, 44
    Validation seeds                             : 10000..10099
    Diagnostic seeds                             : 12000..12199
    Final nominal seeds (OPENED, locked)          : 20000..24999
    Acquisition ablation seeds (OPENED, locked)   : 25000..25999
    Detection-radius sweep seeds (OPENED, locked) : 30000..30999
    MEASUREMENT_SWEEP_SEEDS (OPENED here)         : 31000..31999
    Reserved for later robustness sweeps:
        mobility/speed sweep                       : 32000..32999
        evasion sweep                              : 33000..33999
        dynamics/maneuverability sweep              : 34000..34999
"""

from __future__ import annotations

import dataclasses
import math
from pathlib import Path

from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.final_experiment_protocol import PPO_TRAINING_SEEDS, frozen_model_path

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# MEASUREMENT-VARIANCE GRID -- prespecified BEFORE opening seed 31000
# =============================================================================
MEASUREMENT_VAR_VALUES: tuple[float, ...] = (0.05, 0.25, 0.50, 1.00, 2.00, 4.00)
NOMINAL_MEASUREMENT_VAR = 0.5
assert NOMINAL_MEASUREMENT_VAR in MEASUREMENT_VAR_VALUES

# Kalman process_var is FIXED for every condition -- this is NOT filter tuning.
FIXED_PROCESS_VAR = 1.0
FIXED_LEAD_TIME = 0.0


def measurement_std(variance: float) -> float:
    """Per-axis sensor standard deviation: sigma = sqrt(measurement_var)."""
    return math.sqrt(variance)


def theoretical_raw_measurement_rmse(variance: float) -> float:
    """
    Theoretical 3-D vector RMSE of the raw noisy measurement (sensor
    sanity-check helper, item 25): sqrt(3 * measurement_var), since
    epsilon ~ N(0, measurement_var * I_3) has E[||epsilon||^2] = 3*variance.
    """
    return math.sqrt(3.0 * variance)


# =============================================================================
# MEASUREMENT-NOISE SWEEP SEEDS -- a previously untouched range
# =============================================================================
MEASUREMENT_SWEEP_SEED_OFFSET = 31_000
MEASUREMENT_SWEEP_EPISODES = 1_000
MEASUREMENT_SWEEP_SEEDS: range = range(
    MEASUREMENT_SWEEP_SEED_OFFSET, MEASUREMENT_SWEEP_SEED_OFFSET + MEASUREMENT_SWEEP_EPISODES
)

# Reserved for LATER robustness sweeps -- do NOT evaluate in this task.
MOBILITY_SWEEP_SEED_OFFSET = 32_000
EVASION_SWEEP_SEED_OFFSET = 33_000
DYNAMICS_SWEEP_SEED_OFFSET = 34_000
RESERVED_FUTURE_ROBUSTNESS_SEEDS: range = range(MOBILITY_SWEEP_SEED_OFFSET, 35_000)

# =============================================================================
# STATISTICS PROTOCOL
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def assert_measurement_sweep_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked measurement-noise sweep seed range."""
    if seed not in MEASUREMENT_SWEEP_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in MEASUREMENT_SWEEP_SEEDS "
            f"({MEASUREMENT_SWEEP_SEED_OFFSET}..{MEASUREMENT_SWEEP_SEED_OFFSET + MEASUREMENT_SWEEP_EPISODES - 1})."
        )


# =============================================================================
# SIX PRIMARY PAPER METHODS ONLY (no PN-Kalman / Lead-Kalman in robustness)
# =============================================================================
PRIMARY_PAPER_METHODS: tuple[str, ...] = ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
SCRIPTED_METHODS: tuple[str, ...] = ("greedy", "kalman_greedy", "pn", "lead")
PPO_METHODS: tuple[str, ...] = ("ppo", "ppo_kalman")


def _estimator_for_method(method: str) -> str:
    return "kalman" if method in ("kalman_greedy", "ppo_kalman") else "measurement"


def _track_for_ppo_method(method: str) -> str:
    return "kalman" if method == "ppo_kalman" else "direct"


@dataclasses.dataclass(frozen=True)
class NoisePolicyInstance:
    """One concrete, runnable policy instance evaluated across the noise grid."""

    paper_method: str            # one of the 6 paper-level method keys
    policy_instance: str          # unique id, e.g. "greedy" or "ppo_seed_42"
    estimator: str                # "measurement" | "kalman"
    training_seed: int | None     # None for scripted; 42/43/44 for PPO
    model_path: Path | None       # None for scripted; frozen 400k best_model.zip for PPO


def build_noise_policy_instances() -> tuple[NoisePolicyInstance, ...]:
    instances: list[NoisePolicyInstance] = []
    for method in SCRIPTED_METHODS:
        instances.append(NoisePolicyInstance(
            paper_method=method, policy_instance=method,
            estimator=_estimator_for_method(method), training_seed=None, model_path=None,
        ))
    for method in PPO_METHODS:
        track = _track_for_ppo_method(method)
        for seed in PPO_TRAINING_SEEDS:
            instances.append(NoisePolicyInstance(
                paper_method=method, policy_instance=f"{method}_seed_{seed}",
                estimator=_estimator_for_method(method), training_seed=seed,
                model_path=frozen_model_path(track, seed),
            ))
    return tuple(instances)


NOISE_POLICY_INSTANCES: tuple[NoisePolicyInstance, ...] = build_noise_policy_instances()
EXPECTED_N_POLICY_INSTANCES = 10
EXPECTED_RAW_ROW_COUNT = EXPECTED_N_POLICY_INSTANCES * len(MEASUREMENT_VAR_VALUES) * MEASUREMENT_SWEEP_EPISODES  # 60,000


def build_policy(instance: NoisePolicyInstance):
    """Instantiate the concrete policy for a NoisePolicyInstance. 'greedy' is
    ALWAYS the corrected GreedyInterceptPolicy from commit 0dbdadd -- never
    reconstruct the old escort-only behavior."""
    if instance.model_path is None:
        return get_policy(instance.paper_method)
    return get_policy(instance.paper_method, model_path=str(instance.model_path), deterministic=True)


def build_sweep_env_config(estimator: str, measurement_var: float) -> EnvConfig:
    """
    Nominal EnvConfig with ONLY measurement_var (and, for the Kalman track,
    use_kalman_tracking) changed. process_var and lead_time are explicitly
    passed at their FIXED nominal values (not swept) -- see
    assert_only_measurement_var_differs().
    """
    return EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        measurement_var=measurement_var,
        process_var=FIXED_PROCESS_VAR,
        lead_time=FIXED_LEAD_TIME,
    )


def assert_only_measurement_var_differs(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """
    Programmatically verify the ONLY behavioral EnvConfig fields differing
    between a swept config and the nominal config are measurement_var (and
    use_kalman_tracking, expected between tracks). Raises AssertionError
    otherwise -- in particular this catches any accidental process_var or
    lead_time drift.
    """
    swept_dict = dataclasses.asdict(swept_config)
    nominal_dict = dataclasses.asdict(nominal_config)
    allowed_diff_keys = {"measurement_var", "use_kalman_tracking"}
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    unexpected = diffs - allowed_diff_keys
    assert not unexpected, f"unexpected EnvConfig fields differ from nominal: {unexpected}"
    assert swept_dict["process_var"] == FIXED_PROCESS_VAR == nominal_dict["process_var"]
    assert swept_dict["lead_time"] == FIXED_LEAD_TIME == nominal_dict["lead_time"]


# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE opening seed 31000)
# =============================================================================
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("kalman_greedy_vs_greedy", "kalman_greedy", "greedy"),   # A
    ("ppo_kalman_vs_ppo", "ppo_kalman", "ppo"),                 # B
    ("ppo_vs_lead", "ppo", "lead"),                              # C
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_greedy", "ppo", "greedy"),
    ("ppo_vs_pn", "ppo", "pn"),
    ("lead_vs_greedy", "lead", "greedy"),
    ("lead_vs_pn", "lead", "pn"),
    ("pn_vs_greedy", "pn", "greedy"),
)
ALL_COMPARISONS: tuple[tuple[str, str, str], ...] = PRIMARY_COMPARISONS + SECONDARY_COMPARISONS

# =============================================================================
# OUTPUT LOCATION (gitignored; never committed)
# =============================================================================
ROBUSTNESS_NOISE_OUTPUT_ROOT = PROJECT_ROOT / "results" / "robustness" / "measurement_noise"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
NOISE_SUMMARY_FILENAME = "noise_summary.csv"
PPO_TRAINING_SEED_SUMMARY_FILENAME = "ppo_training_seed_summary.csv"
PAIRWISE_COMPARISONS_FILENAME = "pairwise_comparisons.csv"
ESTIMATOR_SUMMARY_FILENAME = "estimator_summary.csv"
MEASUREMENT_NOISE_SUMMARY_FILENAME = "measurement_noise_summary.json"
