"""
Locked acquisition-behavior ablation protocol -- SINGLE SOURCE OF TRUTH.

Separates PPO's learned PRE-detection acquisition/repositioning behavior
from its POST-detection interception behavior, using the SAME six frozen
400k best_model.zip files evaluated in the final nominal experiment
(results/final_nominal/), on a PREVIOUSLY UNTOUCHED seed range.

Seed ranges (disjoint from all prior protocol ranges -- see
test_acquisition_ablation.py):

    Training seeds                              : 42, 43, 44
    Validation seeds                             : 10000..10099
    Diagnostic seeds                             : 12000..12199
    Final nominal seeds (OPENED, do not reuse)   : 20000..24999
    ACQUISITION_ABLATION_SEEDS (OPENED here)     : 25000..25999
    Untouched gap (reserved for later tasks)     : 26000..29999
    ROBUSTNESS_SEED_OFFSET (reserved)            : 30000+
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from experiments.final_experiment_protocol import (
    PPO_TRAINING_SEEDS,
    ROBUSTNESS_SEED_OFFSET,
    frozen_model_path,
)

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# ACQUISITION ABLATION SEEDS -- a previously untouched range, separate from
# the final nominal test seeds (20000..24999, immutable/opened elsewhere).
# =============================================================================
ACQUISITION_ABLATION_SEED_OFFSET = 25_000
ACQUISITION_ABLATION_EPISODES = 1_000
ACQUISITION_ABLATION_SEEDS: range = range(
    ACQUISITION_ABLATION_SEED_OFFSET, ACQUISITION_ABLATION_SEED_OFFSET + ACQUISITION_ABLATION_EPISODES
)

# 26000..29999 remains an untouched gap in this task (not evaluated here).
UNTOUCHED_GAP_SEEDS: range = range(26_000, ROBUSTNESS_SEED_OFFSET)

# =============================================================================
# STATISTICS PROTOCOL
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def assert_ablation_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked acquisition-ablation seed range."""
    if seed not in ACQUISITION_ABLATION_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in ACQUISITION_ABLATION_SEEDS "
            f"({ACQUISITION_ABLATION_SEED_OFFSET}..{ACQUISITION_ABLATION_SEED_OFFSET + ACQUISITION_ABLATION_EPISODES - 1})."
        )


# =============================================================================
# 14 policy instances: 2 tracks x (1 Lead + 3 Full PPO + 3 Escorted PPO)
# =============================================================================
@dataclass(frozen=True)
class AblationInstance:
    """One concrete, runnable ablation policy instance."""

    condition: str            # "lead" | "full_ppo" | "escorted_ppo"
    track: str                # "direct" | "kalman"
    policy_instance: str      # unique id, e.g. "lead", "ppo_full_seed_42", "ppo_kalman_escorted_seed_44"
    estimator: str            # "measurement" | "kalman"
    training_seed: int | None  # None for lead; 42/43/44 for PPO conditions
    model_path: Path | None    # None for lead; the SAME frozen 400k best_model.zip for both PPO conditions


TRACKS: tuple[tuple[str, str], ...] = (("direct", "measurement"), ("kalman", "kalman"))


def _lead_instance_name(track: str) -> str:
    return "lead" if track == "direct" else "lead_kalman"


def _ppo_instance_name(track: str, condition: str, seed: int) -> str:
    prefix = "ppo" if track == "direct" else "ppo_kalman"
    suffix = "full" if condition == "full_ppo" else "escorted"
    return f"{prefix}_{suffix}_seed_{seed}"


def build_ablation_instances() -> tuple[AblationInstance, ...]:
    instances: list[AblationInstance] = []
    for track, estimator in TRACKS:
        instances.append(AblationInstance(
            condition="lead", track=track, policy_instance=_lead_instance_name(track),
            estimator=estimator, training_seed=None, model_path=None,
        ))
        for seed in PPO_TRAINING_SEEDS:
            # SAME frozen 400k best_model.zip used by both the Full and
            # Escorted conditions -- the wrapper never changes the weights.
            model_path = frozen_model_path(track, seed)
            for condition in ("full_ppo", "escorted_ppo"):
                instances.append(AblationInstance(
                    condition=condition, track=track,
                    policy_instance=_ppo_instance_name(track, condition, seed),
                    estimator=estimator, training_seed=seed, model_path=model_path,
                ))
    return tuple(instances)


ABLATION_INSTANCES: tuple[AblationInstance, ...] = build_ablation_instances()

EXPECTED_N_POLICY_INSTANCES = 14
EXPECTED_RAW_ROW_COUNT = EXPECTED_N_POLICY_INSTANCES * ACQUISITION_ABLATION_EPISODES  # 14,000

# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE opening seed 25000)
# =============================================================================
# (comparison_name, track, condition_a, condition_b) -- reported as A minus B
# (success-rate percentage-point difference).
PRIMARY_COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("A", "direct", "full_ppo", "escorted_ppo"),
    ("B", "direct", "escorted_ppo", "lead"),
    ("C", "kalman", "full_ppo", "escorted_ppo"),
    ("D", "kalman", "escorted_ppo", "lead"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str, str], ...] = (
    ("E", "direct", "full_ppo", "lead"),
    ("F", "kalman", "full_ppo", "lead"),
)
ALL_COMPARISONS: tuple[tuple[str, str, str, str], ...] = PRIMARY_COMPARISONS + SECONDARY_COMPARISONS

# =============================================================================
# OUTPUT LOCATION (gitignored; never committed)
# =============================================================================
ACQUISITION_ABLATION_OUTPUT_ROOT = PROJECT_ROOT / "results" / "acquisition_ablation"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
CONDITION_SUMMARY_FILENAME = "condition_summary.csv"
TRAINING_SEED_SUMMARY_FILENAME = "training_seed_summary.csv"
PAIRED_COMPARISONS_FILENAME = "paired_comparisons.csv"
ACQUISITION_ABLATION_SUMMARY_FILENAME = "acquisition_ablation_summary.json"
