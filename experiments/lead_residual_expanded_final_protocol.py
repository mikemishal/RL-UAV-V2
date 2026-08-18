"""
Lead-Residual PPO expanded final nominal evaluation protocol -- SINGLE SOURCE
OF TRUTH. ONE-TIME, terminal 5000-seed evaluation (61000..65999) -- the
PRIMARY FINAL NOMINAL EVALUATION for the expanded six-method paper (Greedy,
PN, Lead, PPO, PPO-Kalman, Lead-Residual PPO).

training_source_commit (old 6-model PPO/PPO-Kalman campaign) =
    4b0a903bf801c7b439be4671ed822e2f853d72ba
lr_ppo_training_source_commit (LR-PPO-55/56/57 campaign) =
    3b5bc9a48c0d3d52bcf369aea2d97de4c85d6445
lr_ppo_diagnostic_source_commit (200-seed diagnostic, 60100-60299) =
    80e7aef9777039ae27745dc27f585b312bd9d8ce
expanded_final_evaluation_source_commit = recorded at evaluation time (see
    run_lead_residual_expanded_final.py); must be identical before/after.
"""

from __future__ import annotations

from pathlib import Path

from experiments.lead_residual_diagnostic_protocol import (  # noqa: F401 -- single source of truth, re-exported
    FROZEN_MODELS,
    TRAINING_SEEDS,
    LR_TRAINING_ROOTS,
    TRAINING_SOURCE_COMMIT,
    LR_PPO_TRAINING_SOURCE_COMMIT,
    SCRIPTED_METHODS,
    PPO_METHODS,
    LR_PPO_METHOD,
    EXCLUDED_METHODS,
    RESIDUAL_MAX_ANGLE_DEG,
)

PROJECT_ROOT = Path(__file__).parent.parent

LR_PPO_DIAGNOSTIC_SOURCE_COMMIT = "80e7aef9777039ae27745dc27f585b312bd9d8ce"

# Previously-consumed/still-sealed ranges, retained here ONLY for the
# assert_not_reserved_seed() guard below.
DIAGNOSTIC_SEED_RANGE = (60_100, 60_299)
ROBUSTNESS_SEED_START = 66_000
ROBUSTNESS_SEED_END = 99_999

FINAL_SEED_START = 61_000
FINAL_SEED_END = 65_999
FINAL_EPISODES = FINAL_SEED_END - FINAL_SEED_START + 1  # 5000
FINAL_SEEDS: range = range(FINAL_SEED_START, FINAL_SEED_END + 1)
assert FINAL_EPISODES == 5000

PRIMARY_PAPER_METHODS: tuple[str, ...] = ("greedy", "pn", "lead", "ppo", "ppo_kalman", "lr_ppo")

# Same bootstrap seed frozen for BOTH the diagnostic and this final evaluation
# (per this task's explicit instruction) -- fixed BEFORE viewing final results.
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260818
CONFIDENCE_LEVEL = 0.95

# Primary comparisons frozen BEFORE opening final outcomes (item 14) -- must
# not be reclassified after viewing results.
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("LR-PPO family - Lead", "lr_ppo", "lead"),
    ("LR-PPO family - PPO family", "lr_ppo", "ppo"),
)
# Existing-paper (secondary, but required for the expanded-paper final
# evidence set -- item 17) hierarchical family comparisons.
EXISTING_PAPER_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("PPO family - Greedy", "ppo", "greedy"),
    ("PPO family - PN", "ppo", "pn"),
    ("PPO family - Lead", "ppo", "lead"),
    ("PPO-Kalman family - PPO family", "ppo_kalman", "ppo"),
)
# Classical paired comparisons (item 18).
CLASSICAL_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("Lead - Greedy", "lead", "greedy"),
    ("Lead - PN", "lead", "pn"),
    ("PN - Greedy", "pn", "greedy"),
)

# Recorded BEFORE this evaluation (item 25) -- validation (100-seed,
# checkpoint-selection) and 200-seed diagnostic rates -- for a purely
# descriptive validation -> diagnostic -> final generalization triple.
VALIDATION_BEST_SUCCESS_RATES = {
    ("lr_ppo", 55): 0.94,
    ("lr_ppo", 56): 0.96,
    ("lr_ppo", 57): 0.97,
}
DIAGNOSTIC_200_SUCCESS_RATES = {
    ("lr_ppo", 55): 0.855,
    ("lr_ppo", 56): 0.855,
    ("lr_ppo", 57): 0.890,
}

OUTPUT_ROOT = PROJECT_ROOT / "results" / "lead_residual_expanded_final_5000"


def assert_seed_in_final_range(seed: int) -> None:
    if not (FINAL_SEED_START <= seed <= FINAL_SEED_END):
        raise ValueError(f"Seed {seed} is outside the locked final-test range {FINAL_SEED_START}..{FINAL_SEED_END}")


def assert_not_reserved_elsewhere(seed: int) -> None:
    if DIAGNOSTIC_SEED_RANGE[0] <= seed <= DIAGNOSTIC_SEED_RANGE[1]:
        raise ValueError(f"Seed {seed} is in the already-consumed diagnostic range; must not reuse here.")
    if ROBUSTNESS_SEED_START <= seed <= ROBUSTNESS_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (sealed) robustness range; must not be used here.")
