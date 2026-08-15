"""
Multi-seed PPO training protocol constants -- SINGLE SOURCE OF TRUTH.

This module defines the seeds and seed RANGES used across the multi-seed
PPO training campaign (Direct + Kalman, 3 seeds each). It is intentionally
separate from experiments/ppo_hyperparams.py (which holds PPO ALGORITHM
hyperparameters): this module holds PROTOCOL constants -- which seeds are
used for what purpose, and where run artifacts live.

Seed ranges (disjoint, non-overlapping by construction -- see
test_training_protocol.py):

    TRAINING_SEEDS                      : 42, 43, 44
        The three independent PPO training seeds per track. Each is an
        independent run from a freshly-initialized network -- never a
        continuation of another seed.

    VALIDATION_SEED_OFFSET .. +99        : 10000..10099
        The FIXED, IDENTICAL validation set used for in-training best-model
        selection (SuccessRateEvalCallback) for ALL SIX runs (Direct x3,
        Kalman x3). Never offset by training seed -- using the same
        validation seeds for every run holds the model-selection difficulty
        distribution constant across runs.

    DIAGNOSTIC_SEED_OFFSET .. +199       : 12000..12199
        A separate, larger holdout used ONLY for the post-training
        diagnostic summary (this task). NOT the final paper test set, and
        NOT used for model selection.

    FINAL_TEST_SEED_OFFSET               : 20000..
        RESERVATION ONLY in this task. The final paper Monte Carlo
        evaluation will use seeds >= FINAL_TEST_SEED_OFFSET. Do NOT
        evaluate on these seeds until that later phase.
"""

from __future__ import annotations

# =============================================================================
# TRAINING SEEDS
# =============================================================================
# Three independent training seeds per track (Direct and Kalman). Each seed
# initializes SB3/NumPy/PyTorch and the environment's own RNG streams
# (uav_defend.envs.soldier_env.SoldierEnv.reset(seed=...)) independently --
# no seed continues training from another seed's model.
TRAINING_SEEDS: tuple[int, ...] = (42, 43, 44)

# =============================================================================
# VALIDATION SET (in-training best-model selection -- ALL SIX runs share this)
# =============================================================================
VALIDATION_SEED_OFFSET = 10_000
VALIDATION_EPISODES = 100
VALIDATION_SEEDS: range = range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + VALIDATION_EPISODES)

# =============================================================================
# DIAGNOSTIC HOLDOUT (post-training sanity check -- NOT the paper test set)
# =============================================================================
DIAGNOSTIC_SEED_OFFSET = 12_000
DIAGNOSTIC_EPISODES = 200
DIAGNOSTIC_SEEDS: range = range(DIAGNOSTIC_SEED_OFFSET, DIAGNOSTIC_SEED_OFFSET + DIAGNOSTIC_EPISODES)

# =============================================================================
# FINAL PAPER TEST SET -- RESERVATION ONLY, DO NOT USE YET
# =============================================================================
# Do NOT evaluate on any seed >= FINAL_TEST_SEED_OFFSET during the training
# protocol or its diagnostics. Reserved exclusively for the final paper
# Monte Carlo evaluation phase (a later task).
FINAL_TEST_SEED_OFFSET = 20_000


def assert_not_final_test_seed(seed: int) -> None:
    """
    Guard against accidentally evaluating on the reserved final-test seed
    range during training/diagnostics. Raise immediately if violated.
    """
    if seed >= FINAL_TEST_SEED_OFFSET:
        raise ValueError(
            f"Seed {seed} is >= FINAL_TEST_SEED_OFFSET ({FINAL_TEST_SEED_OFFSET}); "
            f"final-test seeds are RESERVED for the final paper evaluation and must "
            f"not be used during training or training diagnostics."
        )


# =============================================================================
# OUTPUT DIRECTORY LAYOUT
# =============================================================================
# <campaign_root>/<track>/seed_<N>/ containing:
#   best_model.zip, final_model.zip, checkpoints/, training_logs/, monitor/,
#   training_manifest.json
#
# `campaign_root` is caller-supplied (not hardcoded here) so that separate
# training campaigns can use separate, non-colliding output roots -- e.g.
# the original 200k campaign uses results/training/, and the 400k
# replication campaign (feature/ppo-400k-replication) uses a completely
# separate results/training_400k/ root. See CAMPAIGN_200K_DIRNAME /
# CAMPAIGN_400K_DIRNAME below.
TRACK_DIRNAMES = {"direct": "direct", "kalman": "kalman"}


def run_output_dir(campaign_root, track: str, seed: int):
    """
    Return the Path to the run-specific output directory for a given track
    and training seed: <campaign_root>/<track>/seed_<seed>/.

    Args:
        campaign_root: Path under which `<track>/seed_<seed>/` should live
            for a given training campaign (e.g. results/training for the
            original 200k campaign, or results/training_400k for the 400k
            replication campaign). NOT the top-level results/ directory
            itself -- callers select the campaign-specific root explicitly.
        track: "direct" or "kalman".
        seed: Training seed (e.g. one of TRAINING_SEEDS).
    """
    from pathlib import Path

    if track not in TRACK_DIRNAMES:
        raise ValueError(f"Unknown track '{track}'; expected one of {list(TRACK_DIRNAMES)}")
    return Path(campaign_root) / TRACK_DIRNAMES[track] / f"seed_{seed}"


# =============================================================================
# 400K REPLICATION CAMPAIGN (feature/ppo-400k-replication)
# =============================================================================
# A completely separate output root from the original 200k campaign so the
# two campaigns can never collide or overwrite one another. The original
# 200k campaign (results/training/) is an immutable record and must not be
# touched by 400k campaign code.
CAMPAIGN_200K_DIRNAME = "training"
CAMPAIGN_400K_DIRNAME = "training_400k"
CAMPAIGN_400K_TIMESTEPS = 400_000
