"""
Post-detection PPO training protocol constants -- SINGLE SOURCE OF TRUTH.

This is the REVISED training protocol for the corrected canonical engagement
(see docs/revised_predetection_protocol.md): the defender remains exactly
co-located with the soldier, at rest, until hostile detection, and controller
authority begins on the NEXT environment step. PPO training therefore begins
at the first detected observation (see experiments/post_detection_training_env.py
:class:`PostDetectionTrainingEnv`), never on meaningless standby transitions.

This module is intentionally separate from experiments/ppo_hyperparams.py
(PPO ALGORITHM hyperparameters, reused UNCHANGED from the legacy campaign --
see module docstring there) and from experiments/training_protocol.py (the
OLD/legacy campaign's seeds, now superseded for this revised protocol).

Seed ranges (disjoint from every legacy/prior protocol range; see
test_post_detection_training_env.py):

    Legacy training seeds (SUPERSEDED)   : 42, 43, 44
    Legacy validation seeds (SUPERSEDED) : 10000..10099
    Legacy diagnostic seeds (SUPERSEDED) : 12000..12199
    Legacy final-test seeds (untouched)  : 20000..24999 (final_nominal), etc.

    TRAINING_SEEDS (NEW, this protocol)   : 45, 46, 47
    VALIDATION_SEEDS (NEW, in-training)   : 40000..40099 (100 seeds)
    DIAGNOSTIC_SEEDS (NEW, reserved)      : 40100..40299 (200 seeds) -- NOT
                                            opened in this task except <=5
                                            dry-run seeds OUTSIDE this range.
    FINAL_TEST_SEEDS (reserved, untouched): 41000..45999 inclusive
    FUTURE_ROBUSTNESS (reserved)          : >= 46000

Development/smoke-test seeds (e.g. root 12345) are explicitly OUTSIDE every
range above and must never be confused with a protocol seed range.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_defend.config import EnvConfig

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# LEGACY PROTOCOL STATUS (see experiments/training_protocol.py for the
# original constants -- NOT modified, NOT reused by this protocol)
# =============================================================================
old_training_42_43_44_status = "legacy_superseded_for_revised_paper"
LEGACY_ACQUISITION_ABLATION_STATUS = (
    "historical_superseded -- the acquisition-ablation finding "
    "('Full PPO learns pre-acquisition positioning') belongs to the OLD "
    "protocol where PPO could independently control the defender before "
    "hostile detection. It is NOT valid evidence under the revised "
    "canonical engagement (defender_standby_until_detection=True), where "
    "PPO never controls the defender before detection at all. "
    "results/acquisition_ablation/ is retained unmodified as a historical "
    "record; it must not be reused or re-cited for the revised paper "
    "without this caveat."
)

# =============================================================================
# NEW TRAINING ROOT SEEDS (fresh; the legacy 42/43/44 models are superseded)
# =============================================================================
TRAINING_SEEDS: tuple[int, ...] = (45, 46, 47)

# =============================================================================
# NEW VALIDATION SET (in-training best-model selection -- ALL SIX runs
# [Direct x3, Kalman x3] share this IDENTICAL seed range)
# =============================================================================
VALIDATION_SEED_OFFSET = 40_000
VALIDATION_EPISODES = 100
VALIDATION_SEEDS: range = range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + VALIDATION_EPISODES)

# =============================================================================
# NEW DIAGNOSTIC HOLDOUT (post-training sanity check only -- NOT the paper
# test set, NOT used for model selection). RESERVED in this task: do not
# open except for <=5 clearly-labeled dry-run seeds OUTSIDE this range.
# =============================================================================
DIAGNOSTIC_SEED_OFFSET = 40_100
DIAGNOSTIC_EPISODES = 200
DIAGNOSTIC_SEEDS: range = range(DIAGNOSTIC_SEED_OFFSET, DIAGNOSTIC_SEED_OFFSET + DIAGNOSTIC_EPISODES)

# =============================================================================
# RESERVED FINAL PAPER TEST SEEDS -- DO NOT EXECUTE, INSPECT, OR USE FOR
# TRAINING/MODEL-SELECTION/DEVELOPMENT IN THIS OR ANY TRAINING TASK.
# =============================================================================
FINAL_TEST_SEED_START = 41_000
FINAL_TEST_SEED_END = 45_999  # inclusive
FINAL_TEST_SEEDS: range = range(FINAL_TEST_SEED_START, FINAL_TEST_SEED_END + 1)

# =============================================================================
# FUTURE CORRECTED ROBUSTNESS RANGES -- reserved, exact sub-ranges allocated
# later. Do not consume in this or any training task.
# =============================================================================
FUTURE_ROBUSTNESS_SEED_START = 46_000


def assert_not_reserved_seed(seed: int) -> None:
    """
    Guard against accidentally training, validating, or diagnosing on a
    seed reserved for the final paper evaluation or future robustness
    studies. Raise immediately if violated.
    """
    if FINAL_TEST_SEED_START <= seed <= FINAL_TEST_SEED_END:
        raise ValueError(
            f"Seed {seed} is within the RESERVED final-test range "
            f"[{FINAL_TEST_SEED_START}, {FINAL_TEST_SEED_END}]; must not be "
            f"used during training, validation, or diagnostics."
        )
    if seed >= FUTURE_ROBUSTNESS_SEED_START:
        raise ValueError(
            f"Seed {seed} is >= FUTURE_ROBUSTNESS_SEED_START "
            f"({FUTURE_ROBUSTNESS_SEED_START}); reserved for future "
            f"corrected robustness studies, not this training protocol."
        )


# =============================================================================
# DETERMINISTIC PER-EPISODE TRAINING SEED DERIVATION
# =============================================================================
# SoldierEnv.reset(seed=None) constructs fresh, OS-entropy-derived RNG
# streams (see SoldierEnv.reset()) -- NOT modified here, per instructions.
# Reproducible per-episode training seeds are instead derived ENTIRELY in
# this module via numpy.random.SeedSequence, which is a pure, deterministic
# function of (root_seed, episode_index):
#
#     same root_seed  -> same derived episode-seed sequence (every time)
#     different roots -> statistically independent sequences (no
#                        mathematical collision guarantee, but overwhelming
#                        improbability of overlap given the 32-bit seed
#                        space -- see test_post_detection_training_env.py's
#                        disjointness check over 1000 samples/root).
#
# Direct and Kalman runs sharing the same training root seed receive the
# IDENTICAL derived episode-seed sequence (paired realizations across
# estimator tracks) because derivation depends ONLY on root_seed and
# episode_index, never on track/estimator.
def build_episode_seed_sequence(root_seed: int, n_episodes: int) -> list[int]:
    """
    Deterministically derive `n_episodes` environment seeds from `root_seed`.
    Pure function: calling this twice with the same arguments always returns
    the identical list (see test_post_detection_training_env.py, tests K/L/M).
    """
    seed_seq = np.random.SeedSequence(root_seed)
    children = seed_seq.spawn(n_episodes)
    return [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]


# Enough derived episode seeds to comfortably cover a 400k-timestep run even
# under short episodes (worst-case episode length after detection is small
# relative to max_steps; this budget is generous headroom, not a hard cap --
# PostDetectionTrainingEnv extends the sequence on demand if ever exhausted).
DEFAULT_EPISODE_SEED_POOL_SIZE = 100_000


# =============================================================================
# ENVIRONMENT CONFIG BUILDER (Direct vs Kalman differ ONLY in the estimator
# flag; defender_standby_until_detection is explicitly True for both)
# =============================================================================
def build_env_config(track: str) -> EnvConfig:
    """
    Return the canonical EnvConfig for the given track. The ONLY difference
    between tracks is use_kalman_tracking -- every other field (including
    defender_standby_until_detection=True, explicit here even though it is
    also the EnvConfig default) is identical for both. See
    assert_only_estimator_differs() and test_post_detection_training_env.py.
    """
    if track == "direct":
        return EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)
    elif track == "kalman":
        return EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True)
    raise ValueError(f"Unknown track '{track}'; expected 'direct' or 'kalman'")


def assert_only_estimator_differs(config_direct: EnvConfig, config_kalman: EnvConfig) -> None:
    """Programmatically verify Direct/Kalman EnvConfig parity except the estimator flag."""
    import dataclasses
    d = dataclasses.asdict(config_direct)
    k = dataclasses.asdict(config_kalman)
    diffs = {key for key in d if d[key] != k[key]}
    assert diffs <= {"use_kalman_tracking"}, f"unexpected EnvConfig fields differ: {diffs}"
    assert config_direct.defender_standby_until_detection is True
    assert config_kalman.defender_standby_until_detection is True


# =============================================================================
# TRAINING-STEP DEFINITION (documented explicitly -- differs from legacy)
# =============================================================================
# A PPO "timestep" in this campaign means ONE POST-DETECTION, PPO-CONTROLLED
# environment step. The standby fast-forward performed inside
# PostDetectionTrainingEnv.reset() happens INSIDE reset(), which SB3/
# Gymnasium never count toward model.num_timesteps -- only calls the
# training loop makes to env.step() count. Consequently "400,000 timesteps"
# under this protocol means 400,000 PPO-controlled post-detection steps,
# NOT 400,000 total environment steps including standby (as an aside: the
# legacy 42/43/44 campaign's "400,000 timesteps" meant something different
# again -- EVERY step from episode start was policy-controlled there, since
# standby did not exist as a concept at all).
POST_DETECTION_TIMESTEP_DEFINITION = (
    "One PPO training timestep = one POST-DETECTION, PPO-controlled "
    "environment step (executed via PostDetectionTrainingEnv.step()). "
    "Standby fast-forward steps performed inside "
    "PostDetectionTrainingEnv.reset() are NEVER counted toward "
    "model.num_timesteps or the training budget."
)

# =============================================================================
# OUTPUT DIRECTORY LAYOUT (NEW namespace -- never overwrites results/training_400k/)
# =============================================================================
CAMPAIGN_DIRNAME = "training_post_detection_400k"
CAMPAIGN_TIMESTEPS = 400_000
TRACK_DIRNAMES = {"direct": "direct", "kalman": "kalman"}


def run_output_dir(campaign_root, track: str, seed: int):
    """<campaign_root>/<track>/seed_<seed>/ -- see experiments/training_protocol.py
    for the identical legacy pattern (deliberately mirrored, NEW root only)."""
    if track not in TRACK_DIRNAMES:
        raise ValueError(f"Unknown track '{track}'; expected one of {list(TRACK_DIRNAMES)}")
    return Path(campaign_root) / TRACK_DIRNAMES[track] / f"seed_{seed}"


# =============================================================================
# CHECKPOINT / MODEL-SELECTION PROTOCOL (documented BEFORE training begins)
# =============================================================================
MODEL_SELECTION_METRIC = "safe_interception_success_rate"
MODEL_SELECTION_TIE_BREAK_RULE = (
    "1. Highest validation success rate (n=100, seeds "
    f"{VALIDATION_SEED_OFFSET}-{VALIDATION_SEED_OFFSET + VALIDATION_EPISODES - 1}, "
    "full canonical SoldierEnv mission, not the training wrapper). "
    "2. If exactly tied, the EARLIEST checkpoint wins (SuccessRateEvalCallback "
    "only overwrites best_model on a STRICT improvement, `>`, never `>=`, "
    "which implements this tie-break automatically)."
)
FORBIDDEN_SELECTION_METRICS = (
    "mean_reward", "training_loss", "tracking_rmse", "mean_detection_time", "intercept_time",
)
