"""
Lead-Residual PPO (LR-PPO) protocol -- SINGLE SOURCE OF TRUTH for the new,
completely separate seed namespace and frozen training/validation protocol
constants for this research track (see docs/lead_residual_ppo_design.md and
docs/lead_residual_training_protocol.md).

IMPORTANT: the OLD baseline-study seed blocks remain permanently reserved
and historical, and are UNRELATED to (never reused by) this new namespace:
    40000..40099   old PPO validation
    40100..40299   old diagnostic
    41000..45999   opened final baseline test (5-controller study)
    >=46000        was reserved for the old study; NOT repurposed here.

This module defines a COMPLETELY SEPARATE namespace for the Lead-Residual
research track:
    TRAINING ROOTS:            55, 56, 57         (LR-PPO-55/56/57)
    NEW RESIDUAL VALIDATION:   60000..60099       (100 seeds -- FREEZE ONLY)
    NEW RESIDUAL DIAGNOSTIC:   60100..60299       (200 seeds -- FREEZE ONLY)
    NEW EXPANDED FINAL TEST:   61000..65999       (5000 seeds -- FREEZE ONLY)
    NEW ROBUSTNESS:            >=66000            (FREEZE ONLY, open-ended)

NONE of 60000+ may be executed until explicitly authorized by a future task.
This module ONLY freezes/documents these ranges -- see
assert_not_opened_yet() for the runtime guard used by this task's tests.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np

from uav_defend.config import EnvConfig

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# NEW, COMPLETELY SEPARATE LR-PPO SEED NAMESPACE
# =============================================================================
TRAINING_ROOTS: tuple[int, ...] = (55, 56, 57)

RESIDUAL_VALIDATION_SEED_START = 60_000
RESIDUAL_VALIDATION_SEED_END = 60_099
RESIDUAL_VALIDATION_EPISODES = RESIDUAL_VALIDATION_SEED_END - RESIDUAL_VALIDATION_SEED_START + 1  # 100

RESIDUAL_DIAGNOSTIC_SEED_START = 60_100
RESIDUAL_DIAGNOSTIC_SEED_END = 60_299
RESIDUAL_DIAGNOSTIC_EPISODES = RESIDUAL_DIAGNOSTIC_SEED_END - RESIDUAL_DIAGNOSTIC_SEED_START + 1  # 200

EXPANDED_FINAL_SEED_START = 61_000
EXPANDED_FINAL_SEED_END = 65_999
EXPANDED_FINAL_EPISODES = EXPANDED_FINAL_SEED_END - EXPANDED_FINAL_SEED_START + 1  # 5000

FUTURE_ROBUSTNESS_SEED_START = 66_000

# Old-study historical ranges -- retained here ONLY as documentation/for the
# disjointness test; never reused, never modified upstream.
OLD_STUDY_VALIDATION_RANGE = (40_000, 40_099)
OLD_STUDY_DIAGNOSTIC_RANGE = (40_100, 40_299)
OLD_STUDY_FINAL_RANGE = (41_000, 45_999)
OLD_STUDY_FUTURE_ROBUSTNESS_START = 46_000


def assert_not_opened_yet(seed: int) -> None:
    """Raises if `seed` falls in ANY (not-yet-opened) LR-PPO evaluation
    range. Used as a guard so implementation/protocol-freeze tasks cannot
    accidentally execute a reserved range."""
    if RESIDUAL_VALIDATION_SEED_START <= seed <= RESIDUAL_VALIDATION_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (not-yet-opened) LR-PPO validation range.")
    if RESIDUAL_DIAGNOSTIC_SEED_START <= seed <= RESIDUAL_DIAGNOSTIC_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (not-yet-opened) LR-PPO diagnostic range.")
    if EXPANDED_FINAL_SEED_START <= seed <= EXPANDED_FINAL_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (not-yet-opened) LR-PPO expanded-final range.")
    if seed >= FUTURE_ROBUSTNESS_SEED_START:
        raise ValueError(f"Seed {seed} is >= FUTURE_ROBUSTNESS_SEED_START ({FUTURE_ROBUSTNESS_SEED_START}); reserved.")


# =============================================================================
# DETERMINISTIC TRAINING-EPISODE-SEED GENERATION -- LR-PPO-SPECIFIC MAPPER
# =============================================================================
# The existing experiments/post_detection_training_protocol.py::
# build_episode_seed_sequence() (used by the already-FROZEN 6-model 400k
# campaign) is REUSED IN MECHANISM but NOT modified or called directly here:
# any change to that function could retroactively alter the meaning of the
# already-frozen campaign's seeds. This module instead defines a SEPARATE,
# LR-PPO-specific function built on the SAME underlying primitive
# (numpy.random.SeedSequence(root_seed).spawn(...)).
#
# All LR-PPO evaluation ranges begin at 60,000 and extend upward (the
# robustness range is explicitly open-ended, ">=66000"). Rather than
# reject-and-retry against an open-ended, not-fully-enumerable set, every
# generated LR-PPO TRAINING episode seed is deterministically folded into
# the half-open interval [0, TRAINING_EPISODE_SEED_NAMESPACE_CEILING) via
# modulo. This provably guarantees, BY CONSTRUCTION, that no training
# episode seed can ever equal or exceed 60,000 -- and therefore can never
# collide with ANY current OR future LR-PPO evaluation range (validation,
# diagnostic, expanded-final, and robustness all live at >=60,000).
# Determinism/reproducibility are preserved: modulo is a pure function, so
# identical (root_seed, n_episodes) always yields the identical list.
TRAINING_EPISODE_SEED_NAMESPACE_CEILING = 60_000

DEFAULT_EPISODE_SEED_POOL_SIZE = 100_000


def build_lr_ppo_episode_seed_sequence(root_seed: int, n_episodes: int) -> list[int]:
    """
    Deterministically derive `n_episodes` environment seeds from `root_seed`
    for LR-PPO training, using the SAME underlying primitive as the existing
    corrected-PPO campaign (SeedSequence(root_seed).spawn(...)), then folding
    each raw 32-bit derived value into [0, TRAINING_EPISODE_SEED_NAMESPACE_CEILING)
    via modulo so it can never collide with any LR-PPO evaluation seed.

    Pure function: calling this twice with the same arguments always returns
    the identical list (see test_lead_residual_training_protocol.py).
    """
    seed_seq = np.random.SeedSequence(root_seed)
    children = seed_seq.spawn(n_episodes)
    raw = [int(child.generate_state(1, dtype=np.uint32)[0]) for child in children]
    return [s % TRAINING_EPISODE_SEED_NAMESPACE_CEILING for s in raw]


# =============================================================================
# FROZEN PPO ARCHITECTURE / HYPERPARAMETERS -- SAME FAMILY AS STANDALONE PPO
# =============================================================================
PPO_HYPERPARAMS = dict(
    learning_rate=3e-4,
    gamma=0.99,
    batch_size=64,
    n_steps=2048,
    n_epochs=10,
    clip_range=0.2,
    ent_coef=0.01,
    vf_coef=0.5,
    max_grad_norm=0.5,
    gae_lambda=0.95,
)
NET_ARCH = dict(pi=[64, 64], vf=[64, 64])
ACTIVATION_FN_NAME = "Tanh"
# Confirmed programmatically (SB3 default MlpPolicy, no policy_kwargs
# override, 19-D obs / 3-D action Box): differs from standalone PPO's 10,759
# only because of the larger (19 vs 16) input layer -- 2 networks (pi, vf) x
# 3 extra input features x 64 first-layer units = 384 extra parameters
# (10,759 + 384 = 11,143). The architecture itself (net_arch/activation) is
# unchanged -- see test_lead_residual_ppo_architecture.py.
EXPECTED_TOTAL_PARAMETERS = 11_143

# Frozen for THIS first minimal experiment (see docs/lead_residual_ppo_design.md
# and uav_defend/policies/residual/lead_residual_composition.py). NOT tuned
# using any old-study or new LR-PPO evaluation seed.
RESIDUAL_MAX_ANGLE_DEG = 30.0

CAMPAIGN_TIMESTEPS = 400_000
VALIDATION_FREQUENCY_TIMESTEPS = 25_000
VALIDATION_EPISODES = 100
CHECKPOINT_SELECTION_METRIC = "safe_interception_success_rate"
CHECKPOINT_TIE_BREAK_RULE = "earliest_training_step_wins_exact_ties"
VALIDATION_DETERMINISTIC_INFERENCE = True

METHOD_NAME = "Lead-Residual PPO"
METHOD_ABBREVIATION = "LR-PPO"

MODEL_OUTPUT_ROOT = PROJECT_ROOT / "models" / "lead_residual_post_detection"
RESULTS_OUTPUT_ROOT = PROJECT_ROOT / "results" / "lead_residual_training"
CAMPAIGN_DIRNAME = "training_lead_residual_post_detection_400k"


def build_env_config() -> EnvConfig:
    """Canonical Direct-track EnvConfig for LR-PPO (measurement estimator
    only; a Kalman variant is deferred to a later ablation)."""
    return EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)
