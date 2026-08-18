"""
Lead-Residual PPO (LR-PPO) protocol -- SINGLE SOURCE OF TRUTH for the new,
completely separate seed namespace and frozen training/validation protocol
constants for this research track (see docs/lead_residual_ppo_design.md and
docs/lead_residual_training_protocol.md).

HISTORICAL RANGE CLARIFICATION (corrected -- see task that fixed the
training-episode-seed mapper): the OLD baseline-study seed blocks that were
ACTUALLY OPENED/EXECUTED remain permanently reserved and historical, and are
UNRELATED to (never reused by) this new namespace:
    40000..40099   old PPO validation (executed)
    40100..40299   old diagnostic (executed)
    41000..45999   opened final baseline test (5-controller study, executed)

The previous version of this docstring additionally claimed ">=46000 was
reserved for the old study". That was INCORRECT/OVERBROAD: ">=46000" was
merely an UNUSED FUTURE PLACEHOLDER name in the old study's protocol module
-- no seed in that placeholder was ever opened/executed for robustness work.
An open-ended ">=X" claim also mathematically overlaps any later open-ended
claim (e.g. the new LR-PPO namespace below), which is why this correction
replaces ALL open-ended (">=") LR-PPO range claims with FINITE blocks. No
historical evidence changes as a result of this clarification -- only the
documentation of an unused placeholder is corrected.

This module defines a COMPLETELY SEPARATE, FINITE namespace for the
Lead-Residual research track:
    TRAINING ROOTS:            55, 56, 57         (LR-PPO-55/56/57 -- unchanged)
    NEW RESIDUAL VALIDATION:   60000..60099       (100 seeds -- FREEZE ONLY)
    NEW RESIDUAL DIAGNOSTIC:   60100..60299       (200 seeds -- FREEZE ONLY)
    NEW EXPANDED FINAL TEST:   61000..65999       (5000 seeds -- FREEZE ONLY)
    NEW ROBUSTNESS:            66000..99999       (FREEZE ONLY, FINITE block,
                                                    was previously the
                                                    open-ended ">=66000")

NONE of 60000-99999 may be executed until explicitly authorized by a future
task. This module ONLY freezes/documents these ranges -- see
assert_not_opened_yet() for the runtime guard used by this task's tests.

Training-episode seeds now live in a SEPARATE, million-scale-per-root
namespace starting at 1,000,000 (see "DETERMINISTIC TRAINING-EPISODE-SEED
GENERATION" below) -- disjoint from ALL of the above by construction, and
far above the small integer values used by historical/dev seeds.
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

# FINITE block (corrected from the previous open-ended ">=66000" claim --
# see module docstring's HISTORICAL RANGE CLARIFICATION).
ROBUSTNESS_SEED_START = 66_000
ROBUSTNESS_SEED_END = 99_999
FUTURE_ROBUSTNESS_SEED_START = ROBUSTNESS_SEED_START  # kept as an alias for callers using the old name

# Old-study historical ranges -- ONLY the ranges that were ACTUALLY OPENED/
# EXECUTED are reserved; retained here ONLY as documentation/for the
# disjointness test; never reused, never modified upstream.
OLD_STUDY_VALIDATION_RANGE = (40_000, 40_099)
OLD_STUDY_DIAGNOSTIC_RANGE = (40_100, 40_299)
OLD_STUDY_FINAL_RANGE = (41_000, 45_999)
# CORRECTED: this was an UNUSED FUTURE PLACEHOLDER in the old study's
# protocol module, never opened/executed for any robustness work -- it is
# NOT a permanent global reservation and must not be treated as disjoint
# from (or claimed to overlap) any other namespace. Retained here ONLY to
# document the correction itself.
OLD_STUDY_FUTURE_ROBUSTNESS_PLACEHOLDER_UNUSED = 46_000


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
    if ROBUSTNESS_SEED_START <= seed <= ROBUSTNESS_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (not-yet-opened) LR-PPO robustness range.")


# =============================================================================
# DETERMINISTIC, COLLISION-FREE TRAINING-EPISODE-SEED GENERATION
# =============================================================================
# CORRECTED MAPPER (supersedes a prior modulo-60000 fold that was found to
# create both within-root duplicates and cross-root collisions -- e.g. over
# the first 5000 seeds/root: root 55 had 213 repeated occurrences, root 56
# had 216, root 57 had 180, with 396/342/422 pairwise cross-root
# intersections. Folding an arbitrary 32-bit value into a 60,000-slot space
# via modulo is NOT injective over more than ~60,000 draws by the pigeonhole
# principle, and provides no cross-root disjointness guarantee at all.)
#
# The existing experiments/post_detection_training_protocol.py::
# build_episode_seed_sequence() (used by the already-FROZEN 6-model 400k
# campaign) is REUSED IN MECHANISM ONLY (SeedSequence-based) but NOT
# modified or called directly here -- this module defines a SEPARATE,
# LR-PPO-specific function.
#
# NEW DESIGN: each training root is assigned its OWN disjoint, million-scale
# integer namespace (far above any historical/dev/evaluation seed):
#     LR-PPO-55: 1,000,000 .. 1,999,999
#     LR-PPO-56: 2,000,000 .. 2,999,999
#     LR-PPO-57: 3,000,000 .. 3,999,999
# Within a root's namespace, episode index i in [0, M) (M = 1,000,000) is
# mapped to an OFFSET via a deterministic AFFINE PERMUTATION:
#
#     offset_i = (a*i + b) mod M
#
# where a and b are derived deterministically from
# numpy.random.SeedSequence(root_seed), and gcd(a, M) = 1 (M = 10^6 =
# 2^6 * 5^6, so this requires only that `a` be odd and not a multiple of 5).
#
# PROOF OF BIJECTIVITY (=> no duplicates within a root, for any i != j < M):
# the map i -> a*i mod M is a group automorphism of Z/MZ (bijective) iff
# gcd(a, M) = 1 (standard modular-arithmetic fact: a has a multiplicative
# inverse mod M). Composing with a fixed translation (+b mod M) preserves
# bijectivity (translations are bijections). Therefore i != j < M implies
# offset_i != offset_j, hence env_seed_i != env_seed_j.
#
# PROOF OF CROSS-ROOT DISJOINTNESS: env_seed = root_namespace_base + offset,
# with offset in [0, M) and root_namespace_base in {1_000_000, 2_000_000,
# 3_000_000} spaced exactly M apart -- so each root's full value range
# [base, base+M) is disjoint from every other root's range and from every
# evaluation range (all of which are far below 1,000,000), independent of
# which permutation (a, b) is used.
#
# Determinism/reproducibility are preserved: (a, b) are pure, deterministic
# functions of root_seed, and offset_i is a pure function of i -- calling
# this twice with the same arguments always returns the identical list, and
# a PREFIX of a longer request is identical to a shorter request's full
# result (since offset_i never depends on n_episodes).
TRAINING_NAMESPACE_SIZE = 1_000_000  # M = 10^6 = 2^6 * 5^6

TRAINING_ROOT_NAMESPACE_BASE: dict[int, int] = {
    55: 1_000_000,
    56: 2_000_000,
    57: 3_000_000,
}

DEFAULT_EPISODE_SEED_POOL_SIZE = 100_000


def _derive_affine_params(root_seed: int) -> tuple[int, int]:
    """Deterministically derive (a, b) for offset_i = (a*i + b) mod M,
    M = TRAINING_NAMESPACE_SIZE (= 2^6 * 5^6). Requires gcd(a, M) = 1, i.e.
    a odd and not a multiple of 5 (M's only prime factors are 2 and 5)."""
    seed_seq = np.random.SeedSequence(root_seed)
    state = seed_seq.generate_state(2, dtype=np.uint64)
    a = int(state[0] % TRAINING_NAMESPACE_SIZE)
    a |= 1  # force odd
    while a % 5 == 0:
        a += 2  # stays odd; terminates within <=4 steps (only 1/5 odd values is a multiple of 5)
    b = int(state[1] % TRAINING_NAMESPACE_SIZE)
    assert np.gcd(a, TRAINING_NAMESPACE_SIZE) == 1, (root_seed, a)  # defense-in-depth
    return a, b


def build_lr_ppo_episode_seed_sequence(root_seed: int, n_episodes: int) -> list[int]:
    """
    Deterministically derive `n_episodes` COLLISION-FREE environment seeds
    from `root_seed` for LR-PPO training, within that root's dedicated
    disjoint 1,000,000-seed namespace (see module docstring for the
    bijectivity/disjointness proof).

    Pure function: calling this twice with the same arguments always
    returns the identical list. A prefix property holds: the first `k`
    elements of build_lr_ppo_episode_seed_sequence(root, n) for any n >= k
    equal build_lr_ppo_episode_seed_sequence(root, k) exactly (offset_i
    depends only on i, never on n_episodes) -- see
    test_lead_residual_training_protocol.py.

    Raises:
        ValueError: if root_seed has no assigned namespace (only 55/56/57
            are currently assigned), or if n_episodes exceeds the
            namespace's TRAINING_NAMESPACE_SIZE (1,000,000) capacity.
    """
    if root_seed not in TRAINING_ROOT_NAMESPACE_BASE:
        raise ValueError(
            f"root_seed={root_seed} has no assigned LR-PPO training namespace; "
            f"expected one of {sorted(TRAINING_ROOT_NAMESPACE_BASE)}."
        )
    if n_episodes > TRAINING_NAMESPACE_SIZE:
        raise ValueError(
            f"n_episodes={n_episodes} exceeds this root's namespace capacity "
            f"({TRAINING_NAMESPACE_SIZE}); reduce n_episodes or extend the namespace design."
        )
    base = TRAINING_ROOT_NAMESPACE_BASE[root_seed]
    a, b = _derive_affine_params(root_seed)
    M = TRAINING_NAMESPACE_SIZE
    return [base + (a * i + b) % M for i in range(n_episodes)]


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
# Locked for official runs alongside the other protocol values (item 2 of the
# pre-launch-lock task) -- purely a logging/checkpoint-density parameter
# (cannot affect training or model selection), but frozen anyway for complete
# reproducibility. Matches experiments/ppo_hyperparams.py::CHECKPOINT_FREQ
# (the standalone-PPO campaign's value).
CHECKPOINT_FREQUENCY_TIMESTEPS = 50_000
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
