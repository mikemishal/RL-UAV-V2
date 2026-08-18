"""LR-PPO seed protocol tests -- FREEZE-ONLY verification (implementation
task; no seed in the reserved evaluation blocks (60000-99999) is executed
here, only generated/inspected as plain integers and checked for
range-membership; training-namespace seeds -- 1,000,000+ -- are likewise
never passed to env.reset() in this file).

Covers the corrected mapper (supersedes the modulo-60000 fold, which was
found to create both within-root duplicates and cross-root collisions):
    - same root -> bit-identical generated training episode-seed sequence
    - different roots -> disjoint generated sequences (root namespaces
      never overlap, by construction)
    - PREFIX STABILITY: sequence(root, 100) == sequence(root, 1000)[:100]
    - SEED CAPACITY: n_episodes > 1,000,000 is rejected explicitly
    - unassigned root is rejected explicitly
    - generated training episode seeds never fall in ANY reserved LR-PPO
      evaluation range (60000-60099, 60100-60299, 61000-65999, 66000-99999)
    - 100,000-seed-per-root FULL UNIQUENESS AUDIT (integer-only; not passed
      to env.reset()): each root's 100k seeds all unique, and the three
      roots' 100k-seed sets are pairwise disjoint
    - LR-PPO namespace disjoint from old-study namespace
    - assert_not_opened_yet() rejects every reserved (finite) evaluation
      seed and accepts ordinary development/training-namespace seeds
    - historical >=46000 documentation correction (unused placeholder, not
      a permanent global reservation) is recorded and does not alter any
      historical range
    - existing (old-study) build_episode_seed_sequence() is untouched/still
      importable and still produces its OWN previously-verified output

Run directly:
    python test_lead_residual_training_protocol.py
"""

from __future__ import annotations

from experiments.lead_residual_training_protocol import (
    TRAINING_ROOTS,
    TRAINING_ROOT_NAMESPACE_BASE,
    TRAINING_NAMESPACE_SIZE,
    RESIDUAL_VALIDATION_SEED_START,
    RESIDUAL_VALIDATION_SEED_END,
    RESIDUAL_DIAGNOSTIC_SEED_START,
    RESIDUAL_DIAGNOSTIC_SEED_END,
    EXPANDED_FINAL_SEED_START,
    EXPANDED_FINAL_SEED_END,
    ROBUSTNESS_SEED_START,
    ROBUSTNESS_SEED_END,
    OLD_STUDY_VALIDATION_RANGE,
    OLD_STUDY_DIAGNOSTIC_RANGE,
    OLD_STUDY_FINAL_RANGE,
    OLD_STUDY_FUTURE_ROBUSTNESS_PLACEHOLDER_UNUSED,
    build_lr_ppo_episode_seed_sequence,
    assert_not_opened_yet,
)
from experiments.post_detection_training_protocol import (
    build_episode_seed_sequence as old_study_build_episode_seed_sequence,
)


def test_training_roots_are_55_56_57():
    assert TRAINING_ROOTS == (55, 56, 57)


def test_root_namespace_bases():
    assert TRAINING_ROOT_NAMESPACE_BASE == {55: 1_000_000, 56: 2_000_000, 57: 3_000_000}
    assert TRAINING_NAMESPACE_SIZE == 1_000_000


def test_same_root_produces_identical_sequence():
    seq_a = build_lr_ppo_episode_seed_sequence(55, 500)
    seq_b = build_lr_ppo_episode_seed_sequence(55, 500)
    assert seq_a == seq_b


def test_different_roots_produce_disjoint_sequences():
    seq_55 = set(build_lr_ppo_episode_seed_sequence(55, 5000))
    seq_56 = set(build_lr_ppo_episode_seed_sequence(56, 5000))
    seq_57 = set(build_lr_ppo_episode_seed_sequence(57, 5000))
    assert seq_55.isdisjoint(seq_56)
    assert seq_55.isdisjoint(seq_57)
    assert seq_56.isdisjoint(seq_57)


def test_prefix_stability():
    for root in TRAINING_ROOTS:
        short = build_lr_ppo_episode_seed_sequence(root, 100)
        long = build_lr_ppo_episode_seed_sequence(root, 1000)
        assert short == long[:100]
        longer = build_lr_ppo_episode_seed_sequence(root, 5000)
        assert long == longer[:1000]
        assert short == longer[:100]


def test_seed_capacity_rejects_oversized_request():
    for root in TRAINING_ROOTS:
        build_lr_ppo_episode_seed_sequence(root, TRAINING_NAMESPACE_SIZE)  # exactly at capacity: OK
        try:
            build_lr_ppo_episode_seed_sequence(root, TRAINING_NAMESPACE_SIZE + 1)
            assert False, f"expected rejection for n_episodes > {TRAINING_NAMESPACE_SIZE}"
        except ValueError:
            pass


def test_unassigned_root_rejected():
    for bad_root in (12345, 58, 0, 42):
        try:
            build_lr_ppo_episode_seed_sequence(bad_root, 10)
            assert False, f"expected rejection for unassigned root {bad_root}"
        except ValueError:
            pass


def test_generated_training_seeds_never_collide_with_any_reserved_range():
    for root in TRAINING_ROOTS:
        base = TRAINING_ROOT_NAMESPACE_BASE[root]
        seeds = build_lr_ppo_episode_seed_sequence(root, 5000)
        for s in seeds:
            assert base <= s < base + TRAINING_NAMESPACE_SIZE
            assert not (RESIDUAL_VALIDATION_SEED_START <= s <= RESIDUAL_VALIDATION_SEED_END)
            assert not (RESIDUAL_DIAGNOSTIC_SEED_START <= s <= RESIDUAL_DIAGNOSTIC_SEED_END)
            assert not (EXPANDED_FINAL_SEED_START <= s <= EXPANDED_FINAL_SEED_END)
            assert not (ROBUSTNESS_SEED_START <= s <= ROBUSTNESS_SEED_END)


def test_100k_full_uniqueness_audit():
    """Item 8: generate >=100,000 seeds per root (integer-only; NEVER passed
    to env.reset()) and verify full within-root uniqueness and pairwise
    cross-root disjointness."""
    n = 100_000
    per_root = {}
    for root in TRAINING_ROOTS:
        seeds = build_lr_ppo_episode_seed_sequence(root, n)
        unique = set(seeds)
        assert len(seeds) == n
        assert len(unique) == n, f"root {root}: {n - len(unique)} duplicates found"
        per_root[root] = unique

    combined = set()
    for s in per_root.values():
        combined |= s
    assert len(combined) == n * len(TRAINING_ROOTS)

    roots = list(TRAINING_ROOTS)
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            inter = per_root[roots[i]] & per_root[roots[j]]
            assert len(inter) == 0, f"{roots[i]} vs {roots[j]}: {len(inter)} overlapping seeds"


def test_lr_ppo_namespace_disjoint_from_old_study_namespace():
    assert RESIDUAL_VALIDATION_SEED_START == 60_000
    assert RESIDUAL_VALIDATION_SEED_END == 60_099
    assert RESIDUAL_DIAGNOSTIC_SEED_START == 60_100
    assert RESIDUAL_DIAGNOSTIC_SEED_END == 60_299
    assert EXPANDED_FINAL_SEED_START == 61_000
    assert EXPANDED_FINAL_SEED_END == 65_999
    assert ROBUSTNESS_SEED_START == 66_000
    assert ROBUSTNESS_SEED_END == 99_999
    assert OLD_STUDY_VALIDATION_RANGE == (40_000, 40_099)
    assert OLD_STUDY_DIAGNOSTIC_RANGE == (40_100, 40_299)
    assert OLD_STUDY_FINAL_RANGE == (41_000, 45_999)
    # Training namespace (1,000,000+) is disjoint from ALL evaluation
    # ranges above (all < 100,000) and from the old study (all < 46,000).
    for base in TRAINING_ROOT_NAMESPACE_BASE.values():
        assert base >= 1_000_000 > ROBUSTNESS_SEED_END


def test_historical_placeholder_correction_documented():
    # The old ">=46000" claim was an unused future placeholder, never
    # opened/executed -- documented here as a plain constant, NOT as a
    # permanent reservation overlapping the new (finite) LR-PPO ranges.
    assert OLD_STUDY_FUTURE_ROBUSTNESS_PLACEHOLDER_UNUSED == 46_000
    # No historical range changed as a result of this correction.
    assert OLD_STUDY_FINAL_RANGE == (41_000, 45_999)


def test_assert_not_opened_yet_rejects_reserved_and_accepts_other_seeds():
    for reserved in (60_000, 60_050, 60_099, 60_100, 60_299, 61_000, 65_999, 66_000, 99_999):
        try:
            assert_not_opened_yet(reserved)
            assert False, f"expected rejection for reserved seed {reserved}"
        except ValueError:
            pass
    # Values >= 100,000 (including the entire training namespace) and
    # ordinary dev seeds are NOT LR-PPO evaluation seeds and must not raise.
    for other_seed in (0, 55, 123, 5000, 40000, 41000, 45999, 100_000, 1_000_000, 3_999_999):
        assert assert_not_opened_yet(other_seed) is None  # must not raise


def test_expected_episode_counts():
    assert RESIDUAL_VALIDATION_SEED_END - RESIDUAL_VALIDATION_SEED_START + 1 == 100
    assert RESIDUAL_DIAGNOSTIC_SEED_END - RESIDUAL_DIAGNOSTIC_SEED_START + 1 == 200
    assert EXPANDED_FINAL_SEED_END - EXPANDED_FINAL_SEED_START + 1 == 5000
    assert ROBUSTNESS_SEED_END - ROBUSTNESS_SEED_START + 1 == 34_000


def test_old_study_seed_sequence_mechanism_untouched():
    # This corrected module must not have modified the existing
    # (frozen-campaign) mechanism -- import it and confirm it still behaves
    # as previously verified (deterministic, same root -> same sequence).
    a = old_study_build_episode_seed_sequence(45, 100)
    b = old_study_build_episode_seed_sequence(45, 100)
    assert a == b
    assert len(a) == 100


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    tests.sort(key=lambda kv: kv[0])
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)

