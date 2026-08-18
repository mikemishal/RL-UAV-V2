"""LR-PPO seed protocol tests -- FREEZE-ONLY verification (implementation
task; no seed in 60000+ is executed here, only generated/inspected as plain
integers and checked for range-membership).

Covers (per the lead-residual-ppo-training-protocol task spec, items 5/6):
    - same root -> identical generated training episode-seed sequence
    - different roots -> different generated sequences
    - generated training episode seeds never fall in ANY reserved LR-PPO
      evaluation range (60000..60099, 60100..60299, 61000..65999, >=66000)
      -- guaranteed by construction (modulo fold into [0, 60000))
    - LR-PPO namespace is disjoint from the old-study namespace
      (40000..40099, 40100..40299, 41000..45999, >=46000)
    - assert_not_opened_yet() rejects every reserved seed and accepts
      ordinary development seeds
    - existing (old-study) build_episode_seed_sequence() is untouched/still
      importable and still produces its OWN previously-verified output
      (i.e. this new module did not modify it)

Run directly:
    python test_lead_residual_training_protocol.py
"""

from __future__ import annotations

from experiments.lead_residual_training_protocol import (
    TRAINING_ROOTS,
    RESIDUAL_VALIDATION_SEED_START,
    RESIDUAL_VALIDATION_SEED_END,
    RESIDUAL_DIAGNOSTIC_SEED_START,
    RESIDUAL_DIAGNOSTIC_SEED_END,
    EXPANDED_FINAL_SEED_START,
    EXPANDED_FINAL_SEED_END,
    FUTURE_ROBUSTNESS_SEED_START,
    OLD_STUDY_VALIDATION_RANGE,
    OLD_STUDY_DIAGNOSTIC_RANGE,
    OLD_STUDY_FINAL_RANGE,
    OLD_STUDY_FUTURE_ROBUSTNESS_START,
    TRAINING_EPISODE_SEED_NAMESPACE_CEILING,
    build_lr_ppo_episode_seed_sequence,
    assert_not_opened_yet,
)
from experiments.post_detection_training_protocol import (
    build_episode_seed_sequence as old_study_build_episode_seed_sequence,
)


def test_training_roots_are_55_56_57():
    assert TRAINING_ROOTS == (55, 56, 57)


def test_same_root_produces_identical_sequence():
    seq_a = build_lr_ppo_episode_seed_sequence(55, 500)
    seq_b = build_lr_ppo_episode_seed_sequence(55, 500)
    assert seq_a == seq_b


def test_different_roots_produce_different_sequences():
    seq_55 = build_lr_ppo_episode_seed_sequence(55, 500)
    seq_56 = build_lr_ppo_episode_seed_sequence(56, 500)
    seq_57 = build_lr_ppo_episode_seed_sequence(57, 500)
    assert seq_55 != seq_56
    assert seq_56 != seq_57
    assert seq_55 != seq_57


def test_generated_training_seeds_never_collide_with_any_reserved_range():
    for root in TRAINING_ROOTS:
        seeds = build_lr_ppo_episode_seed_sequence(root, 5000)
        for s in seeds:
            assert 0 <= s < TRAINING_EPISODE_SEED_NAMESPACE_CEILING
            assert not (RESIDUAL_VALIDATION_SEED_START <= s <= RESIDUAL_VALIDATION_SEED_END)
            assert not (RESIDUAL_DIAGNOSTIC_SEED_START <= s <= RESIDUAL_DIAGNOSTIC_SEED_END)
            assert not (EXPANDED_FINAL_SEED_START <= s <= EXPANDED_FINAL_SEED_END)
            assert s < FUTURE_ROBUSTNESS_SEED_START


def test_lr_ppo_namespace_disjoint_from_old_study_namespace():
    # Every NEW LR-PPO evaluation range must start at/after 60000, strictly
    # above the old study's highest reserved boundary (46000).
    assert RESIDUAL_VALIDATION_SEED_START >= OLD_STUDY_FUTURE_ROBUSTNESS_START
    assert RESIDUAL_VALIDATION_SEED_START == 60_000
    assert RESIDUAL_VALIDATION_SEED_END == 60_099
    assert RESIDUAL_DIAGNOSTIC_SEED_START == 60_100
    assert RESIDUAL_DIAGNOSTIC_SEED_END == 60_299
    assert EXPANDED_FINAL_SEED_START == 61_000
    assert EXPANDED_FINAL_SEED_END == 65_999
    assert FUTURE_ROBUSTNESS_SEED_START == 66_000
    assert OLD_STUDY_VALIDATION_RANGE == (40_000, 40_099)
    assert OLD_STUDY_DIAGNOSTIC_RANGE == (40_100, 40_299)
    assert OLD_STUDY_FINAL_RANGE == (41_000, 45_999)
    assert OLD_STUDY_FUTURE_ROBUSTNESS_START == 46_000


def test_assert_not_opened_yet_rejects_reserved_and_accepts_dev_seeds():
    for reserved in (60_000, 60_050, 60_099, 60_100, 60_299, 61_000, 65_999, 66_000, 100_000):
        try:
            assert_not_opened_yet(reserved)
            assert False, f"expected rejection for reserved seed {reserved}"
        except ValueError:
            pass
    for dev_seed in (0, 55, 123, 5000, 40000, 41000, 45999, 59999):
        assert_not_opened_yet(dev_seed) is None  # must not raise


def test_expected_episode_counts():
    assert RESIDUAL_VALIDATION_SEED_END - RESIDUAL_VALIDATION_SEED_START + 1 == 100
    assert RESIDUAL_DIAGNOSTIC_SEED_END - RESIDUAL_DIAGNOSTIC_SEED_START + 1 == 200
    assert EXPANDED_FINAL_SEED_END - EXPANDED_FINAL_SEED_START + 1 == 5000


def test_old_study_seed_sequence_mechanism_untouched():
    # This new module must not have modified the existing (frozen-campaign)
    # mechanism -- import it and confirm it still behaves as previously
    # verified (deterministic, same root -> same sequence).
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
