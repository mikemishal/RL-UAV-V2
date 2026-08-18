"""Official-run argument lock + output-collision guard tests for the LR-PPO
training launcher (pre-launch protocol lock task). No training/environment
construction occurs in this file -- these test the guard FUNCTIONS directly
with synthetic argparse.Namespace objects and temporary directories.

Run directly:
    python test_lead_residual_official_run_lock.py
"""

from __future__ import annotations

import argparse
import json
import tempfile
from pathlib import Path

from experiments.lead_residual_training_protocol import (
    TRAINING_ROOTS,
    CAMPAIGN_TIMESTEPS,
    VALIDATION_FREQUENCY_TIMESTEPS,
    VALIDATION_EPISODES,
    RESIDUAL_MAX_ANGLE_DEG,
    CHECKPOINT_FREQUENCY_TIMESTEPS,
)
import experiments.train_lead_residual_ppo as launcher


def _official_args(**overrides) -> argparse.Namespace:
    base = dict(
        training_seed=55,
        smoke_test=False,
        total_timesteps=CAMPAIGN_TIMESTEPS,
        eval_freq=VALIDATION_FREQUENCY_TIMESTEPS,
        validation_episodes=VALIDATION_EPISODES,
        residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG,
        checkpoint_freq=CHECKPOINT_FREQUENCY_TIMESTEPS,
        expected_source_commit="deadbeefcafe",
    )
    base.update(overrides)
    return argparse.Namespace(**base)


def _assert_raises(fn, exc_types, *args, **kwargs):
    try:
        fn(*args, **kwargs)
        assert False, f"expected {exc_types} to be raised"
    except exc_types:
        pass


def test_official_root_rejects_wrong_total_timesteps():
    _assert_raises(launcher._enforce_official_run_lock, ValueError, _official_args(total_timesteps=999))


def test_official_root_rejects_wrong_eval_freq():
    _assert_raises(launcher._enforce_official_run_lock, ValueError, _official_args(eval_freq=1000))


def test_official_root_rejects_wrong_validation_episodes():
    _assert_raises(launcher._enforce_official_run_lock, ValueError, _official_args(validation_episodes=50))


def test_official_root_rejects_wrong_residual_angle():
    _assert_raises(launcher._enforce_official_run_lock, ValueError, _official_args(residual_max_angle_deg=45.0))


def test_official_root_rejects_wrong_checkpoint_freq():
    _assert_raises(launcher._enforce_official_run_lock, ValueError, _official_args(checkpoint_freq=10_000))


def test_official_root_requires_expected_source_commit():
    _assert_raises(launcher._enforce_official_run_lock, ValueError, _official_args(expected_source_commit=None))
    _assert_raises(launcher._enforce_official_run_lock, ValueError, _official_args(expected_source_commit=""))


def test_official_root_rejects_mismatched_commit():
    real_get_commit = launcher._get_git_commit
    launcher._get_git_commit = lambda: "actual_commit_sha"
    try:
        _assert_raises(
            launcher._enforce_official_run_lock, RuntimeError,
            _official_args(expected_source_commit="different_sha"),
        )
    finally:
        launcher._get_git_commit = real_get_commit


def test_official_root_rejects_dirty_tree():
    real_get_commit = launcher._get_git_commit
    real_get_clean = launcher._get_git_status_clean
    launcher._get_git_commit = lambda: "abc123"
    launcher._get_git_status_clean = lambda: False
    try:
        _assert_raises(
            launcher._enforce_official_run_lock, RuntimeError,
            _official_args(expected_source_commit="abc123"),
        )
    finally:
        launcher._get_git_commit = real_get_commit
        launcher._get_git_status_clean = real_get_clean


def test_official_root_accepts_correct_values_and_commit():
    real_get_commit = launcher._get_git_commit
    real_get_clean = launcher._get_git_status_clean
    launcher._get_git_commit = lambda: "abc123"
    launcher._get_git_status_clean = lambda: True
    try:
        launcher._enforce_official_run_lock(_official_args(expected_source_commit="abc123"))  # must NOT raise
    finally:
        launcher._get_git_commit = real_get_commit
        launcher._get_git_status_clean = real_get_clean


def test_smoke_test_bypasses_lock():
    launcher._enforce_official_run_lock(_official_args(smoke_test=True, total_timesteps=1))  # must NOT raise


def test_dev_root_bypasses_lock():
    assert 999999 not in TRAINING_ROOTS
    launcher._enforce_official_run_lock(_official_args(training_seed=999999, total_timesteps=1))  # must NOT raise


def test_output_collision_guard_blocks_completed_official_run():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        model_dir = Path(tmp) / "model"
        output_dir.mkdir()
        model_dir.mkdir()
        (output_dir / "COMPLETED.txt").write_text("done")
        (output_dir / "run_manifest.json").write_text(json.dumps({"smoke_test": False}))
        _assert_raises(launcher._check_output_collision_guard, RuntimeError, output_dir, model_dir)


def test_output_collision_guard_blocks_existing_official_model():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        model_dir = Path(tmp) / "model"
        output_dir.mkdir()
        model_dir.mkdir()
        (output_dir / "run_manifest.json").write_text(json.dumps({"smoke_test": False}))
        (model_dir / "best_model.zip").write_bytes(b"fake")
        _assert_raises(launcher._check_output_collision_guard, RuntimeError, output_dir, model_dir)


def test_output_collision_guard_allows_smoke_test_dir():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        model_dir = Path(tmp) / "model"
        output_dir.mkdir()
        model_dir.mkdir()
        (output_dir / "COMPLETED.txt").write_text("done")
        (output_dir / "run_manifest.json").write_text(json.dumps({"smoke_test": True}))
        launcher._check_output_collision_guard(output_dir, model_dir)  # must NOT raise


def test_output_collision_guard_allows_empty_dirs():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        model_dir = Path(tmp) / "model"
        launcher._check_output_collision_guard(output_dir, model_dir)  # neither dir exists yet -- must NOT raise


def test_output_collision_guard_reports_partial_files_without_raising():
    with tempfile.TemporaryDirectory() as tmp:
        output_dir = Path(tmp) / "output"
        model_dir = Path(tmp) / "model"
        output_dir.mkdir()
        model_dir.mkdir()
        (model_dir / "checkpoints").mkdir()
        (model_dir / "checkpoints" / "lr_ppo_seed55_100000_steps.zip").write_bytes(b"partial")
        launcher._check_output_collision_guard(output_dir, model_dir)  # must NOT raise (only reports)


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
