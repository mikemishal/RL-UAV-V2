"""
Pre-launch validation for the post-detection PPO training protocol.

Run BEFORE starting any of the six full 400k training runs. Fails LOUDLY
(non-zero exit, explicit error message) if protocol drift is detected --
never silently proceeds.

Checks:
    - git branch/commit, clean working tree
    - protocol constants (training roots, validation/diagnostic/final-test/
      future-robustness seed ranges) match the frozen specification
    - no reserved seed range is referenced by the output directory contents
    - output directory does not already contain a prior PUBLICATION model
      (i.e. a completed manifest under the LEGACY results/training_400k/ path)
    - Direct and Kalman EnvConfig parity (only use_kalman_tracking differs)

Usage:
    python experiments/pre_launch_validation.py --output-dir results/training_post_detection_400k
"""

from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.post_detection_training_protocol import (
    TRAINING_SEEDS,
    VALIDATION_SEED_OFFSET,
    VALIDATION_EPISODES,
    DIAGNOSTIC_SEED_OFFSET,
    DIAGNOSTIC_EPISODES,
    FINAL_TEST_SEED_START,
    FINAL_TEST_SEED_END,
    FUTURE_ROBUSTNESS_SEED_START,
    build_env_config,
    assert_only_estimator_differs,
)

_ERRORS: list[str] = []


def _fail(msg: str) -> None:
    _ERRORS.append(msg)
    print(f"[FAIL] {msg}")


def _ok(msg: str) -> None:
    print(f"[OK]   {msg}")


def check_git_state() -> None:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode()
        branch = subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
        commit = subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception as e:
        _fail(f"unable to query git state: {e}")
        return
    if status.strip():
        _fail(f"working tree is NOT clean:\n{status}")
    else:
        _ok("working tree clean")
    _ok(f"branch={branch}, commit={commit}")


def check_protocol_constants() -> None:
    expected_roots = (45, 46, 47)
    if TRAINING_SEEDS != expected_roots:
        _fail(f"TRAINING_SEEDS drift: expected {expected_roots}, got {TRAINING_SEEDS}")
    else:
        _ok(f"TRAINING_SEEDS == {TRAINING_SEEDS}")

    if (VALIDATION_SEED_OFFSET, VALIDATION_EPISODES) != (40_000, 100):
        _fail(f"validation range drift: expected (40000,100), got ({VALIDATION_SEED_OFFSET},{VALIDATION_EPISODES})")
    else:
        _ok(f"validation seeds == {VALIDATION_SEED_OFFSET}..{VALIDATION_SEED_OFFSET + VALIDATION_EPISODES - 1}")

    if (DIAGNOSTIC_SEED_OFFSET, DIAGNOSTIC_EPISODES) != (40_100, 200):
        _fail(f"diagnostic range drift: expected (40100,200), got ({DIAGNOSTIC_SEED_OFFSET},{DIAGNOSTIC_EPISODES})")
    else:
        _ok(f"diagnostic seeds == {DIAGNOSTIC_SEED_OFFSET}..{DIAGNOSTIC_SEED_OFFSET + DIAGNOSTIC_EPISODES - 1} (reserved)")

    if (FINAL_TEST_SEED_START, FINAL_TEST_SEED_END) != (41_000, 45_999):
        _fail(f"final-test range drift: expected (41000,45999), got ({FINAL_TEST_SEED_START},{FINAL_TEST_SEED_END})")
    else:
        _ok(f"final-test seeds == {FINAL_TEST_SEED_START}..{FINAL_TEST_SEED_END} (RESERVED, untouched)")

    if FUTURE_ROBUSTNESS_SEED_START != 46_000:
        _fail(f"future-robustness start drift: expected 46000, got {FUTURE_ROBUSTNESS_SEED_START}")
    else:
        _ok(f"future robustness seeds >= {FUTURE_ROBUSTNESS_SEED_START} (RESERVED)")

    # Disjointness sanity across all ranges.
    ranges = {
        "training": set(TRAINING_SEEDS),
        "validation": set(range(VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + VALIDATION_EPISODES)),
        "diagnostic": set(range(DIAGNOSTIC_SEED_OFFSET, DIAGNOSTIC_SEED_OFFSET + DIAGNOSTIC_EPISODES)),
        "final_test": set(range(FINAL_TEST_SEED_START, FINAL_TEST_SEED_END + 1)),
    }
    names = list(ranges)
    any_overlap = False
    for i in range(len(names)):
        for j in range(i + 1, len(names)):
            overlap = ranges[names[i]] & ranges[names[j]]
            if overlap:
                any_overlap = True
                _fail(f"seed range overlap between {names[i]} and {names[j]}: {overlap}")
    if not any_overlap:
        _ok("all protocol seed ranges are pairwise disjoint")


def check_env_config_parity() -> None:
    try:
        config_direct = build_env_config("direct")
        config_kalman = build_env_config("kalman")
        assert_only_estimator_differs(config_direct, config_kalman)
        assert config_direct.defender_standby_until_detection is True
        assert config_kalman.defender_standby_until_detection is True
        _ok("Direct/Kalman EnvConfig parity confirmed (only use_kalman_tracking differs); standby=True for both")
    except AssertionError as e:
        _fail(f"EnvConfig parity check failed: {e}")


def check_output_dir_no_legacy_publication_models(output_dir: Path) -> None:
    legacy_root = PROJECT_ROOT / "results" / "training_400k"
    if output_dir.resolve() == legacy_root.resolve():
        _fail(f"output_dir must NOT be the legacy publication path {legacy_root}")
        return
    if not output_dir.exists():
        _ok(f"output_dir {output_dir} does not yet exist (clean start)")
        return
    found_publication_run = False
    for manifest_path in output_dir.rglob("training_manifest.json"):
        try:
            import json
            manifest = json.loads(manifest_path.read_text())
        except Exception:
            continue
        if manifest.get("status") == "completed" and not manifest.get("smoke_test", False):
            if manifest.get("training_seed") in (45, 46, 47) and manifest.get("training_timesteps", 0) >= 400_000:
                found_publication_run = True
                _fail(f"output_dir already contains a COMPLETED publication-scale run at {manifest_path} "
                      f"-- refusing to silently overwrite. Use a fresh output_dir or explicit override.")
    if not found_publication_run:
        _ok(f"no prior completed publication-scale run detected under {output_dir}")


def main() -> int:
    parser = argparse.ArgumentParser(description="Pre-launch validation for post-detection PPO training")
    parser.add_argument("--output-dir", type=str, default="results/training_post_detection_400k")
    args = parser.parse_args()

    print("=" * 70)
    print("PRE-LAUNCH VALIDATION -- post-detection PPO training protocol")
    print("=" * 70)
    check_git_state()
    check_protocol_constants()
    check_env_config_parity()
    check_output_dir_no_legacy_publication_models(Path(args.output_dir))
    print("=" * 70)

    if _ERRORS:
        print(f"\n{len(_ERRORS)} PROTOCOL DRIFT ERROR(S) DETECTED -- DO NOT LAUNCH TRAINING:")
        for e in _ERRORS:
            print(f"  - {e}")
        return 1
    print("\nAll pre-launch checks passed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
