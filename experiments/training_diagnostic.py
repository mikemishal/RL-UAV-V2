"""
Post-training diagnostic holdout evaluation (multi-seed PPO training protocol).

TRAINING DIAGNOSTIC -- NOT THE FINAL PAPER RESULT. This evaluates the
best_model.zip of each of the six training runs (Direct/Kalman x seeds
42/43/44) on a SEPARATE diagnostic holdout (seeds 12000..12199, 200
episodes, deterministic inference) to sanity-check training quality before
the final paper Monte Carlo evaluation (which will use a different,
reserved seed range >= FINAL_TEST_SEED_OFFSET).

Guardrail: this script refuses to evaluate on any seed >= FINAL_TEST_SEED_OFFSET.

Usage:
    python experiments/training_diagnostic.py
    python experiments/training_diagnostic.py --tracks direct
    python experiments/training_diagnostic.py --seeds 42 43

Output: a gitignored JSON summary under results/training_diagnostic/ (not a
publication CSV -- see experiments/method_specs.py / the later paper
evaluation phase for that).
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.rl.ppo_policy_wrapper import PPOPolicyWrapper
from uav_defend.policies.rl_kalman.ppo_kalman_policy_wrapper import PPOKalmanPolicyWrapper

from experiments.eval_utils import run_episode
from experiments.training_protocol import (
    TRAINING_SEEDS,
    DIAGNOSTIC_SEED_OFFSET,
    DIAGNOSTIC_EPISODES,
    DIAGNOSTIC_SEEDS,
    FINAL_TEST_SEED_OFFSET,
    run_output_dir,
    assert_not_final_test_seed,
)

ALL_TRACKS = ("direct", "kalman")


def _build_env(track: str) -> SoldierEnv:
    return SoldierEnv(config=EnvConfig(use_kalman_tracking=(track == "kalman")))


def _load_policy(track: str, model_path: Path):
    if track == "direct":
        return PPOPolicyWrapper.load(str(model_path), deterministic=True)
    return PPOKalmanPolicyWrapper.load(str(model_path), deterministic=True)


def evaluate_run(track: str, seed: int, output_root: Path, diagnostic_seeds) -> dict | None:
    """
    Evaluate ONE run's best_model.zip on the diagnostic holdout. Returns
    None (with a printed warning) if the run's best_model.zip is missing.
    """
    for s in diagnostic_seeds:
        assert_not_final_test_seed(s)  # guard: never evaluate on the reserved final-test range

    run_dir = run_output_dir(output_root, track, seed)
    best_model_path = run_dir / "best_model.zip"
    if not best_model_path.exists():
        print(f"  [WARN] {track} seed={seed}: best_model.zip not found at {best_model_path}; skipping")
        return None

    env = _build_env(track)
    policy = _load_policy(track, best_model_path)

    outcomes = {"intercepted": 0, "soldier_caught": 0, "unsafe_intercept": 0, "timeout": 0}
    episode_lengths = []
    detection_times = []
    intercept_times = []
    tracking_errors = []

    for s in diagnostic_seeds:
        metrics = run_episode(env, policy, seed=s)
        outcome = metrics["outcome"]
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        episode_lengths.append(metrics["episode_length"])
        if metrics["detection_time"] >= 0:
            detection_times.append(metrics["detection_time"])
        if metrics["intercept_time"] >= 0:
            intercept_times.append(metrics["intercept_time"])
        if metrics.get("mean_tracking_error") is not None:
            tracking_errors.append(metrics["mean_tracking_error"])

    n = len(diagnostic_seeds)
    success_count = outcomes.get("intercepted", 0)
    result = {
        "track": track,
        "seed": seed,
        "run_dir": str(run_dir),
        "n_episodes": n,
        "success_count": success_count,
        "success_rate": success_count / n,
        "soldier_caught_count": outcomes.get("soldier_caught", 0),
        "soldier_caught_rate": outcomes.get("soldier_caught", 0) / n,
        "unsafe_intercept_count": outcomes.get("unsafe_intercept", 0),
        "unsafe_intercept_rate": outcomes.get("unsafe_intercept", 0) / n,
        "timeout_count": outcomes.get("timeout", 0),
        "timeout_rate": outcomes.get("timeout", 0) / n,
        "mean_episode_length": float(np.mean(episode_lengths)) if episode_lengths else None,
        "mean_detection_time": float(np.mean(detection_times)) if detection_times else None,
        "mean_intercept_time": float(np.mean(intercept_times)) if intercept_times else None,
        # Diagnostic ONLY. Direct tracking_error = raw measurement error;
        # Kalman tracking_error = filter estimation error. NOT the same
        # estimator -- do not compare these two numbers as equivalent.
        "mean_tracking_error": float(np.mean(tracking_errors)) if tracking_errors else None,
    }
    return result


def aggregate_track(results: list[dict]) -> dict:
    """
    Training-seed variability across the 3 independently trained models for
    one track: mean/std/min/max of success rate and mean episode length.
    This is NOT the Monte Carlo confidence interval used for the final
    paper -- it is variability ACROSS TRAINING SEEDS.
    """
    success_rates = [r["success_rate"] for r in results]
    ep_lengths = [r["mean_episode_length"] for r in results if r["mean_episode_length"] is not None]
    return {
        "n_seeds": len(results),
        "success_rate_mean": float(np.mean(success_rates)),
        "success_rate_std": float(np.std(success_rates)),
        "success_rate_min": float(np.min(success_rates)),
        "success_rate_max": float(np.max(success_rates)),
        "mean_episode_length_mean": float(np.mean(ep_lengths)) if ep_lengths else None,
        "mean_episode_length_std": float(np.std(ep_lengths)) if ep_lengths else None,
        "mean_episode_length_min": float(np.min(ep_lengths)) if ep_lengths else None,
        "mean_episode_length_max": float(np.max(ep_lengths)) if ep_lengths else None,
    }


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-training diagnostic holdout evaluation (NOT final paper results)")
    parser.add_argument("--tracks", type=str, nargs="+", default=list(ALL_TRACKS), choices=list(ALL_TRACKS))
    parser.add_argument("--seeds", type=int, nargs="+", default=list(TRAINING_SEEDS))
    parser.add_argument("--output-root", type=str, default=str(PROJECT_ROOT / "results"))
    parser.add_argument(
        "--summary-path", type=str,
        default=str(PROJECT_ROOT / "results" / "training_diagnostic" / "diagnostic_summary.json"),
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)
    diagnostic_seeds = list(DIAGNOSTIC_SEEDS)

    print("=" * 70)
    print("TRAINING DIAGNOSTIC HOLDOUT EVALUATION -- NOT FINAL PAPER RESULT")
    print("=" * 70)
    print(f"Diagnostic seeds: {DIAGNOSTIC_SEED_OFFSET}..{DIAGNOSTIC_SEED_OFFSET + DIAGNOSTIC_EPISODES - 1} "
          f"({DIAGNOSTIC_EPISODES} episodes)")
    print(f"Final-test seeds (RESERVED, untouched here): >= {FINAL_TEST_SEED_OFFSET}")
    print("=" * 70)

    all_results: dict[str, list[dict]] = {track: [] for track in args.tracks}
    for track in args.tracks:
        for seed in args.seeds:
            print(f"\nEvaluating {track} seed={seed} on diagnostic holdout ({len(diagnostic_seeds)} episodes)...")
            result = evaluate_run(track, seed, output_root, diagnostic_seeds)
            if result is not None:
                all_results[track].append(result)
                print(f"  success_rate={result['success_rate']:.2%} "
                      f"soldier_caught={result['soldier_caught_rate']:.2%} "
                      f"unsafe_intercept={result['unsafe_intercept_rate']:.2%} "
                      f"timeout={result['timeout_rate']:.2%} "
                      f"mean_ep_len={result['mean_episode_length']:.1f}")

    aggregates = {track: aggregate_track(results) for track, results in all_results.items() if results}

    summary = {
        "label": "TRAINING DIAGNOSTIC -- NOT FINAL PAPER RESULT",
        "diagnostic_seed_offset": DIAGNOSTIC_SEED_OFFSET,
        "diagnostic_episodes": DIAGNOSTIC_EPISODES,
        "final_test_seed_offset_reserved_untouched": FINAL_TEST_SEED_OFFSET,
        "per_run_results": all_results,
        "per_track_aggregate_across_training_seeds": aggregates,
    }

    summary_path = Path(args.summary_path)
    summary_path.parent.mkdir(parents=True, exist_ok=True)
    summary_path.write_text(json.dumps(summary, indent=2, default=str))
    print(f"\nDiagnostic summary written to: {summary_path}")

    print("\n" + "=" * 70)
    print("PER-TRACK AGGREGATE (training-seed variability, NOT Monte Carlo CI)")
    print("=" * 70)
    for track, agg in aggregates.items():
        print(f"{track}: success_rate mean={agg['success_rate_mean']:.2%} "
              f"std={agg['success_rate_std']:.2%} min={agg['success_rate_min']:.2%} "
              f"max={agg['success_rate_max']:.2%}")

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
