"""
Locked acquisition-behavior ablation runner.

Separates PPO's learned pre-detection acquisition/repositioning behavior
from its post-detection interception behavior by comparing:
    - Full PPO (the exact frozen policy from the final nominal experiment)
    - Escorted PPO (SAME frozen weights, wrapped so the built-in scripted
      escort law controls until detection, then hands off completely)
against the strongest Direct/Kalman classical baseline (Lead / Lead-Kalman).

Usage:
    # Full locked ablation (seeds 25000..25999, 1000 episodes/instance):
    python experiments/run_acquisition_ablation.py

    # Small NON-FINAL dry run (diagnostic seeds < 25000 only):
    python experiments/run_acquisition_ablation.py --seed-start 13000 --n-episodes 5 \
        --output-root results/_dry_run_acquisition_ablation
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
import warnings
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy
from uav_defend.policies.rl.ppo_policy_wrapper import PPOPolicyWrapper
from uav_defend.policies.rl_kalman.ppo_kalman_policy_wrapper import PPOKalmanPolicyWrapper

from experiments.policy_wrappers import EscortUntilDetectionPolicy
from experiments.acquisition_ablation_protocol import (
    ACQUISITION_ABLATION_SEED_OFFSET,
    ACQUISITION_ABLATION_EPISODES,
    ACQUISITION_ABLATION_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    ABLATION_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    ACQUISITION_ABLATION_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    CONDITION_SUMMARY_FILENAME,
    TRAINING_SEED_SUMMARY_FILENAME,
    PAIRED_COMPARISONS_FILENAME,
    ACQUISITION_ABLATION_SUMMARY_FILENAME,
)
from experiments.acquisition_eval_utils import run_ablation_episode, sha256_of_file
from experiments.final_stats import (
    wilson_interval,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_paired_bootstrap_matched_seeds,
)

_NOMINAL_ENV_CONFIG = EnvConfig()
_FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def build_env(estimator: str) -> SoldierEnv:
    return SoldierEnv(config=EnvConfig(use_kalman_tracking=(estimator == "kalman")))


def build_policy(instance):
    if instance.condition == "lead":
        return get_policy("lead" if instance.track == "direct" else "lead_kalman")
    wrapper_cls = PPOKalmanPolicyWrapper if instance.track == "kalman" else PPOPolicyWrapper
    underlying = wrapper_cls.load(str(instance.model_path), deterministic=True)
    if instance.condition == "escorted_ppo":
        return EscortUntilDetectionPolicy(underlying)
    return underlying  # full_ppo


def compute_model_hashes_and_verify() -> list[dict]:
    """Hash the 6 frozen 400k models and verify they match the nominal manifest (item 3)."""
    nominal_hashes_by_path = {}
    if _FINAL_NOMINAL_HASHES_PATH.exists():
        for row in json.loads(_FINAL_NOMINAL_HASHES_PATH.read_text()):
            nominal_hashes_by_path[row["model_path"].replace("\\", "/")] = row["sha256"]

    seen_paths: dict[str, str] = {}
    rows = []
    for instance in ABLATION_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if rel in seen_paths:
            continue  # Full/Escorted share the same file; hash it once.
        if not instance.model_path.exists():
            raise FileNotFoundError(f"Frozen model not found: {instance.model_path}")
        digest = sha256_of_file(instance.model_path)
        seen_paths[rel] = digest
        nominal_digest = nominal_hashes_by_path.get(rel)
        rows.append({
            "track": instance.track,
            "training_seed": instance.training_seed,
            "model_path": rel,
            "sha256": digest,
            "matches_final_nominal_hash": (nominal_digest == digest) if nominal_digest else None,
            "final_nominal_sha256": nominal_digest,
        })
    for r in rows:
        if r["matches_final_nominal_hash"] is False:
            raise RuntimeError(
                f"Model hash mismatch vs final nominal manifest for {r['model_path']}: "
                f"{r['sha256']} != {r['final_nominal_sha256']}"
            )
    return rows


def build_manifest(seeds: list[int], output_root: Path, model_hashes: list[dict]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    opened = len(seeds) > 0 and min(seeds) >= ACQUISITION_ABLATION_SEED_OFFSET
    env = build_env("measurement")
    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "ppo_model_hashes": model_hashes,
        "evaluation_seed_start": min(seeds) if seeds else None,
        "evaluation_seed_end": max(seeds) if seeds else None,
        "n_episodes_per_instance": len(seeds),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "ablation_test_opened": opened,
        "ablation_first_seed": ACQUISITION_ABLATION_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    for i, instance in enumerate(ABLATION_INSTANCES, start=1):
        env = build_env(instance.estimator)
        policy = build_policy(instance)
        if verbose:
            print(f"[{i}/{len(ABLATION_INSTANCES)}] {instance.policy_instance} "
                  f"(condition={instance.condition}, track={instance.track}) -- {len(seeds)} episodes...")
        for s in seeds:
            row = run_ablation_episode(env, policy, seed=s, dt=dt)
            row["condition"] = instance.condition
            row["track"] = instance.track
            row["policy_instance"] = instance.policy_instance
            row["estimator"] = instance.estimator
            row["training_seed"] = instance.training_seed if instance.training_seed is not None else float("nan")
            rows.append(row)
        env.close()
        if verbose:
            print(f"  done ({time.time() - t_start:.1f}s elapsed)")

    df = pd.DataFrame(rows)
    output_root.mkdir(parents=True, exist_ok=True)
    df.to_csv(output_root / RAW_EPISODE_RESULTS_FILENAME, index=False)
    return df


def run_integrity_checks(df: pd.DataFrame, seeds: list[int]) -> dict:
    results = {}
    expected_rows = EXPECTED_N_POLICY_INSTANCES * len(seeds)
    results["expected_row_count"] = expected_rows
    results["actual_row_count"] = len(df)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    dup = df.duplicated(subset=["policy_instance", "eval_seed"]).sum()
    results["duplicate_policy_instance_seed_pairs"] = int(dup)
    assert dup == 0, f"found {dup} duplicated (policy_instance, eval_seed) pairs"

    seed_set_expected = set(seeds)
    common_seeds_ok = all(s == seed_set_expected for s in df.groupby("policy_instance")["eval_seed"].apply(set))
    results["common_random_numbers_ok"] = bool(common_seeds_ok)
    assert common_seeds_ok, "not all 14 policy instances share the identical evaluation-seed set"

    outcome_sum = df.groupby("policy_instance")[["success", "soldier_caught", "unsafe_intercept", "timeout"]].sum().sum(axis=1)
    outcome_ok = bool((outcome_sum == len(seeds)).all())
    results["outcome_accounting_ok"] = outcome_ok
    assert outcome_ok, "outcome categories do not sum to n_episodes for every policy instance"

    required = ["success", "soldier_caught", "unsafe_intercept", "timeout", "episode_length_steps", "total_reward"]
    nan_inf_bad = {c: int((~np.isfinite(df[c])).sum()) for c in required if (~np.isfinite(df[c])).sum()}
    results["nan_inf_in_required_fields"] = nan_inf_bad
    assert not nan_inf_bad, f"NaN/Inf found in required-always-defined fields: {nan_inf_bad}"

    n_never_detected = int(df["detection_step"].isna().sum())
    results["episodes_never_detected"] = n_never_detected

    return results


def run_pre_detection_identity_check(df: pd.DataFrame) -> dict:
    """
    Item 21: Lead and all 3 Escorted PPO models (same track) must have
    IDENTICAL pre-detection trajectories under matched seeds -- same
    detection_step, detection_time, defender position at detection.
    """
    tol = 1e-4
    results = {}
    for track in ("direct", "kalman"):
        lead_name = "lead" if track == "direct" else "lead_kalman"
        lead_df = df[df["policy_instance"] == lead_name].sort_values("eval_seed").reset_index(drop=True)
        prefix = "ppo" if track == "direct" else "ppo_kalman"
        mismatches = 0
        checked = 0
        for seed_label in (42, 43, 44):
            esc_name = f"{prefix}_escorted_seed_{seed_label}"
            esc_df = df[df["policy_instance"] == esc_name].sort_values("eval_seed").reset_index(drop=True)
            assert (lead_df["eval_seed"].to_numpy() == esc_df["eval_seed"].to_numpy()).all()
            checked += len(esc_df)
            step_mismatch = ~np.isclose(
                lead_df["detection_step"].to_numpy(), esc_df["detection_step"].to_numpy(),
                equal_nan=True, atol=tol,
            )
            pos_mismatch = ~np.isclose(
                lead_df[["defender_pos_at_detection_x", "defender_pos_at_detection_y", "defender_pos_at_detection_z"]].to_numpy(),
                esc_df[["defender_pos_at_detection_x", "defender_pos_at_detection_y", "defender_pos_at_detection_z"]].to_numpy(),
                equal_nan=True, atol=tol,
            ).any(axis=1)
            mismatches += int((step_mismatch | pos_mismatch).sum())
        results[track] = {"checked_episode_pairs": checked, "mismatches": mismatches}
    return results


def _get_success_vector(df: pd.DataFrame, policy_instance: str) -> np.ndarray:
    sub = df[df["policy_instance"] == policy_instance].sort_values("eval_seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _get_success_matrix(df: pd.DataFrame, track: str, condition: str) -> tuple[np.ndarray, list[int]]:
    prefix = "ppo" if track == "direct" else "ppo_kalman"
    suffix = "full" if condition == "full_ppo" else "escorted"
    seed_labels = [42, 43, 44]
    rows = []
    for seed in seed_labels:
        name = f"{prefix}_{suffix}_seed_{seed}"
        sub = df[df["policy_instance"] == name].sort_values("eval_seed")
        rows.append(sub["success"].to_numpy(dtype=np.float64))
    return np.stack(rows), seed_labels


def _get_metric_matrix(df: pd.DataFrame, track: str, condition: str, column: str) -> np.ndarray:
    prefix = "ppo" if track == "direct" else "ppo_kalman"
    suffix = "full" if condition == "full_ppo" else "escorted"
    rows = []
    for seed in (42, 43, 44):
        name = f"{prefix}_{suffix}_seed_{seed}"
        sub = df[df["policy_instance"] == name].sort_values("eval_seed")
        rows.append(sub[column].to_numpy(dtype=np.float64))
    return np.stack(rows)


def compute_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track in ("direct", "kalman"):
        lead_name = "lead" if track == "direct" else "lead_kalman"
        lead_df = df[df["policy_instance"] == lead_name]
        n = len(lead_df)
        successes = int(lead_df["success"].sum())
        ci_lo, ci_hi = wilson_interval(successes, n)
        rows.append({
            "track": track, "condition": "lead", "n_eval_episodes": n, "n_training_seeds": 0,
            "success_rate": successes / n, "success_ci_low": ci_lo, "success_ci_high": ci_hi,
            "training_seed_sd": float("nan"),
            "mean_detection_time_seconds": float(lead_df["detection_time_seconds"].mean()),
            "mean_post_detection_intercept_seconds": float(lead_df["post_detection_intercept_seconds"].mean()),
            "mean_episode_length_seconds": float(lead_df["episode_length_seconds"].mean()),
        })
        for condition in ("full_ppo", "escorted_ppo"):
            success_matrix, seed_labels = _get_success_matrix(df, track, condition)
            observed, ci_lo, ci_hi = hierarchical_bootstrap_track(success_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            per_seed_rates = success_matrix.mean(axis=1)

            def _ew(col):
                m = _get_metric_matrix(df, track, condition, col)
                with warnings.catch_warnings():
                    warnings.simplefilter("ignore", RuntimeWarning)
                    row_means = np.nanmean(m, axis=1)
                    if np.all(np.isnan(row_means)):
                        return float("nan")
                    return float(np.nanmean(row_means))

            rows.append({
                "track": track, "condition": condition,
                "n_eval_episodes": success_matrix.shape[1], "n_training_seeds": 3,
                "success_rate": observed, "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                "training_seed_sd": float(np.std(per_seed_rates)),
                "mean_detection_time_seconds": _ew("detection_time_seconds"),
                "mean_post_detection_intercept_seconds": _ew("post_detection_intercept_seconds"),
                "mean_episode_length_seconds": _ew("episode_length_seconds"),
            })
    return pd.DataFrame(rows)


def compute_training_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for track in ("direct", "kalman"):
        prefix = "ppo" if track == "direct" else "ppo_kalman"
        for condition in ("full_ppo", "escorted_ppo"):
            suffix = "full" if condition == "full_ppo" else "escorted"
            for seed in (42, 43, 44):
                name = f"{prefix}_{suffix}_seed_{seed}"
                sub = df[df["policy_instance"] == name]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                rows.append({
                    "track": track, "condition": condition, "training_seed": seed,
                    "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
                    "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "mean_detection_time_seconds": float(sub["detection_time_seconds"].mean()),
                })
    return pd.DataFrame(rows)


def compute_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, track, cond_a, cond_b in ALL_COMPARISONS:
        row = {"comparison_name": name, "track": track, "condition_a": cond_a, "condition_b": cond_b,
               "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan")}
        a_is_ppo = cond_a != "lead"
        b_is_ppo = cond_b != "lead"
        if a_is_ppo and b_is_ppo:
            a_matrix, labels_a = _get_success_matrix(df, track, cond_a)
            b_matrix, labels_b = _get_success_matrix(df, track, cond_b)
            assert labels_a == labels_b == [42, 43, 44]
            observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
                a_matrix, b_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
            )
            row.update({
                "comparison_type": "hierarchical_paired_matched_seeds",
                "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                "diff_seed_42_pp": per_seed_diffs[0] * 100,
                "diff_seed_43_pp": per_seed_diffs[1] * 100,
                "diff_seed_44_pp": per_seed_diffs[2] * 100,
            })
        else:
            ppo_cond, lead_cond = (cond_a, cond_b) if a_is_ppo else (cond_b, cond_a)
            ppo_matrix, _ = _get_success_matrix(df, track, ppo_cond)
            lead_name = "lead" if track == "direct" else "lead_kalman"
            lead_vec = _get_success_vector(df, lead_name)
            observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(
                ppo_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
            )
            if not a_is_ppo:
                observed, lo, hi = -observed, -hi, -lo
            row.update({
                "comparison_type": "hierarchical_paired_vs_scripted",
                "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            })
        rows.append(row)

    # Detection-time difference (item 26): Full PPO vs Escorted PPO, per track.
    for track in ("direct", "kalman"):
        full_matrix = _get_metric_matrix(df, track, "full_ppo", "detection_time_seconds")
        esc_matrix = _get_metric_matrix(df, track, "escorted_ppo", "detection_time_seconds")
        observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
            full_matrix, esc_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
        )
        rows.append({
            "comparison_name": f"detection_time_full_minus_escorted_{track}",
            "track": track, "condition_a": "full_ppo", "condition_b": "escorted_ppo",
            "comparison_type": "hierarchical_paired_matched_seeds_detection_time_seconds",
            "observed_diff_pp": float("nan"),
            "ci_low_pp": float("nan"), "ci_high_pp": float("nan"),
            "observed_diff_seconds": observed, "ci_low_seconds": lo, "ci_high_seconds": hi,
            "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan"),
        })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked acquisition-behavior ablation (Full PPO vs Escorted PPO vs Lead)")
    parser.add_argument("--seed-start", type=int, default=ACQUISITION_ABLATION_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=ACQUISITION_ABLATION_EPISODES)
    parser.add_argument("--output-root", type=str, default=str(ACQUISITION_ABLATION_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    output_root = Path(args.output_root)
    is_final = args.seed_start >= ACQUISITION_ABLATION_SEED_OFFSET

    print("=" * 70)
    print("ACQUISITION-BEHAVIOR ABLATION" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/instance)")
    print(f"Policy instances: {len(ABLATION_INSTANCES)}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    print("\nHashing frozen PPO models and verifying vs final nominal manifest...")
    model_hashes = compute_model_hashes_and_verify()
    for h in model_hashes:
        print(f"  {h['track']} seed={h['training_seed']}: sha256={h['sha256'][:16]}... "
              f"matches_final_nominal={h['matches_final_nominal_hash']}")

    manifest = build_manifest(seeds, output_root, model_hashes)
    print(f"\nManifest written. ablation_test_opened={manifest['ablation_test_opened']}")

    print("\nRunning evaluation...")
    df = run_evaluation(seeds, output_root)

    print("\nRunning integrity checks...")
    integrity = run_integrity_checks(df, seeds)
    print(json.dumps(integrity, indent=2, default=str))

    print("\nRunning pre-detection identity check (item 21)...")
    identity_check = run_pre_detection_identity_check(df)
    print(json.dumps(identity_check, indent=2, default=str))

    print("\nComputing summaries...")
    condition_summary = compute_condition_summary(df)
    condition_summary.to_csv(output_root / CONDITION_SUMMARY_FILENAME, index=False)

    training_seed_summary = compute_training_seed_summary(df)
    training_seed_summary.to_csv(output_root / TRAINING_SEED_SUMMARY_FILENAME, index=False)

    paired_comparisons = compute_paired_comparisons(df)
    paired_comparisons.to_csv(output_root / PAIRED_COMPARISONS_FILENAME, index=False)

    final_summary = {
        "label": "ACQUISITION-BEHAVIOR ABLATION" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "pre_detection_identity_check": identity_check,
        "condition_summary": condition_summary.to_dict(orient="records"),
        "training_seed_summary": training_seed_summary.to_dict(orient="records"),
        "paired_comparisons": paired_comparisons.to_dict(orient="records"),
    }
    (output_root / ACQUISITION_ABLATION_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
