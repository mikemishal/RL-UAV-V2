"""
Locked detection-radius robustness sweep runner.

Evaluates the SAME 8 paper-level methods (12 concrete policy instances,
same frozen 400k PPO models as the final nominal experiment) across 6
detection-radius values, on a previously untouched seed range. Detection
radius is the ONLY environment parameter that changes; PPO and the Kalman
filter remain exactly as trained (zero-shot generalization test -- no
retraining, no retuning).

Usage:
    # Full locked sweep (seeds 30000-30999, all 6 radii):
    python experiments/run_robustness_detection_sweep.py

    # Small NON-FINAL dry run (diagnostic seeds < 30000, radius subset):
    python experiments/run_robustness_detection_sweep.py --seed-start 14000 --n-episodes 5 \
        --radii 10.0 15.0 20.0 --output-root results/_dry_run_robustness_detection
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
import time
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

from experiments.robustness_detection_protocol import (
    DETECTION_RADIUS_VALUES,
    NOMINAL_DETECTION_RADIUS,
    DETECTION_SWEEP_SEED_OFFSET,
    DETECTION_SWEEP_EPISODES,
    DETECTION_SWEEP_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    SWEEP_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    SCRIPTED_PRE_DETECTION_IDENTITY_GROUP,
    ROBUSTNESS_DETECTION_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    RADIUS_SUMMARY_FILENAME,
    PPO_TRAINING_SEED_SUMMARY_FILENAME,
    PAIRWISE_COMPARISONS_FILENAME,
    ACQUISITION_SUMMARY_FILENAME,
    ESTIMATOR_SUMMARY_FILENAME,
    DETECTION_RADIUS_SUMMARY_FILENAME,
    build_sweep_env_config,
    assert_only_detection_radius_differs,
)
from experiments.robustness_eval_utils import run_robustness_episode, sha256_of_file
from experiments.final_stats import (
    wilson_interval,
    mcnemar_exact,
    paired_bootstrap_ci,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_paired_bootstrap_matched_seeds,
)

_NOMINAL_ENV_CONFIG = EnvConfig()  # detection_radius=15.0, use_kalman_tracking=False (the training/eval default)
_FINAL_NOMINAL_HASHES_PATH = PROJECT_ROOT / "results" / "final_nominal" / "model_hashes.json"
_FINAL_NOMINAL_METHOD_SUMMARY_PATH = PROJECT_ROOT / "results" / "final_nominal" / "method_summary.csv"

PPO_TRACK_METHODS = ("ppo", "ppo_kalman")


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


def build_env(estimator: str, detection_radius: float) -> SoldierEnv:
    config = build_sweep_env_config(estimator, detection_radius)
    assert_only_detection_radius_differs(config, _NOMINAL_ENV_CONFIG)
    return SoldierEnv(config=config)


def build_policy(instance):
    if instance.model_path is None:
        return get_policy(instance.policy_name)
    wrapper_cls = PPOKalmanPolicyWrapper if instance.estimator == "kalman" else PPOPolicyWrapper
    return wrapper_cls.load(str(instance.model_path), deterministic=True)


def compute_model_hashes_and_verify() -> list[dict]:
    nominal_hashes_by_path = {}
    if _FINAL_NOMINAL_HASHES_PATH.exists():
        for row in json.loads(_FINAL_NOMINAL_HASHES_PATH.read_text()):
            nominal_hashes_by_path[row["model_path"].replace("\\", "/")] = row["sha256"]

    rows = []
    for instance in SWEEP_POLICY_INSTANCES:
        if instance.model_path is None:
            continue
        rel = str(instance.model_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        if not instance.model_path.exists():
            raise FileNotFoundError(f"Frozen model not found: {instance.model_path}")
        digest = sha256_of_file(instance.model_path)
        nominal_digest = nominal_hashes_by_path.get(rel)
        matches = (nominal_digest == digest) if nominal_digest else None
        if matches is False:
            raise RuntimeError(f"Model hash mismatch vs final nominal manifest for {rel}: {digest} != {nominal_digest}")
        rows.append({
            "paper_method": instance.paper_method,
            "policy_instance": instance.policy_instance,
            "training_seed": instance.training_seed,
            "model_path": rel,
            "sha256": digest,
            "matches_final_nominal_hash": matches,
        })
    return rows


def build_manifest(seeds: list[int], radii: list[float], output_root: Path, model_hashes: list[dict]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    opened = len(seeds) > 0 and min(seeds) >= DETECTION_SWEEP_SEED_OFFSET
    env = build_env("measurement", NOMINAL_DETECTION_RADIUS)
    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "nominal_env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "detection_radius_values": list(radii),
        "nominal_detection_radius": NOMINAL_DETECTION_RADIUS,
        "ppo_model_hashes": model_hashes,
        "evaluation_seed_start": min(seeds) if seeds else None,
        "evaluation_seed_end": max(seeds) if seeds else None,
        "n_episodes_per_condition": len(seeds),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "robustness_test_opened": opened,
        "robustness_type": "detection_radius",
        "robustness_first_seed": DETECTION_SWEEP_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], radii: list[float], output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    total = len(radii) * len(SWEEP_POLICY_INSTANCES)
    i = 0
    for radius in radii:
        for instance in SWEEP_POLICY_INSTANCES:
            i += 1
            env = build_env(instance.estimator, radius)
            policy = build_policy(instance)
            if verbose:
                print(f"[{i}/{total}] radius={radius} {instance.policy_instance} "
                      f"(estimator={instance.estimator}) -- {len(seeds)} episodes...")
            for s in seeds:
                row = run_robustness_episode(env, policy, seed=s, dt=dt)
                row["sweep_variable"] = "detection_radius"
                row["detection_radius"] = radius
                row["paper_method"] = instance.paper_method
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


def run_integrity_checks(df: pd.DataFrame, seeds: list[int], radii: list[float]) -> dict:
    results = {}
    expected_rows = EXPECTED_N_POLICY_INSTANCES * len(radii) * len(seeds)
    results["expected_row_count"] = expected_rows
    results["actual_row_count"] = len(df)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    dup = df.duplicated(subset=["detection_radius", "policy_instance", "eval_seed"]).sum()
    results["duplicate_keys"] = int(dup)
    assert dup == 0, f"found {dup} duplicated (detection_radius, policy_instance, eval_seed) keys"

    seed_set_expected = set(seeds)
    combo_seed_sets = df.groupby(["detection_radius", "policy_instance"])["eval_seed"].apply(set)
    common_seeds_ok = all(s == seed_set_expected for s in combo_seed_sets)
    results["common_random_numbers_ok"] = bool(common_seeds_ok)
    assert common_seeds_ok, "not all (radius, policy_instance) combinations share the identical evaluation-seed set"
    results["n_radius_method_combinations"] = int(len(combo_seed_sets))
    assert len(combo_seed_sets) == len(radii) * EXPECTED_N_POLICY_INSTANCES, "missing (radius, policy_instance) combination"

    outcome_sum = df.groupby(["detection_radius", "policy_instance"])[
        ["success", "soldier_caught", "unsafe_intercept", "timeout"]
    ].sum().sum(axis=1)
    outcome_ok = bool((outcome_sum == len(seeds)).all())
    results["outcome_accounting_ok"] = outcome_ok
    assert outcome_ok, "outcome categories do not sum to n_episodes for every (radius, policy_instance)"

    required = ["success", "soldier_caught", "unsafe_intercept", "timeout", "episode_length_steps", "total_reward"]
    nan_inf_bad = {c: int((~np.isfinite(df[c])).sum()) for c in required if (~np.isfinite(df[c])).sum()}
    results["nan_inf_in_required_fields"] = nan_inf_bad
    assert not nan_inf_bad, f"NaN/Inf found in required-always-defined fields: {nan_inf_bad}"

    return results


def run_scripted_pre_detection_identity_check(df: pd.DataFrame, radii: list[float]) -> dict:
    """Item 22: at each fixed radius, all 6 scripted controllers share the same escort law."""
    tol = 1e-4
    results = {}
    for radius in radii:
        sub = df[df["detection_radius"] == radius]
        ref_name = SCRIPTED_PRE_DETECTION_IDENTITY_GROUP[0]
        ref_df = sub[sub["policy_instance"] == ref_name].sort_values("eval_seed").reset_index(drop=True)
        mismatches = 0
        checked = 0
        for name in SCRIPTED_PRE_DETECTION_IDENTITY_GROUP[1:]:
            other_df = sub[sub["policy_instance"] == name].sort_values("eval_seed").reset_index(drop=True)
            assert (ref_df["eval_seed"].to_numpy() == other_df["eval_seed"].to_numpy()).all()
            checked += len(other_df)
            step_mismatch = ~np.isclose(
                ref_df["detection_step"].to_numpy(), other_df["detection_step"].to_numpy(), equal_nan=True, atol=tol
            )
            pos_mismatch = ~np.isclose(
                ref_df[["defender_pos_at_detection_x", "defender_pos_at_detection_y", "defender_pos_at_detection_z"]].to_numpy(),
                other_df[["defender_pos_at_detection_x", "defender_pos_at_detection_y", "defender_pos_at_detection_z"]].to_numpy(),
                equal_nan=True, atol=tol,
            ).any(axis=1)
            mismatches += int((step_mismatch | pos_mismatch).sum())
        results[str(radius)] = {"checked_episode_pairs": checked, "mismatches": mismatches}
    return results


def _success_vector(df: pd.DataFrame, radius: float, policy_instance: str) -> np.ndarray:
    sub = df[(df["detection_radius"] == radius) & (df["policy_instance"] == policy_instance)].sort_values("eval_seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, radius: float, paper_method: str) -> np.ndarray:
    sub = df[(df["detection_radius"] == radius) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s["success"].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _metric_vector(df: pd.DataFrame, radius: float, policy_instance: str, column: str) -> np.ndarray:
    sub = df[(df["detection_radius"] == radius) & (df["policy_instance"] == policy_instance)].sort_values("eval_seed")
    return sub[column].to_numpy(dtype=np.float64)


def _metric_matrix(df: pd.DataFrame, radius: float, paper_method: str, column: str) -> np.ndarray:
    sub = df[(df["detection_radius"] == radius) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s[column].to_numpy(dtype=np.float64))
    return np.stack(rows)


def compute_radius_summary(df: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    from experiments.final_experiment_protocol import PAPER_METHODS, SCRIPTED_METHODS

    rows = []
    for radius in radii:
        for method in PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["detection_radius"] == radius) & (df["policy_instance"] == method)]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                rows.append({
                    "detection_radius": radius, "method": method, "n_eval_episodes": n, "n_training_seeds": 0,
                    "success_rate": successes / n, "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "training_seed_sd": float("nan"),
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                    "detection_rate": float(sub["detected"].mean()),
                })
            else:
                success_matrix = _success_matrix(df, radius, method)
                observed, ci_lo, ci_hi = hierarchical_bootstrap_track(success_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                per_seed_rates = np.nanmean(success_matrix, axis=1)

                def _ew(col):
                    m = _metric_matrix(df, radius, method, col)
                    return float(np.nanmean(np.nanmean(m, axis=1)))

                rows.append({
                    "detection_radius": radius, "method": method,
                    "n_eval_episodes": success_matrix.shape[1], "n_training_seeds": 3,
                    "success_rate": observed, "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "training_seed_sd": float(np.std(per_seed_rates)),
                    "soldier_caught_rate": _ew("soldier_caught"),
                    "unsafe_intercept_rate": _ew("unsafe_intercept"),
                    "timeout_rate": _ew("timeout"),
                    "detection_rate": _ew("detected"),
                })
    return pd.DataFrame(rows)


def compute_ppo_training_seed_summary(df: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    rows = []
    for radius in radii:
        for method in PPO_TRACK_METHODS:
            sub_method = df[(df["detection_radius"] == radius) & (df["paper_method"] == method)]
            for seed in (42, 43, 44):
                sub = sub_method[sub_method["training_seed"] == seed]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                rows.append({
                    "detection_radius": radius, "paper_method": method, "training_seed": seed,
                    "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
                    "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "detection_rate": float(sub["detected"].mean()),
                    "mean_detection_time_seconds": float(sub["detection_time_seconds"].mean()),
                })
    return pd.DataFrame(rows)


def compute_pairwise_comparisons(df: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    rows = []
    for radius in radii:
        for name, method_a, method_b in ALL_COMPARISONS:
            a_is_ppo = method_a in PPO_TRACK_METHODS
            b_is_ppo = method_b in PPO_TRACK_METHODS
            row = {"detection_radius": radius, "comparison_name": name, "method_a": method_a, "method_b": method_b,
                   "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan")}
            if not a_is_ppo and not b_is_ppo:
                vec_a = _success_vector(df, radius, method_a)
                vec_b = _success_vector(df, radius, method_b)
                observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                n10 = int(np.sum((vec_a == 1) & (vec_b == 0)))
                n01 = int(np.sum((vec_a == 0) & (vec_b == 1)))
                p_value = mcnemar_exact(n01=n01, n10=n10)
                row.update({
                    "comparison_type": "fixed_paired",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                    "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
                })
            elif a_is_ppo and b_is_ppo:
                a_matrix = _success_matrix(df, radius, method_a)
                b_matrix = _success_matrix(df, radius, method_b)
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
                ppo_method, scripted_method = (method_a, method_b) if a_is_ppo else (method_b, method_a)
                ppo_matrix = _success_matrix(df, radius, ppo_method)
                scripted_vec = _success_vector(df, radius, scripted_method)
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(
                    ppo_matrix, scripted_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
                )
                if not a_is_ppo:
                    observed, lo, hi = -observed, -hi, -lo
                row.update({
                    "comparison_type": "hierarchical_paired_vs_scripted",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                })
            rows.append(row)
    return pd.DataFrame(rows)


def compute_acquisition_summary(df: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    """Item 29/30: detection probability + conditional detection time for PPO/Lead/PPO-Kalman/Lead-Kalman."""
    rows = []
    for radius in radii:
        entry = {"detection_radius": radius}
        for method, is_scripted in (("lead", True), ("lead_kalman", True), ("ppo", False), ("ppo_kalman", False)):
            if is_scripted:
                sub = df[(df["detection_radius"] == radius) & (df["policy_instance"] == method)]
                detection_rate = float(sub["detected"].mean())
                cond_mean_det = float(sub.loc[sub["detected"] == 1, "detection_time_seconds"].mean())
            else:
                m_detected = _metric_matrix(df, radius, method, "detected")
                m_dettime = _metric_matrix(df, radius, method, "detection_time_seconds")
                detection_rate = float(np.nanmean(np.nanmean(m_detected, axis=1)))
                with np.errstate(invalid="ignore"):
                    per_seed_cond = np.nanmean(m_dettime, axis=1)
                cond_mean_det = float(np.nanmean(per_seed_cond))
            entry[f"{method}_detection_rate"] = detection_rate
            entry[f"{method}_mean_detection_time_seconds_conditional"] = cond_mean_det

        # PPO - Lead detection-time difference (hierarchical paired bootstrap; NaN-safe).
        ppo_dettime = _metric_matrix(df, radius, "ppo", "detection_time_seconds")
        lead_dettime = _metric_vector(df, radius, "lead", "detection_time_seconds")
        obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_dettime, lead_dettime, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        entry["ppo_minus_lead_detection_time_seconds"] = obs
        entry["ppo_minus_lead_detection_time_ci_low"] = lo
        entry["ppo_minus_lead_detection_time_ci_high"] = hi

        ppo_k_dettime = _metric_matrix(df, radius, "ppo_kalman", "detection_time_seconds")
        lead_k_dettime = _metric_vector(df, radius, "lead_kalman", "detection_time_seconds")
        obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_k_dettime, lead_k_dettime, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        entry["ppo_kalman_minus_lead_kalman_detection_time_seconds"] = obs
        entry["ppo_kalman_minus_lead_kalman_detection_time_ci_low"] = lo
        entry["ppo_kalman_minus_lead_kalman_detection_time_ci_high"] = hi

        rows.append(entry)
    return pd.DataFrame(rows)


def compute_estimator_summary(df: pd.DataFrame, radii: list[float]) -> pd.DataFrame:
    from experiments.final_experiment_protocol import PAPER_METHODS, SCRIPTED_METHODS

    rows = []
    for radius in radii:
        for method in PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["detection_radius"] == radius) & (df["policy_instance"] == method)]
                errs = sub["mean_position_estimation_error"].dropna()
                rmse = float(np.sqrt(np.mean(np.square(errs)))) if len(errs) else float("nan")
                rows.append({
                    "detection_radius": radius, "method": method,
                    "mean_position_estimation_error": float(errs.mean()) if len(errs) else float("nan"),
                    "position_estimation_rmse": rmse,
                })
            else:
                m = _metric_matrix(df, radius, method, "mean_position_estimation_error")
                with np.errstate(invalid="ignore"):
                    per_seed_mean = np.nanmean(m, axis=1)
                    per_seed_rmse = np.sqrt(np.nanmean(np.square(m), axis=1))
                rows.append({
                    "detection_radius": radius, "method": method,
                    "mean_position_estimation_error": float(np.nanmean(per_seed_mean)),
                    "position_estimation_rmse": float(np.nanmean(per_seed_rmse)),
                })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked detection-radius robustness sweep")
    parser.add_argument("--seed-start", type=int, default=DETECTION_SWEEP_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=DETECTION_SWEEP_EPISODES)
    parser.add_argument("--radii", type=float, nargs="+", default=list(DETECTION_RADIUS_VALUES))
    parser.add_argument("--output-root", type=str, default=str(ROBUSTNESS_DETECTION_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    radii = args.radii
    output_root = Path(args.output_root)
    is_final = args.seed_start >= DETECTION_SWEEP_SEED_OFFSET

    print("=" * 70)
    print("DETECTION-RADIUS ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/condition)")
    print(f"Radii: {radii}")
    print(f"Policy instances: {len(SWEEP_POLICY_INSTANCES)}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    print("\nHashing frozen PPO models and verifying vs final nominal manifest...")
    model_hashes = compute_model_hashes_and_verify()
    for h in model_hashes:
        print(f"  {h['policy_instance']}: sha256={h['sha256'][:16]}... matches_final_nominal={h['matches_final_nominal_hash']}")

    manifest = build_manifest(seeds, radii, output_root, model_hashes)
    print(f"\nManifest written. robustness_test_opened={manifest['robustness_test_opened']}")

    print("\nRunning evaluation...")
    df = run_evaluation(seeds, radii, output_root)

    print("\nRunning integrity checks...")
    integrity = run_integrity_checks(df, seeds, radii)
    print(json.dumps(integrity, indent=2, default=str))

    print("\nRunning scripted pre-detection identity check (item 22)...")
    identity_check = run_scripted_pre_detection_identity_check(df, radii)
    print(json.dumps(identity_check, indent=2, default=str))

    print("\nComputing summaries...")
    radius_summary = compute_radius_summary(df, radii)
    radius_summary.to_csv(output_root / RADIUS_SUMMARY_FILENAME, index=False)

    ppo_seed_summary = compute_ppo_training_seed_summary(df, radii)
    ppo_seed_summary.to_csv(output_root / PPO_TRAINING_SEED_SUMMARY_FILENAME, index=False)

    pairwise = compute_pairwise_comparisons(df, radii)
    pairwise.to_csv(output_root / PAIRWISE_COMPARISONS_FILENAME, index=False)

    acquisition_summary = compute_acquisition_summary(df, radii)
    acquisition_summary.to_csv(output_root / ACQUISITION_SUMMARY_FILENAME, index=False)

    estimator_summary = compute_estimator_summary(df, radii)
    estimator_summary.to_csv(output_root / ESTIMATOR_SUMMARY_FILENAME, index=False)

    final_summary = {
        "label": "DETECTION-RADIUS ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "scripted_pre_detection_identity_check": identity_check,
        "radius_summary": radius_summary.to_dict(orient="records"),
        "ppo_training_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "pairwise_comparisons": pairwise.to_dict(orient="records"),
        "acquisition_summary": acquisition_summary.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
    }
    (output_root / DETECTION_RADIUS_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
