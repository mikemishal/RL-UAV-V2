"""
Locked hostile-evasion-strength robustness sweep runner.

Evaluates the SIX primary paper methods (10 concrete policy instances) across
5 evasion-gain values, on a previously reserved seed range. Only
enemy_evasion_gain (+ use_kalman_tracking) changes per condition; all
mobility, maneuverability, sensing, and reward parameters are held FIXED at
nominal. enemy_evasion_enabled=True at EVERY gain (including 0.0). PPO/PPO-
Kalman remain exactly as trained/frozen at gain=0.75 -- every other gain is a
zero-shot evasion-strength generalization test.

Usage:
    # Full locked sweep (seeds 33000-33999, all 5 gains):
    python experiments/run_robustness_hostile_evasion_sweep.py

    # Small NON-FINAL dry run (diagnostic seeds < 33000, gain subset):
    python experiments/run_robustness_hostile_evasion_sweep.py --seed-start 19500 --n-episodes 5 \
        --gains 0.0 0.75 1.0 --output-root results/_dry_run_robustness_hostile_evasion
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

from experiments.robustness_hostile_evasion_protocol import (
    EVASION_GAIN_VALUES,
    NOMINAL_EVASION_GAIN,
    EVASION_RADIUS,
    HOSTILE_EVASION_SEED_OFFSET,
    HOSTILE_EVASION_EPISODES,
    HOSTILE_EVASION_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    EVASION_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_ROWS,
    PRIMARY_COMPARISONS,
    ESTIMATOR_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    ROBUSTNESS_HOSTILE_EVASION_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    EVASION_SUMMARY_FILENAME,
    PPO_TRAINING_SEED_SUMMARY_FILENAME,
    PPO_VS_LEAD_COMPARISONS_FILENAME,
    ESTIMATOR_SUCCESS_COMPARISONS_FILENAME,
    WITHIN_METHOD_EVASION_EFFECTS_FILENAME,
    ACQUISITION_EVASION_SUMMARY_FILENAME,
    FAILURE_MODE_SUMMARY_FILENAME,
    OPERATIONAL_SUMMARY_FILENAME,
    ESTIMATOR_SUMMARY_FILENAME,
    HOSTILE_EVASION_SUMMARY_FILENAME,
    build_policy,
    build_evasion_env_config,
    assert_only_evasion_gain_differs,
)
from experiments.hostile_evasion_eval_utils import run_hostile_evasion_episode, sha256_of_file
from experiments.estimator_stats import (
    theoretical_raw_measurement_mean_error,
    theoretical_raw_measurement_rmse,
    sample_weighted_mean_error,
    sample_weighted_rmse,
    equal_weight_across_seeds,
)
from experiments.final_stats import (
    wilson_interval,
    mcnemar_exact,
    paired_bootstrap_ci,
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


def build_env(estimator: str, evasion_gain: float) -> SoldierEnv:
    config = build_evasion_env_config(estimator, evasion_gain)
    assert_only_evasion_gain_differs(config, _NOMINAL_ENV_CONFIG)
    return SoldierEnv(config=config)


def compute_model_hashes_and_verify() -> list[dict]:
    nominal_hashes_by_path = {}
    if _FINAL_NOMINAL_HASHES_PATH.exists():
        for row in json.loads(_FINAL_NOMINAL_HASHES_PATH.read_text()):
            nominal_hashes_by_path[row["model_path"].replace("\\", "/")] = row["sha256"]

    rows = []
    for instance in EVASION_POLICY_INSTANCES:
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
            "paper_method": instance.paper_method, "policy_instance": instance.policy_instance,
            "training_seed": instance.training_seed, "model_path": rel, "sha256": digest,
            "matches_final_nominal_hash": matches,
        })
    return rows


def build_manifest(seeds: list[int], gains, output_root: Path, model_hashes: list[dict]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    opened = len(seeds) > 0 and min(seeds) >= HOSTILE_EVASION_SEED_OFFSET
    env = build_env("measurement", NOMINAL_EVASION_GAIN)
    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "experiment_name": "hostile-evasion-strength robustness sweep",
        "primary_paper_methods": list(PRIMARY_PAPER_METHODS),
        "nominal_env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "evasion_gain_values": list(gains),
        "nominal_evasion_gain": NOMINAL_EVASION_GAIN,
        "evasion_radius_fixed": EVASION_RADIUS,
        "ppo_model_hashes": model_hashes,
        "evaluation_seed_start": min(seeds) if seeds else None,
        "evaluation_seed_end": max(seeds) if seeds else None,
        "n_episodes_per_gain": len(seeds),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "estimator_comparisons": list(ESTIMATOR_COMPARISONS),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "note_only_evasion_gain_swept": (
            "enemy_evasion_gain is the only swept behavioral parameter. All mobility, "
            "acceleration, turn-rate, vertical-rate, sensing, reward, and termination "
            "parameters remain frozen. enemy_evasion_enabled=True at every gain, including 0.0."
        ),
        "note_zero_shot_ppo": (
            "PPO models were trained only at enemy_evasion_gain=0.75. All other gain "
            "conditions are zero-shot evasion-strength generalization evaluations."
        ),
        "note_gain_zero_semantics": (
            "gain=0.0 means ZERO reactive-away evasion CONTRIBUTION, not a non-maneuvering "
            "hostile: the hostile still pursues, weaves, and has stochastic heading noise "
            "under constrained 3-D dynamics. Activation alpha(d) is still computed "
            "geometrically; only effective_weight = gain * alpha(d) is forced to zero."
        ),
        "note_gain_one_semantics": (
            "gain=1.0 is the maximum TESTED reactive-evasion gain under the current "
            "parameterization, not an optimal or adversarially-optimal evasion strategy."
        ),
        "note_evasion_vs_detection_radii": (
            "enemy_evasion_radius=20m exceeds detection_radius=15m, so the hostile can "
            "begin evasive maneuvering before the defender acquires a track on it. This is "
            "intentional and is measured explicitly (acquisition_evasion_summary.csv)."
        ),
        "robustness_test_opened": opened,
        "robustness_type": "hostile_evasion",
        "robustness_first_seed": HOSTILE_EVASION_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], gains, output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    total = len(gains) * len(EVASION_POLICY_INSTANCES)
    i = 0
    for gain in gains:
        for instance in EVASION_POLICY_INSTANCES:
            i += 1
            env = build_env(instance.estimator, gain)
            policy = build_policy(instance)
            if verbose:
                print(f"[{i}/{total}] gain={gain} {instance.policy_instance} "
                      f"(estimator={instance.estimator}) -- {len(seeds)} episodes...")
            for s in seeds:
                row = run_hostile_evasion_episode(env, policy, seed=s, dt=dt, evasion_gain=gain)
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


def run_integrity_checks(df: pd.DataFrame, seeds: list[int], gains) -> dict:
    results = {}
    expected_rows = len(gains) * EXPECTED_N_POLICY_INSTANCES * len(seeds)
    results["expected_row_count"] = expected_rows
    results["actual_row_count"] = len(df)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    dup = df.duplicated(subset=["enemy_evasion_gain", "policy_instance", "eval_seed"]).sum()
    results["duplicate_keys"] = int(dup)
    assert dup == 0, f"found {dup} duplicated (gain, policy_instance, eval_seed) keys"

    seed_set_expected = set(seeds)
    combo_seed_sets = df.groupby(["enemy_evasion_gain", "policy_instance"])["eval_seed"].apply(set)
    common_seeds_ok = all(s == seed_set_expected for s in combo_seed_sets)
    results["common_random_numbers_ok"] = bool(common_seeds_ok)
    assert common_seeds_ok, "not all (gain, policy_instance) combinations share the identical evaluation-seed set"
    results["n_gain_method_combinations"] = int(len(combo_seed_sets))
    assert len(combo_seed_sets) == len(gains) * EXPECTED_N_POLICY_INSTANCES, "missing (gain, policy_instance) combination"

    unique_gains_found = df["enemy_evasion_gain"].drop_duplicates()
    results["n_unique_gains_found"] = int(len(unique_gains_found))
    assert len(unique_gains_found) == len(gains), "missing or extra unique gain value"

    outcome_sum = df.groupby(["enemy_evasion_gain", "policy_instance"])[
        ["success", "soldier_caught", "unsafe_intercept", "timeout"]
    ].sum().sum(axis=1)
    outcome_ok = bool((outcome_sum == len(seeds)).all())
    results["outcome_accounting_ok"] = outcome_ok
    assert outcome_ok, "outcome categories do not sum to n_episodes for every (gain, policy_instance)"

    required = ["success", "soldier_caught", "unsafe_intercept", "timeout", "episode_length_steps", "total_reward"]
    nan_inf_bad = {c: int((~np.isfinite(df[c])).sum()) for c in required if (~np.isfinite(df[c])).sum()}
    results["nan_inf_in_required_fields"] = nan_inf_bad
    assert not nan_inf_bad, f"NaN/Inf found in required-always-defined fields: {nan_inf_bad}"

    return results


def _success_vector(df: pd.DataFrame, gain: float, policy_instance: str) -> np.ndarray:
    sub = df[(df["enemy_evasion_gain"] == gain) & (df["policy_instance"] == policy_instance)].sort_values("eval_seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, gain: float, paper_method: str) -> np.ndarray:
    sub = df[(df["enemy_evasion_gain"] == gain) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s["success"].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _metric_matrix(df: pd.DataFrame, gain: float, paper_method: str, column: str) -> np.ndarray:
    sub = df[(df["enemy_evasion_gain"] == gain) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s[column].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _gain_method_success_row(df: pd.DataFrame, gain: float, method: str) -> dict:
    if method in SCRIPTED_METHODS:
        sub = df[(df["enemy_evasion_gain"] == gain) & (df["policy_instance"] == method)]
        n = len(sub)
        successes = int(sub["success"].sum())
        ci_lo, ci_hi = wilson_interval(successes, n)
        return {
            "n_eval_episodes": n, "success_rate": successes / n,
            "success_ci_low": ci_lo, "success_ci_high": ci_hi,
            "training_seed_sd": float("nan"), "training_seed_min": float("nan"), "training_seed_max": float("nan"),
        }
    success_matrix = _success_matrix(df, gain, method)
    observed, ci_lo, ci_hi = hierarchical_bootstrap_track(success_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    per_seed_rates = np.nanmean(success_matrix, axis=1)
    return {
        "n_eval_episodes": success_matrix.shape[1], "success_rate": observed,
        "success_ci_low": ci_lo, "success_ci_high": ci_hi,
        "training_seed_sd": float(np.std(per_seed_rates)),
        "training_seed_min": float(np.min(per_seed_rates)), "training_seed_max": float(np.max(per_seed_rates)),
    }


def compute_evasion_summary(df: pd.DataFrame, gains) -> pd.DataFrame:
    rows = []
    for gain in gains:
        for method in PRIMARY_PAPER_METHODS:
            row = {"enemy_evasion_gain": gain, "method": method}
            row.update(_gain_method_success_row(df, gain, method))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_ppo_training_seed_summary(df: pd.DataFrame, gains) -> pd.DataFrame:
    rows = []
    for gain in gains:
        for method in PPO_METHODS:
            sub_method = df[(df["enemy_evasion_gain"] == gain) & (df["paper_method"] == method)]
            for seed in (42, 43, 44):
                sub = sub_method[sub_method["training_seed"] == seed]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                rows.append({
                    "enemy_evasion_gain": gain, "paper_method": method, "training_seed": seed,
                    "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
                    "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "mean_episode_length_seconds": float(sub["episode_length_seconds"].mean()),
                    "mean_detection_time_seconds": float(sub["detection_time_seconds"].mean()),
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def _paired_comparison_row(df: pd.DataFrame, gain: float, name: str, method_a: str, method_b: str) -> dict:
    a_is_ppo = method_a in PPO_METHODS
    b_is_ppo = method_b in PPO_METHODS
    row = {"enemy_evasion_gain": gain, "comparison_name": name, "method_a": method_a, "method_b": method_b,
           "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan"),
           "n10_a_wins": float("nan"), "n01_a_loses": float("nan"), "mcnemar_p_value": float("nan")}
    if not a_is_ppo and not b_is_ppo:
        vec_a = _success_vector(df, gain, method_a)
        vec_b = _success_vector(df, gain, method_b)
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
        a_matrix = _success_matrix(df, gain, method_a)
        b_matrix = _success_matrix(df, gain, method_b)
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
        ppo_matrix = _success_matrix(df, gain, ppo_method)
        scripted_vec = _success_vector(df, gain, scripted_method)
        observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(
            ppo_matrix, scripted_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
        )
        if not a_is_ppo:
            observed, lo, hi = -observed, -hi, -lo
        row.update({
            "comparison_type": "hierarchical_paired_vs_scripted",
            "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
        })
    return row


def _classify_ci(row: dict) -> str:
    if row["ci_low_pp"] > 0:
        return "PPO higher, CI excludes 0"
    if row["ci_high_pp"] < 0:
        return "Lead higher, CI excludes 0"
    return "CI includes 0"


def compute_ppo_vs_lead_comparisons(df: pd.DataFrame, gains) -> pd.DataFrame:
    """Primary + secondary comparisons (item 16 + item 34's secondary set)."""
    rows = []
    for gain in gains:
        for name, method_a, method_b in (PRIMARY_COMPARISONS + SECONDARY_COMPARISONS):
            row = _paired_comparison_row(df, gain, name, method_a, method_b)
            if name == "ppo_vs_lead":
                row["classification"] = _classify_ci(row)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_estimator_success_comparisons(df: pd.DataFrame, gains) -> pd.DataFrame:
    """Kalman-Greedy vs Greedy, and PPO-Kalman vs PPO (items 17/18)."""
    rows = []
    for gain in gains:
        for name, method_a, method_b in ESTIMATOR_COMPARISONS:
            rows.append(_paired_comparison_row(df, gain, name, method_a, method_b))
    return pd.DataFrame(rows)


def compute_within_method_evasion_effects(df: pd.DataFrame, gains) -> pd.DataFrame:
    """Item 19/20: paired changes relative to gain=0.0, plus gain=1.0 - gain=0.75."""
    rows = []
    gain0 = 0.0
    assert gain0 in gains
    comparison_gains = [g for g in gains if g != gain0]
    for method in PRIMARY_PAPER_METHODS:
        for target_gain in comparison_gains:
            name = f"gain_{target_gain}_minus_gain_0"
            row = _paired_comparison_row(df, target_gain, name, method, method)
            # Reinterpret: compare method at target_gain vs method at gain0 (both same method,
            # different evasion-gain conditions -- pair on identical evaluation seeds).
            if method in SCRIPTED_METHODS:
                vec_a = _success_vector(df, target_gain, method)
                vec_b = _success_vector(df, gain0, method)
                observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                n10 = int(np.sum((vec_a == 1) & (vec_b == 0)))
                n01 = int(np.sum((vec_a == 0) & (vec_b == 1)))
                p_value = mcnemar_exact(n01=n01, n10=n10)
                rows.append({
                    "method": method, "comparison": name, "gain_a": target_gain, "gain_b": gain0,
                    "comparison_type": "fixed_paired",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                    "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
                })
            else:
                a_matrix = _success_matrix(df, target_gain, method)
                b_matrix = _success_matrix(df, gain0, method)
                observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
                    a_matrix, b_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
                )
                rows.append({
                    "method": method, "comparison": name, "gain_a": target_gain, "gain_b": gain0,
                    "comparison_type": "hierarchical_paired_matched_seeds",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                    "diff_seed_42_pp": per_seed_diffs[0] * 100,
                    "diff_seed_43_pp": per_seed_diffs[1] * 100,
                    "diff_seed_44_pp": per_seed_diffs[2] * 100,
                })
        # gain 1.0 - gain 0.75 (item 20)
        if 1.0 in gains and 0.75 in gains:
            name = "gain_1.0_minus_gain_0.75"
            if method in SCRIPTED_METHODS:
                vec_a = _success_vector(df, 1.0, method)
                vec_b = _success_vector(df, 0.75, method)
                observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
                n10 = int(np.sum((vec_a == 1) & (vec_b == 0)))
                n01 = int(np.sum((vec_a == 0) & (vec_b == 1)))
                p_value = mcnemar_exact(n01=n01, n10=n10)
                rows.append({
                    "method": method, "comparison": name, "gain_a": 1.0, "gain_b": 0.75,
                    "comparison_type": "fixed_paired",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                    "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
                })
            else:
                a_matrix = _success_matrix(df, 1.0, method)
                b_matrix = _success_matrix(df, 0.75, method)
                observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
                    a_matrix, b_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
                )
                rows.append({
                    "method": method, "comparison": name, "gain_a": 1.0, "gain_b": 0.75,
                    "comparison_type": "hierarchical_paired_matched_seeds",
                    "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                    "diff_seed_42_pp": per_seed_diffs[0] * 100,
                    "diff_seed_43_pp": per_seed_diffs[1] * 100,
                    "diff_seed_44_pp": per_seed_diffs[2] * 100,
                })
    return pd.DataFrame(rows)


def compute_acquisition_evasion_summary(df: pd.DataFrame, gains) -> pd.DataFrame:
    rows = []
    for gain in gains:
        for method in PRIMARY_PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["enemy_evasion_gain"] == gain) & (df["policy_instance"] == method)]
            else:
                sub = df[(df["enemy_evasion_gain"] == gain) & (df["paper_method"] == method)]
            pre_detect_evasion = sub["evasion_active_before_detection"].mean()
            both_occur = sub.dropna(subset=["time_evasion_onset_to_detection_seconds"])
            rows.append({
                "enemy_evasion_gain": gain, "method": method,
                "detection_rate": float(sub["detected"].mean()),
                "mean_detection_time_seconds": float(sub["detection_time_seconds"].mean()),
                "fraction_evasion_before_detection": float(pre_detect_evasion),
                "mean_time_evasion_onset_to_detection_seconds": (
                    float(both_occur["time_evasion_onset_to_detection_seconds"].mean()) if len(both_occur) else float("nan")
                ),
                "n_episodes_both_evasion_and_detection": int(len(both_occur)),
            })
    return pd.DataFrame(rows)


def compute_failure_mode_summary(df: pd.DataFrame, gains) -> pd.DataFrame:
    rows = []
    for gain in gains:
        for method in PRIMARY_PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["enemy_evasion_gain"] == gain) & (df["policy_instance"] == method)]
            else:
                sub = df[(df["enemy_evasion_gain"] == gain) & (df["paper_method"] == method)]
            rows.append({
                "enemy_evasion_gain": gain, "method": method,
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_operational_summary(df: pd.DataFrame, gains) -> pd.DataFrame:
    op_cols = [
        "defender_path_length", "mean_defender_speed", "max_defender_speed",
        "mean_defender_accel", "max_defender_accel", "mean_defender_turn_rate", "max_defender_turn_rate",
        "fraction_defender_accel_saturated", "fraction_defender_turn_saturated", "fraction_defender_climb_saturated",
        "mean_enemy_speed", "max_enemy_speed", "mean_enemy_horizontal_speed", "mean_abs_enemy_vertical_speed",
        "fraction_enemy_accel_saturated", "fraction_enemy_turn_saturated", "fraction_enemy_climb_or_vertical_saturated",
        "enemy_path_length", "ground_boundary_hits", "ceiling_hits", "horizontal_wall_hits",
        "evasion_activated_episode", "fraction_steps_evasion_active", "mean_evasion_activation",
        "max_evasion_activation", "mean_effective_evasion_weight", "max_effective_evasion_weight",
        "episode_length_seconds", "intercept_time_seconds", "min_defender_enemy_dist", "min_enemy_soldier_dist",
    ]
    rows = []
    for gain in gains:
        for method in PRIMARY_PAPER_METHODS:
            row = {"enemy_evasion_gain": gain, "method": method}
            if method in SCRIPTED_METHODS:
                sub = df[(df["enemy_evasion_gain"] == gain) & (df["policy_instance"] == method)]
                for col in op_cols:
                    row[col] = float(sub[col].mean())
            else:
                for col in op_cols:
                    m = _metric_matrix(df, gain, method, col)
                    with np.errstate(invalid="ignore"):
                        row[col] = float(np.nanmean(np.nanmean(m, axis=1)))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_estimator_summary(df: pd.DataFrame, gains) -> pd.DataFrame:
    """CORRECTED aggregation (reuses experiments/estimator_stats.py directly). Also
    includes success_rate alongside RMSE for the 'success vs tracking' compact table (item 32)."""
    rows = []
    theoretical_mean = theoretical_raw_measurement_mean_error(_NOMINAL_ENV_CONFIG.measurement_var)
    theoretical_rmse = theoretical_raw_measurement_rmse(_NOMINAL_ENV_CONFIG.measurement_var)
    for gain in gains:
        for method in SCRIPTED_METHODS:
            sub = df[(df["enemy_evasion_gain"] == gain) & (df["policy_instance"] == method)].dropna(
                subset=["mean_position_estimation_error", "position_estimation_rmse"]
            )
            n = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
            m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
            r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
            is_direct = method in ("greedy", "pn", "lead")
            all_sub = df[(df["enemy_evasion_gain"] == gain) & (df["policy_instance"] == method)]
            rows.append({
                "enemy_evasion_gain": gain, "method": method,
                "success_rate": float(all_sub["success"].mean()),
                "n_episodes_with_estimates": int(len(sub)), "n_total_estimation_samples": int(n.sum()),
                "mean_error_sample_weighted": sample_weighted_mean_error(n, m),
                "rmse_sample_weighted": sample_weighted_rmse(n, r),
                "theoretical_raw_mean_error": theoretical_mean if is_direct else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if is_direct else float("nan"),
            })
        for method in PPO_METHODS:
            sub_method = df[(df["enemy_evasion_gain"] == gain) & (df["paper_method"] == method)]
            seed_rmse, seed_mean, seed_success = [], [], []
            n_total = 0
            for seed in (42, 43, 44):
                seed_sub = sub_method[sub_method["training_seed"] == seed]
                seed_sub_valid = seed_sub.dropna(subset=["mean_position_estimation_error", "position_estimation_rmse"])
                n = seed_sub_valid["n_position_estimation_samples"].to_numpy(dtype=np.float64)
                m = seed_sub_valid["mean_position_estimation_error"].to_numpy(dtype=np.float64)
                r = seed_sub_valid["position_estimation_rmse"].to_numpy(dtype=np.float64)
                n_total += int(n.sum())
                seed_mean.append(sample_weighted_mean_error(n, m))
                seed_rmse.append(sample_weighted_rmse(n, r))
                seed_success.append(float(seed_sub["success"].mean()))
            mean_error, _ = equal_weight_across_seeds(np.array(seed_mean))
            rmse, rmse_sd = equal_weight_across_seeds(np.array(seed_rmse))
            rows.append({
                "enemy_evasion_gain": gain, "method": method,
                "success_rate": float(np.mean(seed_success)),
                "n_episodes_with_estimates": float("nan"), "n_total_estimation_samples": n_total,
                "mean_error_sample_weighted": mean_error,
                "rmse_sample_weighted": rmse, "train_seed_rmse_sd": rmse_sd,
                "theoretical_raw_mean_error": theoretical_mean if method == "ppo" else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if method == "ppo" else float("nan"),
            })
    return pd.DataFrame(rows)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked hostile-evasion-strength robustness sweep")
    parser.add_argument("--seed-start", type=int, default=HOSTILE_EVASION_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=HOSTILE_EVASION_EPISODES)
    parser.add_argument("--gains", type=float, nargs="+", default=None,
                         help="Subset of evasion gains (dry-run only); default = all 5 gains.")
    parser.add_argument("--output-root", type=str, default=str(ROBUSTNESS_HOSTILE_EVASION_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    gains = args.gains if args.gains else list(EVASION_GAIN_VALUES)
    output_root = Path(args.output_root)
    is_final = args.seed_start >= HOSTILE_EVASION_SEED_OFFSET

    print("=" * 70)
    print("HOSTILE-EVASION-STRENGTH ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/gain)")
    print(f"Gains ({len(gains)}): {gains}")
    print(f"Policy instances: {len(EVASION_POLICY_INSTANCES)}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    print("\nHashing frozen PPO models and verifying vs final nominal manifest...")
    model_hashes = compute_model_hashes_and_verify()
    for h in model_hashes:
        print(f"  {h['policy_instance']}: sha256={h['sha256'][:16]}... matches_final_nominal={h['matches_final_nominal_hash']}")

    manifest = build_manifest(seeds, gains, output_root, model_hashes)
    print(f"\nManifest written. robustness_test_opened={manifest['robustness_test_opened']}")

    print("\nRunning evaluation...")
    df = run_evaluation(seeds, gains, output_root)

    print("\nRunning integrity checks...")
    integrity = run_integrity_checks(df, seeds, gains)
    print(json.dumps(integrity, indent=2, default=str))

    print("\nComputing summaries...")
    evasion_summary = compute_evasion_summary(df, gains)
    evasion_summary.to_csv(output_root / EVASION_SUMMARY_FILENAME, index=False)

    ppo_seed_summary = compute_ppo_training_seed_summary(df, gains)
    ppo_seed_summary.to_csv(output_root / PPO_TRAINING_SEED_SUMMARY_FILENAME, index=False)

    ppo_vs_lead = compute_ppo_vs_lead_comparisons(df, gains)
    ppo_vs_lead.to_csv(output_root / PPO_VS_LEAD_COMPARISONS_FILENAME, index=False)

    estimator_success_comparisons = compute_estimator_success_comparisons(df, gains)
    estimator_success_comparisons.to_csv(output_root / ESTIMATOR_SUCCESS_COMPARISONS_FILENAME, index=False)

    has_gain0 = 0.0 in gains
    if has_gain0:
        within_method_effects = compute_within_method_evasion_effects(df, gains)
        within_method_effects.to_csv(output_root / WITHIN_METHOD_EVASION_EFFECTS_FILENAME, index=False)
    else:
        within_method_effects = pd.DataFrame()
        print("\n(Skipping within-method evasion effects -- requires gain=0.0 in this run)")

    acquisition_evasion = compute_acquisition_evasion_summary(df, gains)
    acquisition_evasion.to_csv(output_root / ACQUISITION_EVASION_SUMMARY_FILENAME, index=False)

    failure_modes = compute_failure_mode_summary(df, gains)
    failure_modes.to_csv(output_root / FAILURE_MODE_SUMMARY_FILENAME, index=False)

    operational_summary = compute_operational_summary(df, gains)
    operational_summary.to_csv(output_root / OPERATIONAL_SUMMARY_FILENAME, index=False)

    estimator_summary = compute_estimator_summary(df, gains)
    estimator_summary.to_csv(output_root / ESTIMATOR_SUMMARY_FILENAME, index=False)

    final_summary = {
        "label": "HOSTILE-EVASION-STRENGTH ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "evasion_summary": evasion_summary.to_dict(orient="records"),
        "ppo_training_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "ppo_vs_lead_comparisons": ppo_vs_lead.to_dict(orient="records"),
        "estimator_success_comparisons": estimator_success_comparisons.to_dict(orient="records"),
        "within_method_evasion_effects": within_method_effects.to_dict(orient="records"),
        "acquisition_evasion_summary": acquisition_evasion.to_dict(orient="records"),
        "failure_mode_summary": failure_modes.to_dict(orient="records"),
        "operational_summary": operational_summary.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
    }
    (output_root / HOSTILE_EVASION_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
