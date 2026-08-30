"""
Locked mobility robustness sweep runner.

Evaluates the SIX primary paper methods (10 concrete policy instances: 4
scripted + 3 PPO + 3 PPO-Kalman, same frozen 400k PPO models as the final
nominal experiment) across 9 UNIQUE (v_d, v_e) mobility configurations (two
matched 1-D arms sharing one nominal point -- NOT a 25-cell 2-D grid), on a
previously untouched seed range. Only v_d (defender-speed arm) or v_e
(hostile-speed arm) changes per condition; all maneuverability, sensing,
evasion, and reward parameters are held FIXED at their nominal values.
PPO/PPO-Kalman remain exactly as trained/frozen at (v_d=18, v_e=12) --
zero-shot mobility generalization test at every other condition.

Usage:
    # Full locked sweep (seeds 32000-32999, all 9 conditions):
    python experiments/run_robustness_mobility_sweep.py

    # Small NON-FINAL dry run (diagnostic seeds < 32000, condition subset):
    python experiments/run_robustness_mobility_sweep.py --seed-start 17000 --n-episodes 5 \
        --conditions 15,12 18,12 18,14 --output-root results/_dry_run_robustness_mobility
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

from experiments.robustness_mobility_protocol import (
    DEFENDER_SPEED_VALUES,
    HOSTILE_SPEED_VALUES,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
    MOBILITY_CONDITIONS,
    defender_speed_conditions,
    hostile_speed_conditions,
    MOBILITY_SWEEP_SEED_OFFSET,
    MOBILITY_SWEEP_EPISODES,
    MOBILITY_SWEEP_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    MOBILITY_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_N_MOBILITY_CONDITIONS,
    PRIMARY_COMPARISONS,
    ESTIMATOR_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    ROBUSTNESS_MOBILITY_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    DEFENDER_SPEED_SUMMARY_FILENAME,
    HOSTILE_SPEED_SUMMARY_FILENAME,
    MOBILITY_RATIO_SUMMARY_FILENAME,
    PPO_TRAINING_SEED_SUMMARY_FILENAME,
    PAIRWISE_COMPARISONS_FILENAME,
    OPERATIONAL_SUMMARY_FILENAME,
    ESTIMATOR_SUMMARY_FILENAME,
    MOBILITY_SUMMARY_FILENAME,
    build_policy,
    build_mobility_env_config,
    assert_only_mobility_speed_differs,
)
from experiments.mobility_eval_utils import run_mobility_episode, sha256_of_file
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


def build_env(estimator: str, defender_speed: float, hostile_speed: float) -> SoldierEnv:
    config = build_mobility_env_config(estimator, defender_speed, hostile_speed)
    assert_only_mobility_speed_differs(config, _NOMINAL_ENV_CONFIG)
    return SoldierEnv(config=config)


def compute_model_hashes_and_verify() -> list[dict]:
    nominal_hashes_by_path = {}
    if _FINAL_NOMINAL_HASHES_PATH.exists():
        for row in json.loads(_FINAL_NOMINAL_HASHES_PATH.read_text()):
            nominal_hashes_by_path[row["model_path"].replace("\\", "/")] = row["sha256"]

    rows = []
    for instance in MOBILITY_POLICY_INSTANCES:
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


def build_manifest(seeds: list[int], conditions, output_root: Path, model_hashes: list[dict]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    opened = len(seeds) > 0 and min(seeds) >= MOBILITY_SWEEP_SEED_OFFSET
    env = build_env("measurement", NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED)
    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "primary_paper_methods": list(PRIMARY_PAPER_METHODS),
        "nominal_env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "defender_speed_values": list(DEFENDER_SPEED_VALUES),
        "hostile_speed_values": list(HOSTILE_SPEED_VALUES),
        "nominal_defender_speed": NOMINAL_DEFENDER_SPEED,
        "nominal_hostile_speed": NOMINAL_HOSTILE_SPEED,
        "unique_conditions": [dataclasses.asdict(c) for c in conditions],
        "ppo_model_hashes": model_hashes,
        "evaluation_seed_start": min(seeds) if seeds else None,
        "evaluation_seed_end": max(seeds) if seeds else None,
        "n_episodes_per_condition": len(seeds),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "estimator_comparisons": list(ESTIMATOR_COMPARISONS),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "note_only_mobility_speed_swept": (
            "Only v_d (defender-speed arm) or v_e (hostile-speed arm) is swept per condition; "
            "all maneuverability, sensing, evasion, and reward parameters are frozen at nominal."
        ),
        "note_no_2d_grid": (
            "Two matched 1-D arms sharing one nominal point (9 unique configurations), NOT a "
            "25-cell 2-D Cartesian grid."
        ),
        "note_zero_shot_ppo": (
            "Frozen PPO policies were trained only at (v_d=18, v_e=12); all other mobility "
            "conditions are zero-shot generalization tests."
        ),
        "robustness_test_opened": opened,
        "robustness_type": "mobility",
        "robustness_first_seed": MOBILITY_SWEEP_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], conditions, output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    total = len(conditions) * len(MOBILITY_POLICY_INSTANCES)
    i = 0
    for condition in conditions:
        for instance in MOBILITY_POLICY_INSTANCES:
            i += 1
            env = build_env(instance.estimator, condition.defender_speed, condition.hostile_speed)
            policy = build_policy(instance)
            if verbose:
                print(f"[{i}/{total}] (v_d={condition.defender_speed}, v_e={condition.hostile_speed}, "
                      f"axis={condition.sweep_axis}) {instance.policy_instance} "
                      f"(estimator={instance.estimator}) -- {len(seeds)} episodes...")
            for s in seeds:
                row = run_mobility_episode(
                    env, policy, seed=s, dt=dt,
                    defender_speed=condition.defender_speed, hostile_speed=condition.hostile_speed,
                    sweep_axis=condition.sweep_axis,
                )
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


def run_integrity_checks(df: pd.DataFrame, seeds: list[int], conditions) -> dict:
    results = {}
    expected_rows = len(conditions) * EXPECTED_N_POLICY_INSTANCES * len(seeds)
    results["expected_row_count"] = expected_rows
    results["actual_row_count"] = len(df)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    dup = df.duplicated(subset=["defender_speed", "hostile_speed", "policy_instance", "eval_seed"]).sum()
    results["duplicate_keys"] = int(dup)
    assert dup == 0, f"found {dup} duplicated (v_d, v_e, policy_instance, eval_seed) keys"

    seed_set_expected = set(seeds)
    combo_seed_sets = df.groupby(["defender_speed", "hostile_speed", "policy_instance"])["eval_seed"].apply(set)
    common_seeds_ok = all(s == seed_set_expected for s in combo_seed_sets)
    results["common_random_numbers_ok"] = bool(common_seeds_ok)
    assert common_seeds_ok, "not all (v_d, v_e, policy_instance) combinations share the identical evaluation-seed set"
    results["n_condition_method_combinations"] = int(len(combo_seed_sets))
    assert len(combo_seed_sets) == len(conditions) * EXPECTED_N_POLICY_INSTANCES, "missing (v_d, v_e, policy_instance) combination"

    unique_conditions_found = df[["defender_speed", "hostile_speed"]].drop_duplicates()
    results["n_unique_conditions_found"] = int(len(unique_conditions_found))
    assert len(unique_conditions_found) == len(conditions), "missing or extra unique (v_d, v_e) condition"
    if len(conditions) == EXPECTED_N_MOBILITY_CONDITIONS:
        assert len(unique_conditions_found) == EXPECTED_N_MOBILITY_CONDITIONS

    outcome_sum = df.groupby(["defender_speed", "hostile_speed", "policy_instance"])[
        ["success", "soldier_caught", "unsafe_intercept", "timeout"]
    ].sum().sum(axis=1)
    outcome_ok = bool((outcome_sum == len(seeds)).all())
    results["outcome_accounting_ok"] = outcome_ok
    assert outcome_ok, "outcome categories do not sum to n_episodes for every (v_d, v_e, policy_instance)"

    required = ["success", "soldier_caught", "unsafe_intercept", "timeout", "episode_length_steps", "total_reward"]
    nan_inf_bad = {c: int((~np.isfinite(df[c])).sum()) for c in required if (~np.isfinite(df[c])).sum()}
    results["nan_inf_in_required_fields"] = nan_inf_bad
    assert not nan_inf_bad, f"NaN/Inf found in required-always-defined fields: {nan_inf_bad}"

    return results


def _success_vector(df: pd.DataFrame, v_d: float, v_e: float, policy_instance: str) -> np.ndarray:
    sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["policy_instance"] == policy_instance)].sort_values("eval_seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, v_d: float, v_e: float, paper_method: str) -> np.ndarray:
    sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s["success"].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _metric_matrix(df: pd.DataFrame, v_d: float, v_e: float, paper_method: str, column: str) -> np.ndarray:
    sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["paper_method"] == paper_method)]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s[column].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _condition_method_success_row(df: pd.DataFrame, v_d: float, v_e: float, method: str) -> dict:
    if method in SCRIPTED_METHODS:
        sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["policy_instance"] == method)]
        n = len(sub)
        successes = int(sub["success"].sum())
        ci_lo, ci_hi = wilson_interval(successes, n)
        return {
            "n_eval_episodes": n, "success_rate": successes / n,
            "success_ci_low": ci_lo, "success_ci_high": ci_hi,
            "training_seed_sd": float("nan"), "training_seed_min": float("nan"), "training_seed_max": float("nan"),
        }
    success_matrix = _success_matrix(df, v_d, v_e, method)
    observed, ci_lo, ci_hi = hierarchical_bootstrap_track(success_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    per_seed_rates = np.nanmean(success_matrix, axis=1)
    return {
        "n_eval_episodes": success_matrix.shape[1], "success_rate": observed,
        "success_ci_low": ci_lo, "success_ci_high": ci_hi,
        "training_seed_sd": float(np.std(per_seed_rates)),
        "training_seed_min": float(np.min(per_seed_rates)), "training_seed_max": float(np.max(per_seed_rates)),
    }


def compute_defender_speed_summary(df: pd.DataFrame, conditions=None) -> pd.DataFrame:
    wanted = {(c.defender_speed, c.hostile_speed) for c in (conditions or MOBILITY_CONDITIONS)}
    rows = []
    for condition in defender_speed_conditions():
        if (condition.defender_speed, condition.hostile_speed) not in wanted:
            continue
        for method in PRIMARY_PAPER_METHODS:
            row = {"defender_speed": condition.defender_speed, "hostile_speed": condition.hostile_speed,
                   "mobility_ratio": condition.mobility_ratio, "sweep_axis": condition.sweep_axis, "method": method}
            row.update(_condition_method_success_row(df, condition.defender_speed, condition.hostile_speed, method))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_hostile_speed_summary(df: pd.DataFrame, conditions=None) -> pd.DataFrame:
    wanted = {(c.defender_speed, c.hostile_speed) for c in (conditions or MOBILITY_CONDITIONS)}
    rows = []
    for condition in hostile_speed_conditions():
        if (condition.defender_speed, condition.hostile_speed) not in wanted:
            continue
        for method in PRIMARY_PAPER_METHODS:
            row = {"defender_speed": condition.defender_speed, "hostile_speed": condition.hostile_speed,
                   "mobility_ratio": condition.mobility_ratio, "sweep_axis": condition.sweep_axis, "method": method}
            row.update(_condition_method_success_row(df, condition.defender_speed, condition.hostile_speed, method))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_mobility_ratio_summary(df: pd.DataFrame, conditions=None) -> pd.DataFrame:
    rows = []
    for condition in (conditions or MOBILITY_CONDITIONS):
        for method in PRIMARY_PAPER_METHODS:
            row = {"mobility_ratio": condition.mobility_ratio, "defender_speed": condition.defender_speed,
                   "hostile_speed": condition.hostile_speed, "sweep_axis": condition.sweep_axis, "method": method}
            row.update(_condition_method_success_row(df, condition.defender_speed, condition.hostile_speed, method))
            rows.append(row)
    result = pd.DataFrame(rows)
    return result.sort_values(["mobility_ratio", "method"]).reset_index(drop=True)


def compute_ppo_training_seed_summary(df: pd.DataFrame, conditions=None) -> pd.DataFrame:
    rows = []
    for condition in (conditions or MOBILITY_CONDITIONS):
        for method in PPO_METHODS:
            sub_method = df[(df["defender_speed"] == condition.defender_speed) & (df["hostile_speed"] == condition.hostile_speed) & (df["paper_method"] == method)]
            for seed in (42, 43, 44):
                sub = sub_method[sub_method["training_seed"] == seed]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                n_samples = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
                m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
                r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
                valid = ~(np.isnan(m) | np.isnan(r))
                rows.append({
                    "defender_speed": condition.defender_speed, "hostile_speed": condition.hostile_speed,
                    "mobility_ratio": condition.mobility_ratio, "sweep_axis": condition.sweep_axis,
                    "paper_method": method, "training_seed": seed,
                    "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
                    "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "mean_error_sample_weighted": sample_weighted_mean_error(n_samples[valid], m[valid]),
                    "rmse_sample_weighted": sample_weighted_rmse(n_samples[valid], r[valid]),
                    "mean_episode_length_seconds": float(sub["episode_length_seconds"].mean()),
                })
    return pd.DataFrame(rows)


def compute_pairwise_comparisons(df: pd.DataFrame, conditions=None) -> pd.DataFrame:
    rows = []
    for condition in (conditions or MOBILITY_CONDITIONS):
        v_d, v_e = condition.defender_speed, condition.hostile_speed
        for name, method_a, method_b in ALL_COMPARISONS:
            a_is_ppo = method_a in PPO_METHODS
            b_is_ppo = method_b in PPO_METHODS
            row = {"defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": condition.mobility_ratio,
                   "sweep_axis": condition.sweep_axis, "comparison_name": name, "method_a": method_a, "method_b": method_b,
                   "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan"),
                   "n10_a_wins": float("nan"), "n01_a_loses": float("nan"), "mcnemar_p_value": float("nan")}
            if not a_is_ppo and not b_is_ppo:
                vec_a = _success_vector(df, v_d, v_e, method_a)
                vec_b = _success_vector(df, v_d, v_e, method_b)
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
                a_matrix = _success_matrix(df, v_d, v_e, method_a)
                b_matrix = _success_matrix(df, v_d, v_e, method_b)
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
                ppo_matrix = _success_matrix(df, v_d, v_e, ppo_method)
                scripted_vec = _success_vector(df, v_d, v_e, scripted_method)
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


def compute_operational_summary(df: pd.DataFrame, conditions=None) -> pd.DataFrame:
    op_cols = [
        "defender_path_length", "mean_defender_speed", "max_defender_speed",
        "mean_defender_accel", "max_defender_accel", "mean_defender_turn_rate", "max_defender_turn_rate",
        "fraction_defender_accel_saturated", "fraction_defender_turn_saturated", "fraction_defender_climb_saturated",
        "evasion_activated_episode", "fraction_steps_evasion_active", "max_evasion_activation",
        "episode_length_seconds", "intercept_time_seconds", "detected", "detection_time_seconds",
    ]
    rows = []
    for condition in (conditions or MOBILITY_CONDITIONS):
        v_d, v_e = condition.defender_speed, condition.hostile_speed
        for method in PRIMARY_PAPER_METHODS:
            row = {"defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": condition.mobility_ratio,
                   "sweep_axis": condition.sweep_axis, "method": method}
            if method in SCRIPTED_METHODS:
                sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["policy_instance"] == method)]
                for col in op_cols:
                    row[col] = float(sub[col].mean())
            else:
                for col in op_cols:
                    m = _metric_matrix(df, v_d, v_e, method, col)
                    with np.errstate(invalid="ignore"):
                        row[col] = float(np.nanmean(np.nanmean(m, axis=1)))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_estimator_summary(df: pd.DataFrame, conditions=None) -> pd.DataFrame:
    """
    CORRECTED aggregation (reuses experiments/estimator_stats.py directly --
    never reintroduces sqrt(mean(per_episode_mean_error^2)) as RMSE).
    """
    rows = []
    for condition in (conditions or MOBILITY_CONDITIONS):
        v_d, v_e = condition.defender_speed, condition.hostile_speed
        theoretical_mean = theoretical_raw_measurement_mean_error(_NOMINAL_ENV_CONFIG.measurement_var)
        theoretical_rmse = theoretical_raw_measurement_rmse(_NOMINAL_ENV_CONFIG.measurement_var)
        for method in SCRIPTED_METHODS:
            sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["policy_instance"] == method)].dropna(
                subset=["mean_position_estimation_error", "position_estimation_rmse"]
            )
            n = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
            m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
            r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
            is_direct = method in ("greedy", "pn", "lead")
            rows.append({
                "defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": condition.mobility_ratio,
                "sweep_axis": condition.sweep_axis, "method": method,
                "n_episodes_with_estimates": int(len(sub)), "n_total_estimation_samples": int(n.sum()),
                "mean_error_sample_weighted": sample_weighted_mean_error(n, m),
                "rmse_sample_weighted": sample_weighted_rmse(n, r),
                "theoretical_raw_mean_error": theoretical_mean if is_direct else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if is_direct else float("nan"),
            })
        for method in PPO_METHODS:
            sub_method = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["paper_method"] == method)]
            seed_rmse, seed_mean = [], []
            n_total = 0
            for seed in (42, 43, 44):
                seed_sub = sub_method[sub_method["training_seed"] == seed].dropna(
                    subset=["mean_position_estimation_error", "position_estimation_rmse"]
                )
                n = seed_sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
                m = seed_sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
                r = seed_sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
                n_total += int(n.sum())
                seed_mean.append(sample_weighted_mean_error(n, m))
                seed_rmse.append(sample_weighted_rmse(n, r))
            mean_error, _ = equal_weight_across_seeds(np.array(seed_mean))
            rmse, rmse_sd = equal_weight_across_seeds(np.array(seed_rmse))
            rows.append({
                "defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": condition.mobility_ratio,
                "sweep_axis": condition.sweep_axis, "method": method,
                "n_episodes_with_estimates": float("nan"), "n_total_estimation_samples": n_total,
                "mean_error_sample_weighted": mean_error,
                "rmse_sample_weighted": rmse, "train_seed_rmse_sd": rmse_sd,
                "theoretical_raw_mean_error": theoretical_mean if method == "ppo" else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if method == "ppo" else float("nan"),
            })
    return pd.DataFrame(rows)


def parse_condition(s: str):
    parts = s.split(",")
    return float(parts[0]), float(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked mobility robustness sweep")
    parser.add_argument("--seed-start", type=int, default=MOBILITY_SWEEP_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=MOBILITY_SWEEP_EPISODES)
    parser.add_argument("--conditions", type=str, nargs="+", default=None,
                         help="Subset of conditions as 'v_d,v_e' pairs (dry-run only); default = all 9 unique conditions.")
    parser.add_argument("--output-root", type=str, default=str(ROBUSTNESS_MOBILITY_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    if args.conditions:
        wanted = {parse_condition(s) for s in args.conditions}
        conditions = [c for c in MOBILITY_CONDITIONS if (c.defender_speed, c.hostile_speed) in wanted]
    else:
        conditions = list(MOBILITY_CONDITIONS)
    output_root = Path(args.output_root)
    is_final = args.seed_start >= MOBILITY_SWEEP_SEED_OFFSET

    print("=" * 70)
    print("MOBILITY ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/condition)")
    print(f"Conditions ({len(conditions)}): {[(c.defender_speed, c.hostile_speed) for c in conditions]}")
    print(f"Policy instances: {len(MOBILITY_POLICY_INSTANCES)}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    print("\nHashing frozen PPO models and verifying vs final nominal manifest...")
    model_hashes = compute_model_hashes_and_verify()
    for h in model_hashes:
        print(f"  {h['policy_instance']}: sha256={h['sha256'][:16]}... matches_final_nominal={h['matches_final_nominal_hash']}")

    manifest = build_manifest(seeds, conditions, output_root, model_hashes)
    print(f"\nManifest written. robustness_test_opened={manifest['robustness_test_opened']}")

    print("\nRunning evaluation...")
    df = run_evaluation(seeds, conditions, output_root)

    print("\nRunning integrity checks...")
    integrity = run_integrity_checks(df, seeds, conditions)
    print(json.dumps(integrity, indent=2, default=str))

    print("\nComputing summaries...")
    defender_speed_summary = compute_defender_speed_summary(df, conditions)
    defender_speed_summary.to_csv(output_root / DEFENDER_SPEED_SUMMARY_FILENAME, index=False)

    hostile_speed_summary = compute_hostile_speed_summary(df, conditions)
    hostile_speed_summary.to_csv(output_root / HOSTILE_SPEED_SUMMARY_FILENAME, index=False)

    mobility_ratio_summary = compute_mobility_ratio_summary(df, conditions)
    mobility_ratio_summary.to_csv(output_root / MOBILITY_RATIO_SUMMARY_FILENAME, index=False)

    ppo_seed_summary = compute_ppo_training_seed_summary(df, conditions)
    ppo_seed_summary.to_csv(output_root / PPO_TRAINING_SEED_SUMMARY_FILENAME, index=False)

    pairwise = compute_pairwise_comparisons(df, conditions)
    pairwise.to_csv(output_root / PAIRWISE_COMPARISONS_FILENAME, index=False)

    operational_summary = compute_operational_summary(df, conditions)
    operational_summary.to_csv(output_root / OPERATIONAL_SUMMARY_FILENAME, index=False)

    estimator_summary = compute_estimator_summary(df, conditions)
    estimator_summary.to_csv(output_root / ESTIMATOR_SUMMARY_FILENAME, index=False)

    final_summary = {
        "label": "MOBILITY ROBUSTNESS SWEEP" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "defender_speed_summary": defender_speed_summary.to_dict(orient="records"),
        "hostile_speed_summary": hostile_speed_summary.to_dict(orient="records"),
        "mobility_ratio_summary": mobility_ratio_summary.to_dict(orient="records"),
        "ppo_training_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "pairwise_comparisons": pairwise.to_dict(orient="records"),
        "operational_summary": operational_summary.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
    }
    (output_root / MOBILITY_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
