"""
Locked defender-hostile speed interaction grid runner.

Evaluates the SIX primary paper methods (10 concrete policy instances) across
ALL 25 Cartesian (v_d, v_e) grid cells, on a previously untouched seed range.
Only v_d and v_e (+ use_kalman_tracking) change per cell; all maneuverability,
sensing, evasion, and reward parameters are held FIXED at nominal. PPO/PPO-
Kalman remain exactly as trained/frozen at (v_d=18, v_e=12) -- every other
cell is a zero-shot generalization test.

Usage:
    # Full locked grid (seeds 35000-35499, all 25 cells):
    python experiments/run_robustness_mobility_grid_sweep.py

    # Small NON-FINAL dry run (diagnostic seeds < 35000, cell subset):
    python experiments/run_robustness_mobility_grid_sweep.py --seed-start 19000 --n-episodes 5 \
        --cells 15,10 18,12 21,14 24,16 --output-root results/_dry_run_robustness_mobility_grid
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

from experiments.robustness_mobility_grid_protocol import (
    DEFENDER_SPEED_VALUES,
    HOSTILE_SPEED_VALUES,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
    EQUAL_RATIO_MU,
    EQUAL_RATIO_CELLS,
    GRID_CELLS,
    MOBILITY_GRID_SEED_OFFSET,
    MOBILITY_GRID_EPISODES,
    MOBILITY_GRID_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    GRID_POLICY_INSTANCES,
    EXPECTED_POLICY_INSTANCES,
    EXPECTED_GRID_CELLS,
    PRIMARY_COMPARISONS,
    ESTIMATOR_COMPARISONS,
    SECONDARY_COMPARISONS,
    ALL_COMPARISONS,
    ROBUSTNESS_MOBILITY_GRID_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    GRID_SUCCESS_SUMMARY_FILENAME,
    PPO_TRAINING_SEED_SUMMARY_FILENAME,
    PAIRWISE_COMPARISONS_FILENAME,
    FEASIBILITY_REGION_SUMMARY_FILENAME,
    PPO_VS_LEAD_ENVELOPE_FILENAME,
    EQUAL_RATIO_SUMMARY_FILENAME,
    SWEET_SPOT_SUMMARY_FILENAME,
    INTERACTION_SUMMARY_FILENAME,
    FAILURE_MODE_SUMMARY_FILENAME,
    OPERATIONAL_SUMMARY_FILENAME,
    ESTIMATOR_SUMMARY_FILENAME,
    MOBILITY_GRID_SUMMARY_FILENAME,
    build_policy,
    build_grid_env_config,
    assert_only_grid_speed_differs,
)
from experiments.mobility_grid_eval_utils import run_mobility_grid_episode, sha256_of_file
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
    config = build_grid_env_config(estimator, defender_speed, hostile_speed)
    assert_only_grid_speed_differs(config, _NOMINAL_ENV_CONFIG)
    return SoldierEnv(config=config)


def compute_model_hashes_and_verify() -> list[dict]:
    nominal_hashes_by_path = {}
    if _FINAL_NOMINAL_HASHES_PATH.exists():
        for row in json.loads(_FINAL_NOMINAL_HASHES_PATH.read_text()):
            nominal_hashes_by_path[row["model_path"].replace("\\", "/")] = row["sha256"]

    rows = []
    for instance in GRID_POLICY_INSTANCES:
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


def build_manifest(seeds: list[int], cells, output_root: Path, model_hashes: list[dict]) -> dict:
    import gymnasium
    import stable_baselines3
    import torch

    opened = len(seeds) > 0 and min(seeds) >= MOBILITY_GRID_SEED_OFFSET
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
        "experiment_name": "defender-hostile speed interaction grid (two-parameter mobility operating-envelope study)",
        "primary_paper_methods": list(PRIMARY_PAPER_METHODS),
        "nominal_env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "defender_speed_values": list(DEFENDER_SPEED_VALUES),
        "hostile_speed_values": list(HOSTILE_SPEED_VALUES),
        "nominal_defender_speed": NOMINAL_DEFENDER_SPEED,
        "nominal_hostile_speed": NOMINAL_HOSTILE_SPEED,
        "n_grid_cells": len(cells),
        "grid_cells": [{"defender_speed": c.defender_speed, "hostile_speed": c.hostile_speed, "mobility_ratio": c.mobility_ratio} for c in cells],
        "equal_ratio_mu": EQUAL_RATIO_MU,
        "equal_ratio_cells": list(EQUAL_RATIO_CELLS),
        "ppo_model_hashes": model_hashes,
        "evaluation_seed_start": min(seeds) if seeds else None,
        "evaluation_seed_end": max(seeds) if seeds else None,
        "n_episodes_per_cell": len(seeds),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": list(PRIMARY_COMPARISONS),
        "estimator_comparisons": list(ESTIMATOR_COMPARISONS),
        "secondary_comparisons": list(SECONDARY_COMPARISONS),
        "observation_shape": list(env.observation_space.shape),
        "action_shape": list(env.action_space.shape),
        "note_only_grid_speed_swept": (
            "v_d and v_e are the only swept behavioral parameters. All acceleration, turn-rate, "
            "vertical-rate, sensing, reward, evasion, and termination parameters remain frozen."
        ),
        "note_zero_shot_ppo": (
            "PPO models were trained only at v_d=18, v_e=12. All other grid conditions are "
            "zero-shot generalization evaluations."
        ),
        "note_action_semantics": (
            "v_d represents both the nonzero-action desired cruise-speed magnitude and the "
            "defender maximum speed in the current action/dynamics model."
        ),
        "note_not_a_ratio_sweep": (
            "This is the defender-hostile speed interaction grid (a two-parameter mobility "
            "operating-envelope study), NOT merely a mobility-ratio sweep. mu=v_d/v_e is recorded "
            "as a descriptive field only and is not assumed to uniquely determine performance."
        ),
        "note_not_extension_of_1d_mobility": (
            "This grid uses a fresh, disjoint seed range (35000-35499) and is statistically "
            "independent of the prior 1-D mobility experiment (seeds 32000-32999). The two "
            "datasets are never pooled."
        ),
        "robustness_test_opened": opened,
        "robustness_type": "mobility_grid",
        "robustness_first_seed": MOBILITY_GRID_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], cells, output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    total = len(cells) * len(GRID_POLICY_INSTANCES)
    i = 0
    for cell in cells:
        for instance in GRID_POLICY_INSTANCES:
            i += 1
            env = build_env(instance.estimator, cell.defender_speed, cell.hostile_speed)
            policy = build_policy(instance)
            if verbose:
                print(f"[{i}/{total}] (v_d={cell.defender_speed}, v_e={cell.hostile_speed}) "
                      f"{instance.policy_instance} (estimator={instance.estimator}) -- {len(seeds)} episodes...")
            for s in seeds:
                row = run_mobility_grid_episode(
                    env, policy, seed=s, dt=dt,
                    defender_speed=cell.defender_speed, hostile_speed=cell.hostile_speed,
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


def run_integrity_checks(df: pd.DataFrame, seeds: list[int], cells) -> dict:
    results = {}
    expected_rows = len(cells) * EXPECTED_POLICY_INSTANCES * len(seeds)
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
    results["n_cell_method_combinations"] = int(len(combo_seed_sets))
    assert len(combo_seed_sets) == len(cells) * EXPECTED_POLICY_INSTANCES, "missing (v_d, v_e, policy_instance) combination"

    unique_cells_found = df[["defender_speed", "hostile_speed"]].drop_duplicates()
    results["n_unique_cells_found"] = int(len(unique_cells_found))
    assert len(unique_cells_found) == len(cells), "missing or extra unique (v_d, v_e) cell"
    if len(cells) == EXPECTED_GRID_CELLS:
        assert len(unique_cells_found) == EXPECTED_GRID_CELLS

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


def _cell_method_success_row(df: pd.DataFrame, v_d: float, v_e: float, method: str) -> dict:
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


def compute_grid_success_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for cell in cells:
        for method in PRIMARY_PAPER_METHODS:
            row = {"defender_speed": cell.defender_speed, "hostile_speed": cell.hostile_speed,
                   "mobility_ratio": cell.mobility_ratio, "method": method}
            row.update(_cell_method_success_row(df, cell.defender_speed, cell.hostile_speed, method))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_ppo_training_seed_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for cell in cells:
        for method in PPO_METHODS:
            sub_method = df[(df["defender_speed"] == cell.defender_speed) & (df["hostile_speed"] == cell.hostile_speed) & (df["paper_method"] == method)]
            for seed in (42, 43, 44):
                sub = sub_method[sub_method["training_seed"] == seed]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                rows.append({
                    "defender_speed": cell.defender_speed, "hostile_speed": cell.hostile_speed,
                    "mobility_ratio": cell.mobility_ratio, "paper_method": method, "training_seed": seed,
                    "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
                    "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "mean_episode_length_seconds": float(sub["episode_length_seconds"].mean()),
                    "mean_detection_time_seconds": float(sub["detection_time_seconds"].mean()),
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def compute_pairwise_comparisons(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for cell in cells:
        v_d, v_e = cell.defender_speed, cell.hostile_speed
        for name, method_a, method_b in ALL_COMPARISONS:
            a_is_ppo = method_a in PPO_METHODS
            b_is_ppo = method_b in PPO_METHODS
            row = {"defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": cell.mobility_ratio,
                   "comparison_name": name, "method_a": method_a, "method_b": method_b,
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


def compute_feasibility_region_summary(grid_success: pd.DataFrame, n_cells: int) -> pd.DataFrame:
    rows = []
    for method in PRIMARY_PAPER_METHODS:
        sub = grid_success[grid_success["method"] == method]
        for threshold in (0.70, 0.80, 0.90):
            n = int((sub["success_rate"] >= threshold).sum())
            rows.append({
                "method": method, "threshold": threshold,
                "n_cells_meeting_threshold": n, "n_cells_total": n_cells,
                "fraction_cells_meeting_threshold": n / n_cells,
            })
    return pd.DataFrame(rows)


def compute_ppo_vs_lead_envelope(pairwise: pd.DataFrame) -> pd.DataFrame:
    sub = pairwise[pairwise["comparison_name"] == "ppo_vs_lead"].copy()

    def _classify(row):
        if row["ci_low_pp"] > 0:
            return "PPO higher, CI excludes 0"
        if row["ci_high_pp"] < 0:
            return "Lead higher, CI excludes 0"
        return "CI includes 0"

    sub["classification"] = sub.apply(_classify, axis=1)
    return sub[["defender_speed", "hostile_speed", "mobility_ratio", "observed_diff_pp", "ci_low_pp", "ci_high_pp", "classification"]]


def compute_equal_ratio_summary(grid_success: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PRIMARY_PAPER_METHODS:
        successes = []
        for v_d, v_e in EQUAL_RATIO_CELLS:
            row = grid_success[(grid_success["defender_speed"] == v_d) & (grid_success["hostile_speed"] == v_e) & (grid_success["method"] == method)].iloc[0]
            successes.append(float(row["success_rate"]))
            rows.append({
                "method": method, "defender_speed": v_d, "hostile_speed": v_e,
                "mobility_ratio": EQUAL_RATIO_MU, "success_rate": float(row["success_rate"]),
                "success_ci_low": float(row["success_ci_low"]), "success_ci_high": float(row["success_ci_high"]),
            })
        range_val = max(successes) - min(successes)
        for r in rows[-len(EQUAL_RATIO_CELLS):]:
            r["range_across_equal_ratio_cells"] = range_val
    return pd.DataFrame(rows)


def compute_sweet_spot_summary(grid_success: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PRIMARY_PAPER_METHODS:
        sub = grid_success[grid_success["method"] == method]
        for v_e in HOSTILE_SPEED_VALUES:
            fixed_ve = sub[sub["hostile_speed"] == v_e]
            best = fixed_ve.loc[fixed_ve["success_rate"].idxmax()]
            rows.append({
                "axis": "hostile_speed_fixed", "fixed_value": v_e, "method": method,
                "best_tested_defender_speed": float(best["defender_speed"]), "success_rate": float(best["success_rate"]),
            })
        for v_d in DEFENDER_SPEED_VALUES:
            fixed_vd = sub[sub["defender_speed"] == v_d]
            best = fixed_vd.loc[fixed_vd["success_rate"].idxmax()]
            rows.append({
                "axis": "defender_speed_fixed", "fixed_value": v_d, "method": method,
                "best_tested_hostile_speed": float(best["hostile_speed"]), "success_rate": float(best["success_rate"]),
            })
    return pd.DataFrame(rows)


def compute_interaction_summary(grid_success: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PRIMARY_PAPER_METHODS:
        sub = grid_success[grid_success["method"] == method]
        for v_e in HOSTILE_SPEED_VALUES:
            p15 = sub[(sub["defender_speed"] == 15.0) & (sub["hostile_speed"] == v_e)]["success_rate"].iloc[0]
            p18 = sub[(sub["defender_speed"] == 18.0) & (sub["hostile_speed"] == v_e)]["success_rate"].iloc[0]
            rows.append({
                "method": method, "metric": "delta_d", "hostile_speed": v_e, "defender_speed": float("nan"),
                "value": float(p15 - p18),
                "detail": f"P(v_d=15,v_e={v_e}) - P(v_d=18,v_e={v_e})",
            })
        for v_d in DEFENDER_SPEED_VALUES:
            p16 = sub[(sub["defender_speed"] == v_d) & (sub["hostile_speed"] == 16.0)]["success_rate"].iloc[0]
            p8 = sub[(sub["defender_speed"] == v_d) & (sub["hostile_speed"] == 8.0)]["success_rate"].iloc[0]
            rows.append({
                "method": method, "metric": "delta_e", "hostile_speed": float("nan"), "defender_speed": v_d,
                "value": float(p16 - p8),
                "detail": f"P(v_d={v_d},v_e=16) - P(v_d={v_d},v_e=8)",
            })
    return pd.DataFrame(rows)


def compute_failure_mode_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for cell in cells:
        for method in PRIMARY_PAPER_METHODS:
            v_d, v_e = cell.defender_speed, cell.hostile_speed
            if method in SCRIPTED_METHODS:
                sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["policy_instance"] == method)]
                rows.append({
                    "defender_speed": v_d, "hostile_speed": v_e, "method": method,
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
            else:
                sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["paper_method"] == method)]
                rows.append({
                    "defender_speed": v_d, "hostile_speed": v_e, "method": method,
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def compute_operational_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    op_cols = [
        "defender_path_length", "mean_defender_speed", "max_defender_speed",
        "mean_defender_accel", "max_defender_accel", "mean_defender_turn_rate", "max_defender_turn_rate",
        "fraction_defender_accel_saturated", "fraction_defender_turn_saturated", "fraction_defender_climb_saturated",
        "mean_enemy_speed", "max_enemy_speed", "mean_enemy_horizontal_speed", "mean_abs_enemy_vertical_speed",
        "fraction_enemy_accel_saturated", "fraction_enemy_turn_saturated", "fraction_enemy_climb_or_vertical_saturated",
        "enemy_path_length", "ground_boundary_hits", "ceiling_hits", "horizontal_wall_hits",
        "min_horizontal_enemy_soldier_dist", "altitude_at_min_horizontal_enemy_soldier_dist",
        "min_full_3d_enemy_soldier_dist", "entered_2m_horizontal_while_vertical_gt_2m",
        "evasion_activated_episode", "fraction_steps_evasion_active", "max_evasion_activation",
        "episode_length_seconds", "intercept_time_seconds", "detected", "detection_time_seconds",
    ]
    rows = []
    for cell in cells:
        v_d, v_e = cell.defender_speed, cell.hostile_speed
        for method in PRIMARY_PAPER_METHODS:
            row = {"defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": cell.mobility_ratio, "method": method}
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


def compute_estimator_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    """CORRECTED aggregation (reuses experiments/estimator_stats.py directly)."""
    rows = []
    theoretical_mean = theoretical_raw_measurement_mean_error(_NOMINAL_ENV_CONFIG.measurement_var)
    theoretical_rmse = theoretical_raw_measurement_rmse(_NOMINAL_ENV_CONFIG.measurement_var)
    for cell in cells:
        v_d, v_e = cell.defender_speed, cell.hostile_speed
        for method in SCRIPTED_METHODS:
            sub = df[(df["defender_speed"] == v_d) & (df["hostile_speed"] == v_e) & (df["policy_instance"] == method)].dropna(
                subset=["mean_position_estimation_error", "position_estimation_rmse"]
            )
            n = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
            m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
            r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
            is_direct = method in ("greedy", "pn", "lead")
            rows.append({
                "defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": cell.mobility_ratio, "method": method,
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
                "defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": cell.mobility_ratio, "method": method,
                "n_episodes_with_estimates": float("nan"), "n_total_estimation_samples": n_total,
                "mean_error_sample_weighted": mean_error,
                "rmse_sample_weighted": rmse, "train_seed_rmse_sd": rmse_sd,
                "theoretical_raw_mean_error": theoretical_mean if method == "ppo" else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if method == "ppo" else float("nan"),
            })
    return pd.DataFrame(rows)


def parse_cell(s: str):
    parts = s.split(",")
    return float(parts[0]), float(parts[1])


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked defender-hostile speed interaction grid")
    parser.add_argument("--seed-start", type=int, default=MOBILITY_GRID_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=MOBILITY_GRID_EPISODES)
    parser.add_argument("--cells", type=str, nargs="+", default=None,
                         help="Subset of cells as 'v_d,v_e' pairs (dry-run only); default = all 25 grid cells.")
    parser.add_argument("--output-root", type=str, default=str(ROBUSTNESS_MOBILITY_GRID_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    if args.cells:
        wanted = {parse_cell(s) for s in args.cells}
        cells = [c for c in GRID_CELLS if (c.defender_speed, c.hostile_speed) in wanted]
    else:
        cells = list(GRID_CELLS)
    output_root = Path(args.output_root)
    is_final = args.seed_start >= MOBILITY_GRID_SEED_OFFSET

    print("=" * 70)
    print("DEFENDER-HOSTILE SPEED INTERACTION GRID" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/cell)")
    print(f"Cells ({len(cells)}): {[(c.defender_speed, c.hostile_speed) for c in cells]}")
    print(f"Policy instances: {len(GRID_POLICY_INSTANCES)}")
    print(f"Output root: {output_root}")
    print("=" * 70)

    print("\nHashing frozen PPO models and verifying vs final nominal manifest...")
    model_hashes = compute_model_hashes_and_verify()
    for h in model_hashes:
        print(f"  {h['policy_instance']}: sha256={h['sha256'][:16]}... matches_final_nominal={h['matches_final_nominal_hash']}")

    manifest = build_manifest(seeds, cells, output_root, model_hashes)
    print(f"\nManifest written. robustness_test_opened={manifest['robustness_test_opened']}")

    print("\nRunning evaluation...")
    df = run_evaluation(seeds, cells, output_root)

    print("\nRunning integrity checks...")
    integrity = run_integrity_checks(df, seeds, cells)
    print(json.dumps(integrity, indent=2, default=str))

    print("\nComputing summaries...")
    grid_success = compute_grid_success_summary(df, cells)
    grid_success.to_csv(output_root / GRID_SUCCESS_SUMMARY_FILENAME, index=False)

    ppo_seed_summary = compute_ppo_training_seed_summary(df, cells)
    ppo_seed_summary.to_csv(output_root / PPO_TRAINING_SEED_SUMMARY_FILENAME, index=False)

    pairwise = compute_pairwise_comparisons(df, cells)
    pairwise.to_csv(output_root / PAIRWISE_COMPARISONS_FILENAME, index=False)

    feasibility = compute_feasibility_region_summary(grid_success, len(cells))
    feasibility.to_csv(output_root / FEASIBILITY_REGION_SUMMARY_FILENAME, index=False)

    ppo_vs_lead_envelope = compute_ppo_vs_lead_envelope(pairwise)
    ppo_vs_lead_envelope.to_csv(output_root / PPO_VS_LEAD_ENVELOPE_FILENAME, index=False)

    is_full_grid = len(cells) == EXPECTED_GRID_CELLS
    if is_full_grid:
        equal_ratio = compute_equal_ratio_summary(grid_success)
        equal_ratio.to_csv(output_root / EQUAL_RATIO_SUMMARY_FILENAME, index=False)

        sweet_spot = compute_sweet_spot_summary(grid_success)
        sweet_spot.to_csv(output_root / SWEET_SPOT_SUMMARY_FILENAME, index=False)

        interaction = compute_interaction_summary(grid_success)
        interaction.to_csv(output_root / INTERACTION_SUMMARY_FILENAME, index=False)
    else:
        equal_ratio = pd.DataFrame()
        sweet_spot = pd.DataFrame()
        interaction = pd.DataFrame()
        print("\n(Skipping equal-ratio/sweet-spot/interaction summaries -- require the full 25-cell grid)")

    failure_modes = compute_failure_mode_summary(df, cells)
    failure_modes.to_csv(output_root / FAILURE_MODE_SUMMARY_FILENAME, index=False)

    operational_summary = compute_operational_summary(df, cells)
    operational_summary.to_csv(output_root / OPERATIONAL_SUMMARY_FILENAME, index=False)

    estimator_summary = compute_estimator_summary(df, cells)
    estimator_summary.to_csv(output_root / ESTIMATOR_SUMMARY_FILENAME, index=False)

    final_summary = {
        "label": "DEFENDER-HOSTILE SPEED INTERACTION GRID" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "grid_success_summary": grid_success.to_dict(orient="records"),
        "ppo_training_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "pairwise_comparisons": pairwise.to_dict(orient="records"),
        "feasibility_region_summary": feasibility.to_dict(orient="records"),
        "ppo_vs_lead_envelope": ppo_vs_lead_envelope.to_dict(orient="records"),
        "equal_ratio_summary": equal_ratio.to_dict(orient="records"),
        "sweet_spot_summary": sweet_spot.to_dict(orient="records"),
        "interaction_summary": interaction.to_dict(orient="records"),
        "failure_mode_summary": failure_modes.to_dict(orient="records"),
        "operational_summary": operational_summary.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
    }
    (output_root / MOBILITY_GRID_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
