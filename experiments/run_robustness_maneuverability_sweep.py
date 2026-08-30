"""
Locked defender-maneuverability (accel x turn-rate) interaction grid runner.

Evaluates the SIX primary paper methods (10 concrete policy instances) across
9 Cartesian (defender_max_accel, defender_max_turn_rate_deg) cells, on a
previously reserved seed range. Only defender_max_accel/defender_max_turn_rate_deg
(+ use_kalman_tracking) change per condition; defender speed, vertical-rate
limits, hostile dynamics, sensing, evasion, and reward parameters are held
FIXED at nominal. PPO/PPO-Kalman remain exactly as trained/frozen at
(accel=8, turn=90) -- every other cell is a zero-shot dynamics
generalization test. Maneuverability limits are NOT exposed to the PPO
observation (item 44) -- this file does not touch the observation builder.

Usage:
    # Full locked sweep (seeds 34000-34999, all 9 cells):
    python experiments/run_robustness_maneuverability_sweep.py

    # Small NON-FINAL dry run (diagnostic seeds < 34000, cell subset):
    python experiments/run_robustness_maneuverability_sweep.py --seed-start 19600 --n-episodes 5 \
        --cells 4,45 4,135 8,90 12,45 12,135 --output-root results/_dry_run_robustness_maneuverability
"""

from __future__ import annotations

import argparse
import dataclasses
import itertools
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

from experiments.robustness_maneuverability_protocol import (
    DEFENDER_MAX_ACCEL_VALUES,
    DEFENDER_MAX_TURN_RATE_VALUES,
    NOMINAL_DEFENDER_MAX_ACCEL,
    NOMINAL_DEFENDER_MAX_TURN_RATE_DEG,
    MANEUVERABILITY_CELLS,
    RESTRICTIVE_CELL,
    PERMISSIVE_CELL,
    NOMINAL_CELL,
    MANEUVERABILITY_SEED_OFFSET,
    MANEUVERABILITY_EPISODES,
    MANEUVERABILITY_SEEDS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    MANEUVERABILITY_POLICY_INSTANCES,
    EXPECTED_N_POLICY_INSTANCES,
    EXPECTED_MANEUVERABILITY_CELLS,
    EXPECTED_RAW_ROWS,
    PRIMARY_COMPARISONS,
    ESTIMATOR_COMPARISONS,
    SECONDARY_COMPARISONS,
    FEASIBILITY_SUCCESS_THRESHOLDS,
    ROBUSTNESS_MANEUVERABILITY_OUTPUT_ROOT,
    EXPERIMENT_MANIFEST_FILENAME,
    MODEL_HASHES_FILENAME,
    RAW_EPISODE_RESULTS_FILENAME,
    MANEUVERABILITY_SUCCESS_SUMMARY_FILENAME,
    PPO_TRAINING_SEED_SUMMARY_FILENAME,
    PPO_VS_LEAD_COMPARISONS_FILENAME,
    ESTIMATOR_COMPARISONS_FILENAME,
    ACCELERATION_EFFECT_SUMMARY_FILENAME,
    TURN_RATE_EFFECT_SUMMARY_FILENAME,
    INTERACTION_SUMMARY_FILENAME,
    FEASIBILITY_SUMMARY_FILENAME,
    ACQUISITION_SUMMARY_FILENAME,
    POST_DETECTION_SUMMARY_FILENAME,
    FAILURE_MODE_SUMMARY_FILENAME,
    OPERATIONAL_SUMMARY_FILENAME,
    ESTIMATOR_SUMMARY_FILENAME,
    MANEUVERABILITY_SUMMARY_FILENAME,
    build_policy,
    build_maneuverability_env_config,
    assert_only_maneuverability_differs,
)
from experiments.maneuverability_eval_utils import run_maneuverability_episode, sha256_of_file
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


def build_env(estimator: str, accel: float, turn: float) -> SoldierEnv:
    config = build_maneuverability_env_config(estimator, accel, turn)
    assert_only_maneuverability_differs(config, _NOMINAL_ENV_CONFIG)
    return SoldierEnv(config=config)


def compute_model_hashes_and_verify() -> list[dict]:
    nominal_hashes_by_path = {}
    if _FINAL_NOMINAL_HASHES_PATH.exists():
        for row in json.loads(_FINAL_NOMINAL_HASHES_PATH.read_text()):
            nominal_hashes_by_path[row["model_path"].replace("\\", "/")] = row["sha256"]

    rows = []
    for instance in MANEUVERABILITY_POLICY_INSTANCES:
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

    opened = len(seeds) > 0 and min(seeds) >= MANEUVERABILITY_SEED_OFFSET
    env = build_env("measurement", NOMINAL_DEFENDER_MAX_ACCEL, NOMINAL_DEFENDER_MAX_TURN_RATE_DEG)
    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "gymnasium_version": gymnasium.__version__,
        "experiment_name": "defender-maneuverability (accel x turn-rate) robustness grid",
        "primary_paper_methods": list(PRIMARY_PAPER_METHODS),
        "nominal_env_config": dataclasses.asdict(_NOMINAL_ENV_CONFIG),
        "defender_max_accel_values": list(DEFENDER_MAX_ACCEL_VALUES),
        "defender_max_turn_rate_values": list(DEFENDER_MAX_TURN_RATE_VALUES),
        "nominal_cell": {"defender_max_accel": NOMINAL_DEFENDER_MAX_ACCEL, "defender_max_turn_rate_deg": NOMINAL_DEFENDER_MAX_TURN_RATE_DEG},
        "cells_evaluated": [{"defender_max_accel": c[0], "defender_max_turn_rate_deg": c[1]} for c in cells],
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
        "note_only_maneuverability_swept": (
            "The only swept behavioral parameters are defender_max_accel and "
            "defender_max_turn_rate_deg. Defender speed, vertical-rate limits, hostile "
            "dynamics, sensing, evasion, rewards, and termination conditions remain frozen."
        ),
        "note_zero_shot_ppo": (
            "PPO models were trained only at defender_max_accel=8 m/s^2 and "
            "defender_max_turn_rate_deg=90 deg/s. All other cells are zero-shot dynamics "
            "generalization conditions."
        ),
        "note_observation_contract": (
            "Maneuverability limits are NOT added to the PPO observation. PPO performs "
            "zero-shot control under altered plant constraints; observation dimensionality "
            "and semantics are unchanged."
        ),
        "note_not_hardware_validated": (
            "The acceleration and turn-rate grid values are simulation parameters for a "
            "robustness study, not hardware-validated or claimed-realistic airframe specifications."
        ),
        "research_questions": [
            "Q1. How does interception success depend jointly on defender acceleration and turn rate?",
            "Q2. Which constraint is more influential over the tested range?",
            "Q3. Is there a strong acceleration x turn-rate interaction?",
            "Q4. Does PPO degrade more slowly than Lead as defender maneuverability becomes restrictive?",
            "Q5. Does Lead outperform PPO in any constrained-dynamics region?",
            "Q6. Does PPO outperform Lead in any constrained-dynamics region?",
            "Q7. Does Kalman filtering help under any defender-maneuverability condition?",
            "Q8. How much does performance improve when maneuverability is relaxed beyond the training condition?",
        ],
        "robustness_test_opened": opened,
        "robustness_type": "maneuverability",
        "robustness_first_seed": MANEUVERABILITY_SEED_OFFSET if opened else None,
    }
    output_root.mkdir(parents=True, exist_ok=True)
    (output_root / EXPERIMENT_MANIFEST_FILENAME).write_text(json.dumps(manifest, indent=2, default=str))
    (output_root / MODEL_HASHES_FILENAME).write_text(json.dumps(model_hashes, indent=2, default=str))
    return manifest


def run_evaluation(seeds: list[int], cells, output_root: Path, verbose: bool = True) -> pd.DataFrame:
    rows = []
    dt = _NOMINAL_ENV_CONFIG.dt
    t_start = time.time()
    total = len(cells) * len(MANEUVERABILITY_POLICY_INSTANCES)
    i = 0
    for accel, turn in cells:
        for instance in MANEUVERABILITY_POLICY_INSTANCES:
            i += 1
            env = build_env(instance.estimator, accel, turn)
            policy = build_policy(instance)
            if verbose:
                print(f"[{i}/{total}] accel={accel} turn={turn} {instance.policy_instance} "
                      f"(estimator={instance.estimator}) -- {len(seeds)} episodes...")
            for s in seeds:
                row = run_maneuverability_episode(env, policy, seed=s, dt=dt)
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
    expected_rows = len(cells) * EXPECTED_N_POLICY_INSTANCES * len(seeds)
    results["expected_row_count"] = expected_rows
    results["actual_row_count"] = len(df)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    dup = df.duplicated(subset=["defender_max_accel", "defender_max_turn_rate_deg", "policy_instance", "eval_seed"]).sum()
    results["duplicate_keys"] = int(dup)
    assert dup == 0, f"found {dup} duplicated (accel, turn, policy_instance, eval_seed) keys"

    seed_set_expected = set(seeds)
    combo_seed_sets = df.groupby(["defender_max_accel", "defender_max_turn_rate_deg", "policy_instance"])["eval_seed"].apply(set)
    common_seeds_ok = all(s == seed_set_expected for s in combo_seed_sets)
    results["common_random_numbers_ok"] = bool(common_seeds_ok)
    assert common_seeds_ok, "not all (accel, turn, policy_instance) combinations share the identical evaluation-seed set"
    results["n_cell_method_combinations"] = int(len(combo_seed_sets))
    assert len(combo_seed_sets) == len(cells) * EXPECTED_N_POLICY_INSTANCES, "missing (accel, turn, policy_instance) combination"

    unique_cells_found = df[["defender_max_accel", "defender_max_turn_rate_deg"]].drop_duplicates()
    results["n_unique_cells_found"] = int(len(unique_cells_found))
    assert len(unique_cells_found) == len(cells), "missing or extra unique cell"

    outcome_sum = df.groupby(["defender_max_accel", "defender_max_turn_rate_deg", "policy_instance"])[
        ["success", "soldier_caught", "unsafe_intercept", "timeout"]
    ].sum().sum(axis=1)
    outcome_ok = bool((outcome_sum == len(seeds)).all())
    results["outcome_accounting_ok"] = outcome_ok
    assert outcome_ok, "outcome categories do not sum to n_episodes for every (accel, turn, policy_instance)"

    required = ["success", "soldier_caught", "unsafe_intercept", "timeout", "episode_length_steps", "total_reward"]
    nan_inf_bad = {c: int((~np.isfinite(df[c])).sum()) for c in required if (~np.isfinite(df[c])).sum()}
    results["nan_inf_in_required_fields"] = nan_inf_bad
    assert not nan_inf_bad, f"NaN/Inf found in required-always-defined fields: {nan_inf_bad}"

    return results


def _success_vector(df: pd.DataFrame, accel: float, turn: float, policy_instance: str) -> np.ndarray:
    sub = df[
        (df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn)
        & (df["policy_instance"] == policy_instance)
    ].sort_values("eval_seed")
    return sub["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, accel: float, turn: float, paper_method: str) -> np.ndarray:
    sub = df[
        (df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn)
        & (df["paper_method"] == paper_method)
    ]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s["success"].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _metric_matrix(df: pd.DataFrame, accel: float, turn: float, paper_method: str, column: str) -> np.ndarray:
    sub = df[
        (df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn)
        & (df["paper_method"] == paper_method)
    ]
    rows = []
    for seed in (42, 43, 44):
        s = sub[sub["training_seed"] == seed].sort_values("eval_seed")
        rows.append(s[column].to_numpy(dtype=np.float64))
    return np.stack(rows)


def _cell_method_success_row(df: pd.DataFrame, accel: float, turn: float, method: str) -> dict:
    if method in SCRIPTED_METHODS:
        sub = df[
            (df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn)
            & (df["policy_instance"] == method)
        ]
        n = len(sub)
        successes = int(sub["success"].sum())
        ci_lo, ci_hi = wilson_interval(successes, n)
        return {
            "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
            "success_ci_low": ci_lo, "success_ci_high": ci_hi,
            "training_seed_sd": float("nan"), "training_seed_min": float("nan"), "training_seed_max": float("nan"),
        }
    success_matrix = _success_matrix(df, accel, turn, method)
    observed, ci_lo, ci_hi = hierarchical_bootstrap_track(success_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    per_seed_rates = np.nanmean(success_matrix, axis=1)
    return {
        "n_eval_episodes": success_matrix.shape[1], "success_count": float("nan"), "success_rate": observed,
        "success_ci_low": ci_lo, "success_ci_high": ci_hi,
        "training_seed_sd": float(np.std(per_seed_rates)),
        "training_seed_min": float(np.min(per_seed_rates)), "training_seed_max": float(np.max(per_seed_rates)),
    }


def compute_success_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for accel, turn in cells:
        for method in PRIMARY_PAPER_METHODS:
            row = {"defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "method": method,
                   "is_nominal_cell": (accel == NOMINAL_DEFENDER_MAX_ACCEL and turn == NOMINAL_DEFENDER_MAX_TURN_RATE_DEG)}
            row.update(_cell_method_success_row(df, accel, turn, method))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_ppo_training_seed_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for accel, turn in cells:
        for method in PPO_METHODS:
            sub_method = df[
                (df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn)
                & (df["paper_method"] == method)
            ]
            for seed in (42, 43, 44):
                sub = sub_method[sub_method["training_seed"] == seed]
                n = len(sub)
                successes = int(sub["success"].sum())
                ci_lo, ci_hi = wilson_interval(successes, n)
                rows.append({
                    "defender_max_accel": accel, "defender_max_turn_rate_deg": turn,
                    "paper_method": method, "training_seed": seed,
                    "n_eval_episodes": n, "success_count": successes, "success_rate": successes / n,
                    "success_ci_low": ci_lo, "success_ci_high": ci_hi,
                    "mean_episode_length_seconds": float(sub["episode_length_seconds"].mean()),
                    "mean_detection_time_seconds": float(sub["detection_time_seconds"].mean()),
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def _paired_comparison_row(df: pd.DataFrame, accel: float, turn: float, name: str, method_a: str, method_b: str) -> dict:
    a_is_ppo = method_a in PPO_METHODS
    b_is_ppo = method_b in PPO_METHODS
    row = {"defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "comparison_name": name,
           "method_a": method_a, "method_b": method_b,
           "diff_seed_42_pp": float("nan"), "diff_seed_43_pp": float("nan"), "diff_seed_44_pp": float("nan"),
           "n10_a_wins": float("nan"), "n01_a_loses": float("nan"), "mcnemar_p_value": float("nan")}
    if not a_is_ppo and not b_is_ppo:
        vec_a = _success_vector(df, accel, turn, method_a)
        vec_b = _success_vector(df, accel, turn, method_b)
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
        a_matrix = _success_matrix(df, accel, turn, method_a)
        b_matrix = _success_matrix(df, accel, turn, method_b)
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
        ppo_matrix = _success_matrix(df, accel, turn, ppo_method)
        scripted_vec = _success_vector(df, accel, turn, scripted_method)
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


def compute_ppo_vs_lead_comparisons(df: pd.DataFrame, cells) -> pd.DataFrame:
    """Primary + secondary comparisons at all 9 cells (item 19/23's peers)."""
    rows = []
    for accel, turn in cells:
        for name, method_a, method_b in (PRIMARY_COMPARISONS + SECONDARY_COMPARISONS):
            row = _paired_comparison_row(df, accel, turn, name, method_a, method_b)
            if name == "ppo_vs_lead":
                row["classification"] = _classify_ci(row)
            rows.append(row)
    return pd.DataFrame(rows)


def compute_estimator_comparisons(df: pd.DataFrame, cells) -> pd.DataFrame:
    """Kalman-Greedy vs Greedy, and PPO-Kalman vs PPO, at all 9 cells (items 34/35)."""
    rows = []
    for accel, turn in cells:
        for name, method_a, method_b in ESTIMATOR_COMPARISONS:
            rows.append(_paired_comparison_row(df, accel, turn, name, method_a, method_b))
    return pd.DataFrame(rows)


def _method_success_series(df: pd.DataFrame, method: str, accel: float, turn: float) -> np.ndarray:
    """Aligned-by-seed success vector for a method at one cell (mean over PPO training seeds if PPO)."""
    if method in SCRIPTED_METHODS:
        return _success_vector(df, accel, turn, method)
    matrix = _success_matrix(df, accel, turn, method)
    return np.nanmean(matrix, axis=0)


def _paired_diff_generic(df: pd.DataFrame, method: str, accel_a: float, turn_a: float, accel_b: float, turn_b: float) -> dict:
    """Paired success difference between two cells for one method, matched by evaluation seed
    (and by training seed for PPO methods)."""
    if method in SCRIPTED_METHODS:
        vec_a = _success_vector(df, accel_a, turn_a, method)
        vec_b = _success_vector(df, accel_b, turn_b, method)
        observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        n10 = int(np.sum((vec_a == 1) & (vec_b == 0)))
        n01 = int(np.sum((vec_a == 0) & (vec_b == 1)))
        p_value = mcnemar_exact(n01=n01, n10=n10)
        return {
            "comparison_type": "fixed_paired",
            "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
        }
    a_matrix = _success_matrix(df, accel_a, turn_a, method)
    b_matrix = _success_matrix(df, accel_b, turn_b, method)
    observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
        a_matrix, b_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED
    )
    return {
        "comparison_type": "hierarchical_paired_matched_seeds",
        "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
        "diff_seed_42_pp": per_seed_diffs[0] * 100, "diff_seed_43_pp": per_seed_diffs[1] * 100,
        "diff_seed_44_pp": per_seed_diffs[2] * 100,
    }


def compute_acceleration_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    """At each fixed turn rate: accel8-accel4, accel12-accel8, accel12-accel4, per method (item 23)."""
    rows = []
    pairs = [("accel8_minus_accel4", 8.0, 4.0), ("accel12_minus_accel8", 12.0, 8.0), ("accel12_minus_accel4", 12.0, 4.0)]
    for turn in DEFENDER_MAX_TURN_RATE_VALUES:
        for method in PRIMARY_PAPER_METHODS:
            for name, accel_hi, accel_lo in pairs:
                row = {"defender_max_turn_rate_deg": turn, "method": method, "comparison": name,
                       "accel_a": accel_hi, "accel_b": accel_lo}
                row.update(_paired_diff_generic(df, method, accel_hi, turn, accel_lo, turn))
                rows.append(row)
    return pd.DataFrame(rows)


def compute_turn_rate_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    """At each fixed acceleration: turn90-turn45, turn135-turn90, turn135-turn45, per method (item 24)."""
    rows = []
    pairs = [("turn90_minus_turn45", 90.0, 45.0), ("turn135_minus_turn90", 135.0, 90.0), ("turn135_minus_turn45", 135.0, 45.0)]
    for accel in DEFENDER_MAX_ACCEL_VALUES:
        for method in PRIMARY_PAPER_METHODS:
            for name, turn_hi, turn_lo in pairs:
                row = {"defender_max_accel": accel, "method": method, "comparison": name,
                       "turn_a": turn_hi, "turn_b": turn_lo}
                row.update(_paired_diff_generic(df, method, accel, turn_hi, accel, turn_lo))
                rows.append(row)
    return pd.DataFrame(rows)


def compute_interaction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Difference-in-differences interaction (item 25):
    I_accel = [P(12,135)-P(4,135)] - [P(12,45)-P(4,45)]  (accel effect changes with turn rate)
    I_turn  = [P(12,135)-P(12,45)] - [P(4,135)-P(4,45)]  (turn effect changes with accel) -- algebraically identical.
    Reported descriptively only; no complex model fit.
    """
    rows = []
    for method in PRIMARY_PAPER_METHODS:
        p_hi_hi = float(np.mean(_method_success_series(df, method, 12.0, 135.0)))
        p_hi_lo = float(np.mean(_method_success_series(df, method, 12.0, 45.0)))
        p_lo_hi = float(np.mean(_method_success_series(df, method, 4.0, 135.0)))
        p_lo_lo = float(np.mean(_method_success_series(df, method, 4.0, 45.0)))
        interaction_pp = ((p_hi_hi - p_lo_hi) - (p_hi_lo - p_lo_lo)) * 100
        rows.append({
            "method": method,
            "P_accel12_turn135": p_hi_hi, "P_accel12_turn45": p_hi_lo,
            "P_accel4_turn135": p_lo_hi, "P_accel4_turn45": p_lo_lo,
            "accel_effect_at_turn135_pp": (p_hi_hi - p_lo_hi) * 100,
            "accel_effect_at_turn45_pp": (p_hi_lo - p_lo_lo) * 100,
            "interaction_diff_in_diff_pp": interaction_pp,
            "large_interaction": bool(abs(interaction_pp) >= 10.0),
        })
    return pd.DataFrame(rows)


def compute_feasibility_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    """Discrete measured-cell coverage only -- no interpolation (item 36)."""
    rows = []
    for method in PRIMARY_PAPER_METHODS:
        cell_success = []
        for accel, turn in cells:
            cell_success.append(float(np.mean(_method_success_series(df, method, accel, turn))))
        cell_success = np.array(cell_success)
        row = {"method": method, "n_cells": len(cells)}
        for thresh in FEASIBILITY_SUCCESS_THRESHOLDS:
            n_ok = int(np.sum(cell_success >= thresh))
            row[f"n_cells_success_ge_{int(thresh*100)}pct"] = n_ok
            row[f"fraction_cells_success_ge_{int(thresh*100)}pct"] = n_ok / len(cells)
        rows.append(row)
    return pd.DataFrame(rows)


def compute_acquisition_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for accel, turn in cells:
        for method in PRIMARY_PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["policy_instance"] == method)]
            else:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["paper_method"] == method)]
            rows.append({
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "method": method,
                "detection_rate": float(sub["detected"].mean()),
                "mean_detection_time_seconds": float(sub["detection_time_seconds"].mean()),
                "mean_defender_enemy_distance_at_detection": float(sub["defender_enemy_distance_at_detection"].mean()),
                "mean_enemy_soldier_distance_at_detection": float(sub["enemy_soldier_distance_at_detection"].mean()),
            })
    return pd.DataFrame(rows)


def compute_post_detection_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for accel, turn in cells:
        for method in PRIMARY_PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["policy_instance"] == method)]
            else:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["paper_method"] == method)]
            rows.append({
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "method": method,
                "mean_post_detection_intercept_seconds": float(sub["post_detection_intercept_seconds"].mean()),
                "mean_min_defender_enemy_dist_post_detection": float(sub["min_defender_enemy_dist_post_detection"].mean()),
                "mean_min_enemy_soldier_dist_post_detection": float(sub["min_enemy_soldier_dist_post_detection"].mean()),
                "mean_defender_path_length_post_detection": float(sub["defender_path_length_post_detection"].mean()),
                "mean_post_detection_closing_efficiency": float(sub["post_detection_closing_efficiency"].mean()),
            })
    return pd.DataFrame(rows)


def compute_failure_mode_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    rows = []
    for accel, turn in cells:
        for method in PRIMARY_PAPER_METHODS:
            if method in SCRIPTED_METHODS:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["policy_instance"] == method)]
            else:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["paper_method"] == method)]
            rows.append({
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "method": method,
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
        "mean_accel_utilization", "mean_turn_rate_utilization",
        "mean_enemy_speed", "max_enemy_speed",
        "fraction_enemy_accel_saturated", "fraction_enemy_turn_saturated", "fraction_enemy_vertical_saturated",
        "enemy_path_length", "evasion_activated_episode", "fraction_steps_evasion_active", "mean_evasion_activation",
        "episode_length_seconds", "min_defender_enemy_dist", "min_enemy_soldier_dist",
    ]
    rows = []
    for accel, turn in cells:
        for method in PRIMARY_PAPER_METHODS:
            row = {"defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "method": method}
            if method in SCRIPTED_METHODS:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["policy_instance"] == method)]
                for col in op_cols:
                    row[col] = float(sub[col].mean())
            else:
                for col in op_cols:
                    m = _metric_matrix(df, accel, turn, method, col)
                    with np.errstate(invalid="ignore"):
                        row[col] = float(np.nanmean(np.nanmean(m, axis=1)))
            rows.append(row)
    return pd.DataFrame(rows)


def compute_estimator_summary(df: pd.DataFrame, cells) -> pd.DataFrame:
    """CORRECTED aggregation (reuses experiments/estimator_stats.py directly)."""
    rows = []
    theoretical_mean = theoretical_raw_measurement_mean_error(_NOMINAL_ENV_CONFIG.measurement_var)
    theoretical_rmse = theoretical_raw_measurement_rmse(_NOMINAL_ENV_CONFIG.measurement_var)
    for accel, turn in cells:
        for method in SCRIPTED_METHODS:
            sub_all = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["policy_instance"] == method)]
            sub = sub_all.dropna(subset=["mean_position_estimation_error", "position_estimation_rmse"])
            n = sub["n_position_estimation_samples"].to_numpy(dtype=np.float64)
            m = sub["mean_position_estimation_error"].to_numpy(dtype=np.float64)
            r = sub["position_estimation_rmse"].to_numpy(dtype=np.float64)
            is_direct = method in ("greedy", "pn", "lead")
            rows.append({
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "method": method,
                "success_rate": float(sub_all["success"].mean()),
                "n_episodes_with_estimates": int(len(sub)), "n_total_estimation_samples": int(n.sum()),
                "mean_error_sample_weighted": sample_weighted_mean_error(n, m),
                "rmse_sample_weighted": sample_weighted_rmse(n, r),
                "theoretical_raw_mean_error": theoretical_mean if is_direct else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if is_direct else float("nan"),
            })
        for method in PPO_METHODS:
            sub_method = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn) & (df["paper_method"] == method)]
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
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn, "method": method,
                "success_rate": float(np.mean(seed_success)),
                "n_episodes_with_estimates": float("nan"), "n_total_estimation_samples": n_total,
                "mean_error_sample_weighted": mean_error,
                "rmse_sample_weighted": rmse, "train_seed_rmse_sd": rmse_sd,
                "theoretical_raw_mean_error": theoretical_mean if method == "ppo" else float("nan"),
                "theoretical_raw_rmse": theoretical_rmse if method == "ppo" else float("nan"),
            })
    return pd.DataFrame(rows)


def _parse_cell(text: str) -> tuple[float, float]:
    a, t = text.split(",")
    return float(a), float(t)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Locked defender-maneuverability robustness grid sweep")
    parser.add_argument("--seed-start", type=int, default=MANEUVERABILITY_SEED_OFFSET)
    parser.add_argument("--n-episodes", type=int, default=MANEUVERABILITY_EPISODES)
    parser.add_argument("--cells", type=_parse_cell, nargs="+", default=None,
                         help="Subset of 'accel,turn' cells (dry-run only); default = all 9 cells.")
    parser.add_argument("--output-root", type=str, default=str(ROBUSTNESS_MANEUVERABILITY_OUTPUT_ROOT))
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    seeds = list(range(args.seed_start, args.seed_start + args.n_episodes))
    cells = args.cells if args.cells else [(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS]
    output_root = Path(args.output_root)
    is_final = args.seed_start >= MANEUVERABILITY_SEED_OFFSET

    print("=" * 70)
    print("DEFENDER-MANEUVERABILITY ROBUSTNESS GRID" if is_final else "DRY RUN (non-final seeds)")
    print("=" * 70)
    print(f"Seeds: {seeds[0]}..{seeds[-1]} ({len(seeds)} episodes/cell)")
    print(f"Cells ({len(cells)}): {cells}")
    print(f"Policy instances: {len(MANEUVERABILITY_POLICY_INSTANCES)}")
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
    success_summary = compute_success_summary(df, cells)
    success_summary.to_csv(output_root / MANEUVERABILITY_SUCCESS_SUMMARY_FILENAME, index=False)

    ppo_seed_summary = compute_ppo_training_seed_summary(df, cells)
    ppo_seed_summary.to_csv(output_root / PPO_TRAINING_SEED_SUMMARY_FILENAME, index=False)

    ppo_vs_lead = compute_ppo_vs_lead_comparisons(df, cells)
    ppo_vs_lead.to_csv(output_root / PPO_VS_LEAD_COMPARISONS_FILENAME, index=False)

    estimator_comparisons = compute_estimator_comparisons(df, cells)
    estimator_comparisons.to_csv(output_root / ESTIMATOR_COMPARISONS_FILENAME, index=False)

    has_full_grid = set(cells) == {(c.defender_max_accel, c.defender_max_turn_rate_deg) for c in MANEUVERABILITY_CELLS}
    if has_full_grid:
        accel_effect = compute_acceleration_effect_summary(df)
        accel_effect.to_csv(output_root / ACCELERATION_EFFECT_SUMMARY_FILENAME, index=False)
        turn_effect = compute_turn_rate_effect_summary(df)
        turn_effect.to_csv(output_root / TURN_RATE_EFFECT_SUMMARY_FILENAME, index=False)
        interaction = compute_interaction_summary(df)
        interaction.to_csv(output_root / INTERACTION_SUMMARY_FILENAME, index=False)
    else:
        accel_effect = turn_effect = interaction = pd.DataFrame()
        print("\n(Skipping acceleration/turn-rate effect + interaction summaries -- require the full 9-cell grid)")

    feasibility = compute_feasibility_summary(df, cells)
    feasibility.to_csv(output_root / FEASIBILITY_SUMMARY_FILENAME, index=False)

    acquisition = compute_acquisition_summary(df, cells)
    acquisition.to_csv(output_root / ACQUISITION_SUMMARY_FILENAME, index=False)

    post_detection = compute_post_detection_summary(df, cells)
    post_detection.to_csv(output_root / POST_DETECTION_SUMMARY_FILENAME, index=False)

    failure_modes = compute_failure_mode_summary(df, cells)
    failure_modes.to_csv(output_root / FAILURE_MODE_SUMMARY_FILENAME, index=False)

    operational_summary = compute_operational_summary(df, cells)
    operational_summary.to_csv(output_root / OPERATIONAL_SUMMARY_FILENAME, index=False)

    estimator_summary = compute_estimator_summary(df, cells)
    estimator_summary.to_csv(output_root / ESTIMATOR_SUMMARY_FILENAME, index=False)

    final_summary = {
        "label": "DEFENDER-MANEUVERABILITY ROBUSTNESS GRID" if is_final else "DRY RUN (non-final seeds, diagnostic only)",
        "integrity_checks": integrity,
        "success_summary": success_summary.to_dict(orient="records"),
        "ppo_training_seed_summary": ppo_seed_summary.to_dict(orient="records"),
        "ppo_vs_lead_comparisons": ppo_vs_lead.to_dict(orient="records"),
        "estimator_comparisons": estimator_comparisons.to_dict(orient="records"),
        "acceleration_effect_summary": accel_effect.to_dict(orient="records"),
        "turn_rate_effect_summary": turn_effect.to_dict(orient="records"),
        "interaction_summary": interaction.to_dict(orient="records"),
        "feasibility_summary": feasibility.to_dict(orient="records"),
        "acquisition_summary": acquisition.to_dict(orient="records"),
        "post_detection_summary": post_detection.to_dict(orient="records"),
        "failure_mode_summary": failure_modes.to_dict(orient="records"),
        "operational_summary": operational_summary.to_dict(orient="records"),
        "estimator_summary": estimator_summary.to_dict(orient="records"),
    }
    (output_root / MANEUVERABILITY_SUMMARY_FILENAME).write_text(json.dumps(final_summary, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
