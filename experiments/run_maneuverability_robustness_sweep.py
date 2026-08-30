"""
Defender-maneuverability expanded robustness sweep -- LOCKED, ONE-TIME
evaluation of the 12 paper-facing method instances across the frozen 3x3
(defender_max_accel x defender_max_turn_rate_deg) grid on the reserved seed
range 67000..67999 (1000 matched seeds).

Scientific framing: this is a bounded-acceleration and bounded-horizontal-
turn-rate SENSITIVITY STUDY over the SAME point-mass dynamics already used
throughout this project. It does NOT introduce new dynamics and these values
are NOT "hardware-validated limits", "real aircraft specifications",
"six-DOF dynamics", or "aerodynamic constraints" -- they are simply
alternative settings of the existing constrained point-mass model.

Because the defender is co-located/stationary with the soldier until natural
detection (defender_standby_until_detection=True), changing
defender_max_accel/defender_max_turn_rate_deg affects ONLY post-detection
defender dynamics -- pre-detection acquisition trajectories must be identical
not only across the 12 methods (as in the hostile-evasion sweep) but also
ACROSS ALL 9 maneuverability cells for a matched seed. Both invariants are
verified explicitly below.

Reuses the SAME validated building blocks as the hostile-evasion sweep
(episode runner, sanitizer, frozen model table, hierarchical bootstrap
statistics, policy construction) -- only the swept parameter, seed range,
and grid-specific descriptive summaries (acceleration effect, turn-rate
effect, interaction, representative-cell contrast) are new.

Usage:
    python experiments/run_maneuverability_robustness_sweep.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv

from experiments.expanded_robustness_protocol import (
    build_method_instances,
    SCRIPTED_METHODS,
    LR_PPO_METHOD,
    TRAINING_SEEDS,
    LR_TRAINING_ROOTS,
    RESIDUAL_MAX_ANGLE_DEG,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    MANEUVERABILITY_SEED_START,
    MANEUVERABILITY_SEED_END,
    MANEUVERABILITY_ACCEL_VALUES,
    MANEUVERABILITY_TURN_RATE_VALUES,
    MANEUVERABILITY_GRID,
    NOMINAL_MANEUVERABILITY_ACCEL,
    NOMINAL_MANEUVERABILITY_TURN_RATE,
    build_maneuverability_config,
    assert_seed_in_maneuverability_range,
    assert_seed_not_yet_opened,
    assert_disjoint_from_all_prior_ranges,
)
from experiments.run_hostile_evasion_robustness_sweep import build_policy_for_instance
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode, sha256_of_file
from experiments.lead_residual_diagnostic_eval_utils import run_lr_ppo_diagnostic_episode
from experiments.run_lead_residual_diagnostic import (
    verify_model_hashes,
    compute_lead_rescue_harm,
    compute_residual_angle_summary,
    compute_residual_angle_by_outcome,
    compute_physical_validity,
)
from experiments.run_post_detection_diagnostic import (
    compute_post_detection_summary,
    audit_ground_truth_leakage,
    _get_git_commit,
    _get_git_branch,
    _get_git_status_clean,
)
from experiments.final_stats import (
    wilson_interval,
    standard_error,
    paired_bootstrap_ci,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_bootstrap_independent_families,
    hierarchical_paired_bootstrap_matched_seeds,
)

OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_robustness" / "maneuverability"
SEEDS = list(range(MANEUVERABILITY_SEED_START, MANEUVERABILITY_SEED_END + 1))
NOMINAL_CELL = (NOMINAL_MANEUVERABILITY_ACCEL, NOMINAL_MANEUVERABILITY_TURN_RATE)
RESTRICTIVE_CELL = (4.0, 45.0)
PERMISSIVE_CELL = (12.0, 135.0)


def run_full_sweep(instances, seeds: list[int], production: bool = True):
    """Runs all (cell x instance x seed) missions. Returns
    (episode_results_df, trajectories_by_cell, angle_data_by_cell).

    production=True (default, used for the real 67000-67999 run): each seed
    must fall exactly WITHIN the locked maneuverability range (positive
    membership check -- this is what authorizes opening the range).
    production=False (used for dev/smoke tasks on seeds like 123-127): each
    seed must NOT fall in ANY reserved (not-yet-opened) range. Both modes
    also enforce disjointness from all prior/already-consumed ranges
    (including the now-frozen hostile_evasion range)."""
    for s in seeds:
        if production:
            assert_seed_in_maneuverability_range(s)
        else:
            assert_seed_not_yet_opened(s)
        assert_disjoint_from_all_prior_ranges(s)

    rows = []
    trajectories_by_cell: dict[tuple[float, float], dict] = {}
    angle_data_by_cell: dict[str, dict] = {"by_instance": {}, "by_instance_seed": {}}

    total = len(MANEUVERABILITY_GRID) * len(instances)
    i = 0
    for accel, turn_rate in MANEUVERABILITY_GRID:
        trajectories = {}
        for instance in instances:
            i += 1
            estimator = "kalman" if instance.family == "ppo_kalman" else "measurement"
            config = build_maneuverability_config(estimator, accel, turn_rate)
            env = SoldierEnv(config=config)
            policy = build_policy_for_instance(instance, config)
            dt = config.dt
            print(f"[{i}/{total}] cell=({accel},{turn_rate}) {instance.display_name} ({instance.family}) -- {len(seeds)} episodes...", flush=True)
            instance_angles = []
            for seed in seeds:
                if instance.family == "lr_ppo":
                    row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                    step_angles = row.pop("step_angles")
                    instance_angles.extend(step_angles)
                    angle_data_by_cell["by_instance_seed"][((accel, turn_rate), instance.display_name, seed)] = step_angles
                else:
                    row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                trajectory = row.pop("pre_detection_trajectory")
                trajectories[(seed, instance.display_name)] = trajectory
                row["method"] = instance.name
                row["instance"] = instance.display_name
                row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
                row["defender_max_accel"] = accel
                row["defender_max_turn_rate_deg"] = turn_rate
                rows.append(row)
            if instance.family == "lr_ppo":
                key = (accel, turn_rate), instance.display_name
                angle_data_by_cell["by_instance"][key] = angle_data_by_cell["by_instance"].get(key, []) + instance_angles
            env.close()
        trajectories_by_cell[(accel, turn_rate)] = trajectories

    df = pd.DataFrame(rows)
    return df, trajectories_by_cell, angle_data_by_cell


def compute_acquisition_fairness_within_cell(trajectories_by_cell: dict, instance_names: list[str], seeds: list[int]) -> pd.DataFrame:
    """For every (cell, seed), all 12 methods must have identical pre-detection trajectories."""
    rows = []
    for cell, trajectories in trajectories_by_cell.items():
        for seed in seeds:
            trajs = {name: trajectories[(seed, name)] for name in instance_names}
            ref_name = instance_names[0]
            ref_traj = trajs[ref_name]
            detection_steps = {name: (t[-1]["step"] if t[-1]["enemy_detected"] else None) for name, t in trajs.items()}
            same_length = all(len(trajs[name]) == len(ref_traj) for name in instance_names)
            max_discrepancy = 0.0
            mismatch = not same_length
            if same_length:
                for idx in range(len(ref_traj)):
                    ref_state = ref_traj[idx]
                    for name in instance_names[1:]:
                        s = trajs[name][idx]
                        for field in ("soldier_pos", "defender_pos", "defender_vel", "enemy_pos", "enemy_vel"):
                            diff = float(np.max(np.abs(ref_state[field] - s[field])))
                            max_discrepancy = max(max_discrepancy, diff)
                        if ref_state["enemy_detected"] != s["enemy_detected"]:
                            mismatch = True
            if max_discrepancy > 1e-6:
                mismatch = True
            final_state = ref_traj[-1]
            sep = float(np.linalg.norm(final_state["defender_pos"] - final_state["soldier_pos"]))
            rows.append({
                "defender_max_accel": cell[0], "defender_max_turn_rate_deg": cell[1], "environment_seed": seed,
                "detection_step": detection_steps[ref_name],
                "all_detection_steps_equal": len(set(detection_steps.values())) == 1,
                "max_physical_discrepancy": max_discrepancy,
                "defender_soldier_separation_at_detection": sep,
                "defender_speed_at_detection": float(np.linalg.norm(final_state["defender_vel"])),
                "mismatch": mismatch,
            })
    return pd.DataFrame(rows)


def compute_cross_cell_acquisition_invariance(trajectories_by_cell: dict, seeds: list[int], cells: list[tuple], reference_instance: str = "Greedy") -> pd.DataFrame:
    """Stronger invariant: acquisition history (pre-detection) for a fixed
    seed must be identical ACROSS ALL 9 maneuverability cells, since the
    defender is stationary/co-located before detection regardless of its
    post-detection accel/turn-rate limits. Uses one representative instance
    (Greedy) per seed since within-cell fairness already proves the 12
    instances agree with each other at any single cell (transitivity)."""
    rows = []
    ref_cell = cells[0]
    for seed in seeds:
        ref_traj = trajectories_by_cell[ref_cell][(seed, reference_instance)]
        max_discrepancy = 0.0
        mismatch = False
        detection_steps = {}
        for cell in cells:
            traj = trajectories_by_cell[cell][(seed, reference_instance)]
            detection_steps[cell] = traj[-1]["step"] if traj[-1]["enemy_detected"] else None
            if len(traj) != len(ref_traj):
                mismatch = True
                continue
            for idx in range(len(ref_traj)):
                ref_state = ref_traj[idx]
                s = traj[idx]
                for field in ("soldier_pos", "defender_pos", "defender_vel", "enemy_pos", "enemy_vel"):
                    diff = float(np.max(np.abs(ref_state[field] - s[field])))
                    max_discrepancy = max(max_discrepancy, diff)
                if ref_state["enemy_detected"] != s["enemy_detected"]:
                    mismatch = True
        if max_discrepancy > 1e-6:
            mismatch = True
        rows.append({
            "environment_seed": seed,
            "all_cells_detection_step_equal": len(set(detection_steps.values())) == 1,
            "max_physical_discrepancy_across_cells": max_discrepancy,
            "mismatch": mismatch,
        })
    return pd.DataFrame(rows)


def _success_vector(df: pd.DataFrame, cell: tuple, method: str, training_seed=None) -> np.ndarray:
    sub = df[(df["defender_max_accel"] == cell[0]) & (df["defender_max_turn_rate_deg"] == cell[1]) & (df["method"] == method)]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, cell: tuple, method: str, training_seeds) -> np.ndarray:
    return np.stack([_success_vector(df, cell, method, s) for s in training_seeds])


def compute_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        for method in SCRIPTED_METHODS:
            sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn_rate) & (df["method"] == method)]
            n = len(sub)
            successes = int(sub["success"].sum())
            lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
            rows.append({
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate, "instance": method, "method": method,
                "n": n, "success_count": successes, "success_rate": successes / n, "standard_error": standard_error(successes, n),
                "ci_low": lo, "ci_high": hi,
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_learned_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            for seed in seeds:
                sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn_rate) & (df["method"] == method) & (df["training_seed"] == seed)]
                n = len(sub)
                successes = int(sub["success"].sum())
                lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
                rows.append({
                    "defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate,
                    "instance": sub["instance"].iloc[0], "method": method, "training_seed": seed,
                    "n": n, "success_count": successes, "success_rate": successes / n,
                    "standard_error": standard_error(successes, n), "ci_low": lo, "ci_high": hi,
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def compute_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            matrix = _success_matrix(df, (accel, turn_rate), method, seeds)
            rates = matrix.mean(axis=1)
            observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate, "method": method, "n_training_seeds": len(seeds),
                "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
                "min_seed_success": float(rates.min()), "max_seed_success": float(rates.max()),
                "range": float(rates.max() - rates.min()),
                "hierarchical_ci_low": lo, "hierarchical_ci_high": hi,
            })
    return pd.DataFrame(rows)


def compute_primary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        cell = (accel, turn_rate)
        lead_vec = _success_vector(df, cell, "lead")
        lr_matrix = _success_matrix(df, cell, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        ppo_matrix = _success_matrix(df, cell, "ppo", TRAINING_SEEDS)
        for name, method_a, method_b in PRIMARY_COMPARISONS:
            if method_b == "lead":
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            else:
                observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate, "comparison": name,
                         "diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_secondary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        cell = (accel, turn_rate)
        greedy_vec = _success_vector(df, cell, "greedy")
        pn_vec = _success_vector(df, cell, "pn")
        lead_vec = _success_vector(df, cell, "lead")
        ppo_matrix = _success_matrix(df, cell, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, cell, "ppo_kalman", TRAINING_SEEDS)
        for name, method_a, method_b in SECONDARY_COMPARISONS:
            if method_a == "ppo_kalman" and method_b == "ppo":
                observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppok_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            elif method_a == "ppo":
                scripted_vec = {"greedy": greedy_vec, "pn": pn_vec, "lead": lead_vec}[method_b]
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_matrix, scripted_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            else:
                vec_a = {"greedy": greedy_vec, "pn": pn_vec, "lead": lead_vec}[method_a]
                vec_b = {"greedy": greedy_vec, "pn": pn_vec, "lead": lead_vec}[method_b]
                observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate, "comparison": name,
                         "diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_within_method_maneuverability_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lead_nominal = _success_vector(df, NOMINAL_CELL, "lead")
    ppo_nominal = _success_matrix(df, NOMINAL_CELL, "ppo", TRAINING_SEEDS)
    lr_nominal = _success_matrix(df, NOMINAL_CELL, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    for accel, turn_rate in MANEUVERABILITY_GRID:
        cell = (accel, turn_rate)
        if cell == NOMINAL_CELL:
            continue
        lead_vec = _success_vector(df, cell, "lead")
        observed, lo, hi = paired_bootstrap_ci(lead_vec, lead_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lead", "defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate,
                     "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        ppo_matrix = _success_matrix(df, cell, "ppo", TRAINING_SEEDS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppo_matrix, ppo_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "ppo", "defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate,
                     "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        lr_matrix = _success_matrix(df, cell, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(lr_matrix, lr_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lr_ppo", "defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate,
                     "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})
    return pd.DataFrame(rows)


def _method_success_rate(df: pd.DataFrame, cell: tuple, method: str) -> float:
    if method == "lead":
        return float(_success_vector(df, cell, "lead").mean())
    if method == "ppo":
        return float(_success_matrix(df, cell, "ppo", TRAINING_SEEDS).mean())
    if method == LR_PPO_METHOD:
        return float(_success_matrix(df, cell, LR_PPO_METHOD, LR_TRAINING_ROOTS).mean())
    raise ValueError(method)


def compute_acceleration_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    """For each fixed turn rate, success as accel goes 4 -> 8 -> 12, for Lead/PPO/LR-PPO."""
    rows = []
    for turn_rate in MANEUVERABILITY_TURN_RATE_VALUES:
        for method in ("lead", "ppo", LR_PPO_METHOD):
            rates = [_method_success_rate(df, (a, turn_rate), method) for a in MANEUVERABILITY_ACCEL_VALUES]
            rows.append({
                "defender_max_turn_rate_deg": turn_rate, "method": method,
                "success_at_accel_4": rates[0], "success_at_accel_8": rates[1], "success_at_accel_12": rates[2],
                "diff_8_minus_4": rates[1] - rates[0], "diff_12_minus_8": rates[2] - rates[1],
            })
    return pd.DataFrame(rows)


def compute_turn_rate_effect_summary(df: pd.DataFrame) -> pd.DataFrame:
    """For each fixed accel, success as turn authority goes 45 -> 90 -> 135, for Lead/PPO/LR-PPO."""
    rows = []
    for accel in MANEUVERABILITY_ACCEL_VALUES:
        for method in ("lead", "ppo", LR_PPO_METHOD):
            rates = [_method_success_rate(df, (accel, t), method) for t in MANEUVERABILITY_TURN_RATE_VALUES]
            rows.append({
                "defender_max_accel": accel, "method": method,
                "success_at_turn_45": rates[0], "success_at_turn_90": rates[1], "success_at_turn_135": rates[2],
                "diff_90_minus_45": rates[1] - rates[0], "diff_135_minus_90": rates[2] - rates[1],
            })
    return pd.DataFrame(rows)


def compute_interaction_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Descriptive/exploratory 3x3 matrix + restrictive-vs-permissive-corner
    difference-in-differences for Lead/PPO/LR-PPO. NOT a formal inferential
    interaction test (none was pre-specified)."""
    rows = []
    for method in ("lead", "ppo", LR_PPO_METHOD):
        matrix = {(a, t): _method_success_rate(df, (a, t), method) for a, t in MANEUVERABILITY_GRID}
        dd = (matrix[(12.0, 135.0)] - matrix[(4.0, 135.0)]) - (matrix[(12.0, 45.0)] - matrix[(4.0, 45.0)])
        row = {"method": method, "interaction_diff_in_diff": dd}
        for a in MANEUVERABILITY_ACCEL_VALUES:
            for t in MANEUVERABILITY_TURN_RATE_VALUES:
                row[f"success_a{a}_t{t}"] = matrix[(a, t)]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_representative_cell_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Pre-specified RESTRICTIVE (4,45) / NOMINAL (8,90) / PERMISSIVE (12,135)
    contrast, with hierarchical intervals for the primary comparisons."""
    rows = []
    for label, cell in (("restrictive", RESTRICTIVE_CELL), ("nominal", NOMINAL_CELL), ("permissive", PERMISSIVE_CELL)):
        lead_rate = _method_success_rate(df, cell, "lead")
        ppo_rate = _method_success_rate(df, cell, "ppo")
        lr_rate = _method_success_rate(df, cell, LR_PPO_METHOD)
        lead_vec = _success_vector(df, cell, "lead")
        lr_matrix = _success_matrix(df, cell, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        ppo_matrix = _success_matrix(df, cell, "ppo", TRAINING_SEEDS)
        d_lead, lo_lead, hi_lead = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        d_ppo, lo_ppo, hi_ppo = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({
            "cell_label": label, "defender_max_accel": cell[0], "defender_max_turn_rate_deg": cell[1],
            "lead_success": lead_rate, "ppo_family_success": ppo_rate, "lr_ppo_family_success": lr_rate,
            "lr_ppo_minus_lead_pp": d_lead * 100, "lr_ppo_minus_lead_ci_low_pp": lo_lead * 100, "lr_ppo_minus_lead_ci_high_pp": hi_lead * 100,
            "lr_ppo_minus_ppo_pp": d_ppo * 100, "lr_ppo_minus_ppo_ci_low_pp": lo_ppo * 100, "lr_ppo_minus_ppo_ci_high_pp": hi_ppo * 100,
        })
    return pd.DataFrame(rows)


def compute_lead_rescue_harm_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn_rate)]
        r = compute_lead_rescue_harm(sub)
        r.insert(0, "defender_max_accel", accel)
        r.insert(1, "defender_max_turn_rate_deg", turn_rate)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_summary_by_cell(angle_data_by_instance: dict) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        cell = (accel, turn_rate)
        scoped = {name: vals for (c, name), vals in angle_data_by_instance.items() if c == cell}
        r = compute_residual_angle_summary(scoped)
        r.insert(0, "defender_max_accel", accel)
        r.insert(1, "defender_max_turn_rate_deg", turn_rate)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_by_outcome_by_cell(df: pd.DataFrame, angle_data_by_instance_seed: dict) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        cell = (accel, turn_rate)
        sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn_rate)]
        scoped = {(name, seed): vals for (c, name, seed), vals in angle_data_by_instance_seed.items() if c == cell}
        r = compute_residual_angle_by_outcome(sub, scoped)
        r.insert(0, "defender_max_accel", accel)
        r.insert(1, "defender_max_turn_rate_deg", turn_rate)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_failure_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn_rate)]
        for instance in sub["instance"].unique():
            inst_sub = sub[sub["instance"] == instance]
            n = len(inst_sub)
            rows.append({
                "defender_max_accel": accel, "defender_max_turn_rate_deg": turn_rate, "instance": instance, "n": n,
                "safe_intercept_rate": float(inst_sub["success"].mean()),
                "soldier_caught_rate": float(inst_sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(inst_sub["unsafe_intercept"].mean()),
                "timeout_rate": float(inst_sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_post_detection_summary_by_cell(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for accel, turn_rate in MANEUVERABILITY_GRID:
        sub = df[(df["defender_max_accel"] == accel) & (df["defender_max_turn_rate_deg"] == turn_rate)]
        r = compute_post_detection_summary(sub)
        r.insert(0, "defender_max_accel", accel)
        r.insert(1, "defender_max_turn_rate_deg", turn_rate)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_training_seed_stability_by_cell(family_summary: pd.DataFrame) -> pd.DataFrame:
    return family_summary[["defender_max_accel", "defender_max_turn_rate_deg", "method", "between_seed_sd"]].copy()


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commit_before = _get_git_commit()
    branch = _get_git_branch()
    tree_clean_before = _get_git_status_clean()
    print(f"branch={branch} commit_before={commit_before} tree_clean_before={tree_clean_before}")

    print("\nVerifying model hashes (9 learned models)...")
    hash_results_before = verify_model_hashes()
    print(json.dumps(hash_results_before, indent=2))
    assert all(v["match"] for v in hash_results_before.values()), "MODEL HASH MISMATCH -- STOP BEFORE SEED 67000"

    print("\nGround-truth leakage audit...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_method_instances()
    instance_names = [i.display_name for i in instances]
    assert len(instances) == 12

    print(f"\nRunning MANEUVERABILITY robustness sweep: {len(MANEUVERABILITY_GRID)} cells x "
          f"{len(instances)} instances x {len(SEEDS)} seeds (seeds {SEEDS[0]}..{SEEDS[-1]})...")
    df, trajectories_by_cell, angle_data = run_full_sweep(instances, SEEDS)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)
    expected_rows = len(MANEUVERABILITY_GRID) * len(instances) * len(SEEDS)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    print("\nComputing within-cell acquisition fairness (9 cells x 1000 seeds = 9000 combinations)...")
    fairness_df = compute_acquisition_fairness_within_cell(trajectories_by_cell, instance_names, SEEDS)
    fairness_df.to_csv(OUTPUT_ROOT / "acquisition_fairness.csv", index=False)
    n_mismatched = int(fairness_df["mismatch"].sum())
    max_discrepancy_overall = float(fairness_df["max_physical_discrepancy"].max())
    print(f"  within-cell combinations checked: {len(fairness_df)}, mismatched: {n_mismatched}, max discrepancy: {max_discrepancy_overall}")

    print("\nVerifying STRONGER cross-cell acquisition invariance (1000 seeds x 9 cells)...")
    cross_cell_df = compute_cross_cell_acquisition_invariance(trajectories_by_cell, SEEDS, list(MANEUVERABILITY_GRID))
    cross_cell_df.to_csv(OUTPUT_ROOT / "cross_cell_acquisition_invariance.csv", index=False)
    n_cross_mismatched = int(cross_cell_df["mismatch"].sum())
    max_cross_discrepancy = float(cross_cell_df["max_physical_discrepancy_across_cells"].max())
    print(f"  cross-cell seeds checked: {len(cross_cell_df)}, mismatched: {n_cross_mismatched}, max discrepancy: {max_cross_discrepancy}")

    classical_summary = compute_classical_summary(df)
    classical_summary.to_csv(OUTPUT_ROOT / "classical_summary.csv", index=False)

    learned_seed_summary = compute_learned_seed_summary(df)
    learned_seed_summary.to_csv(OUTPUT_ROOT / "learned_seed_summary.csv", index=False)

    family_summary = compute_family_summary(df)
    family_summary.to_csv(OUTPUT_ROOT / "family_summary.csv", index=False)

    primary_comparisons = compute_primary_comparisons(df)
    primary_comparisons.to_csv(OUTPUT_ROOT / "primary_comparisons.csv", index=False)

    secondary_comparisons = compute_secondary_comparisons(df)
    secondary_comparisons.to_csv(OUTPUT_ROOT / "secondary_comparisons.csv", index=False)

    within_method_effects = compute_within_method_maneuverability_effects(df)
    within_method_effects.to_csv(OUTPUT_ROOT / "within_method_maneuverability_effects.csv", index=False)

    acceleration_effect_summary = compute_acceleration_effect_summary(df)
    acceleration_effect_summary.to_csv(OUTPUT_ROOT / "acceleration_effect_summary.csv", index=False)

    turn_rate_effect_summary = compute_turn_rate_effect_summary(df)
    turn_rate_effect_summary.to_csv(OUTPUT_ROOT / "turn_rate_effect_summary.csv", index=False)

    interaction_summary = compute_interaction_summary(df)
    interaction_summary.to_csv(OUTPUT_ROOT / "interaction_summary.csv", index=False)

    representative_cell_summary = compute_representative_cell_summary(df)
    representative_cell_summary.to_csv(OUTPUT_ROOT / "representative_cell_summary.csv", index=False)

    rescue_harm = compute_lead_rescue_harm_by_cell(df)
    rescue_harm.to_csv(OUTPUT_ROOT / "lead_rescue_harm.csv", index=False)

    angle_summary = compute_residual_angle_summary_by_cell(angle_data["by_instance"])
    angle_summary.to_csv(OUTPUT_ROOT / "residual_angle_summary.csv", index=False)

    angle_by_outcome = compute_residual_angle_by_outcome_by_cell(df, angle_data["by_instance_seed"])
    angle_by_outcome.to_csv(OUTPUT_ROOT / "residual_angle_by_outcome.csv", index=False)

    failure_mode_summary = compute_failure_mode_summary(df)
    failure_mode_summary.to_csv(OUTPUT_ROOT / "failure_mode_summary.csv", index=False)

    post_detection_summary = compute_post_detection_summary_by_cell(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "post_detection_summary.csv", index=False)

    training_seed_stability = compute_training_seed_stability_by_cell(family_summary)
    training_seed_stability.to_csv(OUTPUT_ROOT / "training_seed_stability.csv", index=False)

    physical_validity = compute_physical_validity(df)
    (OUTPUT_ROOT / "physical_validity.json").write_text(json.dumps(physical_validity, indent=2, default=str))
    if physical_validity["total_violations"] > 0:
        print("\n*** PHYSICAL LIMIT VIOLATION DETECTED -- STOP PER PROTOCOL. DO NOT PATCH AND RERUN. ***")

    (OUTPUT_ROOT / "ground_truth_audit.json").write_text(json.dumps(leakage_audit, indent=2, default=str))

    total_violations = int(df["physical_limit_violations"].sum())

    commit_after = _get_git_commit()
    tree_clean_after = _get_git_status_clean()
    hash_results_after = verify_model_hashes()
    hashes_identical = all(hash_results_before[k]["actual"] == hash_results_after[k]["actual"] for k in hash_results_before)

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "maneuverability_source_commit_before": commit_before,
        "maneuverability_source_commit_after": commit_after,
        "source_immutable": commit_before == commit_after,
        "branch": branch,
        "working_tree_clean_before": tree_clean_before,
        "working_tree_clean_after": tree_clean_after,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "n_episodes_per_condition_per_instance": len(SEEDS),
        "maneuverability_grid": list(MANEUVERABILITY_GRID),
        "nominal_cell": list(NOMINAL_CELL),
        "restrictive_cell": list(RESTRICTIVE_CELL),
        "permissive_cell": list(PERMISSIVE_CELL),
        "n_instances": len(instances),
        "instances": instance_names,
        "expected_total_mission_evaluations": expected_rows,
        "actual_total_mission_evaluations": len(df),
        "model_hash_verification_before": hash_results_before,
        "model_hash_verification_after": hash_results_after,
        "model_hashes_identical_before_after": hashes_identical,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness_within_cell": {
            "cell_seed_combinations_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall, "all_pass": n_mismatched == 0,
        },
        "acquisition_invariance_cross_cell": {
            "seeds_checked": len(cross_cell_df), "n_mismatched": n_cross_mismatched,
            "max_physical_discrepancy": max_cross_discrepancy, "all_pass": n_cross_mismatched == 0,
        },
        "physical_validity": physical_validity,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": [c[0] for c in PRIMARY_COMPARISONS],
        "residual_max_angle_deg": RESIDUAL_MAX_ANGLE_DEG,
        "is_robustness_sweep": True,
        "sweep_name": "maneuverability",
        "scientific_framing": "bounded-acceleration and bounded-horizontal-turn-rate sensitivity study over existing constrained point-mass dynamics; NOT hardware-validated limits, real aircraft specifications, six-DOF dynamics, or aerodynamic constraints.",
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUTPUT_ROOT / "model_hashes.json").write_text(json.dumps(hash_results_before, indent=2, default=str))

    core_files = [
        "episode_results.csv", "classical_summary.csv", "learned_seed_summary.csv",
        "family_summary.csv", "primary_comparisons.csv", "secondary_comparisons.csv",
        "lead_rescue_harm.csv", "representative_cell_summary.csv", "protocol_manifest.json",
    ]
    evidence_hashes = {f: sha256_of_file(OUTPUT_ROOT / f) for f in core_files}
    (OUTPUT_ROOT / "evidence_hashes.json").write_text(json.dumps(evidence_hashes, indent=2))

    print(f"\nSource immutable: {commit_before == commit_after}")
    print(f"Model hashes identical before/after: {hashes_identical}")
    print(f"Within-cell acquisition fairness: {n_mismatched} mismatched (expected 0)")
    print(f"Cross-cell acquisition invariance: {n_cross_mismatched} mismatched (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
