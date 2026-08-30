"""
Measurement-noise expanded robustness sweep -- LOCKED, ONE-TIME evaluation
of the 12 paper-facing method instances across the frozen 6-point
measurement_var grid on the reserved seed range 68000..68999 (1000 matched
seeds).

Scientific framing: this evaluates ZERO-SHOT sensor-noise generalization of
the FROZEN controllers/filter. measurement_var is the ONLY swept behavioral
parameter; process_var=1.0 (Kalman process-noise assumption) is held fixed
at every condition -- this is NOT a Kalman-retuning study. Estimator mode
(Direct vs Kalman) is fixed by controller instance, not itself a sweep axis.

Three distinct scientific questions are addressed (do not conflate):
  A. Does Kalman filtering reduce target-state estimation error?
     -> tracking_error_summary.csv (pooled mean error / RMSE, Direct vs Kalman,
        vs. theoretical raw-noise benchmarks from experiments/estimator_stats.py)
  B. Does lower tracking error translate into higher PPO interception success?
     -> estimator_comparisons.csv (PPO-Kalman family - PPO family, per variance)
  C. Does LR-PPO's predictive-Lead structure remain useful as sensor
     uncertainty increases?
     -> primary_comparisons.csv (LR-PPO family - Lead / - PPO family, per variance)

Unlike the maneuverability sweep, measurement noise CAN legitimately change
acquisition/detection timing across DIFFERENT variance conditions (noisy
sensing affects detection itself) -- fairness is required only ACROSS THE
12 METHODS at a fixed (variance, seed), never across variances.

Reuses the SAME validated building blocks as the hostile-evasion and
maneuverability sweeps (episode runner, sanitizer, frozen model table,
hierarchical bootstrap statistics, policy construction) plus the existing
experiments/estimator_stats.py pooling/theoretical-benchmark helpers for
tracking-error analysis -- only the swept parameter, seed range, and the
noise-specific analyses (estimator ablation, tracking error, representative
noise contrast) are new.

Usage:
    python experiments/run_measurement_noise_robustness_sweep.py
"""

from __future__ import annotations

import json
import math
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
    MEASUREMENT_NOISE_SEED_START,
    MEASUREMENT_NOISE_SEED_END,
    MEASUREMENT_VAR_VALUES,
    NOMINAL_MEASUREMENT_VAR,
    FIXED_PROCESS_VAR,
    build_measurement_noise_config,
    assert_process_var_fixed,
    assert_seed_in_measurement_noise_range,
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
from experiments.estimator_stats import (
    theoretical_raw_measurement_mean_error,
    theoretical_raw_measurement_rmse,
    sample_weighted_mean_error,
    sample_weighted_rmse,
    episode_weighted_mean_error,
    mean_episode_rmse,
)

OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_robustness" / "measurement_noise"
SEEDS = list(range(MEASUREMENT_NOISE_SEED_START, MEASUREMENT_NOISE_SEED_END + 1))
LOW_NOISE_VAR = 0.05
HIGH_NOISE_VAR = 4.00
DIRECT_TRACK_METHODS = ("greedy", "pn", "lead", "ppo", LR_PPO_METHOD)
KALMAN_TRACK_METHODS = ("ppo_kalman",)


def run_full_sweep(instances, seeds: list[int], production: bool = True):
    """Runs all (measurement_var x instance x seed) missions. Returns
    (episode_results_df, trajectories_by_var, angle_data_by_var).

    production=True (default, used for the real 68000-68999 run): each seed
    must fall exactly WITHIN the locked measurement_noise range (positive
    membership check). production=False (dev/smoke on seeds like 123-127):
    each seed must NOT fall in ANY reserved (not-yet-opened) range. Both
    modes enforce disjointness from all prior/already-consumed ranges
    (hostile_evasion, maneuverability -- both now permanently frozen)."""
    for s in seeds:
        if production:
            assert_seed_in_measurement_noise_range(s)
        else:
            assert_seed_not_yet_opened(s)
        assert_disjoint_from_all_prior_ranges(s)

    rows = []
    trajectories_by_var: dict[float, dict] = {}
    angle_data_by_var: dict[str, dict] = {"by_instance": {}, "by_instance_seed": {}}

    total = len(MEASUREMENT_VAR_VALUES) * len(instances)
    i = 0
    for var in MEASUREMENT_VAR_VALUES:
        trajectories = {}
        for instance in instances:
            i += 1
            estimator = "kalman" if instance.family == "ppo_kalman" else "measurement"
            config = build_measurement_noise_config(estimator, var)
            assert_process_var_fixed(config)
            env = SoldierEnv(config=config)
            policy = build_policy_for_instance(instance, config)
            dt = config.dt
            print(f"[{i}/{total}] var={var} {instance.display_name} ({instance.family}) -- {len(seeds)} episodes...", flush=True)
            instance_angles = []
            for seed in seeds:
                if instance.family == "lr_ppo":
                    row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                    step_angles = row.pop("step_angles")
                    instance_angles.extend(step_angles)
                    angle_data_by_var["by_instance_seed"][(var, instance.display_name, seed)] = step_angles
                else:
                    row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                trajectory = row.pop("pre_detection_trajectory")
                trajectories[(seed, instance.display_name)] = trajectory
                row["method"] = instance.name
                row["instance"] = instance.display_name
                row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
                row["measurement_var"] = var
                rows.append(row)
            if instance.family == "lr_ppo":
                key = (var, instance.display_name)
                angle_data_by_var["by_instance"][key] = angle_data_by_var["by_instance"].get(key, []) + instance_angles
            env.close()
        trajectories_by_var[var] = trajectories

    df = pd.DataFrame(rows)
    return df, trajectories_by_var, angle_data_by_var


def compute_acquisition_fairness_all_vars(trajectories_by_var: dict, instance_names: list[str], seeds: list[int]) -> pd.DataFrame:
    """Fairness required ONLY across the 12 methods at a fixed (variance, seed)
    -- NOT across different variances (noise legitimately changes acquisition
    timing itself)."""
    rows = []
    for var, trajectories in trajectories_by_var.items():
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
                "measurement_var": var, "environment_seed": seed,
                "detection_step": detection_steps[ref_name],
                "all_detection_steps_equal": len(set(detection_steps.values())) == 1,
                "max_physical_discrepancy": max_discrepancy,
                "defender_soldier_separation_at_detection": sep,
                "defender_speed_at_detection": float(np.linalg.norm(final_state["defender_vel"])),
                "mismatch": mismatch,
            })
    return pd.DataFrame(rows)


def _success_vector(df: pd.DataFrame, var: float, method: str, training_seed=None) -> np.ndarray:
    sub = df[(df["measurement_var"] == var) & (df["method"] == method)]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, var: float, method: str, training_seeds) -> np.ndarray:
    return np.stack([_success_vector(df, var, method, s) for s in training_seeds])


def compute_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        for method in SCRIPTED_METHODS:
            sub = df[(df["measurement_var"] == var) & (df["method"] == method)]
            n = len(sub)
            successes = int(sub["success"].sum())
            lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
            rows.append({
                "measurement_var": var, "instance": method, "method": method, "n": n, "success_count": successes,
                "success_rate": successes / n, "standard_error": standard_error(successes, n),
                "ci_low": lo, "ci_high": hi,
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_learned_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            for seed in seeds:
                sub = df[(df["measurement_var"] == var) & (df["method"] == method) & (df["training_seed"] == seed)]
                n = len(sub)
                successes = int(sub["success"].sum())
                lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
                rows.append({
                    "measurement_var": var, "instance": sub["instance"].iloc[0], "method": method, "training_seed": seed,
                    "n": n, "success_count": successes, "success_rate": successes / n,
                    "standard_error": standard_error(successes, n), "ci_low": lo, "ci_high": hi,
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def compute_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            matrix = _success_matrix(df, var, method, seeds)
            rates = matrix.mean(axis=1)
            observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({
                "measurement_var": var, "method": method, "n_training_seeds": len(seeds),
                "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
                "min_seed_success": float(rates.min()), "max_seed_success": float(rates.max()),
                "range": float(rates.max() - rates.min()),
                "hierarchical_ci_low": lo, "hierarchical_ci_high": hi,
            })
    return pd.DataFrame(rows)


def compute_primary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        lead_vec = _success_vector(df, var, "lead")
        lr_matrix = _success_matrix(df, var, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        ppo_matrix = _success_matrix(df, var, "ppo", TRAINING_SEEDS)
        for name, method_a, method_b in PRIMARY_COMPARISONS:
            if method_b == "lead":
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            else:
                observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"measurement_var": var, "comparison": name, "diff_pp": observed * 100,
                         "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_secondary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        greedy_vec = _success_vector(df, var, "greedy")
        pn_vec = _success_vector(df, var, "pn")
        lead_vec = _success_vector(df, var, "lead")
        ppo_matrix = _success_matrix(df, var, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, var, "ppo_kalman", TRAINING_SEEDS)
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
            rows.append({"measurement_var": var, "comparison": name, "diff_pp": observed * 100,
                         "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_estimator_ablation(df: pd.DataFrame) -> pd.DataFrame:
    """Item 21: PPO-Kalman family - PPO family (hierarchical, matched seeds)
    plus per-root descriptive-only comparisons (PPOK45 vs PPO45, etc.).
    Do not interpret any single condition as a general Kalman-benefit claim."""
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        ppo_matrix = _success_matrix(df, var, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, var, "ppo_kalman", TRAINING_SEEDS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppok_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"measurement_var": var, "comparison": "PPO-Kalman family - PPO family", "training_seed": None,
                     "diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
        for seed in TRAINING_SEEDS:
            ppo_vec = _success_vector(df, var, "ppo", seed)
            ppok_vec = _success_vector(df, var, "ppo_kalman", seed)
            observed, lo, hi = paired_bootstrap_ci(ppok_vec, ppo_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"measurement_var": var, "comparison": f"PPOK{seed} vs PPO{seed} (descriptive, same root)", "training_seed": seed,
                         "diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_tracking_error_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 22/23-A: pooled (sample-weighted) mean error and RMSE for the
    Direct track (greedy/pn/lead/ppo/lr_ppo pooled) and the Kalman track
    (ppo_kalman pooled), vs. theoretical raw-noise benchmarks (which apply
    ONLY to the unfiltered/Direct track)."""
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        sigma = math.sqrt(var)
        for track, methods in (("direct", DIRECT_TRACK_METHODS), ("kalman", KALMAN_TRACK_METHODS)):
            sub = df[(df["measurement_var"] == var) & (df["method"].isin(methods))]
            sub = sub[sub["n_tracking_samples"] > 0]
            n = sub["n_tracking_samples"].to_numpy(dtype=np.float64)
            m = sub["mean_tracking_error"].to_numpy(dtype=np.float64)
            r = sub["tracking_rmse"].to_numpy(dtype=np.float64)
            rows.append({
                "measurement_var": var, "sigma": sigma, "track": track, "n_episodes": len(sub),
                "n_pooled_timestep_samples": int(n.sum()),
                "sample_weighted_mean_error": sample_weighted_mean_error(n, m),
                "sample_weighted_rmse": sample_weighted_rmse(n, r),
                "episode_weighted_mean_error": episode_weighted_mean_error(m),
                "mean_episode_rmse_NOT_pooled_rmse": mean_episode_rmse(r),
                "theoretical_raw_noise_mean_error": theoretical_raw_measurement_mean_error(var),
                "theoretical_raw_noise_rmse": theoretical_raw_measurement_rmse(var),
            })
    return pd.DataFrame(rows)


def compute_within_method_noise_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lead_nominal = _success_vector(df, NOMINAL_MEASUREMENT_VAR, "lead")
    ppo_nominal = _success_matrix(df, NOMINAL_MEASUREMENT_VAR, "ppo", TRAINING_SEEDS)
    ppok_nominal = _success_matrix(df, NOMINAL_MEASUREMENT_VAR, "ppo_kalman", TRAINING_SEEDS)
    lr_nominal = _success_matrix(df, NOMINAL_MEASUREMENT_VAR, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    for var in MEASUREMENT_VAR_VALUES:
        if var == NOMINAL_MEASUREMENT_VAR:
            continue
        lead_vec = _success_vector(df, var, "lead")
        observed, lo, hi = paired_bootstrap_ci(lead_vec, lead_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lead", "measurement_var": var, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        ppo_matrix = _success_matrix(df, var, "ppo", TRAINING_SEEDS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppo_matrix, ppo_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "ppo", "measurement_var": var, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        ppok_matrix = _success_matrix(df, var, "ppo_kalman", TRAINING_SEEDS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppok_matrix, ppok_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "ppo_kalman", "measurement_var": var, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        lr_matrix = _success_matrix(df, var, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(lr_matrix, lr_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lr_ppo", "measurement_var": var, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})
    return pd.DataFrame(rows)


def compute_representative_noise_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 26: pre-specified LOW (0.05) / NOMINAL (0.50) / HIGH (4.00)
    contrast for Lead/PPO/PPO-Kalman/LR-PPO with frozen comparison intervals."""
    rows = []
    for label, var in (("low_noise", LOW_NOISE_VAR), ("nominal", NOMINAL_MEASUREMENT_VAR), ("high_noise", HIGH_NOISE_VAR)):
        lead_vec = _success_vector(df, var, "lead")
        ppo_matrix = _success_matrix(df, var, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, var, "ppo_kalman", TRAINING_SEEDS)
        lr_matrix = _success_matrix(df, var, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        d_lead, lo_lead, hi_lead = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        d_ppo, lo_ppo, hi_ppo = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        d_est, lo_est, hi_est, _ = hierarchical_paired_bootstrap_matched_seeds(ppok_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({
            "condition_label": label, "measurement_var": var, "sigma": math.sqrt(var),
            "lead_success": float(lead_vec.mean()), "ppo_family_success": float(ppo_matrix.mean()),
            "ppo_kalman_family_success": float(ppok_matrix.mean()), "lr_ppo_family_success": float(lr_matrix.mean()),
            "lr_ppo_minus_lead_pp": d_lead * 100, "lr_ppo_minus_lead_ci_low_pp": lo_lead * 100, "lr_ppo_minus_lead_ci_high_pp": hi_lead * 100,
            "lr_ppo_minus_ppo_pp": d_ppo * 100, "lr_ppo_minus_ppo_ci_low_pp": lo_ppo * 100, "lr_ppo_minus_ppo_ci_high_pp": hi_ppo * 100,
            "ppo_kalman_minus_ppo_pp": d_est * 100, "ppo_kalman_minus_ppo_ci_low_pp": lo_est * 100, "ppo_kalman_minus_ppo_ci_high_pp": hi_est * 100,
        })
    return pd.DataFrame(rows)


def compute_lead_rescue_harm_by_var(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        sub = df[df["measurement_var"] == var]
        r = compute_lead_rescue_harm(sub)
        r.insert(0, "measurement_var", var)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_summary_by_var(angle_data_by_instance: dict) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        scoped = {name: vals for (v, name), vals in angle_data_by_instance.items() if v == var}
        r = compute_residual_angle_summary(scoped)
        r.insert(0, "measurement_var", var)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_by_outcome_by_var(df: pd.DataFrame, angle_data_by_instance_seed: dict) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        sub = df[df["measurement_var"] == var]
        scoped = {(name, seed): vals for (v, name, seed), vals in angle_data_by_instance_seed.items() if v == var}
        r = compute_residual_angle_by_outcome(sub, scoped)
        r.insert(0, "measurement_var", var)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_failure_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        sub = df[df["measurement_var"] == var]
        for instance in sub["instance"].unique():
            inst_sub = sub[sub["instance"] == instance]
            n = len(inst_sub)
            rows.append({
                "measurement_var": var, "instance": instance, "n": n,
                "safe_intercept_rate": float(inst_sub["success"].mean()),
                "soldier_caught_rate": float(inst_sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(inst_sub["unsafe_intercept"].mean()),
                "timeout_rate": float(inst_sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_post_detection_summary_by_var(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for var in MEASUREMENT_VAR_VALUES:
        sub = df[df["measurement_var"] == var]
        r = compute_post_detection_summary(sub)
        r.insert(0, "measurement_var", var)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_training_seed_stability_by_var(family_summary: pd.DataFrame) -> pd.DataFrame:
    return family_summary[["measurement_var", "method", "between_seed_sd"]].copy()


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commit_before = _get_git_commit()
    branch = _get_git_branch()
    tree_clean_before = _get_git_status_clean()
    print(f"branch={branch} commit_before={commit_before} tree_clean_before={tree_clean_before}")

    print("\nVerifying model hashes (9 learned models)...")
    hash_results_before = verify_model_hashes()
    print(json.dumps(hash_results_before, indent=2))
    assert all(v["match"] for v in hash_results_before.values()), "MODEL HASH MISMATCH -- STOP BEFORE SEED 68000"

    print("\nGround-truth leakage audit...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_method_instances()
    instance_names = [i.display_name for i in instances]
    assert len(instances) == 12

    print(f"\nRunning MEASUREMENT-NOISE robustness sweep: {len(MEASUREMENT_VAR_VALUES)} variances x "
          f"{len(instances)} instances x {len(SEEDS)} seeds (seeds {SEEDS[0]}..{SEEDS[-1]})...")
    df, trajectories_by_var, angle_data = run_full_sweep(instances, SEEDS)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)
    expected_rows = len(MEASUREMENT_VAR_VALUES) * len(instances) * len(SEEDS)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    print("\nComputing acquisition fairness (6 variances x 1000 seeds = 6000 combinations, WITHIN-variance only)...")
    fairness_df = compute_acquisition_fairness_all_vars(trajectories_by_var, instance_names, SEEDS)
    fairness_df.to_csv(OUTPUT_ROOT / "acquisition_fairness.csv", index=False)
    n_mismatched = int(fairness_df["mismatch"].sum())
    max_discrepancy_overall = float(fairness_df["max_physical_discrepancy"].max())
    print(f"  combinations checked: {len(fairness_df)}, mismatched: {n_mismatched}, max discrepancy: {max_discrepancy_overall}")

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

    estimator_comparisons = compute_estimator_ablation(df)
    estimator_comparisons.to_csv(OUTPUT_ROOT / "estimator_comparisons.csv", index=False)

    tracking_error_summary = compute_tracking_error_summary(df)
    tracking_error_summary.to_csv(OUTPUT_ROOT / "tracking_error_summary.csv", index=False)

    within_method_effects = compute_within_method_noise_effects(df)
    within_method_effects.to_csv(OUTPUT_ROOT / "within_method_noise_effects.csv", index=False)

    representative_noise_summary = compute_representative_noise_summary(df)
    representative_noise_summary.to_csv(OUTPUT_ROOT / "representative_noise_summary.csv", index=False)

    rescue_harm = compute_lead_rescue_harm_by_var(df)
    rescue_harm.to_csv(OUTPUT_ROOT / "lead_rescue_harm.csv", index=False)

    angle_summary = compute_residual_angle_summary_by_var(angle_data["by_instance"])
    angle_summary.to_csv(OUTPUT_ROOT / "residual_angle_summary.csv", index=False)

    angle_by_outcome = compute_residual_angle_by_outcome_by_var(df, angle_data["by_instance_seed"])
    angle_by_outcome.to_csv(OUTPUT_ROOT / "residual_angle_by_outcome.csv", index=False)

    failure_mode_summary = compute_failure_mode_summary(df)
    failure_mode_summary.to_csv(OUTPUT_ROOT / "failure_mode_summary.csv", index=False)

    post_detection_summary = compute_post_detection_summary_by_var(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "post_detection_summary.csv", index=False)

    training_seed_stability = compute_training_seed_stability_by_var(family_summary)
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
        "measurement_noise_source_commit_before": commit_before,
        "measurement_noise_source_commit_after": commit_after,
        "source_immutable": commit_before == commit_after,
        "branch": branch,
        "working_tree_clean_before": tree_clean_before,
        "working_tree_clean_after": tree_clean_after,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "n_episodes_per_condition_per_instance": len(SEEDS),
        "measurement_var_values": list(MEASUREMENT_VAR_VALUES),
        "sigma_values": [math.sqrt(v) for v in MEASUREMENT_VAR_VALUES],
        "nominal_measurement_var": NOMINAL_MEASUREMENT_VAR,
        "fixed_process_var": FIXED_PROCESS_VAR,
        "low_noise_var": LOW_NOISE_VAR,
        "high_noise_var": HIGH_NOISE_VAR,
        "n_instances": len(instances),
        "instances": instance_names,
        "expected_total_mission_evaluations": expected_rows,
        "actual_total_mission_evaluations": len(df),
        "model_hash_verification_before": hash_results_before,
        "model_hash_verification_after": hash_results_after,
        "model_hashes_identical_before_after": hashes_identical,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness": {
            "var_seed_combinations_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall, "all_pass": n_mismatched == 0,
            "note": "fairness required WITHIN a fixed variance only, not across variances",
        },
        "physical_validity": physical_validity,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": [c[0] for c in PRIMARY_COMPARISONS],
        "residual_max_angle_deg": RESIDUAL_MAX_ANGLE_DEG,
        "is_robustness_sweep": True,
        "sweep_name": "measurement_noise",
        "scientific_framing": "zero-shot sensor-noise generalization of the frozen controllers/filter; process_var held fixed at 1.0 (NOT a Kalman-retuning study); measurement_var is the only swept behavioral parameter.",
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUTPUT_ROOT / "model_hashes.json").write_text(json.dumps(hash_results_before, indent=2, default=str))

    core_files = [
        "episode_results.csv", "classical_summary.csv", "learned_seed_summary.csv",
        "family_summary.csv", "primary_comparisons.csv", "secondary_comparisons.csv",
        "estimator_comparisons.csv", "tracking_error_summary.csv", "lead_rescue_harm.csv", "protocol_manifest.json",
    ]
    evidence_hashes = {f: sha256_of_file(OUTPUT_ROOT / f) for f in core_files}
    (OUTPUT_ROOT / "evidence_hashes.json").write_text(json.dumps(evidence_hashes, indent=2))

    print(f"\nSource immutable: {commit_before == commit_after}")
    print(f"Model hashes identical before/after: {hashes_identical}")
    print(f"Acquisition fairness (within-variance): {n_mismatched} mismatched (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
