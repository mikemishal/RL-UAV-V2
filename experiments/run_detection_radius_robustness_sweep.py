"""
Detection-radius expanded robustness sweep -- LOCKED, ONE-TIME evaluation of
the 12 paper-facing method instances across the frozen 6-point detection_radius
grid on the reserved seed range 70000..70999 (1000 matched seeds).

THIS IS THE FINAL PLANNED ROBUSTNESS SWEEP OF THE CONFERENCE EXPERIMENTAL
CAMPAIGN. Seeds 71000..99999 remain permanently sealed for future work.

Scientific question: how does sensing/acquisition range affect the relative
performance of predictive Lead guidance, standalone PPO, and Lead+Residual
PPO under the same constrained maneuvering engagement? detection_radius is
NOT a validated physical sensor specification -- it is a swept behavioral
parameter of the frozen environment model.

Evasion/detection interaction (intentional, NOT normalized away): hostile
reactive evasion radius is fixed at 20m regardless of detection_radius, so
the sweep covers three qualitatively distinct regimes:
  - detection_radius < 20 (5, 10, 15): hostile may begin reactive evasion
    BEFORE defender acquisition.
  - detection_radius == 20: acquisition and evasion-onset thresholds
    coincide geometrically, subject to discrete-time dynamics (not exact).
  - detection_radius > 20 (25, 30): defender may acquire the hostile BEFORE
    it enters its own reactive-evasion radius.
Hostile evasion logic itself is never modified and never depends on the
defender's detection state.

Policy config pass-through (item 8 of this task, carried over from the
mobility audit's finding): reuses run_mobility_robustness_sweep's CORRECTED
build_policy_for_instance (which explicitly passes config=config to PN/Lead,
and to LR-PPO's wrapper) rather than the older shared hostile-evasion helper
that silently defaults scripted policies to EnvConfig(). v_d/v_e are nominal
here (only detection_radius is swept), so this would not have numerically
mattered for Lead's speed assumption -- but the corrected pattern is used
throughout regardless, per instruction.

Reuses the SAME validated building blocks as prior sweeps otherwise (episode
runner, sanitizer, frozen model table, hierarchical bootstrap statistics).

Usage:
    python experiments/run_detection_radius_robustness_sweep.py
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
    DETECTION_RADIUS_SEED_START,
    DETECTION_RADIUS_SEED_END,
    DETECTION_RADIUS_VALUES,
    NOMINAL_DETECTION_RADIUS,
    build_detection_radius_config,
    assert_seed_in_detection_radius_range,
    assert_seed_not_yet_opened,
    assert_disjoint_from_all_prior_ranges,
)
from experiments.run_mobility_robustness_sweep import build_policy_for_instance
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

OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_robustness" / "detection_radius"
SEEDS = list(range(DETECTION_RADIUS_SEED_START, DETECTION_RADIUS_SEED_END + 1))
ENEMY_EVASION_RADIUS = 20.0
SHORT_WARNING_RADIUS = 5.0
LONG_WARNING_RADIUS = 30.0
REPRESENTATIVE_RADII = (
    ("short_warning", SHORT_WARNING_RADIUS),
    ("nominal", NOMINAL_DETECTION_RADIUS),
    ("evasion_threshold", ENEMY_EVASION_RADIUS),
    ("long_warning", LONG_WARNING_RADIUS),
)


def run_full_sweep(instances, seeds: list[int], production: bool = True):
    """Runs all (radius x instance x seed) missions. Returns
    (episode_results_df, trajectories_by_radius, angle_data_by_radius).

    production=True (default, used for the real 70000-70999 run): each seed
    must fall exactly WITHIN the locked detection_radius range (positive
    membership check). production=False (dev/smoke on seeds like 123-127):
    each seed must NOT fall in ANY reserved (not-yet-opened) range. Both
    modes enforce disjointness from all prior/already-consumed ranges
    (hostile_evasion, maneuverability, measurement_noise, mobility -- all
    now permanently frozen)."""
    for s in seeds:
        if production:
            assert_seed_in_detection_radius_range(s)
        else:
            assert_seed_not_yet_opened(s)
        assert_disjoint_from_all_prior_ranges(s)

    rows = []
    trajectories_by_radius: dict[float, dict] = {}
    angle_data_by_radius: dict[str, dict] = {"by_instance": {}, "by_instance_seed": {}}

    total = len(DETECTION_RADIUS_VALUES) * len(instances)
    i = 0
    for radius in DETECTION_RADIUS_VALUES:
        trajectories = {}
        for instance in instances:
            i += 1
            estimator = "kalman" if instance.family == "ppo_kalman" else "measurement"
            config = build_detection_radius_config(estimator, radius)
            env = SoldierEnv(config=config)
            policy = build_policy_for_instance(instance, config)
            dt = config.dt
            print(f"[{i}/{total}] radius={radius} {instance.display_name} ({instance.family}) -- {len(seeds)} episodes...", flush=True)
            instance_angles = []
            for seed in seeds:
                if instance.family == "lr_ppo":
                    row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                    step_angles = row.pop("step_angles")
                    instance_angles.extend(step_angles)
                    angle_data_by_radius["by_instance_seed"][(radius, instance.display_name, seed)] = step_angles
                else:
                    row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                trajectory = row.pop("pre_detection_trajectory")
                trajectories[(seed, instance.display_name)] = trajectory
                row["method"] = instance.name
                row["instance"] = instance.display_name
                row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
                row["detection_radius"] = radius
                rows.append(row)
            if instance.family == "lr_ppo":
                key = (radius, instance.display_name)
                angle_data_by_radius["by_instance"][key] = angle_data_by_radius["by_instance"].get(key, []) + instance_angles
            env.close()
        trajectories_by_radius[radius] = trajectories

    df = pd.DataFrame(rows)
    return df, trajectories_by_radius, angle_data_by_radius


def compute_acquisition_fairness_all_radii(trajectories_by_radius: dict, instance_names: list[str], seeds: list[int]) -> pd.DataFrame:
    """Fairness required ONLY across the 12 methods at a fixed detection
    radius -- NOT across different radii (different radii intentionally
    terminate standby at different times; once launched, subsequent hostile
    motion may diverge because hostile evasion depends on defender position)."""
    rows = []
    for radius, trajectories in trajectories_by_radius.items():
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
                "detection_radius": radius, "environment_seed": seed,
                "detection_step": detection_steps[ref_name],
                "all_detection_steps_equal": len(set(detection_steps.values())) == 1,
                "max_physical_discrepancy": max_discrepancy,
                "defender_soldier_separation_at_detection": sep,
                "defender_speed_at_detection": float(np.linalg.norm(final_state["defender_vel"])),
                "mismatch": mismatch,
            })
    return pd.DataFrame(rows)


def _success_vector(df: pd.DataFrame, radius: float, method: str, training_seed=None) -> np.ndarray:
    sub = df[(df["detection_radius"] == radius) & (df["method"] == method)]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, radius: float, method: str, training_seeds) -> np.ndarray:
    return np.stack([_success_vector(df, radius, method, s) for s in training_seeds])


def _method_success_rate(df: pd.DataFrame, radius: float, method: str) -> float:
    if method == "lead":
        return float(_success_vector(df, radius, "lead").mean())
    if method == "ppo":
        return float(_success_matrix(df, radius, "ppo", TRAINING_SEEDS).mean())
    if method == "ppo_kalman":
        return float(_success_matrix(df, radius, "ppo_kalman", TRAINING_SEEDS).mean())
    if method == LR_PPO_METHOD:
        return float(_success_matrix(df, radius, LR_PPO_METHOD, LR_TRAINING_ROOTS).mean())
    if method in ("greedy", "pn"):
        return float(_success_vector(df, radius, method).mean())
    raise ValueError(method)


def compute_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        for method in SCRIPTED_METHODS:
            sub = df[(df["detection_radius"] == radius) & (df["method"] == method)]
            n = len(sub)
            successes = int(sub["success"].sum())
            lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
            rows.append({
                "detection_radius": radius, "instance": method, "method": method, "n": n, "success_count": successes,
                "success_rate": successes / n, "standard_error": standard_error(successes, n),
                "ci_low": lo, "ci_high": hi,
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_learned_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            for seed in seeds:
                sub = df[(df["detection_radius"] == radius) & (df["method"] == method) & (df["training_seed"] == seed)]
                n = len(sub)
                successes = int(sub["success"].sum())
                lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
                rows.append({
                    "detection_radius": radius, "instance": sub["instance"].iloc[0], "method": method, "training_seed": seed,
                    "n": n, "success_count": successes, "success_rate": successes / n,
                    "standard_error": standard_error(successes, n), "ci_low": lo, "ci_high": hi,
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def compute_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            matrix = _success_matrix(df, radius, method, seeds)
            rates = matrix.mean(axis=1)
            observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({
                "detection_radius": radius, "method": method, "n_training_seeds": len(seeds),
                "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
                "min_seed_success": float(rates.min()), "max_seed_success": float(rates.max()),
                "range": float(rates.max() - rates.min()),
                "hierarchical_ci_low": lo, "hierarchical_ci_high": hi,
            })
    return pd.DataFrame(rows)


def compute_primary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        lead_vec = _success_vector(df, radius, "lead")
        lr_matrix = _success_matrix(df, radius, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        ppo_matrix = _success_matrix(df, radius, "ppo", TRAINING_SEEDS)
        for name, method_a, method_b in PRIMARY_COMPARISONS:
            if method_b == "lead":
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            else:
                observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"detection_radius": radius, "comparison": name, "diff_pp": observed * 100,
                         "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_secondary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        greedy_vec = _success_vector(df, radius, "greedy")
        pn_vec = _success_vector(df, radius, "pn")
        lead_vec = _success_vector(df, radius, "lead")
        ppo_matrix = _success_matrix(df, radius, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, radius, "ppo_kalman", TRAINING_SEEDS)
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
            rows.append({"detection_radius": radius, "comparison": name, "diff_pp": observed * 100,
                         "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_within_method_detection_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lead_nominal = _success_vector(df, NOMINAL_DETECTION_RADIUS, "lead")
    ppo_nominal = _success_matrix(df, NOMINAL_DETECTION_RADIUS, "ppo", TRAINING_SEEDS)
    ppok_nominal = _success_matrix(df, NOMINAL_DETECTION_RADIUS, "ppo_kalman", TRAINING_SEEDS)
    lr_nominal = _success_matrix(df, NOMINAL_DETECTION_RADIUS, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    for radius in DETECTION_RADIUS_VALUES:
        if radius == NOMINAL_DETECTION_RADIUS:
            continue
        lead_vec = _success_vector(df, radius, "lead")
        observed, lo, hi = paired_bootstrap_ci(lead_vec, lead_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lead", "detection_radius": radius, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        ppo_matrix = _success_matrix(df, radius, "ppo", TRAINING_SEEDS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppo_matrix, ppo_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "ppo", "detection_radius": radius, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        ppok_matrix = _success_matrix(df, radius, "ppo_kalman", TRAINING_SEEDS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppok_matrix, ppok_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "ppo_kalman", "detection_radius": radius, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        lr_matrix = _success_matrix(df, radius, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(lr_matrix, lr_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lr_ppo", "detection_radius": radius, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})
    return pd.DataFrame(rows)


def compute_representative_radius_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 25: pre-specified SHORT (5) / NOMINAL (15) / EVASION-THRESHOLD
    (20) / LONG (30) contrast, with hierarchical intervals."""
    rows = []
    for label, radius in REPRESENTATIVE_RADII:
        lead_vec = _success_vector(df, radius, "lead")
        ppo_matrix = _success_matrix(df, radius, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, radius, "ppo_kalman", TRAINING_SEEDS)
        lr_matrix = _success_matrix(df, radius, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        d_lead, lo_lead, hi_lead = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        d_ppo, lo_ppo, hi_ppo = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        d_est, lo_est, hi_est, _ = hierarchical_paired_bootstrap_matched_seeds(ppok_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({
            "condition_label": label, "detection_radius": radius,
            "lead_success": float(lead_vec.mean()), "ppo_family_success": float(ppo_matrix.mean()),
            "ppo_kalman_family_success": float(ppok_matrix.mean()), "lr_ppo_family_success": float(lr_matrix.mean()),
            "lr_ppo_minus_lead_pp": d_lead * 100, "lr_ppo_minus_lead_ci_low_pp": lo_lead * 100, "lr_ppo_minus_lead_ci_high_pp": hi_lead * 100,
            "lr_ppo_minus_ppo_pp": d_ppo * 100, "lr_ppo_minus_ppo_ci_low_pp": lo_ppo * 100, "lr_ppo_minus_ppo_ci_high_pp": hi_ppo * 100,
            "ppo_kalman_minus_ppo_pp": d_est * 100, "ppo_kalman_minus_ppo_ci_low_pp": lo_est * 100, "ppo_kalman_minus_ppo_ci_high_pp": hi_est * 100,
        })
    return pd.DataFrame(rows)


def compute_evasion_relative_regime_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 26: DESCRIPTIVE aggregation only (no new inferential headline) of
    the three evasion-relative regimes: R_det<20 (5,10,15), R_det=20
    (threshold), R_det>20 (25,30)."""
    regimes = {
        "pre_evasion_acquisition (R_det>20)": (25.0, 30.0),
        "threshold (R_det=20)": (20.0,),
        "post_evasion_onset_acquisition (R_det<20)": (5.0, 10.0, 15.0),
    }
    rows = []
    for regime_label, radii in regimes.items():
        for method in ("lead", "ppo", LR_PPO_METHOD):
            rates = [_method_success_rate(df, r, method) for r in radii]
            rows.append({"regime": regime_label, "method": method, "radii_included": list(radii), "mean_success": float(np.mean(rates))})
    return pd.DataFrame(rows)


def compute_marginal_range_benefit(df: pd.DataFrame) -> pd.DataFrame:
    """Item 27: adjacent success changes (10-5, 15-10, 20-15, 25-20, 30-25)
    for Lead/PPO/LR-PPO -- descriptive, characterizes diminishing returns or
    non-monotonicity."""
    rows = []
    for method in ("lead", "ppo", LR_PPO_METHOD):
        rates = [_method_success_rate(df, r, method) for r in DETECTION_RADIUS_VALUES]
        row = {"method": method}
        for r, rate in zip(DETECTION_RADIUS_VALUES, rates):
            row[f"success_at_{r:g}m"] = rate
        for i in range(1, len(DETECTION_RADIUS_VALUES)):
            r_lo, r_hi = DETECTION_RADIUS_VALUES[i - 1], DETECTION_RADIUS_VALUES[i]
            row[f"diff_{r_hi:g}_minus_{r_lo:g}"] = rates[i] - rates[i - 1]
        rows.append(row)
    return pd.DataFrame(rows)


def compute_detection_time_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 18: detection time distribution + remaining nominal horizon per
    radius (across all methods pooled, since acquisition is method-agnostic
    at a fixed radius per the fairness guarantee)."""
    rows = []
    max_steps = 2000  # EnvConfig default, unchanged across the sweep (config isolation verified)
    for radius in DETECTION_RADIUS_VALUES:
        sub = df[df["detection_radius"] == radius]
        det_time = sub["detection_time_seconds"].dropna().to_numpy(dtype=np.float64)
        det_step = sub["detection_step"].dropna().to_numpy(dtype=np.float64)
        remaining_steps = max_steps - det_step
        rows.append({
            "detection_radius": radius, "n": len(det_time),
            "mean_detection_time_s": float(det_time.mean()), "median_detection_time_s": float(np.median(det_time)),
            "sd_detection_time_s": float(det_time.std(ddof=0)),
            "p25_detection_time_s": float(np.percentile(det_time, 25)), "p75_detection_time_s": float(np.percentile(det_time, 75)),
            "p95_detection_time_s": float(np.percentile(det_time, 95)), "max_detection_time_s": float(det_time.max()),
            "mean_remaining_mission_steps_at_detection": float(remaining_steps.mean()),
        })
    return pd.DataFrame(rows)


def compute_lead_rescue_harm_by_radius(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        sub = df[df["detection_radius"] == radius]
        r = compute_lead_rescue_harm(sub)
        r.insert(0, "detection_radius", radius)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_summary_by_radius(angle_data_by_instance: dict) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        scoped = {name: vals for (r, name), vals in angle_data_by_instance.items() if r == radius}
        r_df = compute_residual_angle_summary(scoped)
        r_df.insert(0, "detection_radius", radius)
        rows.append(r_df)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_by_outcome_by_radius(df: pd.DataFrame, angle_data_by_instance_seed: dict) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        sub = df[df["detection_radius"] == radius]
        scoped = {(name, seed): vals for (r, name, seed), vals in angle_data_by_instance_seed.items() if r == radius}
        r_df = compute_residual_angle_by_outcome(sub, scoped)
        r_df.insert(0, "detection_radius", radius)
        rows.append(r_df)
    return pd.concat(rows, ignore_index=True)


def compute_failure_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        sub = df[df["detection_radius"] == radius]
        for instance in sub["instance"].unique():
            inst_sub = sub[sub["instance"] == instance]
            n = len(inst_sub)
            rows.append({
                "detection_radius": radius, "instance": instance, "n": n,
                "safe_intercept_rate": float(inst_sub["success"].mean()),
                "soldier_caught_rate": float(inst_sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(inst_sub["unsafe_intercept"].mean()),
                "timeout_rate": float(inst_sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_post_detection_summary_by_radius(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for radius in DETECTION_RADIUS_VALUES:
        sub = df[df["detection_radius"] == radius]
        r = compute_post_detection_summary(sub)
        r.insert(0, "detection_radius", radius)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_training_seed_stability_by_radius(family_summary: pd.DataFrame) -> pd.DataFrame:
    return family_summary[["detection_radius", "method", "between_seed_sd"]].copy()


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commit_before = _get_git_commit()
    branch = _get_git_branch()
    tree_clean_before = _get_git_status_clean()
    print(f"branch={branch} commit_before={commit_before} tree_clean_before={tree_clean_before}")

    print("\nVerifying model hashes (9 learned models)...")
    hash_results_before = verify_model_hashes()
    print(json.dumps(hash_results_before, indent=2))
    assert all(v["match"] for v in hash_results_before.values()), "MODEL HASH MISMATCH -- STOP BEFORE SEED 70000"

    print("\nGround-truth leakage audit...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_method_instances()
    instance_names = [i.display_name for i in instances]
    assert len(instances) == 12

    print(f"\nRunning DETECTION-RADIUS robustness sweep: {len(DETECTION_RADIUS_VALUES)} radii x "
          f"{len(instances)} instances x {len(SEEDS)} seeds (seeds {SEEDS[0]}..{SEEDS[-1]})...")
    df, trajectories_by_radius, angle_data = run_full_sweep(instances, SEEDS)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)
    expected_rows = len(DETECTION_RADIUS_VALUES) * len(instances) * len(SEEDS)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    print("\nComputing acquisition fairness (6 radii x 1000 seeds = 6000 combinations, WITHIN-radius only)...")
    fairness_df = compute_acquisition_fairness_all_radii(trajectories_by_radius, instance_names, SEEDS)
    fairness_df.to_csv(OUTPUT_ROOT / "acquisition_fairness.csv", index=False)
    n_mismatched = int(fairness_df["mismatch"].sum())
    max_discrepancy_overall = float(fairness_df["max_physical_discrepancy"].max())
    print(f"  combinations checked: {len(fairness_df)}, mismatched: {n_mismatched}, max discrepancy: {max_discrepancy_overall}")

    detection_time_summary = compute_detection_time_summary(df)
    detection_time_summary.to_csv(OUTPUT_ROOT / "detection_time_summary.csv", index=False)

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

    within_method_effects = compute_within_method_detection_effects(df)
    within_method_effects.to_csv(OUTPUT_ROOT / "within_method_detection_effects.csv", index=False)

    representative_radius_summary = compute_representative_radius_summary(df)
    representative_radius_summary.to_csv(OUTPUT_ROOT / "representative_radius_summary.csv", index=False)

    evasion_relative_regime_summary = compute_evasion_relative_regime_summary(df)
    evasion_relative_regime_summary.to_csv(OUTPUT_ROOT / "evasion_relative_regime_summary.csv", index=False)

    marginal_range_benefit = compute_marginal_range_benefit(df)
    marginal_range_benefit.to_csv(OUTPUT_ROOT / "marginal_range_benefit.csv", index=False)

    rescue_harm = compute_lead_rescue_harm_by_radius(df)
    rescue_harm.to_csv(OUTPUT_ROOT / "lead_rescue_harm.csv", index=False)

    angle_summary = compute_residual_angle_summary_by_radius(angle_data["by_instance"])
    angle_summary.to_csv(OUTPUT_ROOT / "residual_angle_summary.csv", index=False)

    angle_by_outcome = compute_residual_angle_by_outcome_by_radius(df, angle_data["by_instance_seed"])
    angle_by_outcome.to_csv(OUTPUT_ROOT / "residual_angle_by_outcome.csv", index=False)

    failure_mode_summary = compute_failure_mode_summary(df)
    failure_mode_summary.to_csv(OUTPUT_ROOT / "failure_mode_summary.csv", index=False)

    post_detection_summary = compute_post_detection_summary_by_radius(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "post_detection_summary.csv", index=False)

    training_seed_stability = compute_training_seed_stability_by_radius(family_summary)
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
        "detection_radius_source_commit_before": commit_before,
        "detection_radius_source_commit_after": commit_after,
        "source_immutable": commit_before == commit_after,
        "branch": branch,
        "working_tree_clean_before": tree_clean_before,
        "working_tree_clean_after": tree_clean_after,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "n_episodes_per_condition_per_instance": len(SEEDS),
        "detection_radius_values": list(DETECTION_RADIUS_VALUES),
        "nominal_detection_radius": NOMINAL_DETECTION_RADIUS,
        "enemy_evasion_radius": ENEMY_EVASION_RADIUS,
        "n_instances": len(instances),
        "instances": instance_names,
        "expected_total_mission_evaluations": expected_rows,
        "actual_total_mission_evaluations": len(df),
        "model_hash_verification_before": hash_results_before,
        "model_hash_verification_after": hash_results_after,
        "model_hashes_identical_before_after": hashes_identical,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness": {
            "radius_seed_combinations_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall, "all_pass": n_mismatched == 0,
            "note": "fairness required WITHIN a fixed detection radius only, not across radii",
        },
        "physical_validity": physical_validity,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": [c[0] for c in PRIMARY_COMPARISONS],
        "residual_max_angle_deg": RESIDUAL_MAX_ANGLE_DEG,
        "is_robustness_sweep": True,
        "sweep_name": "detection_radius",
        "is_final_robustness_sweep_of_conference_campaign": True,
        "scientific_framing": "detection_radius is a swept behavioral parameter of the frozen environment model, NOT a validated physical sensor specification; enemy_evasion_radius (20m) is held fixed and independent of defender detection state throughout.",
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUTPUT_ROOT / "model_hashes.json").write_text(json.dumps(hash_results_before, indent=2, default=str))

    core_files = [
        "episode_results.csv", "classical_summary.csv", "learned_seed_summary.csv",
        "family_summary.csv", "primary_comparisons.csv", "secondary_comparisons.csv",
        "representative_radius_summary.csv", "lead_rescue_harm.csv", "protocol_manifest.json",
    ]
    evidence_hashes = {f: sha256_of_file(OUTPUT_ROOT / f) for f in core_files}
    (OUTPUT_ROOT / "evidence_hashes.json").write_text(json.dumps(evidence_hashes, indent=2))

    print(f"\nSource immutable: {commit_before == commit_after}")
    print(f"Model hashes identical before/after: {hashes_identical}")
    print(f"Acquisition fairness (within-radius): {n_mismatched} mismatched (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
