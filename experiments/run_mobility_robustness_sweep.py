"""
Mobility expanded robustness sweep -- LOCKED, ONE-TIME evaluation of the 12
paper-facing method instances across the frozen 9-condition (v_d, v_e) grid
(two matched 1-D arms sharing one nominal condition) on the reserved seed
range 69000..69999 (1000 matched seeds).

Scientific questions (kept distinct, per protocol):
  A. How does interception change as DEFENDER speed changes, hostile speed
     fixed at 12 m/s?                              -> defender_speed_summary.csv
  B. How does interception change as HOSTILE speed changes, defender speed
     fixed at 18 m/s?                               -> hostile_speed_summary.csv
  C. Does LR-PPO generalize more consistently than standalone PPO away from
     its (18,12) training condition?                -> family_summary.csv / primary comparisons
  D. At which of the 9 TESTED conditions does interception become difficult
     or infeasible (measured-condition coverage only, no interpolation)?
                                                     -> feasibility_summary.csv

acceleration/turn-rate/vertical-rate limits are held FIXED at nominal values
throughout (never scaled with speed) -- a faster vehicle intentionally gets
different effective turn/acceleration geometry; this is not normalized away.

CRITICAL FIX (item 10 of this task's audit): the hostile-evasion/
maneuverability/measurement-noise runners' shared `build_policy_for_instance`
constructs scripted policies via `get_policy(instance.name)` WITHOUT passing
`config` -- harmless there because v_d was never swept in those sweeps
(Lead/PN's default EnvConfig().v_d == the actual nominal v_d == 18.0 in every
condition they tested). For MOBILITY, v_d (and v_e) ARE the swept parameter,
so Lead/PN must be constructed with the CURRENT condition's config or they
would silently assume the wrong interceptor speed at 8 of 9 conditions. This
module therefore defines its OWN corrected build_policy_for_instance and does
NOT import the shared (unfixed-for-this-purpose) one -- verified by
test_lead_uses_current_condition_v_d_not_hardcoded_nominal and
test_observation_normalization_uses_current_condition_speeds in
test_expanded_robustness_protocol.py.

Reuses the SAME validated building blocks as prior sweeps (episode runner,
sanitizer, frozen model table, hierarchical bootstrap statistics) otherwise.

Usage:
    python experiments/run_mobility_robustness_sweep.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.policies.registry import get_policy
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.expanded_robustness_protocol import (
    build_method_instances,
    FROZEN_MODELS,
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
    MOBILITY_SEED_START,
    MOBILITY_SEED_END,
    DEFENDER_SPEED_ARM_VALUES,
    HOSTILE_SPEED_ARM_VALUES,
    MOBILITY_GRID,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
    build_mobility_config,
    assert_seed_in_mobility_range,
    assert_seed_not_yet_opened,
    assert_disjoint_from_all_prior_ranges,
)
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

OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_robustness" / "mobility"
SEEDS = list(range(MOBILITY_SEED_START, MOBILITY_SEED_END + 1))
NOMINAL_CONDITION = (NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED)
REPRESENTATIVE_CONDITIONS = (
    ("low_defender_authority", (12.0, 12.0)),
    ("nominal", (18.0, 12.0)),
    ("high_defender_authority", (24.0, 12.0)),
    ("slow_hostile", (18.0, 8.0)),
    ("fast_hostile", (18.0, 16.0)),
)
FEASIBILITY_THRESHOLDS = (0.70, 0.80, 0.90)


def build_policy_for_instance(instance, config):
    """Corrected for mobility: PN/Lead MUST receive the current per-condition
    config (see module docstring) since v_d is the swept parameter here."""
    if instance.family == "scripted":
        if instance.name == "greedy":
            return get_policy("greedy")  # direction-only; env scales by config.v_d itself
        return get_policy(instance.name, config=config)  # pn / lead: v_d-dependent
    if instance.family == "ppo":
        model_path = FROZEN_MODELS[("direct", instance.training_seed)]["path"]
        return get_policy("ppo", model_path=str(model_path), deterministic=True)
    if instance.family == "ppo_kalman":
        model_path = FROZEN_MODELS[("kalman", instance.training_seed)]["path"]
        return get_policy("ppo_kalman", model_path=str(model_path), deterministic=True)
    if instance.family == "lr_ppo":
        model_path = FROZEN_MODELS[("lr_ppo", instance.training_seed)]["path"]
        model = PPO.load(str(model_path), device="cpu")
        return LeadResidualPPOPolicyWrapper(model, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG, config=config, deterministic=True)
    raise ValueError(instance.family)


def run_full_sweep(instances, seeds: list[int], production: bool = True):
    """Runs all (condition x instance x seed) missions. Returns
    (episode_results_df, trajectories_by_condition, angle_data_by_condition).

    production=True (default, used for the real 69000-69999 run): each seed
    must fall exactly WITHIN the locked mobility range (positive membership
    check). production=False (dev/smoke on seeds like 123-127): each seed
    must NOT fall in ANY reserved (not-yet-opened) range. Both modes enforce
    disjointness from all prior/already-consumed ranges (hostile_evasion,
    maneuverability, measurement_noise -- all now permanently frozen)."""
    for s in seeds:
        if production:
            assert_seed_in_mobility_range(s)
        else:
            assert_seed_not_yet_opened(s)
        assert_disjoint_from_all_prior_ranges(s)

    rows = []
    trajectories_by_condition: dict[tuple[float, float], dict] = {}
    angle_data_by_condition: dict[str, dict] = {"by_instance": {}, "by_instance_seed": {}}

    total = len(MOBILITY_GRID) * len(instances)
    i = 0
    for v_d, v_e in MOBILITY_GRID:
        trajectories = {}
        for instance in instances:
            i += 1
            estimator = "kalman" if instance.family == "ppo_kalman" else "measurement"
            config = build_mobility_config(estimator, v_d, v_e)
            env = SoldierEnv(config=config)
            policy = build_policy_for_instance(instance, config)
            dt = config.dt
            print(f"[{i}/{total}] (v_d={v_d},v_e={v_e}) {instance.display_name} ({instance.family}) -- {len(seeds)} episodes...", flush=True)
            instance_angles = []
            for seed in seeds:
                if instance.family == "lr_ppo":
                    row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                    step_angles = row.pop("step_angles")
                    instance_angles.extend(step_angles)
                    angle_data_by_condition["by_instance_seed"][((v_d, v_e), instance.display_name, seed)] = step_angles
                else:
                    row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                trajectory = row.pop("pre_detection_trajectory")
                trajectories[(seed, instance.display_name)] = trajectory
                row["method"] = instance.name
                row["instance"] = instance.display_name
                row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
                row["v_d"] = v_d
                row["v_e"] = v_e
                rows.append(row)
            if instance.family == "lr_ppo":
                key = (v_d, v_e), instance.display_name
                angle_data_by_condition["by_instance"][key] = angle_data_by_condition["by_instance"].get(key, []) + instance_angles
            env.close()
        trajectories_by_condition[(v_d, v_e)] = trajectories

    df = pd.DataFrame(rows)
    return df, trajectories_by_condition, angle_data_by_condition


def compute_acquisition_fairness_all_conditions(trajectories_by_condition: dict, instance_names: list[str], seeds: list[int]) -> pd.DataFrame:
    """Fairness required ONLY across the 12 methods at a fixed condition --
    NOT across different mobility conditions (hostile speed legitimately
    changes pre-detection acquisition timing)."""
    rows = []
    for (v_d, v_e), trajectories in trajectories_by_condition.items():
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
                "v_d": v_d, "v_e": v_e, "environment_seed": seed,
                "detection_step": detection_steps[ref_name],
                "all_detection_steps_equal": len(set(detection_steps.values())) == 1,
                "max_physical_discrepancy": max_discrepancy,
                "defender_soldier_separation_at_detection": sep,
                "defender_speed_at_detection": float(np.linalg.norm(final_state["defender_vel"])),
                "mismatch": mismatch,
            })
    return pd.DataFrame(rows)


def _success_vector(df: pd.DataFrame, condition: tuple, method: str, training_seed=None) -> np.ndarray:
    sub = df[(df["v_d"] == condition[0]) & (df["v_e"] == condition[1]) & (df["method"] == method)]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, condition: tuple, method: str, training_seeds) -> np.ndarray:
    return np.stack([_success_vector(df, condition, method, s) for s in training_seeds])


def _method_success_rate(df: pd.DataFrame, condition: tuple, method: str) -> float:
    if method == "lead":
        return float(_success_vector(df, condition, "lead").mean())
    if method == "ppo":
        return float(_success_matrix(df, condition, "ppo", TRAINING_SEEDS).mean())
    if method == "ppo_kalman":
        return float(_success_matrix(df, condition, "ppo_kalman", TRAINING_SEEDS).mean())
    if method == LR_PPO_METHOD:
        return float(_success_matrix(df, condition, LR_PPO_METHOD, LR_TRAINING_ROOTS).mean())
    if method in ("greedy", "pn"):
        return float(_success_vector(df, condition, method).mean())
    raise ValueError(method)


def compute_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        for method in SCRIPTED_METHODS:
            sub = df[(df["v_d"] == v_d) & (df["v_e"] == v_e) & (df["method"] == method)]
            n = len(sub)
            successes = int(sub["success"].sum())
            lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
            rows.append({
                "v_d": v_d, "v_e": v_e, "instance": method, "method": method, "n": n, "success_count": successes,
                "success_rate": successes / n, "standard_error": standard_error(successes, n),
                "ci_low": lo, "ci_high": hi,
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_learned_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            for seed in seeds:
                sub = df[(df["v_d"] == v_d) & (df["v_e"] == v_e) & (df["method"] == method) & (df["training_seed"] == seed)]
                n = len(sub)
                successes = int(sub["success"].sum())
                lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
                rows.append({
                    "v_d": v_d, "v_e": v_e, "instance": sub["instance"].iloc[0], "method": method, "training_seed": seed,
                    "n": n, "success_count": successes, "success_rate": successes / n,
                    "standard_error": standard_error(successes, n), "ci_low": lo, "ci_high": hi,
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def compute_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            matrix = _success_matrix(df, (v_d, v_e), method, seeds)
            rates = matrix.mean(axis=1)
            observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({
                "v_d": v_d, "v_e": v_e, "method": method, "n_training_seeds": len(seeds),
                "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
                "min_seed_success": float(rates.min()), "max_seed_success": float(rates.max()),
                "range": float(rates.max() - rates.min()),
                "hierarchical_ci_low": lo, "hierarchical_ci_high": hi,
            })
    return pd.DataFrame(rows)


def compute_primary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        condition = (v_d, v_e)
        lead_vec = _success_vector(df, condition, "lead")
        lr_matrix = _success_matrix(df, condition, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        ppo_matrix = _success_matrix(df, condition, "ppo", TRAINING_SEEDS)
        for name, method_a, method_b in PRIMARY_COMPARISONS:
            if method_b == "lead":
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            else:
                observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"v_d": v_d, "v_e": v_e, "comparison": name, "diff_pp": observed * 100,
                         "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_secondary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        condition = (v_d, v_e)
        greedy_vec = _success_vector(df, condition, "greedy")
        pn_vec = _success_vector(df, condition, "pn")
        lead_vec = _success_vector(df, condition, "lead")
        ppo_matrix = _success_matrix(df, condition, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, condition, "ppo_kalman", TRAINING_SEEDS)
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
            rows.append({"v_d": v_d, "v_e": v_e, "comparison": name, "diff_pp": observed * 100,
                         "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_defender_speed_arm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 24: success vs v_d (v_e=12 fixed), for Lead/PPO/PPO-Kalman/LR-PPO,
    with adjacent deltas and delta-from-nominal (v_d=18). No monotonicity
    assumed."""
    rows = []
    for method in ("lead", "ppo", "ppo_kalman", LR_PPO_METHOD):
        rates = [_method_success_rate(df, (v_d, NOMINAL_HOSTILE_SPEED), method) for v_d in DEFENDER_SPEED_ARM_VALUES]
        nominal_rate = _method_success_rate(df, NOMINAL_CONDITION, method)
        row = {"method": method}
        for v_d, rate in zip(DEFENDER_SPEED_ARM_VALUES, rates):
            row[f"success_at_v_d_{v_d:g}"] = rate
        row["diff_15_minus_12"] = rates[1] - rates[0]
        row["diff_18_minus_15"] = rates[2] - rates[1]
        row["diff_21_minus_18"] = rates[3] - rates[2]
        row["diff_24_minus_21"] = rates[4] - rates[3]
        for v_d, rate in zip(DEFENDER_SPEED_ARM_VALUES, rates):
            row[f"delta_from_nominal_v_d_{v_d:g}"] = rate - nominal_rate
        rows.append(row)
    return pd.DataFrame(rows)


def compute_hostile_speed_arm_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 25: success vs v_e (v_d=18 fixed), for Lead/PPO/PPO-Kalman/LR-PPO,
    with adjacent deltas and delta-from-nominal (v_e=12). No monotonicity
    assumed."""
    rows = []
    for method in ("lead", "ppo", "ppo_kalman", LR_PPO_METHOD):
        rates = [_method_success_rate(df, (NOMINAL_DEFENDER_SPEED, v_e), method) for v_e in HOSTILE_SPEED_ARM_VALUES]
        nominal_rate = _method_success_rate(df, NOMINAL_CONDITION, method)
        row = {"method": method}
        for v_e, rate in zip(HOSTILE_SPEED_ARM_VALUES, rates):
            row[f"success_at_v_e_{v_e:g}"] = rate
        row["diff_10_minus_8"] = rates[1] - rates[0]
        row["diff_12_minus_10"] = rates[2] - rates[1]
        row["diff_14_minus_12"] = rates[3] - rates[2]
        row["diff_16_minus_14"] = rates[4] - rates[3]
        for v_e, rate in zip(HOSTILE_SPEED_ARM_VALUES, rates):
            row[f"delta_from_nominal_v_e_{v_e:g}"] = rate - nominal_rate
        rows.append(row)
    return pd.DataFrame(rows)


def compute_mobility_ratio_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 26: descriptive success vs rho=v_d/v_e, tagged by which arm each
    condition belongs to -- never collapsing different absolute-speed
    configurations solely because rho is similar."""
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        condition = (v_d, v_e)
        if (v_d, v_e) == NOMINAL_CONDITION:
            arm = "nominal (both arms)"
        elif v_e == NOMINAL_HOSTILE_SPEED:
            arm = "defender_speed_arm"
        else:
            arm = "hostile_speed_arm"
        rows.append({
            "v_d": v_d, "v_e": v_e, "rho": v_d / v_e, "arm": arm,
            "lead_success": _method_success_rate(df, condition, "lead"),
            "ppo_family_success": _method_success_rate(df, condition, "ppo"),
            "lr_ppo_family_success": _method_success_rate(df, condition, LR_PPO_METHOD),
        })
    return pd.DataFrame(rows).sort_values("rho").reset_index(drop=True)


def compute_representative_condition_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 27: 5 pre-specified representative conditions with hierarchical
    primary-comparison intervals."""
    rows = []
    for label, condition in REPRESENTATIVE_CONDITIONS:
        lead_vec = _success_vector(df, condition, "lead")
        ppo_matrix = _success_matrix(df, condition, "ppo", TRAINING_SEEDS)
        lr_matrix = _success_matrix(df, condition, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        d_lead, lo_lead, hi_lead = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        d_ppo, lo_ppo, hi_ppo = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({
            "condition_label": label, "v_d": condition[0], "v_e": condition[1], "rho": condition[0] / condition[1],
            "lead_success": float(lead_vec.mean()), "ppo_family_success": float(ppo_matrix.mean()), "lr_ppo_family_success": float(lr_matrix.mean()),
            "lr_ppo_minus_lead_pp": d_lead * 100, "lr_ppo_minus_lead_ci_low_pp": lo_lead * 100, "lr_ppo_minus_lead_ci_high_pp": hi_lead * 100,
            "lr_ppo_minus_ppo_pp": d_ppo * 100, "lr_ppo_minus_ppo_ci_low_pp": lo_ppo * 100, "lr_ppo_minus_ppo_ci_high_pp": hi_ppo * 100,
        })
    return pd.DataFrame(rows)


def compute_feasibility_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 28: 'measured-condition coverage'/'tested-condition feasibility'
    -- which of the 9 TESTED conditions each method/family clears each
    threshold. NOT an interpolated or continuous feasibility boundary; NOT a
    claim about untested speed combinations."""
    rows = []
    for method in ("greedy", "pn", "lead", "ppo", "ppo_kalman", LR_PPO_METHOD):
        rates = {(v_d, v_e): _method_success_rate(df, (v_d, v_e), method) for v_d, v_e in MOBILITY_GRID}
        for threshold in FEASIBILITY_THRESHOLDS:
            met = [f"({v_d:g},{v_e:g})" for (v_d, v_e), r in rates.items() if r >= threshold]
            rows.append({
                "method": method, "threshold": threshold,
                "n_conditions_meeting_threshold": len(met), "n_conditions_tested": len(MOBILITY_GRID),
                "conditions_meeting_threshold": ";".join(met),
            })
    return pd.DataFrame(rows)


def compute_lead_rescue_harm_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        sub = df[(df["v_d"] == v_d) & (df["v_e"] == v_e)]
        r = compute_lead_rescue_harm(sub)
        r.insert(0, "v_d", v_d)
        r.insert(1, "v_e", v_e)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_summary_by_condition(angle_data_by_instance: dict) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        condition = (v_d, v_e)
        scoped = {name: vals for (c, name), vals in angle_data_by_instance.items() if c == condition}
        r = compute_residual_angle_summary(scoped)
        r.insert(0, "v_d", v_d)
        r.insert(1, "v_e", v_e)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_by_outcome_by_condition(df: pd.DataFrame, angle_data_by_instance_seed: dict) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        condition = (v_d, v_e)
        sub = df[(df["v_d"] == v_d) & (df["v_e"] == v_e)]
        scoped = {(name, seed): vals for (c, name, seed), vals in angle_data_by_instance_seed.items() if c == condition}
        r = compute_residual_angle_by_outcome(sub, scoped)
        r.insert(0, "v_d", v_d)
        r.insert(1, "v_e", v_e)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_failure_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        sub = df[(df["v_d"] == v_d) & (df["v_e"] == v_e)]
        for instance in sub["instance"].unique():
            inst_sub = sub[sub["instance"] == instance]
            n = len(inst_sub)
            rows.append({
                "v_d": v_d, "v_e": v_e, "instance": instance, "n": n,
                "safe_intercept_rate": float(inst_sub["success"].mean()),
                "soldier_caught_rate": float(inst_sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(inst_sub["unsafe_intercept"].mean()),
                "timeout_rate": float(inst_sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_post_detection_summary_by_condition(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for v_d, v_e in MOBILITY_GRID:
        sub = df[(df["v_d"] == v_d) & (df["v_e"] == v_e)]
        r = compute_post_detection_summary(sub)
        r.insert(0, "v_d", v_d)
        r.insert(1, "v_e", v_e)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_training_seed_stability_by_condition(family_summary: pd.DataFrame) -> pd.DataFrame:
    return family_summary[["v_d", "v_e", "method", "between_seed_sd"]].copy()


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commit_before = _get_git_commit()
    branch = _get_git_branch()
    tree_clean_before = _get_git_status_clean()
    print(f"branch={branch} commit_before={commit_before} tree_clean_before={tree_clean_before}")

    print("\nVerifying model hashes (9 learned models)...")
    hash_results_before = verify_model_hashes()
    print(json.dumps(hash_results_before, indent=2))
    assert all(v["match"] for v in hash_results_before.values()), "MODEL HASH MISMATCH -- STOP BEFORE SEED 69000"

    print("\nGround-truth leakage audit...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_method_instances()
    instance_names = [i.display_name for i in instances]
    assert len(instances) == 12

    print(f"\nRunning MOBILITY robustness sweep: {len(MOBILITY_GRID)} conditions x "
          f"{len(instances)} instances x {len(SEEDS)} seeds (seeds {SEEDS[0]}..{SEEDS[-1]})...")
    df, trajectories_by_condition, angle_data = run_full_sweep(instances, SEEDS)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)
    expected_rows = len(MOBILITY_GRID) * len(instances) * len(SEEDS)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    print("\nComputing acquisition fairness (9 conditions x 1000 seeds = 9000 combinations, WITHIN-condition only)...")
    fairness_df = compute_acquisition_fairness_all_conditions(trajectories_by_condition, instance_names, SEEDS)
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

    defender_speed_summary = compute_defender_speed_arm_summary(df)
    defender_speed_summary.to_csv(OUTPUT_ROOT / "defender_speed_summary.csv", index=False)

    hostile_speed_summary = compute_hostile_speed_arm_summary(df)
    hostile_speed_summary.to_csv(OUTPUT_ROOT / "hostile_speed_summary.csv", index=False)

    mobility_ratio_summary = compute_mobility_ratio_summary(df)
    mobility_ratio_summary.to_csv(OUTPUT_ROOT / "mobility_ratio_summary.csv", index=False)

    representative_condition_summary = compute_representative_condition_summary(df)
    representative_condition_summary.to_csv(OUTPUT_ROOT / "representative_condition_summary.csv", index=False)

    feasibility_summary = compute_feasibility_summary(df)
    feasibility_summary.to_csv(OUTPUT_ROOT / "feasibility_summary.csv", index=False)

    rescue_harm = compute_lead_rescue_harm_by_condition(df)
    rescue_harm.to_csv(OUTPUT_ROOT / "lead_rescue_harm.csv", index=False)

    angle_summary = compute_residual_angle_summary_by_condition(angle_data["by_instance"])
    angle_summary.to_csv(OUTPUT_ROOT / "residual_angle_summary.csv", index=False)

    angle_by_outcome = compute_residual_angle_by_outcome_by_condition(df, angle_data["by_instance_seed"])
    angle_by_outcome.to_csv(OUTPUT_ROOT / "residual_angle_by_outcome.csv", index=False)

    failure_mode_summary = compute_failure_mode_summary(df)
    failure_mode_summary.to_csv(OUTPUT_ROOT / "failure_mode_summary.csv", index=False)

    post_detection_summary = compute_post_detection_summary_by_condition(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "post_detection_summary.csv", index=False)

    training_seed_stability = compute_training_seed_stability_by_condition(family_summary)
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
        "mobility_source_commit_before": commit_before,
        "mobility_source_commit_after": commit_after,
        "source_immutable": commit_before == commit_after,
        "branch": branch,
        "working_tree_clean_before": tree_clean_before,
        "working_tree_clean_after": tree_clean_after,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "n_episodes_per_condition_per_instance": len(SEEDS),
        "mobility_grid": [list(c) for c in MOBILITY_GRID],
        "nominal_condition": list(NOMINAL_CONDITION),
        "representative_conditions": {label: list(cond) for label, cond in REPRESENTATIVE_CONDITIONS},
        "n_instances": len(instances),
        "instances": instance_names,
        "expected_total_mission_evaluations": expected_rows,
        "actual_total_mission_evaluations": len(df),
        "model_hash_verification_before": hash_results_before,
        "model_hash_verification_after": hash_results_after,
        "model_hashes_identical_before_after": hashes_identical,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness": {
            "condition_seed_combinations_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall, "all_pass": n_mismatched == 0,
            "note": "fairness required WITHIN a fixed mobility condition only, not across conditions",
        },
        "physical_validity": physical_validity,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": [c[0] for c in PRIMARY_COMPARISONS],
        "residual_max_angle_deg": RESIDUAL_MAX_ANGLE_DEG,
        "is_robustness_sweep": True,
        "sweep_name": "mobility",
        "scientific_framing": "two matched 1-D speed arms (defender-speed, hostile-speed) sharing one nominal condition; acceleration/turn-rate/vertical-rate limits held fixed and NOT scaled with speed; Lead/PN constructed with the CURRENT per-condition config so their speed assumption tracks config.v_d, not a hardcoded nominal.",
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUTPUT_ROOT / "model_hashes.json").write_text(json.dumps(hash_results_before, indent=2, default=str))

    core_files = [
        "episode_results.csv", "classical_summary.csv", "learned_seed_summary.csv",
        "family_summary.csv", "primary_comparisons.csv", "secondary_comparisons.csv",
        "defender_speed_summary.csv", "hostile_speed_summary.csv", "feasibility_summary.csv",
        "lead_rescue_harm.csv", "protocol_manifest.json",
    ]
    evidence_hashes = {f: sha256_of_file(OUTPUT_ROOT / f) for f in core_files}
    (OUTPUT_ROOT / "evidence_hashes.json").write_text(json.dumps(evidence_hashes, indent=2))

    print(f"\nSource immutable: {commit_before == commit_after}")
    print(f"Model hashes identical before/after: {hashes_identical}")
    print(f"Acquisition fairness (within-condition): {n_mismatched} mismatched (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
