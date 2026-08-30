"""
Hostile-evasion expanded robustness sweep -- LOCKED, ONE-TIME evaluation of
the 12 paper-facing method instances across 5 enemy_evasion_gain conditions
on the previously reserved seed range 66000..66999 (1000 matched seeds).

Scientific question: does LR-PPO's advantage over analytical Lead persist as
the reactive hostile-evasion component changes in strength? gain=0.0 means
zero REACTIVE-AWAY contribution -- NOT a non-maneuvering hostile (pursuit,
weave, and stochastic heading/process perturbation remain active at every
gain). enemy_evasion_enabled=True at every gain including 0.0.

Reuses the SAME validated building blocks as the expanded final evaluation
(episode runner, sanitizer, frozen model table, hierarchical bootstrap
statistics) -- only the swept parameter and seed range are new.

Usage:
    python experiments/run_hostile_evasion_robustness_sweep.py
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
    TRAINING_SOURCE_COMMIT,
    LR_PPO_TRAINING_SOURCE_COMMIT,
    LR_PPO_DIAGNOSTIC_SOURCE_COMMIT,
    EXPANDED_FINAL_SOURCE_COMMIT,
    SCRIPTED_METHODS,
    PPO_METHODS,
    LR_PPO_METHOD,
    TRAINING_SEEDS,
    LR_TRAINING_ROOTS,
    RESIDUAL_MAX_ANGLE_DEG,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    HOSTILE_EVASION_SEED_START,
    HOSTILE_EVASION_SEED_END,
    HOSTILE_EVASION_EPISODES,
    EVASION_GAIN_VALUES,
    NOMINAL_EVASION_GAIN,
    NOMINAL_EVASION_RADIUS,
    build_hostile_evasion_config,
    assert_seed_in_hostile_evasion_range,
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
    compute_detection_time_summary,
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

OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_robustness" / "hostile_evasion"
SEEDS = list(range(HOSTILE_EVASION_SEED_START, HOSTILE_EVASION_SEED_END + 1))


def build_policy_for_instance(instance, config):
    if instance.family == "scripted":
        return get_policy(instance.name)
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
    """Runs all (gain x instance x seed) missions. Returns
    (episode_results_df, trajectories_by_gain, angle_data_by_gain).

    production=True (default, used for the real 66000-66999 run): each seed
    must fall exactly WITHIN the locked hostile_evasion range (positive
    membership check -- this is what authorizes opening the range).
    production=False (used for dev/smoke tasks on seeds like 123-127): each
    seed must NOT fall in ANY reserved (not-yet-opened) range. Both modes
    also enforce disjointness from all prior/already-consumed ranges."""
    for s in seeds:
        if production:
            assert_seed_in_hostile_evasion_range(s)
        else:
            assert_seed_not_yet_opened(s)
        assert_disjoint_from_all_prior_ranges(s)

    rows = []
    trajectories_by_gain: dict[float, dict] = {}
    angle_data_by_gain: dict[float, dict] = {"by_instance": {}, "by_instance_seed": {}}

    total = len(EVASION_GAIN_VALUES) * len(instances)
    i = 0
    for gain in EVASION_GAIN_VALUES:
        trajectories = {}
        for instance in instances:
            i += 1
            estimator = "kalman" if instance.family == "ppo_kalman" else "measurement"
            config = build_hostile_evasion_config(estimator, gain)
            env = SoldierEnv(config=config)
            policy = build_policy_for_instance(instance, config)
            dt = config.dt
            print(f"[{i}/{total}] gain={gain} {instance.display_name} ({instance.family}) -- {len(seeds)} episodes...", flush=True)
            instance_angles = []
            for seed in seeds:
                if instance.family == "lr_ppo":
                    row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                    step_angles = row.pop("step_angles")
                    instance_angles.extend(step_angles)
                    angle_data_by_gain["by_instance_seed"][(gain, instance.display_name, seed)] = step_angles
                else:
                    row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                trajectory = row.pop("pre_detection_trajectory")
                trajectories[(seed, instance.display_name)] = trajectory
                row["method"] = instance.name
                row["instance"] = instance.display_name
                row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
                row["gain"] = gain
                rows.append(row)
            if instance.family == "lr_ppo":
                key = (gain, instance.display_name)
                angle_data_by_gain["by_instance"][key] = angle_data_by_gain["by_instance"].get(key, []) + instance_angles
            env.close()
        trajectories_by_gain[gain] = trajectories

    df = pd.DataFrame(rows)
    return df, trajectories_by_gain, angle_data_by_gain


def compute_acquisition_fairness_all_gains(trajectories_by_gain: dict, instance_names: list[str], seeds: list[int]) -> pd.DataFrame:
    rows = []
    for gain, trajectories in trajectories_by_gain.items():
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
                "gain": gain, "environment_seed": seed,
                "detection_step": detection_steps[ref_name],
                "all_detection_steps_equal": len(set(detection_steps.values())) == 1,
                "max_physical_discrepancy": max_discrepancy,
                "defender_soldier_separation_at_detection": sep,
                "defender_speed_at_detection": float(np.linalg.norm(final_state["defender_vel"])),
                "mismatch": mismatch,
            })
    return pd.DataFrame(rows)


def _success_vector(df: pd.DataFrame, gain: float, method: str, training_seed=None) -> np.ndarray:
    sub = df[(df["gain"] == gain) & (df["method"] == method)]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, gain: float, method: str, training_seeds) -> np.ndarray:
    return np.stack([_success_vector(df, gain, method, s) for s in training_seeds])


def compute_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        for method in SCRIPTED_METHODS:
            sub = df[(df["gain"] == gain) & (df["method"] == method)]
            n = len(sub)
            successes = int(sub["success"].sum())
            lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
            rows.append({
                "gain": gain, "instance": method, "method": method, "n": n, "success_count": successes,
                "success_rate": successes / n, "standard_error": standard_error(successes, n),
                "ci_low": lo, "ci_high": hi,
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_learned_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            for seed in seeds:
                sub = df[(df["gain"] == gain) & (df["method"] == method) & (df["training_seed"] == seed)]
                n = len(sub)
                successes = int(sub["success"].sum())
                lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
                rows.append({
                    "gain": gain, "instance": sub["instance"].iloc[0], "method": method, "training_seed": seed,
                    "n": n, "success_count": successes, "success_rate": successes / n,
                    "standard_error": standard_error(successes, n), "ci_low": lo, "ci_high": hi,
                    "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                    "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                    "timeout_rate": float(sub["timeout"].mean()),
                })
    return pd.DataFrame(rows)


def compute_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
            matrix = _success_matrix(df, gain, method, seeds)
            rates = matrix.mean(axis=1)
            observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({
                "gain": gain, "method": method, "n_training_seeds": len(seeds),
                "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
                "min_seed_success": float(rates.min()), "max_seed_success": float(rates.max()),
                "range": float(rates.max() - rates.min()),
                "hierarchical_ci_low": lo, "hierarchical_ci_high": hi,
            })
    return pd.DataFrame(rows)


def compute_primary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        lead_vec = _success_vector(df, gain, "lead")
        lr_matrix = _success_matrix(df, gain, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        ppo_matrix = _success_matrix(df, gain, "ppo", TRAINING_SEEDS)
        for name, method_a, method_b in PRIMARY_COMPARISONS:
            if method_b == "lead":
                observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            else:
                observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
            rows.append({"gain": gain, "comparison": name, "diff_pp": observed * 100, "ci_low_pp": lo * 100,
                         "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_secondary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        greedy_vec = _success_vector(df, gain, "greedy")
        pn_vec = _success_vector(df, gain, "pn")
        lead_vec = _success_vector(df, gain, "lead")
        ppo_matrix = _success_matrix(df, gain, "ppo", TRAINING_SEEDS)
        ppok_matrix = _success_matrix(df, gain, "ppo_kalman", TRAINING_SEEDS)
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
            rows.append({"gain": gain, "comparison": name, "diff_pp": observed * 100, "ci_low_pp": lo * 100,
                         "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_within_method_evasion_effects(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    lead_nominal = _success_vector(df, NOMINAL_EVASION_GAIN, "lead")
    ppo_nominal = _success_matrix(df, NOMINAL_EVASION_GAIN, "ppo", TRAINING_SEEDS)
    lr_nominal = _success_matrix(df, NOMINAL_EVASION_GAIN, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    for gain in EVASION_GAIN_VALUES:
        if gain == NOMINAL_EVASION_GAIN:
            continue
        lead_vec = _success_vector(df, gain, "lead")
        observed, lo, hi = paired_bootstrap_ci(lead_vec, lead_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lead", "gain": gain, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        ppo_matrix = _success_matrix(df, gain, "ppo", TRAINING_SEEDS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(ppo_matrix, ppo_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "ppo", "gain": gain, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})

        lr_matrix = _success_matrix(df, gain, LR_PPO_METHOD, LR_TRAINING_ROOTS)
        observed, lo, hi, _ = hierarchical_paired_bootstrap_matched_seeds(lr_matrix, lr_nominal, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": "lr_ppo", "gain": gain, "delta_success_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100})
    return pd.DataFrame(rows)


def compute_lead_rescue_harm_by_gain(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        sub = df[df["gain"] == gain]
        r = compute_lead_rescue_harm(sub)
        r.insert(0, "gain", gain)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_summary_by_gain(angle_data_by_instance: dict) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        scoped = {name: vals for (g, name), vals in angle_data_by_instance.items() if g == gain}
        r = compute_residual_angle_summary(scoped)
        r.insert(0, "gain", gain)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_residual_angle_by_outcome_by_gain(df: pd.DataFrame, angle_data_by_instance_seed: dict) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        sub = df[df["gain"] == gain]
        scoped = {(name, seed): vals for (g, name, seed), vals in angle_data_by_instance_seed.items() if g == gain}
        r = compute_residual_angle_by_outcome(sub, scoped)
        r.insert(0, "gain", gain)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_failure_mode_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        for instance in df[df["gain"] == gain]["instance"].unique():
            sub = df[(df["gain"] == gain) & (df["instance"] == instance)]
            n = len(sub)
            rows.append({
                "gain": gain, "instance": instance, "n": n,
                "safe_intercept_rate": float(sub["success"].mean()),
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
            })
    return pd.DataFrame(rows)


def compute_post_detection_summary_by_gain(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for gain in EVASION_GAIN_VALUES:
        sub = df[df["gain"] == gain]
        r = compute_post_detection_summary(sub)
        r.insert(0, "gain", gain)
        rows.append(r)
    return pd.concat(rows, ignore_index=True)


def compute_training_seed_stability_by_gain(family_summary: pd.DataFrame) -> pd.DataFrame:
    return family_summary[["gain", "method", "between_seed_sd"]].copy()


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commit_before = _get_git_commit()
    branch = _get_git_branch()
    tree_clean_before = _get_git_status_clean()
    print(f"branch={branch} commit_before={commit_before} tree_clean_before={tree_clean_before}")

    print("\nVerifying model hashes (9 learned models)...")
    hash_results_before = verify_model_hashes()
    print(json.dumps(hash_results_before, indent=2))
    assert all(v["match"] for v in hash_results_before.values()), "MODEL HASH MISMATCH -- STOP BEFORE SEED 66000"

    print("\nGround-truth leakage audit...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_method_instances()
    instance_names = [i.display_name for i in instances]
    assert len(instances) == 12

    print(f"\nRunning HOSTILE-EVASION robustness sweep: {len(EVASION_GAIN_VALUES)} gains x "
          f"{len(instances)} instances x {len(SEEDS)} seeds (seeds {SEEDS[0]}..{SEEDS[-1]})...")
    df, trajectories_by_gain, angle_data = run_full_sweep(instances, SEEDS, production=True)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)
    expected_rows = len(EVASION_GAIN_VALUES) * len(instances) * len(SEEDS)
    assert len(df) == expected_rows, f"expected {expected_rows} rows, got {len(df)}"

    print("\nComputing acquisition fairness (5 gains x 1000 seeds = 5000 combinations)...")
    fairness_df = compute_acquisition_fairness_all_gains(trajectories_by_gain, instance_names, SEEDS)
    fairness_df.to_csv(OUTPUT_ROOT / "acquisition_fairness.csv", index=False)
    n_mismatched = int(fairness_df["mismatch"].sum())
    max_discrepancy_overall = float(fairness_df["max_physical_discrepancy"].max())
    print(f"  gain-seed combinations checked: {len(fairness_df)}, mismatched: {n_mismatched}, max discrepancy: {max_discrepancy_overall}")

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

    within_method_effects = compute_within_method_evasion_effects(df)
    within_method_effects.to_csv(OUTPUT_ROOT / "within_method_evasion_effects.csv", index=False)

    rescue_harm = compute_lead_rescue_harm_by_gain(df)
    rescue_harm.to_csv(OUTPUT_ROOT / "lead_rescue_harm.csv", index=False)

    angle_summary = compute_residual_angle_summary_by_gain(angle_data["by_instance"])
    angle_summary.to_csv(OUTPUT_ROOT / "residual_angle_summary.csv", index=False)

    angle_by_outcome = compute_residual_angle_by_outcome_by_gain(df, angle_data["by_instance_seed"])
    angle_by_outcome.to_csv(OUTPUT_ROOT / "residual_angle_by_outcome.csv", index=False)

    failure_mode_summary = compute_failure_mode_summary(df)
    failure_mode_summary.to_csv(OUTPUT_ROOT / "failure_mode_summary.csv", index=False)

    post_detection_summary = compute_post_detection_summary_by_gain(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "post_detection_summary.csv", index=False)

    training_seed_stability = compute_training_seed_stability_by_gain(family_summary)
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
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "lr_ppo_training_source_commit": LR_PPO_TRAINING_SOURCE_COMMIT,
        "lr_ppo_diagnostic_source_commit": LR_PPO_DIAGNOSTIC_SOURCE_COMMIT,
        "expanded_final_source_commit": EXPANDED_FINAL_SOURCE_COMMIT,
        "hostile_evasion_source_commit_before": commit_before,
        "hostile_evasion_source_commit_after": commit_after,
        "source_immutable": commit_before == commit_after,
        "branch": branch,
        "working_tree_clean_before": tree_clean_before,
        "working_tree_clean_after": tree_clean_after,
        "seed_range": [SEEDS[0], SEEDS[-1]],
        "n_episodes_per_condition_per_instance": len(SEEDS),
        "evasion_gain_values": list(EVASION_GAIN_VALUES),
        "nominal_evasion_gain": NOMINAL_EVASION_GAIN,
        "evasion_radius": NOMINAL_EVASION_RADIUS,
        "n_instances": len(instances),
        "instances": instance_names,
        "expected_total_mission_evaluations": expected_rows,
        "actual_total_mission_evaluations": len(df),
        "model_hash_verification_before": hash_results_before,
        "model_hash_verification_after": hash_results_after,
        "model_hashes_identical_before_after": hashes_identical,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness": {
            "gain_seed_combinations_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall, "all_pass": n_mismatched == 0,
        },
        "physical_validity": physical_validity,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": [c[0] for c in PRIMARY_COMPARISONS],
        "residual_max_angle_deg": RESIDUAL_MAX_ANGLE_DEG,
        "is_robustness_sweep": True,
        "sweep_name": "hostile_evasion",
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUTPUT_ROOT / "model_hashes.json").write_text(json.dumps(hash_results_before, indent=2, default=str))

    core_files = [
        "episode_results.csv", "classical_summary.csv", "learned_seed_summary.csv",
        "family_summary.csv", "primary_comparisons.csv", "secondary_comparisons.csv",
        "lead_rescue_harm.csv", "protocol_manifest.json",
    ]
    evidence_hashes = {f: sha256_of_file(OUTPUT_ROOT / f) for f in core_files}
    (OUTPUT_ROOT / "evidence_hashes.json").write_text(json.dumps(evidence_hashes, indent=2))

    print(f"\nSource immutable: {commit_before == commit_after}")
    print(f"Model hashes identical before/after: {hashes_identical}")
    print(f"Acquisition fairness: {n_mismatched} mismatched (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
