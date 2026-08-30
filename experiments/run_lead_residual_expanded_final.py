"""
LOCKED expanded final nominal evaluation runner for Lead-Residual PPO --
ONE-TIME, terminal 5000-seed evaluation (seeds 61000..65999) of the 12
paper-facing method instances (Greedy, PN, Lead, PPO-45/46/47,
PPOK-45/46/47, LR-PPO-55/56/57). This IS the primary final nominal
evaluation for the expanded six-method paper.

Reuses the SAME validated building blocks as the 200-seed diagnostic
(experiments/run_lead_residual_diagnostic.py, lead_residual_diagnostic_eval_utils.py)
-- same canonical mission runner, same policy-info sanitizer, same frozen
model table, same residual-composition/angle tracking -- scaled to the full
locked seed range, with additional required analyses (classical paired
comparisons, PPO-Kalman-vs-PPO family comparison, McNemar descriptive
statistic, validation/diagnostic/final generalization triple, final data
tables).

THIS IS A LOCKED, ONE-TIME RUN. No source modification is permitted between
the start and end of this script's execution.

Usage:
    python experiments/run_lead_residual_expanded_final.py
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

from experiments.lead_residual_expanded_final_protocol import (
    FINAL_SEED_START,
    FINAL_SEED_END,
    FINAL_EPISODES,
    FINAL_SEEDS,
    TRAINING_SEEDS,
    LR_TRAINING_ROOTS,
    FROZEN_MODELS,
    TRAINING_SOURCE_COMMIT,
    LR_PPO_TRAINING_SOURCE_COMMIT,
    LR_PPO_DIAGNOSTIC_SOURCE_COMMIT,
    SCRIPTED_METHODS,
    PPO_METHODS,
    LR_PPO_METHOD,
    RESIDUAL_MAX_ANGLE_DEG,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
    PRIMARY_COMPARISONS,
    EXISTING_PAPER_COMPARISONS,
    CLASSICAL_COMPARISONS,
    VALIDATION_BEST_SUCCESS_RATES,
    DIAGNOSTIC_200_SUCCESS_RATES,
    OUTPUT_ROOT,
    assert_seed_in_final_range,
    assert_not_reserved_elsewhere,
)
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode, sha256_of_file
from experiments.lead_residual_diagnostic_eval_utils import run_lr_ppo_diagnostic_episode
from experiments.run_lead_residual_diagnostic import (
    MethodInstance,
    build_instances,
    build_policy_and_env,
    compute_acquisition_fairness,
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
    mcnemar_exact,
    paired_bootstrap_ci,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_bootstrap_independent_families,
)

_CFG_DIRECT_DT = 0.5  # matches EnvConfig default; re-derived from env below to avoid hardcoding assumption
from uav_defend.config import EnvConfig  # noqa: E402
_CFG_DIRECT_DT = EnvConfig().dt


def run_final_evaluation(instances) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (episode_results_df, trajectories_by_seed_and_instance, angle_data)."""
    rows = []
    trajectories = {}
    angles_by_instance: dict[str, list[float]] = {}
    angles_by_instance_seed: dict[tuple[str, int], list[float]] = {}
    dt = _CFG_DIRECT_DT
    seeds = list(FINAL_SEEDS)
    assert len(seeds) == FINAL_EPISODES
    for seed in seeds:
        assert_seed_in_final_range(seed)
        assert_not_reserved_elsewhere(seed)

    total = len(instances)
    for i, instance in enumerate(instances):
        print(f"[{i+1}/{total}] {instance.display_name} ({instance.family}) -- {len(seeds)} episodes...", flush=True)
        env, policy = build_policy_and_env(instance)
        instance_angles = []
        for seed in seeds:
            if instance.family == "lr_ppo":
                row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
                step_angles = row.pop("step_angles")
                instance_angles.extend(step_angles)
                angles_by_instance_seed[(instance.display_name, seed)] = step_angles
            else:
                row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
            trajectory = row.pop("pre_detection_trajectory")
            trajectories[(seed, instance.display_name)] = trajectory
            row["method"] = instance.name
            row["instance"] = instance.display_name
            row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
            rows.append(row)
        if instance.family == "lr_ppo":
            angles_by_instance[instance.display_name] = instance_angles
        env.close()
        print(f"  ... done", flush=True)
    df = pd.DataFrame(rows)
    return df, trajectories, {"by_instance": angles_by_instance, "by_instance_seed": angles_by_instance_seed}


def compute_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in SCRIPTED_METHODS:
        sub = df[df["method"] == method]
        n = len(sub)
        successes = int(sub["success"].sum())
        lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
        rows.append({
            "instance": method, "method": method, "n_episodes": n, "success_count": successes,
            "success_rate": successes / n, "standard_error": standard_error(successes, n),
            "success_ci_low": lo, "success_ci_high": hi,
            "soldier_caught_rate": float(sub["soldier_caught"].mean()),
            "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
            "timeout_rate": float(sub["timeout"].mean()),
        })
    return pd.DataFrame(rows)


def compute_model_seed_summary(df: pd.DataFrame, method: str, training_seeds) -> pd.DataFrame:
    rows = []
    for seed in training_seeds:
        sub = df[(df["method"] == method) & (df["training_seed"] == seed)]
        n = len(sub)
        successes = int(sub["success"].sum())
        lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
        rows.append({
            "instance": sub["instance"].iloc[0], "method": method, "training_seed": seed,
            "n_episodes": n, "success_count": successes, "success_rate": successes / n,
            "standard_error": standard_error(successes, n), "success_ci_low": lo, "success_ci_high": hi,
            "soldier_caught_rate": float(sub["soldier_caught"].mean()),
            "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
            "timeout_rate": float(sub["timeout"].mean()),
        })
    return pd.DataFrame(rows)


def _success_vector(df: pd.DataFrame, method: str, training_seed=None) -> np.ndarray:
    sub = df[df["method"] == method]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, method: str, training_seeds) -> np.ndarray:
    return np.stack([_success_vector(df, method, s) for s in training_seeds])


def compute_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    families = [("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)]
    for method, seeds in families:
        matrix = _success_matrix(df, method, seeds)
        rates = matrix.mean(axis=1)
        observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({
            "method": method, "n_training_seeds": len(seeds),
            "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
            "min_seed_success": float(rates.min()), "max_seed_success": float(rates.max()),
            "range": float(rates.max() - rates.min()),
            "hierarchical_ci_low": lo, "hierarchical_ci_high": hi,
        })
    return pd.DataFrame(rows)


def compute_primary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    lead_vec = _success_vector(df, "lead")
    lr_matrix = _success_matrix(df, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    ppo_matrix = _success_matrix(df, "ppo", TRAINING_SEEDS)

    rows = []
    for name, method_a, method_b in PRIMARY_COMPARISONS:
        if method_b == "lead":
            observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        elif method_b == "ppo":
            observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        else:
            raise ValueError(name)
        rows.append({
            "comparison": name, "priority": "primary", "diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi),
        })
    return pd.DataFrame(rows)


def compute_existing_paper_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    greedy_vec = _success_vector(df, "greedy")
    pn_vec = _success_vector(df, "pn")
    lead_vec = _success_vector(df, "lead")
    ppo_matrix = _success_matrix(df, "ppo", TRAINING_SEEDS)
    ppok_matrix = _success_matrix(df, "ppo_kalman", TRAINING_SEEDS)

    rows = []
    for name, method_a, method_b in EXISTING_PAPER_COMPARISONS:
        if method_a == "ppo_kalman" and method_b == "ppo":
            from experiments.final_stats import hierarchical_paired_bootstrap_matched_seeds
            observed, lo, hi, per_seed_diffs = hierarchical_paired_bootstrap_matched_seeds(
                ppok_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL,
            )
        else:
            scripted_vec = {"greedy": greedy_vec, "pn": pn_vec, "lead": lead_vec}[method_b]
            observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_matrix, scripted_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({
            "comparison": name, "priority": "secondary_existing_paper", "diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi),
        })
    return pd.DataFrame(rows)


def compute_classical_paired_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for name, method_a, method_b in CLASSICAL_COMPARISONS:
        vec_a = _success_vector(df, method_a)
        vec_b = _success_vector(df, method_b)
        observed, lo, hi = paired_bootstrap_ci(vec_a, vec_b, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({
            "comparison": name, "priority": "secondary_classical", "diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi),
        })
    return pd.DataFrame(rows)


def compute_lr_root_vs_lead(df: pd.DataFrame) -> pd.DataFrame:
    lead_vec = _success_vector(df, "lead")
    rows = []
    for seed in LR_TRAINING_ROOTS:
        lr_vec = _success_vector(df, LR_PPO_METHOD, seed)
        observed, lo, hi = paired_bootstrap_ci(lr_vec, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        n10 = int(np.sum((lr_vec == 1) & (lead_vec == 0)))  # rescued
        n01 = int(np.sum((lr_vec == 0) & (lead_vec == 1)))  # harmed
        p_value = mcnemar_exact(n01=n01, n10=n10)
        rows.append({
            "comparison": f"LR-PPO-{seed} vs Lead", "priority": "secondary_per_root", "diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_includes_zero": bool(lo <= 0 <= hi),
            "rescued_b": n10, "harmed_c": n01, "mcnemar_exact_p_value": p_value,
        })
    return pd.DataFrame(rows)


def compute_generalization_summary(df: pd.DataFrame, lr_seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in lr_seed_summary.iterrows():
        seed = int(r["training_seed"])
        key = (LR_PPO_METHOD, seed)
        val = VALIDATION_BEST_SUCCESS_RATES[key]
        diag = DIAGNOSTIC_200_SUCCESS_RATES[key]
        final = r["success_rate"]
        rows.append({
            "instance": r["instance"], "validation_best": val, "diagnostic_200": diag, "final_5000": final,
            "diagnostic_minus_validation_pp": (diag - val) * 100,
            "final_minus_diagnostic_pp": (final - diag) * 100,
            "final_minus_validation_pp": (final - val) * 100,
        })
    return pd.DataFrame(rows)


def compute_final_paper_table(classical_summary: pd.DataFrame, family_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in classical_summary.iterrows():
        label = "PN" if r["method"] == "pn" else r["method"].capitalize()
        rows.append({
            "method": label, "success_rate": r["success_rate"], "ci_low": r["success_ci_low"], "ci_high": r["success_ci_high"],
            "soldier_caught_rate": r["soldier_caught_rate"], "unsafe_intercept_rate": r["unsafe_intercept_rate"],
            "timeout_rate": r["timeout_rate"],
        })
    label_map = {"ppo": "PPO", "ppo_kalman": "PPO-Kalman", "lr_ppo": "LR-PPO"}
    for _, r in family_summary.iterrows():
        rows.append({
            "method": label_map[r["method"]], "success_rate": r["mean_success"],
            "ci_low": r["hierarchical_ci_low"], "ci_high": r["hierarchical_ci_high"],
            "soldier_caught_rate": np.nan, "unsafe_intercept_rate": np.nan, "timeout_rate": np.nan,
        })
    return pd.DataFrame(rows)


def compute_final_comparison_table(primary: pd.DataFrame, existing: pd.DataFrame, classical: pd.DataFrame) -> pd.DataFrame:
    cols = ["comparison", "diff_pp", "ci_low_pp", "ci_high_pp"]
    return pd.concat([primary[cols], existing[cols], classical[cols]], ignore_index=True)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    commit_before = _get_git_commit()
    branch = _get_git_branch()
    tree_clean_before = _get_git_status_clean()
    print(f"branch={branch} commit_before={commit_before} tree_clean_before={tree_clean_before}")

    print("\nVerifying model hashes (9 learned models)...")
    hash_results = verify_model_hashes()
    print(json.dumps(hash_results, indent=2))

    print("\nGround-truth leakage audit (measurement/kalman sanitizer -- LR-PPO reuses 'measurement')...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_instances()
    instance_names = [i.display_name for i in instances]
    assert len(instances) == 12

    print(f"\nRunning EXPANDED FINAL evaluation: {len(instances)} instances x {FINAL_EPISODES} episodes "
          f"(seeds {FINAL_SEED_START}..{FINAL_SEED_END})...")
    df, trajectories, angle_data = run_final_evaluation(instances)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)
    assert len(df) == len(instances) * FINAL_EPISODES

    print("\nComputing acquisition fairness (5000 seeds)...")
    fairness_df = compute_acquisition_fairness(trajectories, instance_names, list(FINAL_SEEDS))
    fairness_df.to_csv(OUTPUT_ROOT / "acquisition_fairness.csv", index=False)
    n_mismatched = int(fairness_df["mismatch"].sum())
    max_discrepancy_overall = float(fairness_df["max_physical_discrepancy"].max())
    detection_step_equal = bool(fairness_df["all_detection_steps_equal"].all())
    print(f"  seeds checked: {len(fairness_df)}, mismatched: {n_mismatched}, max discrepancy: {max_discrepancy_overall}")

    classical_summary = compute_classical_summary(df)
    ppo_seed_summary = compute_model_seed_summary(df, "ppo", TRAINING_SEEDS)
    ppok_seed_summary = compute_model_seed_summary(df, "ppo_kalman", TRAINING_SEEDS)
    lr_seed_summary = compute_model_seed_summary(df, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    learned_seed_summary = pd.concat([ppo_seed_summary, ppok_seed_summary, lr_seed_summary], ignore_index=True)
    classical_summary.to_csv(OUTPUT_ROOT / "classical_summary.csv", index=False)
    learned_seed_summary.to_csv(OUTPUT_ROOT / "learned_seed_summary.csv", index=False)

    family_summary = compute_family_summary(df)
    family_summary.to_csv(OUTPUT_ROOT / "family_summary.csv", index=False)

    primary_comparisons = compute_primary_comparisons(df)
    primary_comparisons.to_csv(OUTPUT_ROOT / "primary_comparisons.csv", index=False)

    existing_paper_comparisons = compute_existing_paper_comparisons(df)
    classical_paired = compute_classical_paired_comparisons(df)
    lr_root_vs_lead = compute_lr_root_vs_lead(df)
    secondary_comparisons = pd.concat(
        [existing_paper_comparisons, classical_paired, lr_root_vs_lead], ignore_index=True, sort=False,
    )
    secondary_comparisons.to_csv(OUTPUT_ROOT / "secondary_comparisons.csv", index=False)

    rescue_harm = compute_lead_rescue_harm(df)
    rescue_harm.to_csv(OUTPUT_ROOT / "lead_rescue_harm.csv", index=False)

    angle_summary = compute_residual_angle_summary(angle_data["by_instance"])
    angle_summary.to_csv(OUTPUT_ROOT / "residual_angle_summary.csv", index=False)

    angle_by_outcome = compute_residual_angle_by_outcome(df, angle_data["by_instance_seed"])
    angle_by_outcome.to_csv(OUTPUT_ROOT / "residual_angle_by_outcome.csv", index=False)

    detection_summary = compute_detection_time_summary(df)
    (OUTPUT_ROOT / "detection_time_summary.json").write_text(json.dumps(detection_summary, indent=2, default=str))

    post_detection_summary = compute_post_detection_summary(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "post_detection_summary.csv", index=False)

    physical_validity = compute_physical_validity(df)
    (OUTPUT_ROOT / "physical_validity.json").write_text(json.dumps(physical_validity, indent=2, default=str))
    if physical_validity["total_violations"] > 0:
        print("\n*** PHYSICAL LIMIT VIOLATION DETECTED -- STOP PER PROTOCOL. DO NOT PATCH AND RERUN. ***")

    generalization_summary = compute_generalization_summary(df, lr_seed_summary)
    generalization_summary.to_csv(OUTPUT_ROOT / "generalization_summary.csv", index=False)

    final_paper_table = compute_final_paper_table(classical_summary, family_summary)
    final_paper_table.to_csv(OUTPUT_ROOT / "final_paper_table.csv", index=False)

    final_comparison_table = compute_final_comparison_table(primary_comparisons, existing_paper_comparisons, classical_paired)
    final_comparison_table.to_csv(OUTPUT_ROOT / "final_comparison_table.csv", index=False)

    (OUTPUT_ROOT / "ground_truth_audit.json").write_text(json.dumps(leakage_audit, indent=2, default=str))

    total_violations = int(df["physical_limit_violations"].sum())

    commit_after = _get_git_commit()
    tree_clean_after = _get_git_status_clean()
    hash_results_after = verify_model_hashes()
    hashes_identical_after = all(
        hash_results[k]["actual"] == hash_results_after[k]["actual"] for k in hash_results
    )

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "lr_ppo_training_source_commit": LR_PPO_TRAINING_SOURCE_COMMIT,
        "lr_ppo_diagnostic_source_commit": LR_PPO_DIAGNOSTIC_SOURCE_COMMIT,
        "expanded_final_evaluation_source_commit_before": commit_before,
        "expanded_final_evaluation_source_commit_after": commit_after,
        "source_immutable": commit_before == commit_after,
        "branch": branch,
        "working_tree_clean_before": tree_clean_before,
        "working_tree_clean_after": tree_clean_after,
        "final_seed_range": [FINAL_SEED_START, FINAL_SEED_END],
        "n_episodes_per_instance": FINAL_EPISODES,
        "n_instances": len(instances),
        "instances": instance_names,
        "total_mission_evaluations": len(df),
        "model_hash_verification_before": hash_results,
        "model_hash_verification_after": hash_results_after,
        "model_hashes_identical_before_after": hashes_identical_after,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness": {
            "seeds_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall,
            "detection_step_equality": detection_step_equal,
            "all_pass": n_mismatched == 0,
        },
        "detection_time_summary": detection_summary,
        "physical_validity": physical_validity,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": [c[0] for c in PRIMARY_COMPARISONS],
        "residual_max_angle_deg": RESIDUAL_MAX_ANGLE_DEG,
        "is_final_nominal_paper_evaluation": True,
        "not_robustness_experiment": True,
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUTPUT_ROOT / "model_hashes.json").write_text(json.dumps(hash_results, indent=2, default=str))

    core_files = [
        "episode_results.csv", "classical_summary.csv", "learned_seed_summary.csv",
        "family_summary.csv", "primary_comparisons.csv", "secondary_comparisons.csv",
        "lead_rescue_harm.csv", "protocol_manifest.json",
    ]
    evidence_hashes = {f: sha256_of_file(OUTPUT_ROOT / f) for f in core_files}
    (OUTPUT_ROOT / "evidence_hashes.json").write_text(json.dumps(evidence_hashes, indent=2))

    print(f"\nSource immutable (commit before == after): {commit_before == commit_after}")
    print(f"Model hashes identical before/after: {hashes_identical_after}")
    print(f"Acquisition fairness: {n_mismatched} mismatched seeds (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
