"""
Lead-Residual PPO (LR-PPO) 200-seed independent diagnostic evaluation runner
-- ONE-TIME integrity/generalization check on the previously reserved
diagnostic range (60100-60299). NOT the final expanded-paper test
(61000-65999, still sealed). See
experiments/lead_residual_diagnostic_protocol.py for the locked protocol.

Evaluates 12 method instances (Greedy, PN, Lead, PPO-45/46/47,
PPOK-45/46/47, LR-PPO-55/56/57) on the SAME 200 environment seeds.

Usage:
    python experiments/run_lead_residual_diagnostic.py
"""

from __future__ import annotations

import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.lead_residual_diagnostic_protocol import (
    DIAGNOSTIC_SEEDS,
    DIAGNOSTIC_EPISODES,
    TRAINING_SEEDS,
    LR_TRAINING_ROOTS,
    FROZEN_MODELS,
    TRAINING_SOURCE_COMMIT,
    LR_PPO_TRAINING_SOURCE_COMMIT,
    SCRIPTED_METHODS,
    PPO_METHODS,
    LR_PPO_METHOD,
    RESIDUAL_MAX_ANGLE_DEG,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    PRIMARY_COMPARISONS,
    SECONDARY_COMPARISONS,
    VALIDATION_BEST_SUCCESS_RATES,
    OUTPUT_ROOT,
    assert_not_reserved_seed,
)
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode, sha256_of_file
from experiments.lead_residual_diagnostic_eval_utils import run_lr_ppo_diagnostic_episode
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

_CFG_DIRECT = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)
_CFG_KALMAN = EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True)


def verify_model_hashes() -> dict:
    results = {}
    mismatches = []
    for (track, seed), spec in FROZEN_MODELS.items():
        path = Path(spec["path"])
        if not path.exists():
            raise FileNotFoundError(f"Frozen model not found: {path}")
        actual = sha256_of_file(path)
        match = actual == spec["sha256"]
        results[f"{track}_{seed}"] = {"expected": spec["sha256"], "actual": actual, "match": match}
        if not match:
            mismatches.append((track, seed, spec["sha256"], actual))
    if mismatches:
        raise RuntimeError(f"MODEL HASH MISMATCH -- STOP, DO NOT EVALUATE: {mismatches}")
    return results


@dataclasses.dataclass(frozen=True)
class MethodInstance:
    name: str
    display_name: str
    family: str  # "scripted" | "ppo" | "ppo_kalman" | "lr_ppo"
    training_seed: int | None


def build_instances() -> tuple[MethodInstance, ...]:
    instances = [
        MethodInstance("greedy", "Greedy", "scripted", None),
        MethodInstance("pn", "PN", "scripted", None),
        MethodInstance("lead", "Lead", "scripted", None),
    ]
    for seed in TRAINING_SEEDS:
        instances.append(MethodInstance("ppo", f"PPO-{seed}", "ppo", seed))
    for seed in TRAINING_SEEDS:
        instances.append(MethodInstance("ppo_kalman", f"PPOK-{seed}", "ppo_kalman", seed))
    for seed in LR_TRAINING_ROOTS:
        instances.append(MethodInstance("lr_ppo", f"LR-PPO-{seed}", "lr_ppo", seed))
    return tuple(instances)


def build_policy_and_env(instance: MethodInstance):
    if instance.family == "scripted":
        env = SoldierEnv(config=_CFG_DIRECT)
        policy = get_policy(instance.name)
    elif instance.family == "ppo":
        env = SoldierEnv(config=_CFG_DIRECT)
        model_path = FROZEN_MODELS[("direct", instance.training_seed)]["path"]
        policy = get_policy("ppo", model_path=str(model_path), deterministic=True)
    elif instance.family == "ppo_kalman":
        env = SoldierEnv(config=_CFG_KALMAN)
        model_path = FROZEN_MODELS[("kalman", instance.training_seed)]["path"]
        policy = get_policy("ppo_kalman", model_path=str(model_path), deterministic=True)
    elif instance.family == "lr_ppo":
        env = SoldierEnv(config=_CFG_DIRECT)
        model_path = FROZEN_MODELS[("lr_ppo", instance.training_seed)]["path"]
        model = PPO.load(str(model_path), device="cpu")
        policy = LeadResidualPPOPolicyWrapper(
            model, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG, config=_CFG_DIRECT, deterministic=True,
        )
    else:
        raise ValueError(instance.family)
    return env, policy


def run_evaluation(instances) -> tuple[pd.DataFrame, dict, dict]:
    """Returns (episode_results_df, trajectories_by_seed_and_instance, angles_by_instance)."""
    rows = []
    trajectories = {}
    angles_by_instance: dict[str, list[float]] = {}
    angles_by_instance_seed: dict[tuple[str, int], list[float]] = {}
    dt = _CFG_DIRECT.dt
    seeds = list(DIAGNOSTIC_SEEDS)
    for seed in seeds:
        assert_not_reserved_seed(seed)

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
    df = pd.DataFrame(rows)
    return df, trajectories, {"by_instance": angles_by_instance, "by_instance_seed": angles_by_instance_seed}


def compute_acquisition_fairness(trajectories: dict, instance_names: list[str], seeds) -> pd.DataFrame:
    """Parametrized by an explicit seed list (unlike run_post_detection_diagnostic's
    module-constant-bound version) -- see run_final_nominal_post_detection_evaluation.py
    for the same pattern."""
    rows = []
    for seed in seeds:
        trajs = {name: trajectories[(seed, name)] for name in instance_names}
        ref_name = instance_names[0]
        ref_traj = trajs[ref_name]
        detection_steps = {name: (t[-1]["step"] if t[-1]["enemy_detected"] else None) for name, t in trajs.items()}
        same_length = all(len(trajs[name]) == len(ref_traj) for name in instance_names)
        max_discrepancy = 0.0
        mismatch = not same_length
        if same_length:
            for i in range(len(ref_traj)):
                ref_state = ref_traj[i]
                for name in instance_names[1:]:
                    s = trajs[name][i]
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
            "environment_seed": seed,
            "detection_step": detection_steps[ref_name],
            "all_detection_steps_equal": len(set(detection_steps.values())) == 1,
            "max_physical_discrepancy": max_discrepancy,
            "defender_soldier_separation_at_detection": sep,
            "defender_speed_at_detection": float(np.linalg.norm(final_state["defender_vel"])),
            "mismatch": mismatch,
        })
    return pd.DataFrame(rows)


def compute_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in SCRIPTED_METHODS:
        sub = df[df["method"] == method]
        n = len(sub)
        successes = int(sub["success"].sum())
        lo, hi = wilson_interval(successes, n)
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
        lo, hi = wilson_interval(successes, n)
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
        observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        rows.append({
            "method": method, "n_training_seeds": len(seeds),
            "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
            "min_seed_success": float(rates.min()), "max_seed_success": float(rates.max()),
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
            observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        elif method_b == "ppo":
            observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        else:
            raise ValueError(name)
        rows.append({
            "comparison": name, "priority": "primary", "diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows)


def compute_secondary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    lead_vec = _success_vector(df, "lead")
    pn_vec = _success_vector(df, "pn")
    greedy_vec = _success_vector(df, "greedy")
    ppo_matrix = _success_matrix(df, "ppo", TRAINING_SEEDS)

    rows = []
    for seed in LR_TRAINING_ROOTS:
        lr_vec = _success_vector(df, LR_PPO_METHOD, seed)
        observed, lo, hi = paired_bootstrap_ci(lr_vec, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        rows.append({
            "comparison": f"LR-PPO-{seed} vs Lead", "priority": "secondary", "diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })

    for name, method_a, method_b in SECONDARY_COMPARISONS:
        scripted_vec = {"greedy": greedy_vec, "pn": pn_vec, "lead": lead_vec}[method_b]
        observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_matrix, scripted_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        rows.append({
            "comparison": name, "priority": "secondary", "diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100, "ci_excludes_zero": bool(lo > 0 or hi < 0),
        })
    return pd.DataFrame(rows)


def compute_lead_rescue_harm(df: pd.DataFrame) -> pd.DataFrame:
    lead_sub = df[df["method"] == "lead"].sort_values("environment_seed").set_index("environment_seed")["success"]
    rows = []
    for seed in LR_TRAINING_ROOTS:
        lr_sub = df[(df["method"] == LR_PPO_METHOD) & (df["training_seed"] == seed)].sort_values("environment_seed").set_index("environment_seed")["success"]
        both_success = int(((lead_sub == 1) & (lr_sub == 1)).sum())
        rescued = int(((lead_sub == 0) & (lr_sub == 1)).sum())
        harmed = int(((lead_sub == 1) & (lr_sub == 0)).sum())
        both_failure = int(((lead_sub == 0) & (lr_sub == 0)).sum())
        n = len(lead_sub)
        lr_rate = float(lr_sub.mean())
        lead_rate = float(lead_sub.mean())
        expected_diff = (rescued - harmed) / n
        actual_diff = lr_rate - lead_rate
        rows.append({
            "lr_instance": f"LR-PPO-{seed}", "n": n,
            "both_success": both_success, "both_success_pct": 100 * both_success / n,
            "rescued_by_residual": rescued, "rescued_pct": 100 * rescued / n,
            "harmed_by_residual": harmed, "harmed_pct": 100 * harmed / n,
            "both_failure": both_failure, "both_failure_pct": 100 * both_failure / n,
            "lr_success_rate": lr_rate, "lead_success_rate": lead_rate,
            "actual_diff": actual_diff, "expected_diff_rescued_minus_harmed_over_n": expected_diff,
            "algebra_check_pass": bool(abs(actual_diff - expected_diff) < 1e-9),
        })
    return pd.DataFrame(rows)


def compute_residual_angle_summary(angles_by_instance: dict) -> pd.DataFrame:
    rows = []
    for seed in LR_TRAINING_ROOTS:
        name = f"LR-PPO-{seed}"
        angles = np.asarray(angles_by_instance[name], dtype=np.float64)
        rows.append({
            "instance": name, "n_controlled_steps": len(angles),
            "mean_deg": float(angles.mean()), "median_deg": float(np.median(angles)),
            "p25_deg": float(np.percentile(angles, 25)), "p75_deg": float(np.percentile(angles, 75)),
            "p95_deg": float(np.percentile(angles, 95)), "max_deg": float(angles.max()),
            "frac_gt15": float(np.mean(angles > 15.0)), "frac_gt25": float(np.mean(angles > 25.0)),
            "frac_gt29": float(np.mean(angles > 29.0)),
            "angular_violations_gt30": int(np.sum(angles > RESIDUAL_MAX_ANGLE_DEG + 1e-3)),
        })
    return pd.DataFrame(rows)


def compute_residual_angle_by_outcome(df: pd.DataFrame, angles_by_instance_seed: dict) -> pd.DataFrame:
    lead_sub = df[df["method"] == "lead"].sort_values("environment_seed").set_index("environment_seed")["success"]
    rows = []
    for seed in LR_TRAINING_ROOTS:
        name = f"LR-PPO-{seed}"
        lr_sub = df[(df["method"] == LR_PPO_METHOD) & (df["training_seed"] == seed)].sort_values("environment_seed").set_index("environment_seed")["success"]
        classes = {
            "both_success": (lead_sub == 1) & (lr_sub == 1),
            "rescued": (lead_sub == 0) & (lr_sub == 1),
            "harmed": (lead_sub == 1) & (lr_sub == 0),
            "both_failure": (lead_sub == 0) & (lr_sub == 0),
        }
        for cls_name, mask in classes.items():
            env_seeds = mask[mask].index.tolist()
            pooled = []
            for env_seed in env_seeds:
                pooled.extend(angles_by_instance_seed.get((name, env_seed), []))
            pooled = np.asarray(pooled, dtype=np.float64)
            rows.append({
                "instance": name, "outcome_class": cls_name, "n_episodes": len(env_seeds),
                "n_controlled_steps": len(pooled),
                "mean_deg": float(pooled.mean()) if len(pooled) else None,
                "median_deg": float(np.median(pooled)) if len(pooled) else None,
            })
    return pd.DataFrame(rows)


def compute_physical_validity(df: pd.DataFrame) -> dict:
    total_violations = int(df["physical_limit_violations"].sum())
    affected = df[df["physical_limit_violations"] > 0]
    return {
        "total_mission_evaluations": len(df),
        "total_violations": total_violations,
        "max_exceedance": 0.0 if total_violations == 0 else "UNQUANTIFIED -- SEE affected_rows; STOP PER PROTOCOL",
        "affected_rows": affected[["environment_seed", "instance", "physical_limit_violations"]].to_dict(orient="records"),
    }


def compute_generalization_check(family_summary: pd.DataFrame, model_seed_summaries: dict) -> pd.DataFrame:
    rows = []
    lr_seed_summary = model_seed_summaries[LR_PPO_METHOD]
    for _, r in lr_seed_summary.iterrows():
        key = (LR_PPO_METHOD, int(r["training_seed"]))
        val = VALIDATION_BEST_SUCCESS_RATES[key]
        rows.append({
            "instance": r["instance"], "diagnostic_rate": r["success_rate"], "validation_best_rate": val,
            "diagnostic_minus_validation_pp": (r["success_rate"] - val) * 100,
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Verifying model hashes (9 learned models)...")
    hash_results = verify_model_hashes()
    print(json.dumps(hash_results, indent=2))

    print("\nGround-truth leakage audit (measurement/kalman sanitizer -- LR-PPO reuses 'measurement')...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_instances()
    instance_names = [i.display_name for i in instances]
    assert len(instances) == 12

    print(f"\nRunning diagnostic evaluation: {len(instances)} instances x {DIAGNOSTIC_EPISODES} episodes...")
    df, trajectories, angle_data = run_evaluation(instances)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)
    assert len(df) == len(instances) * DIAGNOSTIC_EPISODES

    print("\nComputing acquisition fairness...")
    fairness_df = compute_acquisition_fairness(trajectories, instance_names, list(DIAGNOSTIC_SEEDS))
    fairness_df.to_csv(OUTPUT_ROOT / "acquisition_fairness.csv", index=False)
    n_mismatched = int(fairness_df["mismatch"].sum())
    max_discrepancy_overall = float(fairness_df["max_physical_discrepancy"].max())
    print(f"  seeds checked: {len(fairness_df)}, mismatched: {n_mismatched}, max discrepancy: {max_discrepancy_overall}")

    classical_summary = compute_classical_summary(df)
    ppo_seed_summary = compute_model_seed_summary(df, "ppo", TRAINING_SEEDS)
    ppok_seed_summary = compute_model_seed_summary(df, "ppo_kalman", TRAINING_SEEDS)
    lr_seed_summary = compute_model_seed_summary(df, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    method_summary = pd.concat([classical_summary, ppo_seed_summary, ppok_seed_summary, lr_seed_summary], ignore_index=True)
    method_summary.to_csv(OUTPUT_ROOT / "method_summary.csv", index=False)

    family_summary = compute_family_summary(df)
    family_summary.to_csv(OUTPUT_ROOT / "family_summary.csv", index=False)

    primary_comparisons = compute_primary_comparisons(df)
    secondary_comparisons = compute_secondary_comparisons(df)
    pd.concat([primary_comparisons, secondary_comparisons], ignore_index=True).to_csv(OUTPUT_ROOT / "primary_comparisons.csv", index=False)

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

    model_seed_summaries = {"ppo": ppo_seed_summary, "ppo_kalman": ppok_seed_summary, LR_PPO_METHOD: lr_seed_summary}
    generalization_check = compute_generalization_check(family_summary, model_seed_summaries)
    generalization_check.to_csv(OUTPUT_ROOT / "generalization_check.csv", index=False)

    total_violations = int(df["physical_limit_violations"].sum())

    evaluation_commit_before = _get_git_commit()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "lr_ppo_training_source_commit": LR_PPO_TRAINING_SOURCE_COMMIT,
        "diagnostic_source_commit": evaluation_commit_before,
        "branch": _get_git_branch(),
        "diagnostic_seeds": [DIAGNOSTIC_SEEDS.start, DIAGNOSTIC_SEEDS.stop - 1],
        "n_episodes_per_instance": DIAGNOSTIC_EPISODES,
        "n_instances": len(instances),
        "instances": instance_names,
        "model_hash_verification": hash_results,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness": {
            "seeds_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall, "all_pass": n_mismatched == 0,
        },
        "detection_time_summary": detection_summary,
        "total_physical_limit_violations": total_violations,
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
        "primary_comparisons": [c[0] for c in PRIMARY_COMPARISONS],
        "residual_max_angle_deg": RESIDUAL_MAX_ANGLE_DEG,
        "is_diagnostic_only": True,
        "not_final_paper_evaluation": True,
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (OUTPUT_ROOT / "model_hashes.json").write_text(json.dumps(hash_results, indent=2, default=str))

    core_files = [
        "episode_results.csv", "method_summary.csv", "family_summary.csv",
        "primary_comparisons.csv", "lead_rescue_harm.csv", "protocol_manifest.json",
    ]
    evidence_hashes = {f: sha256_of_file(OUTPUT_ROOT / f) for f in core_files}
    (OUTPUT_ROOT / "evidence_hashes.json").write_text(json.dumps(evidence_hashes, indent=2))

    print(f"\nAcquisition fairness: {n_mismatched} mismatched seeds (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
