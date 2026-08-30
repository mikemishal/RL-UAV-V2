"""
Post-detection diagnostic (200-seed) evaluation runner -- ONE-TIME integrity
check on the previously reserved diagnostic seed range (40100-40299). NOT the
final paper evaluation. See experiments/post_detection_diagnostic_protocol.py.

Usage:
    python experiments/run_post_detection_diagnostic.py
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.post_detection_diagnostic_protocol import (
    DIAGNOSTIC_SEED_OFFSET,
    DIAGNOSTIC_EPISODES,
    DIAGNOSTIC_SEEDS,
    TRAINING_SEEDS,
    FROZEN_MODELS,
    TRAINING_SOURCE_COMMIT,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    OUTPUT_ROOT,
    assert_not_reserved_seed,
)
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode, sha256_of_file
from experiments.final_stats import wilson_interval, mcnemar_exact, paired_bootstrap_ci

_NOMINAL_CONFIG_DIRECT = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)
_NOMINAL_CONFIG_KALMAN = EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True)


def _get_git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def _get_git_branch() -> str:
    return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def _get_git_status_clean() -> bool:
    status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode()
    return not status.strip()


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
    estimator: str
    training_seed: int | None
    model_path: Path | None


def build_instances() -> tuple[MethodInstance, ...]:
    instances = [
        MethodInstance("greedy", "Greedy", "measurement", None, None),
        MethodInstance("pn", "PN", "measurement", None, None),
        MethodInstance("lead", "Lead", "measurement", None, None),
    ]
    for seed in TRAINING_SEEDS:
        instances.append(MethodInstance("ppo", f"PPO-{seed}", "measurement", seed, FROZEN_MODELS[("direct", seed)]["path"]))
    for seed in TRAINING_SEEDS:
        instances.append(MethodInstance("ppo_kalman", f"PPOK-{seed}", "kalman", seed, FROZEN_MODELS[("kalman", seed)]["path"]))
    return tuple(instances)


def build_policy_and_env(instance: MethodInstance):
    config = _NOMINAL_CONFIG_KALMAN if instance.estimator == "kalman" else _NOMINAL_CONFIG_DIRECT
    env = SoldierEnv(config=config)
    if instance.model_path is not None:
        policy = get_policy(instance.name, model_path=str(instance.model_path), deterministic=True)
    else:
        policy = get_policy(instance.name)
    return env, policy


def run_evaluation(instances) -> tuple[pd.DataFrame, dict]:
    """Returns (episode_results_df, trajectories_by_seed_and_instance)."""
    rows = []
    trajectories = {}  # (seed, instance_name) -> trajectory list
    dt = _NOMINAL_CONFIG_DIRECT.dt
    seeds = list(DIAGNOSTIC_SEEDS)
    for seed in seeds:
        assert_not_reserved_seed(seed)

    total = len(instances)
    for i, instance in enumerate(instances):
        print(f"[{i+1}/{total}] {instance.display_name} ({instance.estimator}) -- {len(seeds)} episodes...")
        env, policy = build_policy_and_env(instance)
        for seed in seeds:
            row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=True)
            trajectory = row.pop("pre_detection_trajectory")
            trajectories[(seed, instance.display_name)] = trajectory
            row["method"] = instance.name
            row["instance"] = instance.display_name
            row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
            rows.append(row)
        env.close()
    df = pd.DataFrame(rows)
    return df, trajectories


def compute_acquisition_fairness(trajectories: dict, instance_names: list[str]) -> pd.DataFrame:
    rows = []
    for seed in DIAGNOSTIC_SEEDS:
        trajs = {name: trajectories[(seed, name)] for name in instance_names}
        ref_name = instance_names[0]
        ref_traj = trajs[ref_name]
        detection_steps = {name: t[-1]["step"] if t[-1]["enemy_detected"] else None for name, t in trajs.items()}
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
            "mismatch": mismatch,
        })
    return pd.DataFrame(rows)


def compute_method_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in SCRIPTED_METHODS:
        sub = df[df["method"] == method]
        n = len(sub)
        successes = int(sub["success"].sum())
        lo, hi = wilson_interval(successes, n)
        rows.append({
            "instance": method, "method": method, "n_episodes": n, "success_count": successes,
            "success_rate": successes / n, "success_ci_low": lo, "success_ci_high": hi,
            "soldier_caught_rate": float(sub["soldier_caught"].mean()),
            "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
            "timeout_rate": float(sub["timeout"].mean()),
        })
    return pd.DataFrame(rows)


def compute_ppo_seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PPO_METHODS:
        for seed in TRAINING_SEEDS:
            sub = df[(df["method"] == method) & (df["training_seed"] == seed)]
            n = len(sub)
            successes = int(sub["success"].sum())
            lo, hi = wilson_interval(successes, n)
            rows.append({
                "instance": f"{'PPOK' if method == 'ppo_kalman' else 'PPO'}-{seed}", "method": method,
                "training_seed": seed, "n_episodes": n, "success_count": successes,
                "success_rate": successes / n, "success_ci_low": lo, "success_ci_high": hi,
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
                "mean_tracking_error": float(sub["mean_tracking_error"].mean()),
                "tracking_rmse_mean_of_episode_rmse": float(sub["tracking_rmse"].mean()),
            })
    return pd.DataFrame(rows)


def compute_rl_family_summary(ppo_seed_summary: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in PPO_METHODS:
        sub = ppo_seed_summary[ppo_seed_summary["method"] == method]
        rates = sub["success_rate"].to_numpy(dtype=float)
        rows.append({
            "method": method, "n_training_seeds": len(rates),
            "mean_success": float(np.mean(rates)), "between_seed_sd": float(np.std(rates, ddof=0)),
            "min_seed_success": float(np.min(rates)), "max_seed_success": float(np.max(rates)),
        })
    return pd.DataFrame(rows)


def _success_vector(df: pd.DataFrame, method: str, training_seed=None) -> np.ndarray:
    sub = df[df["method"] == method]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def compute_pairwise_diagnostics(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for seed in TRAINING_SEEDS:
        ppo_vec = _success_vector(df, "ppo", seed)
        for classical in SCRIPTED_METHODS:
            classical_vec = _success_vector(df, classical)
            observed, lo, hi = paired_bootstrap_ci(ppo_vec, classical_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
            n10 = int(np.sum((ppo_vec == 1) & (classical_vec == 0)))
            n01 = int(np.sum((ppo_vec == 0) & (classical_vec == 1)))
            p_value = mcnemar_exact(n01=n01, n10=n10)
            rows.append({
                "comparison": f"PPO-{seed} vs {classical.capitalize()}", "training_seed": seed,
                "method_a": f"ppo_{seed}", "method_b": classical,
                "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
            })
        # PPO-Kalman vs PPO, matched by root
        ppok_vec = _success_vector(df, "ppo_kalman", seed)
        observed, lo, hi = paired_bootstrap_ci(ppok_vec, ppo_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        n10 = int(np.sum((ppok_vec == 1) & (ppo_vec == 0)))
        n01 = int(np.sum((ppok_vec == 0) & (ppo_vec == 1)))
        p_value = mcnemar_exact(n01=n01, n10=n10)
        rows.append({
            "comparison": f"PPOK-{seed} vs PPO-{seed}", "training_seed": seed,
            "method_a": f"ppo_kalman_{seed}", "method_b": f"ppo_{seed}",
            "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
        })
    return pd.DataFrame(rows)


def compute_detection_time_summary(df: pd.DataFrame) -> dict:
    grouped = df.groupby("instance")["detection_time_seconds"].agg(["mean", "median", "std", "min", "max"])
    common = grouped.iloc[0]
    max_deviation = float((grouped["mean"] - common["mean"]).abs().max())
    return {
        "mean_detection_time_seconds": float(common["mean"]), "median_detection_time_seconds": float(common["median"]),
        "std_detection_time_seconds": float(common["std"]) if not np.isnan(common["std"]) else 0.0,
        "min_detection_time_seconds": float(common["min"]), "max_detection_time_seconds": float(common["max"]),
        "max_deviation_across_instances_seconds": max_deviation,
        "detection_time_identical_across_methods": bool(max_deviation < 1e-9),
        "per_instance": grouped.reset_index().to_dict(orient="records"),
    }


def compute_post_detection_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for instance in df["instance"].unique():
        sub = df[df["instance"] == instance]
        successful = sub[sub["success"] == 1]
        rows.append({
            "instance": instance,
            "mean_post_detection_steps": float(sub["post_detection_steps"].mean()),
            "mean_total_steps": float(sub["total_steps"].mean()),
            "mean_successful_intercept_time_seconds": float(successful["intercept_time_seconds"].mean()) if len(successful) else None,
            "min_enemy_soldier_dist_mean": float(sub["min_enemy_soldier_dist"].mean()),
        })
    return pd.DataFrame(rows)


def audit_ground_truth_leakage() -> dict:
    """Confirms build_policy_info() (the SAME sanitizer used for every
    instance in this diagnostic) never exposes enemy_pos/enemy_vel."""
    from uav_defend.policies.sanitize import build_policy_info
    fake_info = {
        "enemy_pos": np.array([999.0, 999.0, 999.0]), "enemy_vel": np.array([999.0, 999.0, 999.0]),
        "soldier_pos": np.zeros(3), "defender_pos": np.zeros(3), "defender_vel": np.zeros(3),
        "enemy_detected": True, "e_hat": np.zeros(3), "v_hat": np.zeros(3),
        "enemy_measurement": np.zeros(3), "enemy_measurement_velocity": np.zeros(3),
    }
    result = {}
    for mode in ("measurement", "kalman"):
        sanitized = build_policy_info(fake_info, mode)
        leaked = "enemy_pos" in sanitized or "enemy_vel" in sanitized
        result[mode] = {"leaked_enemy_pos_or_vel": leaked, "pass": not leaked}
    return result


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("Verifying model hashes...")
    hash_results = verify_model_hashes()
    print(json.dumps(hash_results, indent=2))

    print("\nGround-truth leakage audit...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_instances()
    instance_names = [i.display_name for i in instances]

    print(f"\nRunning diagnostic evaluation: {len(instances)} instances x {DIAGNOSTIC_EPISODES} episodes...")
    df, trajectories = run_evaluation(instances)
    df.to_csv(OUTPUT_ROOT / "episode_results.csv", index=False)

    print("\nComputing acquisition fairness...")
    fairness_df = compute_acquisition_fairness(trajectories, instance_names)
    fairness_df.to_csv(OUTPUT_ROOT / "acquisition_fairness.csv", index=False)
    n_mismatched = int(fairness_df["mismatch"].sum())
    max_discrepancy_overall = float(fairness_df["max_physical_discrepancy"].max())
    print(f"  seeds checked: {len(fairness_df)}, mismatched: {n_mismatched}, max discrepancy: {max_discrepancy_overall}")

    method_summary = compute_method_summary(df)
    method_summary.to_csv(OUTPUT_ROOT / "method_summary.csv", index=False)

    ppo_seed_summary = compute_ppo_seed_summary(df)
    ppo_seed_summary.to_csv(OUTPUT_ROOT / "ppo_seed_summary.csv", index=False)

    rl_family_summary = compute_rl_family_summary(ppo_seed_summary)
    rl_family_summary.to_csv(OUTPUT_ROOT / "rl_family_summary.csv", index=False)

    pairwise = compute_pairwise_diagnostics(df)
    pairwise.to_csv(OUTPUT_ROOT / "pairwise_diagnostics.csv", index=False)

    detection_summary = compute_detection_time_summary(df)
    (OUTPUT_ROOT / "detection_time_summary.json").write_text(json.dumps(detection_summary, indent=2, default=str))

    post_detection_summary = compute_post_detection_summary(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "post_detection_summary.csv", index=False)

    total_violations = int(df["physical_limit_violations"].sum())

    evaluation_commit_before = _get_git_commit()
    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "training_source_commit": TRAINING_SOURCE_COMMIT,
        "evaluation_source_commit": evaluation_commit_before,
        "branch": _get_git_branch(),
        "diagnostic_seeds": [DIAGNOSTIC_SEED_OFFSET, DIAGNOSTIC_SEED_OFFSET + DIAGNOSTIC_EPISODES - 1],
        "n_episodes_per_instance": DIAGNOSTIC_EPISODES,
        "n_instances": len(instances),
        "instances": instance_names,
        "model_hash_verification": hash_results,
        "ground_truth_leakage_audit": leakage_audit,
        "acquisition_fairness": {
            "seeds_checked": len(fairness_df), "n_mismatched": n_mismatched,
            "max_physical_discrepancy": max_discrepancy_overall,
            "all_pass": n_mismatched == 0,
        },
        "detection_time_summary": detection_summary,
        "total_physical_limit_violations": total_violations,
        "is_diagnostic_only": True,
        "not_final_paper_evaluation": True,
    }
    (OUTPUT_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nAcquisition fairness: {n_mismatched} mismatched seeds (expected 0)")
    print(f"Physical limit violations: {total_violations} (expected 0)")
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
