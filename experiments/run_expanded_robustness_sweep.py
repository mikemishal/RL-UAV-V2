"""
Generic expanded-robustness sweep runner -- shared implementation for all
five sweeps (hostile_evasion, maneuverability, measurement_noise, mobility,
detection_radius). Parametrized by --sweep; the actual swept parameter
values come exclusively from experiments/expanded_robustness_protocol.py.

THIS TASK ONLY SMOKE-TESTS this script on ordinary development seeds
(123-127) and a small condition subset -- see --conditions/--seed-start/
--n-episodes overrides. NO robustness seed (66000-99999) is executed here.

Usage (FUTURE full sweep -- NOT run in this task):
    python experiments/run_expanded_robustness_sweep.py --sweep hostile_evasion

    # Smoke test (development seeds, small condition subset):
    python experiments/run_expanded_robustness_sweep.py --sweep hostile_evasion \\
        --seed-start 123 --n-episodes 5 --conditions 0.0 0.75 --allow-dev-seeds \\
        --output-root results/_smoke_expanded_robustness/hostile_evasion
"""

from __future__ import annotations

import argparse
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
    SWEEP_SEED_RANGES,
    SWEEP_CONFIG_BUILDERS,
    SWEEP_OUTPUT_DIRS,
    HOSTILE_EVASION_EPISODES,
    MANEUVERABILITY_GRID,
    MEASUREMENT_VAR_VALUES,
    MOBILITY_GRID,
    DETECTION_RADIUS_VALUES,
    EVASION_GAIN_VALUES,
    assert_seed_not_yet_opened,
    assert_disjoint_from_all_prior_ranges,
)
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode
from experiments.lead_residual_diagnostic_eval_utils import run_lr_ppo_diagnostic_episode
from experiments.run_lead_residual_diagnostic import verify_model_hashes
from experiments.run_post_detection_diagnostic import audit_ground_truth_leakage
from experiments.final_stats import (
    wilson_interval,
    standard_error,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_bootstrap_independent_families,
)

# Sweep -> list of ALL its conditions, as (label, kwargs-for-config-builder) tuples.
_SWEEP_ALL_CONDITIONS = {
    "hostile_evasion": [(f"gain_{g}", {"evasion_gain": g}) for g in EVASION_GAIN_VALUES],
    "maneuverability": [(f"accel_{a}_turn_{t}", {"defender_max_accel": a, "defender_max_turn_rate_deg": t}) for a, t in MANEUVERABILITY_GRID],
    "measurement_noise": [(f"mvar_{v}", {"measurement_var": v}) for v in MEASUREMENT_VAR_VALUES],
    "mobility": [(f"vd_{vd}_ve_{ve}", {"v_d": vd, "v_e": ve}) for vd, ve in MOBILITY_GRID],
    "detection_radius": [(f"radius_{r}", {"detection_radius": r}) for r in DETECTION_RADIUS_VALUES],
}


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


def run_condition(sweep: str, condition_label: str, condition_kwargs: dict, seeds: list[int], instances) -> tuple[pd.DataFrame, dict]:
    """Runs all 12 instances on the SAME `seeds` for one sweep condition.
    Returns (episode_rows_df_for_this_condition, angles_by_instance)."""
    rows = []
    angles_by_instance: dict[str, list[float]] = {}
    for instance in instances:
        estimator = "kalman" if instance.family == "ppo_kalman" else "measurement"
        config = SWEEP_CONFIG_BUILDERS[sweep](estimator, **condition_kwargs)
        env = SoldierEnv(config=config)
        policy = build_policy_for_instance(instance, config)
        dt = config.dt
        instance_angles = []
        for seed in seeds:
            if instance.family == "lr_ppo":
                row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=False)
                instance_angles.extend(row.pop("step_angles"))
            else:
                row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=False)
                row.pop("pre_detection_trajectory", None)
            row["method"] = instance.name
            row["instance"] = instance.display_name
            row["training_seed"] = instance.training_seed if instance.training_seed is not None else np.nan
            row["condition"] = condition_label
            rows.append(row)
        if instance.family == "lr_ppo":
            angles_by_instance[instance.display_name] = angles_by_instance.get(instance.display_name, []) + instance_angles
        env.close()
    return pd.DataFrame(rows), angles_by_instance


def _success_vector(df: pd.DataFrame, method: str, training_seed=None) -> np.ndarray:
    sub = df[df["method"] == method]
    if training_seed is not None:
        sub = sub[sub["training_seed"] == training_seed]
    return sub.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def _success_matrix(df: pd.DataFrame, method: str, training_seeds) -> np.ndarray:
    return np.stack([_success_vector(df, method, s) for s in training_seeds])


def compute_condition_classical_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method in SCRIPTED_METHODS:
        sub = df[df["method"] == method]
        n = len(sub)
        successes = int(sub["success"].sum())
        lo, hi = wilson_interval(successes, n, CONFIDENCE_LEVEL)
        rows.append({"method": method, "n": n, "success_count": successes, "success_rate": successes / n,
                      "standard_error": standard_error(successes, n), "ci_low": lo, "ci_high": hi})
    return pd.DataFrame(rows)


def compute_condition_family_summary(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for method, seeds in (("ppo", TRAINING_SEEDS), ("ppo_kalman", TRAINING_SEEDS), (LR_PPO_METHOD, LR_TRAINING_ROOTS)):
        matrix = _success_matrix(df, method, seeds)
        rates = matrix.mean(axis=1)
        observed, lo, hi = hierarchical_bootstrap_track(matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"method": method, "mean_success": float(rates.mean()), "between_seed_sd": float(rates.std(ddof=0)),
                     "hierarchical_ci_low": lo, "hierarchical_ci_high": hi})
    return pd.DataFrame(rows)


def compute_condition_primary_comparisons(df: pd.DataFrame) -> pd.DataFrame:
    lead_vec = _success_vector(df, "lead")
    lr_matrix = _success_matrix(df, LR_PPO_METHOD, LR_TRAINING_ROOTS)
    ppo_matrix = _success_matrix(df, "ppo", TRAINING_SEEDS)
    rows = []
    for name, method_a, method_b in PRIMARY_COMPARISONS:
        if method_b == "lead":
            observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_matrix, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        else:
            observed, lo, hi = hierarchical_bootstrap_independent_families(lr_matrix, ppo_matrix, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append({"comparison": name, "diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
                     "ci_includes_zero": bool(lo <= 0 <= hi)})
    return pd.DataFrame(rows)


def compute_condition_acquisition_fairness(df: pd.DataFrame, instance_names: list[str]) -> dict:
    """Lightweight fairness check using the recorded detection_step column
    (full trajectory-level fairness is exercised by the diagnostic/final
    scripts; this confirms detection is method-independent per seed)."""
    mismatches = 0
    for seed, group in df.groupby("environment_seed"):
        steps = group.set_index("instance")["detection_step"].reindex(instance_names)
        if steps.nunique() != 1:
            mismatches += 1
    return {"n_seeds_checked": int(df["environment_seed"].nunique()), "n_mismatched": int(mismatches)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generic expanded-robustness sweep runner")
    parser.add_argument("--sweep", required=True, choices=list(SWEEP_SEED_RANGES.keys()))
    parser.add_argument("--seed-start", type=int, default=None, help="Defaults to the sweep's protocol seed start.")
    parser.add_argument("--n-episodes", type=int, default=None, help="Defaults to the sweep's protocol episode count.")
    parser.add_argument("--conditions", type=str, nargs="*", default=None,
                         help="Subset of condition VALUES to evaluate (development smoke-test only). "
                              "E.g. --sweep hostile_evasion --conditions 0.0 0.75")
    parser.add_argument("--output-root", type=str, default=None)
    parser.add_argument("--allow-dev-seeds", action="store_true",
                         help="Bypass the reserved-seed guard for a development/smoke-test run "
                              "(seed-start must then be < 66000).")
    return parser.parse_args()


def _select_conditions(sweep: str, requested: list[str] | None):
    all_conditions = _SWEEP_ALL_CONDITIONS[sweep]
    if requested is None:
        return all_conditions
    if sweep == "maneuverability":
        # Simple float-subset selection isn't well-defined for a 2-parameter
        # grid; smoke tests should omit --conditions for this sweep (all 9
        # small dev-seed cells are cheap) or pass explicit accel/turn pairs
        # via a future extension. For now, fall back to the full grid.
        return all_conditions
    requested_floats = {float(v) for v in requested}
    selected = [
        (label, kwargs) for label, kwargs in all_conditions
        if any(abs(v - r) < 1e-9 for v in kwargs.values() for r in requested_floats)
    ]
    return selected if selected else all_conditions


def main() -> int:
    args = parse_args()
    seed_start_default, _seed_end_default = SWEEP_SEED_RANGES[args.sweep]
    seed_start = args.seed_start if args.seed_start is not None else seed_start_default
    n_episodes = args.n_episodes if args.n_episodes is not None else HOSTILE_EVASION_EPISODES
    output_root = Path(args.output_root) if args.output_root else SWEEP_OUTPUT_DIRS[args.sweep]
    output_root.mkdir(parents=True, exist_ok=True)

    seeds = list(range(seed_start, seed_start + n_episodes))
    if not args.allow_dev_seeds:
        for s in seeds:
            assert_seed_not_yet_opened(s)
            assert_disjoint_from_all_prior_ranges(s)
    else:
        assert seed_start < 66_000, "--allow-dev-seeds requires seed_start < 66000 (development seeds only)"

    print("Verifying model hashes (9 learned models)...")
    hash_results = verify_model_hashes()
    print(json.dumps(hash_results, indent=2))

    print("\nGround-truth leakage audit...")
    leakage_audit = audit_ground_truth_leakage()
    print(json.dumps(leakage_audit, indent=2))
    assert all(v["pass"] for v in leakage_audit.values()), "GROUND TRUTH LEAKAGE DETECTED -- STOP"

    instances = build_method_instances()
    instance_names = [i.display_name for i in instances]

    conditions = _select_conditions(args.sweep, args.conditions)
    print(f"\nRunning sweep '{args.sweep}': {len(conditions)} condition(s) x {len(instances)} instances x {len(seeds)} seeds...")

    all_rows = []
    for label, kwargs in conditions:
        print(f"  condition {label} ({kwargs})...")
        df_cond, _angles = run_condition(args.sweep, label, kwargs, seeds, instances)
        all_rows.append(df_cond)

    df = pd.concat(all_rows, ignore_index=True)
    df.to_csv(output_root / "episode_results.csv", index=False)

    total_violations = int(df["physical_limit_violations"].sum())
    print(f"\nPhysical limit violations: {total_violations} (expected 0)")

    summary_rows = []
    for label, _ in conditions:
        sub = df[df["condition"] == label]
        classical = compute_condition_classical_summary(sub)
        family = compute_condition_family_summary(sub)
        primary = compute_condition_primary_comparisons(sub)
        fairness = compute_condition_acquisition_fairness(sub, instance_names)
        summary_rows.append({"condition": label, "classical": classical.to_dict(orient="records"),
                              "family": family.to_dict(orient="records"), "primary": primary.to_dict(orient="records"),
                              "fairness": fairness})

    (output_root / "condition_summaries.json").write_text(json.dumps(summary_rows, indent=2, default=str))

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "sweep": args.sweep,
        "seeds": [seeds[0], seeds[-1]] if seeds else None,
        "n_episodes": len(seeds),
        "n_conditions": len(conditions),
        "n_instances": len(instances),
        "model_hash_verification": hash_results,
        "ground_truth_leakage_audit": leakage_audit,
        "total_physical_limit_violations": total_violations,
        "is_smoke_test_or_dev_run": bool(args.allow_dev_seeds),
    }
    (output_root / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    print(f"\nAll outputs written to {output_root}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
