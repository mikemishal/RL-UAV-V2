"""
POST-HOC / EXPLORATORY audit of the mobility hostile-speed-arm inversion
(v_e=8 substantially harder than v_e=10,12,14,16 at fixed v_d=18).

DATA-ONLY: reads exclusively from the already-frozen, hashed evidence at
results/expanded_robustness/mobility/episode_results.csv. Does NOT construct
SoldierEnv, does NOT execute any episode, does NOT touch seeds 69000-99999
in any simulation sense (plain-integer seed values already recorded in the
existing CSV are simply read as data). Writes ONLY new derived-summary CSVs
under results/expanded_robustness/mobility/posthoc_audit/ -- never modifies
the original hashed evidence files.

Usage:
    python experiments/run_mobility_posthoc_audit.py
"""

from __future__ import annotations

import json
import math
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
MOBILITY_ROOT = PROJECT_ROOT / "results" / "expanded_robustness" / "mobility"
OUTPUT_ROOT = MOBILITY_ROOT / "posthoc_audit"

HOSTILE_SPEED_ARM_VALUES = (8.0, 10.0, 12.0, 14.0, 16.0)
V_D_FIXED = 18.0
MAX_STEPS = 2000  # EnvConfig default, unchanged across the mobility sweep (config-isolation verified)
DT = 0.5          # EnvConfig default, unchanged across the mobility sweep
ENEMY_MAX_ACCEL = 6.0
ENEMY_MAX_TURN_RATE_DEG = 75.0

TRAINING_SEEDS = (45, 46, 47)
LR_TRAINING_ROOTS = (55, 56, 57)


def load_episode_results() -> pd.DataFrame:
    df = pd.read_csv(MOBILITY_ROOT / "episode_results.csv")
    return df[df["v_d"] == V_D_FIXED].copy()


def _method_success_rate(df: pd.DataFrame, v_e: float, method: str) -> float:
    sub = df[(df["v_e"] == v_e) & (df["method"] == method)]
    return float(sub["success"].mean())


def compute_hostile_speed_arm_detection_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 3: per v_e, detection time/step distributional summary + remaining
    mission budget at detection (derived only, no new simulation)."""
    rows = []
    for v_e in HOSTILE_SPEED_ARM_VALUES:
        sub = df[df["v_e"] == v_e]
        det_step = sub["detection_step"].dropna().to_numpy(dtype=np.float64)
        det_time = sub["detection_time_seconds"].dropna().to_numpy(dtype=np.float64)
        remaining_steps = MAX_STEPS - det_step
        remaining_time = remaining_steps * DT
        rows.append({
            "v_e": v_e, "n_episodes_with_detection": len(det_step),
            "detection_step_mean": float(det_step.mean()), "detection_step_median": float(np.median(det_step)),
            "detection_step_sd": float(det_step.std(ddof=0)), "detection_step_p95": float(np.percentile(det_step, 95)),
            "detection_step_max": float(det_step.max()),
            "detection_time_mean_s": float(det_time.mean()), "detection_time_median_s": float(np.median(det_time)),
            "detection_time_sd_s": float(det_time.std(ddof=0)), "detection_time_p95_s": float(np.percentile(det_time, 95)),
            "detection_time_max_s": float(det_time.max()),
            "remaining_steps_at_detection_mean": float(remaining_steps.mean()),
            "remaining_steps_at_detection_median": float(np.median(remaining_steps)),
            "remaining_steps_at_detection_min": float(remaining_steps.min()),
            "remaining_time_at_detection_mean_s": float(remaining_time.mean()),
            "remaining_time_at_detection_median_s": float(np.median(remaining_time)),
            "remaining_time_at_detection_min_s": float(remaining_time.min()),
        })
    return pd.DataFrame(rows)


def compute_horizon_confound_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 4: per v_e x method/family, outcome breakdown incl. timeout rate,
    to help rule out (or not) a global-horizon confound. No causal claim."""
    rows = []
    method_groups = [
        ("greedy", None), ("pn", None), ("lead", None),
        ("ppo", None), ("ppo_kalman", None), ("lr_ppo", None),
    ]
    for v_e in HOSTILE_SPEED_ARM_VALUES:
        for method, _ in method_groups:
            sub = df[(df["v_e"] == v_e) & (df["method"] == method)]
            n = len(sub)
            rows.append({
                "v_e": v_e, "method": method, "n": n,
                "safe_intercept_rate": float(sub["success"].mean()),
                "soldier_caught_rate": float(sub["soldier_caught"].mean()),
                "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
                "timeout_rate": float(sub["timeout"].mean()),
                "timeout_count": int(sub["timeout"].sum()),
            })
    return pd.DataFrame(rows)


def compute_post_detection_difficulty_summary(df: pd.DataFrame) -> pd.DataFrame:
    """Item 5: per v_e x method/family, post-detection difficulty metrics
    (mean/median post-detection steps, mean post-detection duration, mean
    successful post-detection duration, mean successful intercept time after
    detection, min-enemy-soldier-distance summary already recorded)."""
    rows = []
    for v_e in HOSTILE_SPEED_ARM_VALUES:
        for method in ("greedy", "pn", "lead", "ppo", "ppo_kalman", "lr_ppo"):
            sub = df[(df["v_e"] == v_e) & (df["method"] == method)]
            pdet = sub["post_detection_steps"].dropna().to_numpy(dtype=np.float64)
            succ = sub[sub["success"] == 1]
            succ_pdet = succ["post_detection_steps"].dropna().to_numpy(dtype=np.float64)
            # successful post-detection duration = time from detection to intercept (seconds);
            # computed row-wise on the SAME filtered rows to preserve alignment.
            succ_valid = succ.dropna(subset=["intercept_time_seconds", "detection_time_seconds"])
            succ_post_detection_duration_s = (
                succ_valid["intercept_time_seconds"] - succ_valid["detection_time_seconds"]
            ).to_numpy(dtype=np.float64)
            min_dist = sub["min_enemy_soldier_dist"].dropna().to_numpy(dtype=np.float64)
            rows.append({
                "v_e": v_e, "method": method, "n": len(sub),
                "mean_post_detection_steps": float(np.mean(pdet)) if len(pdet) else None,
                "median_post_detection_steps": float(np.median(pdet)) if len(pdet) else None,
                "mean_post_detection_duration_s": float(np.mean(pdet) * DT) if len(pdet) else None,
                "n_successful": len(succ),
                "mean_successful_post_detection_steps": float(np.mean(succ_pdet)) if len(succ_pdet) else None,
                "mean_successful_post_detection_duration_s": float(np.mean(succ_pdet) * DT) if len(succ_pdet) else None,
                "mean_successful_interception_time_after_detection_s": (
                    float(np.mean(succ_post_detection_duration_s)) if len(succ_post_detection_duration_s) else None
                ),
                "min_enemy_soldier_dist_mean": float(np.mean(min_dist)) if len(min_dist) else None,
                "min_enemy_soldier_dist_median": float(np.median(min_dist)) if len(min_dist) else None,
            })
    return pd.DataFrame(rows)


def compute_dynamic_scaling_descriptors() -> pd.DataFrame:
    """Item 7: descriptive dimensionless/geometry quantities from the FROZEN,
    unchanged (config-isolation verified) enemy dynamics constants. These are
    explanatory descriptors, NOT exact aircraft coordinated-turn models."""
    omega_max_rad_s = math.radians(ENEMY_MAX_TURN_RATE_DEG)
    max_delta_v_per_step = ENEMY_MAX_ACCEL * DT
    rows = []
    for v_e in HOSTILE_SPEED_ARM_VALUES:
        r_characteristic = v_e / omega_max_rad_s
        accel_to_speed_ratio = ENEMY_MAX_ACCEL / v_e
        speed_change_timescale = v_e / ENEMY_MAX_ACCEL
        # Chord-based achievable per-step heading change when the acceleration
        # budget (not the nominal turn-rate cap) is the binding constraint --
        # derived algebraically from _advance_velocity's sequential
        # turn-rate-then-acceleration clipping (see soldier_env.py); this is
        # a demonstrated property of the frozen implementation, not a
        # measured/logged diagnostic (accel_saturated was not recorded to CSV).
        ratio = max_delta_v_per_step / (2.0 * v_e)
        accel_limited_heading_change_deg_per_step = math.degrees(2.0 * math.asin(min(ratio, 1.0)))
        accel_limited_turn_rate_deg_s = accel_limited_heading_change_deg_per_step / DT
        rows.append({
            "v_e": v_e,
            "omega_max_rad_s": omega_max_rad_s,
            "R_characteristic_m": r_characteristic,
            "accel_to_speed_ratio_a_over_v": accel_to_speed_ratio,
            "speed_change_timescale_v_over_a_s": speed_change_timescale,
            "max_delta_v_per_step_m_s": max_delta_v_per_step,
            "accel_limited_heading_change_deg_per_step": accel_limited_heading_change_deg_per_step,
            "accel_limited_effective_turn_rate_deg_s": accel_limited_turn_rate_deg_s,
            "nominal_turn_rate_deg_s": ENEMY_MAX_TURN_RATE_DEG,
            "turn_binding_constraint": "acceleration" if accel_limited_turn_rate_deg_s < ENEMY_MAX_TURN_RATE_DEG else "turn_rate",
        })
    return pd.DataFrame(rows)


def compute_expected_table(df: pd.DataFrame, detection_summary: pd.DataFrame, post_detection_summary: pd.DataFrame,
                            horizon_summary: pd.DataFrame, scaling: pd.DataFrame) -> pd.DataFrame:
    """Item 8: the single combined table requested."""
    rows = []
    for v_e in HOSTILE_SPEED_ARM_VALUES:
        det = detection_summary[detection_summary["v_e"] == v_e].iloc[0]
        scale = scaling[scaling["v_e"] == v_e].iloc[0]
        timeout_rows = horizon_summary[horizon_summary["v_e"] == v_e]
        overall_timeout_rate = float(df[df["v_e"] == v_e]["timeout"].mean())
        pdet_all = post_detection_summary[(post_detection_summary["v_e"] == v_e)]
        mean_pdet_duration = float(pdet_all["mean_post_detection_duration_s"].mean())
        rows.append({
            "v_e": v_e,
            "success_lead": _method_success_rate(df, v_e, "lead"),
            "success_ppo_family": float(df[(df["v_e"] == v_e) & (df["method"] == "ppo")]["success"].mean()),
            "success_lr_ppo_family": float(df[(df["v_e"] == v_e) & (df["method"] == "lr_ppo")]["success"].mean()),
            "mean_detection_time_s": det["detection_time_mean_s"],
            "remaining_time_at_detection_mean_s": det["remaining_time_at_detection_mean_s"],
            "overall_timeout_rate": overall_timeout_rate,
            "mean_post_detection_duration_s_all_methods": mean_pdet_duration,
            "R_characteristic_m": scale["R_characteristic_m"],
            "a_max_over_v_e": scale["accel_to_speed_ratio_a_over_v"],
            "v_e_over_a_max_s": scale["speed_change_timescale_v_over_a_s"],
        })
    return pd.DataFrame(rows)


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    df = load_episode_results()

    detection_summary = compute_hostile_speed_arm_detection_summary(df)
    detection_summary.to_csv(OUTPUT_ROOT / "posthoc_detection_time_summary.csv", index=False)

    horizon_summary = compute_horizon_confound_summary(df)
    horizon_summary.to_csv(OUTPUT_ROOT / "posthoc_horizon_confound_summary.csv", index=False)

    post_detection_summary = compute_post_detection_difficulty_summary(df)
    post_detection_summary.to_csv(OUTPUT_ROOT / "posthoc_post_detection_difficulty_summary.csv", index=False)

    scaling = compute_dynamic_scaling_descriptors()
    scaling.to_csv(OUTPUT_ROOT / "posthoc_dynamic_scaling_descriptors.csv", index=False)

    expected_table = compute_expected_table(df, detection_summary, post_detection_summary, horizon_summary, scaling)
    expected_table.to_csv(OUTPUT_ROOT / "posthoc_expected_table.csv", index=False)

    manifest = {
        "label": "POST-HOC / EXPLORATORY -- derived summaries ONLY, no new simulation",
        "source_evidence": "results/expanded_robustness/mobility/episode_results.csv (unchanged, hash-verified)",
        "no_episodes_executed": True,
        "no_seeds_opened_beyond_69999": True,
        "max_steps": MAX_STEPS,
        "dt": DT,
        "enemy_max_accel": ENEMY_MAX_ACCEL,
        "enemy_max_turn_rate_deg": ENEMY_MAX_TURN_RATE_DEG,
        "hostile_speed_arm_values": list(HOSTILE_SPEED_ARM_VALUES),
        "v_d_fixed": V_D_FIXED,
    }
    (OUTPUT_ROOT / "posthoc_audit_manifest.json").write_text(json.dumps(manifest, indent=2))

    print("Detection/horizon summary:")
    print(detection_summary.to_string(index=False))
    print("\nExpected combined table:")
    print(expected_table.to_string(index=False))
    print(f"\nAll POST-HOC / EXPLORATORY outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
