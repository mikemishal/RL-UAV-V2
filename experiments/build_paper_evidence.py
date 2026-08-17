"""
Read-only paper-evidence-freeze builder.

Reads ONLY existing, already-locked result files under results/ and writes
consolidated paper-facing summary files under results/paper_evidence/. This
script performs NO simulation of any kind:

    - NEVER instantiates SoldierEnv
    - NEVER loads a PPO/Stable-Baselines3 model for inference
    - NEVER runs an episode
    - NEVER consumes an evaluation seed

test_paper_evidence_freeze.py statically asserts this file's source text
contains no forbidden tokens (SoldierEnv, PPO.load, env.step, env.reset,
stable_baselines3 import) and that importing this module does not import
uav_defend.envs or stable_baselines3.

Usage:
    python experiments/build_paper_evidence.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
RESULTS = PROJECT_ROOT / "results"
OUTPUT_ROOT = RESULTS / "paper_evidence"

SIX_METHODS = ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
EXCLUDED_METHODS = ("pn_kalman", "lead_kalman")
METHOD_DISPLAY_NAME = {
    "greedy": "Greedy", "kalman_greedy": "Kalman-Greedy", "pn": "PN",
    "lead": "Lead", "ppo": "PPO", "ppo_kalman": "PPO-Kalman",
}

# =============================================================================
# SOURCE FILE LOCATIONS (canonical unless noted otherwise)
# =============================================================================
FINAL_NOMINAL = RESULTS / "final_nominal"
CORRECTED_GREEDY_NOMINAL = RESULTS / "corrections" / "direct_greedy" / "final_nominal"
CORRECTED_GREEDY_DETECTION = RESULTS / "corrections" / "direct_greedy" / "detection_radius"
CORRECTED_MEASUREMENT_ESTIMATOR = RESULTS / "corrections" / "measurement_noise_estimator"
ACQUISITION_ABLATION = RESULTS / "acquisition_ablation"
DETECTION_RADIUS = RESULTS / "robustness" / "detection_radius"
MEASUREMENT_NOISE = RESULTS / "robustness" / "measurement_noise"
MOBILITY_1D = RESULTS / "robustness" / "mobility"
MOBILITY_GRID = RESULTS / "robustness" / "mobility_grid"
HOSTILE_EVASION = RESULTS / "robustness" / "hostile_evasion"
MANEUVERABILITY = RESULTS / "robustness" / "maneuverability"
MOBILITY_AUDIT = RESULTS / "audits" / "mobility_dynamics"


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


# =============================================================================
# 1. PROVENANCE REGISTRY
# =============================================================================
def build_registry() -> pd.DataFrame:
    """One row per evidence item used (or explicitly excluded) by this freeze."""
    rows = []

    def add(evidence_id, experiment, result_type, method, comparison, parameter, parameter_value,
             seed_range, n_environment_seeds, ppo_training_seeds, source_directory, source_filename,
             status, paper_use, notes):
        rows.append({
            "evidence_id": evidence_id, "experiment": experiment, "result_type": result_type,
            "method": method, "comparison": comparison, "parameter": parameter,
            "parameter_value": parameter_value, "seed_range": seed_range,
            "n_environment_seeds": n_environment_seeds, "ppo_training_seeds": ppo_training_seeds,
            "source_directory": source_directory, "source_filename": source_filename,
            "source_commit_if_known": "unknown_pre_freeze_experiment",
            "status": status, "paper_use": paper_use, "notes": notes,
        })

    # --- Final nominal: canonical (corrected Greedy) ---
    add("EV001", "final_nominal", "success_summary", "greedy", "", "none", "nominal",
        "20000-24999", 5000, 0, str(CORRECTED_GREEDY_NOMINAL.relative_to(PROJECT_ROOT)),
        "corrected_greedy_summary.json", "canonical", "main",
        "Corrected Greedy (post commit 0dbdadd fix). Original Direct Greedy nominal value is INVALID (escort-mode bug).")
    for m in ("kalman_greedy", "pn", "lead", "ppo", "ppo_kalman"):
        add("EV002_" + m, "final_nominal", "success_summary", m, "", "none", "nominal",
            "20000-24999", 5000, 3 if m in ("ppo", "ppo_kalman") else 0,
            str(FINAL_NOMINAL.relative_to(PROJECT_ROOT)), "method_summary.csv", "canonical", "main",
            "Unaffected by the Direct Greedy bug; used as-is.")
    add("EV003", "final_nominal", "invalid_reference", "greedy", "", "none", "nominal",
        "20000-24999", 5000, 0, str(FINAL_NOMINAL.relative_to(PROJECT_ROOT)), "method_summary.csv",
        "invalid", "none",
        "Original (uncorrected) Direct Greedy row in this file (success_rate~0.679) is INVALID -- Direct Greedy "
        "incorrectly remained in escort mode post-detection (expected e_hat, contract exposed enemy_measurement).")
    add("EV004", "final_nominal", "pairwise_comparison", "ppo_vs_greedy", "ppo_vs_greedy(A)/ppo_vs_pn(B)/ppo_vs_lead(C)",
        "none", "nominal", "20000-24999", 5000, 3, str(FINAL_NOMINAL.relative_to(PROJECT_ROOT)),
        "pairwise_comparisons.csv", "canonical", "main",
        "Rows A (ppo_vs_greedy), B (ppo_vs_pn), C (ppo_vs_lead), D (ppo_kalman_vs_kalman_greedy), G (ppo_kalman_vs_ppo) "
        "are valid (do not involve the buggy Greedy baseline). Row H (kalman_greedy_vs_greedy) is INVALID -- see EV005.")
    add("EV005", "final_nominal", "invalid_reference", "kalman_greedy_vs_greedy", "kalman_greedy_vs_greedy", "none",
        "nominal", "20000-24999", 5000, 0, str(FINAL_NOMINAL.relative_to(PROJECT_ROOT)), "pairwise_comparisons.csv",
        "invalid", "none",
        "Row H uses the INVALID (uncorrected) Greedy as method_b (+1.18pp, CI includes 0). Superseded by EV006.")
    add("EV006", "final_nominal", "pairwise_comparison", "kalman_greedy_vs_corrected_greedy", "kalman_greedy_vs_corrected_greedy",
        "none", "nominal", "20000-24999", 5000, 0, str(CORRECTED_GREEDY_NOMINAL.relative_to(PROJECT_ROOT)),
        "corrected_nominal_comparisons.csv", "canonical", "main",
        "Authoritative Kalman-Greedy vs corrected-Greedy comparison for nominal (-6.64pp, CI excludes 0).")
    add("EV007", "final_nominal", "pairwise_comparison", "ppo_vs_corrected_greedy", "ppo_vs_corrected_greedy", "none",
        "nominal", "20000-24999", 5000, 3, str(CORRECTED_GREEDY_NOMINAL.relative_to(PROJECT_ROOT)),
        "corrected_nominal_comparisons.csv", "canonical", "main",
        "Authoritative PPO vs corrected-Greedy comparison for nominal (+9.63pp, CI excludes 0).")

    # --- Acquisition ablation ---
    add("EV010", "acquisition_ablation", "success_summary", "full_ppo/escorted_ppo/lead", "", "none", "nominal",
        "25000-25999", 1000, 3, str(ACQUISITION_ABLATION.relative_to(PROJECT_ROOT)), "condition_summary.csv",
        "canonical", "main", "Direct track only used as canonical; kalman track retained as secondary support only.")
    add("EV011", "acquisition_ablation", "pairwise_comparison", "full_ppo_vs_escorted_ppo/escorted_ppo_vs_lead",
        "A/B/C/D/E/F", "none", "nominal", "25000-25999", 1000, 3,
        str(ACQUISITION_ABLATION.relative_to(PROJECT_ROOT)), "paired_comparisons.csv", "canonical", "main",
        "Do NOT propagate lead_kalman as a main-paper baseline (kalman-track rows C/D retained as supplemental only).")

    # --- Detection radius ---
    add("EV020", "detection_radius", "success_summary", "greedy", "", "detection_radius", "5,10,15,20,25,30",
        "30000-30999", 1000, 0, str(CORRECTED_GREEDY_DETECTION.relative_to(PROJECT_ROOT)),
        "corrected_greedy_radius_summary.csv", "canonical", "main",
        "Corrected Greedy. Original robustness/detection_radius greedy row (constant 0.643 across all radii) is INVALID.")
    add("EV021", "detection_radius", "invalid_reference", "greedy", "", "detection_radius", "5,10,15,20,25,30",
        "30000-30999", 1000, 0, str(DETECTION_RADIUS.relative_to(PROJECT_ROOT)), "radius_summary.csv",
        "invalid", "none", "Original (uncorrected) Direct Greedy rows -- constant 0.643 success at every radius, "
        "an artifact of the escort-mode bug. Do not use.")
    for m in ("pn", "lead", "kalman_greedy", "ppo", "ppo_kalman"):
        add("EV022_" + m, "detection_radius", "success_summary", m, "", "detection_radius", "5,10,15,20,25,30",
            "30000-30999", 1000, 3 if m in ("ppo", "ppo_kalman") else 0,
            str(DETECTION_RADIUS.relative_to(PROJECT_ROOT)), "radius_summary.csv", "canonical", "main",
            "Unaffected by the Direct Greedy bug.")
    add("EV023", "detection_radius", "invalid_reference", "kalman_greedy_vs_greedy", "kalman_greedy_vs_greedy",
        "detection_radius", "5,10,15,20,25,30", "30000-30999", 1000, 0,
        str(DETECTION_RADIUS.relative_to(PROJECT_ROOT)), "pairwise_comparisons.csv", "invalid", "none",
        "Uses the INVALID uncorrected Greedy baseline; produces a spurious apparent Kalman-Greedy ADVANTAGE at "
        "several radii (e.g. +3.5pp at 10m, +4.2pp at 25m, +4.7pp at 30m, all CI excludes 0). Superseded by EV024. "
        "SOURCE-CONSISTENCY-AUDIT FINDING: sign of this comparison flips once the correct Greedy baseline is used.")
    add("EV024", "detection_radius", "pairwise_comparison", "kalman_greedy_vs_corrected_greedy",
        "kalman_greedy_vs_corrected_greedy", "detection_radius", "5,10,15,20,25,30", "30000-30999", 1000, 0,
        str(CORRECTED_GREEDY_DETECTION.relative_to(PROJECT_ROOT)), "corrected_greedy_pairwise_comparisons.csv",
        "canonical", "main",
        "Authoritative: Kalman-Greedy is significantly WORSE than corrected Greedy at every radius (-5.3 to -7.6pp, "
        "all CI exclude 0) -- consistent with every other canonical experiment in this campaign.")
    add("EV025", "detection_radius", "pairwise_comparison", "ppo_vs_lead", "ppo_vs_lead", "detection_radius",
        "5,10,15,20,25,30", "30000-30999", 1000, 3, str(DETECTION_RADIUS.relative_to(PROJECT_ROOT)),
        "pairwise_comparisons.csv", "canonical", "main",
        "PPO significantly higher at 5m and 15m; unresolved (CI includes 0) at 10/20/25/30m. Does not involve Greedy.")
    add("EV026", "detection_radius", "invalid_reference", "estimator_rmse", "", "detection_radius",
        "5,10,15,20,25,30", "30000-30999", 1000, 0, str(DETECTION_RADIUS.relative_to(PROJECT_ROOT)),
        "estimator_summary.csv", "invalid", "none",
        "Uses the old episode-level RMSE aggregation bug (sqrt(mean(per-episode-mean-error^2)) instead of the "
        "correct pooled sample-weighted RMSE). Raw data lacks sufficient n_position_estimation_samples for exact "
        "reconstruction. Do not use for paper estimator-accuracy claims.")

    # --- Measurement noise ---
    add("EV030", "measurement_noise", "success_summary", "all_six", "", "measurement_var",
        "0.05,0.25,0.5,1,2,4", "31000-31999", 1000, 3,
        str(MEASUREMENT_NOISE.relative_to(PROJECT_ROOT)), "noise_summary.csv", "canonical", "main",
        "Run after the Direct Greedy fix (branch feature/robustness-measurement-noise postdates fix commit "
        "bff3927) -- greedy values here are already corrected; no separate correction needed.")
    add("EV031", "measurement_noise", "pairwise_comparison", "ppo_vs_lead/kalman_greedy_vs_greedy/ppo_kalman_vs_ppo",
        "", "measurement_var", "0.05,0.25,0.5,1,2,4", "31000-31999", 1000, 3,
        str(MEASUREMENT_NOISE.relative_to(PROJECT_ROOT)), "pairwise_comparisons.csv", "canonical", "main", "")
    add("EV032", "measurement_noise", "invalid_reference", "estimator_rmse", "", "measurement_var",
        "0.05,0.25,0.5,1,2,4", "31000-31999", 1000, 0, str(MEASUREMENT_NOISE.relative_to(PROJECT_ROOT)),
        "estimator_summary.csv", "invalid", "none",
        "Uses the old episode-level RMSE aggregation bug. Superseded by EV033 (corrected).")
    add("EV033", "measurement_noise", "estimator_summary", "all", "", "measurement_var",
        "0.05,0.25,0.5,1,2,4", "31000-31999", 1000, 3, str(CORRECTED_MEASUREMENT_ESTIMATOR.relative_to(PROJECT_ROOT)),
        "corrected_estimator_summary.csv", "canonical", "main",
        "Corrected sample-weighted aggregation using experiments/estimator_stats.py; includes "
        "n_position_estimation_samples and matches sqrt(3*measurement_var) theoretical raw RMSE closely.")

    # --- Mobility (1-D, secondary/superseded for primary claims) ---
    add("EV040", "mobility_1d", "success_summary", "all_six", "", "v_d_or_v_e", "varies (1-D arms)",
        "32000-32999", 1000, 3, str(MOBILITY_1D.relative_to(PROJECT_ROOT)), "defender_speed_summary.csv/hostile_speed_summary.csv",
        "supplemental", "supplement",
        "Valid as a secondary zero-shot 1-D experiment (individual v_d-only / v_e-only arms). Must NOT be used to "
        "derive mobility-ratio thresholds, monotonic mu claims, or claims that speed ratio alone controls "
        "feasibility (see mobility_dynamics_audit).")
    add("EV041", "mobility_1d", "primary_mobility_analysis_role", "all_six", "", "v_d_or_v_e", "varies (1-D arms)",
        "32000-32999", 1000, 3, str(MOBILITY_1D.relative_to(PROJECT_ROOT)), "mobility_summary.json",
        "superseded", "none",
        "SUPERSEDED as the PRIMARY mobility-vs-performance analysis by the 5x5 mobility_grid experiment (EV050+), "
        "which jointly varies v_d and v_e on a confirmatory Cartesian grid with a fresh seed range. Do not cite "
        "the 1-D sweep as the primary mobility evidence in the main paper.")

    # --- Mobility dynamics audit (diagnostic only) ---
    add("EV045", "mobility_dynamics_audit", "diagnostic", "greedy", "", "v_d_or_v_e", "diagnostic",
        "18000-18099 (non-publication)", 100, 0, str(MOBILITY_AUDIT.relative_to(PROJECT_ROOT)),
        "mobility_dynamics_audit.json", "diagnostic", "supplement",
        "Diagnostic/model-verification evidence only -- confirmed no software bug, root-caused the 1-D mobility "
        "anomalies to intended model semantics (fixed-speed-command; vertical-rate cap decoupling). Use only to "
        "support interpretation/limitations, NOT as an independent controller-performance experiment.")

    # --- Mobility grid (primary mobility experiment) ---
    add("EV050", "mobility_grid", "success_summary", "all_six", "", "v_d,v_e", "5x5 grid (12-24 x 8-16 m/s)",
        "35000-35499", 500, 3, str(MOBILITY_GRID.relative_to(PROJECT_ROOT)), "grid_success_summary.csv",
        "canonical", "main", "PRIMARY mobility experiment (supersedes mobility_1d, EV040).")
    add("EV051", "mobility_grid", "pairwise_comparison", "ppo_vs_lead", "ppo_vs_lead_envelope", "v_d,v_e",
        "5x5 grid", "35000-35499", 500, 3, str(MOBILITY_GRID.relative_to(PROJECT_ROOT)), "ppo_vs_lead_envelope.csv",
        "canonical", "main", "4/25 PPO significantly higher, 4/25 Lead significantly higher, 17/25 unresolved.")
    add("EV052", "mobility_grid", "equal_ratio_summary", "all_six", "", "mu=v_d/v_e", "1.5",
        "35000-35499", 500, 3, str(MOBILITY_GRID.relative_to(PROJECT_ROOT)), "equal_ratio_summary.csv",
        "canonical", "main", "Demonstrates absolute-speed dependence at fixed ratio mu=1.5.")
    add("EV053", "mobility_grid", "feasibility_summary", "all_six", "", "success_threshold", "0.7/0.8/0.9",
        "35000-35499", 500, 3, str(MOBILITY_GRID.relative_to(PROJECT_ROOT)), "feasibility_region_summary.csv",
        "canonical", "main", "PPO has the largest >=70% and >=80% measured-cell coverage.")

    # --- Hostile evasion ---
    add("EV060", "hostile_evasion", "success_summary", "all_six", "", "enemy_evasion_gain", "0,.25,.5,.75,1",
        "33000-33999", 1000, 3, str(HOSTILE_EVASION.relative_to(PROJECT_ROOT)), "evasion_summary.csv",
        "canonical", "main", "")
    add("EV061", "hostile_evasion", "pairwise_comparison", "ppo_vs_lead", "ppo_vs_lead", "enemy_evasion_gain",
        "0,.25,.5,.75,1", "33000-33999", 1000, 3, str(HOSTILE_EVASION.relative_to(PROJECT_ROOT)),
        "ppo_vs_lead_comparisons.csv", "canonical", "main",
        "PPO significantly higher at gain 0/0.25/0.5; unresolved at gain 0.75 (nominal) and gain 1.0.")
    add("EV062", "hostile_evasion", "acquisition_summary", "all_six", "", "enemy_evasion_gain", "0,.25,.5,.75,1",
        "33000-33999", 1000, 3, str(HOSTILE_EVASION.relative_to(PROJECT_ROOT)), "acquisition_evasion_summary.csv",
        "canonical", "supplement", "Pre-detection evasion-onset timing; supports interpretation of gain=0 semantics.")

    # --- Maneuverability ---
    add("EV070", "maneuverability", "success_summary", "all_six", "", "defender_max_accel,defender_max_turn_rate_deg",
        "3x3 grid (4-12 m/s^2 x 45-135 deg/s)", "34000-34999", 1000, 3,
        str(MANEUVERABILITY.relative_to(PROJECT_ROOT)), "maneuverability_success_summary.csv", "canonical", "main", "")
    add("EV071", "maneuverability", "pairwise_comparison", "ppo_vs_lead", "ppo_vs_lead", "defender_max_accel,defender_max_turn_rate_deg",
        "3x3 grid", "34000-34999", 1000, 3, str(MANEUVERABILITY.relative_to(PROJECT_ROOT)), "ppo_vs_lead_comparisons.csv",
        "canonical", "main", "Lead significantly higher at 6/9 cells, PPO significantly higher at 0/9, 3/9 unresolved.")

    # --- Kalman consolidation source list ---
    add("EV080", "kalman_consolidation", "cross_experiment_summary", "kalman_greedy,ppo_kalman", "", "all", "all",
        "20000-35499 (all canonical experiments)", "varies", 3,
        "results/paper_evidence", "kalman_evidence_summary.csv", "canonical", "main",
        "Aggregates EV006/EV024/EV031/EV051-equivalent/EV061-equivalent/EV071-equivalent Kalman comparisons across "
        "every canonical experiment; excludes EV005/EV023 (invalid, buggy-Greedy-contaminated comparisons).")

    return pd.DataFrame(rows)


# =============================================================================
# 2. FINAL NOMINAL TABLE + COMPARISONS
# =============================================================================
def build_final_nominal_table() -> pd.DataFrame:
    method_summary = pd.read_csv(FINAL_NOMINAL / "method_summary.csv")
    corrected_greedy = json.loads((CORRECTED_GREEDY_NOMINAL / "corrected_greedy_summary.json").read_text())

    rows = []
    for m in SIX_METHODS:
        if m == "greedy":
            success_rate = corrected_greedy["success_rate"]
            ci_low = corrected_greedy["success_ci_low"]
            ci_high = corrected_greedy["success_ci_high"]
            soldier_caught = corrected_greedy["soldier_caught_rate"]
            unsafe = corrected_greedy["unsafe_intercept_rate"]
            timeout = corrected_greedy["timeout_rate"]
            mean_ep = corrected_greedy["mean_episode_length_seconds"]
            mean_det = corrected_greedy["mean_detection_time_seconds"]
            mean_int = corrected_greedy["mean_intercept_time_seconds"]
            seed_sd = seed_min = seed_max = float("nan")
        else:
            row = method_summary[method_summary["method"] == m].iloc[0]
            success_rate, ci_low, ci_high = row["success_rate"], row["success_ci_low"], row["success_ci_high"]
            soldier_caught, unsafe, timeout = row["soldier_caught_rate"], row["unsafe_intercept_rate"], row["timeout_rate"]
            mean_ep, mean_det, mean_int = row["mean_episode_length_seconds"], row["mean_detection_time_seconds"], row["mean_intercept_time_seconds"]
            seed_sd = row["training_seed_sd"] if m in ("ppo", "ppo_kalman") else float("nan")
            seed_min = seed_max = float("nan")

        rows.append({
            "method": METHOD_DISPLAY_NAME[m], "method_key": m,
            "success_rate": success_rate, "success_ci_low": ci_low, "success_ci_high": ci_high,
            "soldier_caught_rate": soldier_caught, "unsafe_intercept_rate": unsafe, "timeout_rate": timeout,
            "mean_episode_time_seconds": mean_ep, "mean_detection_time_seconds": mean_det,
            "mean_intercept_time_seconds": mean_int, "training_seed_sd": seed_sd,
            "training_seed_min": seed_min, "training_seed_max": seed_max,
        })
    df = pd.DataFrame(rows)

    # Backfill PPO/PPO-Kalman training-seed min/max from the per-seed summary if available.
    ppo_seed_path = FINAL_NOMINAL / "ppo_training_seed_summary.csv"
    if ppo_seed_path.exists():
        seed_df = pd.read_csv(ppo_seed_path)
        for m in ("ppo", "ppo_kalman"):
            sub = seed_df[seed_df["method"] == m] if "method" in seed_df.columns else seed_df[seed_df.get("paper_method", "") == m]
            if len(sub):
                rates = sub["success_rate"].to_numpy(dtype=float)
                df.loc[df["method_key"] == m, "training_seed_min"] = float(np.min(rates))
                df.loc[df["method_key"] == m, "training_seed_max"] = float(np.max(rates))

    return df


def _verify_final_nominal_table(df: pd.DataFrame) -> list[str]:
    expected = {"greedy": 0.7576, "kalman_greedy": 0.6912, "pn": 0.7304, "lead": 0.8310, "ppo": 0.8539, "ppo_kalman": 0.8157}
    discrepancies = []
    for method_key, exp_val in expected.items():
        actual = float(df[df["method_key"] == method_key]["success_rate"].iloc[0])
        if abs(actual - exp_val) > 0.005:  # 0.5pp tolerance
            discrepancies.append(f"{method_key}: expected~{exp_val}, actual={actual}")
    return discrepancies


def build_final_nominal_comparisons() -> pd.DataFrame:
    orig = pd.read_csv(FINAL_NOMINAL / "pairwise_comparisons.csv")
    corrected = pd.read_csv(CORRECTED_GREEDY_NOMINAL / "corrected_nominal_comparisons.csv")

    def orig_row(name):
        r = orig[orig["comparison_name"] == name].iloc[0]
        return r["observed_diff_pp"], r["ci_low_pp"], r["ci_high_pp"]

    rows = []
    labels = [
        ("PPO - corrected Greedy", "ppo_vs_corrected_greedy", corrected),
        ("PPO - PN", "B", orig),
        ("PPO - Lead", "C", orig),
        ("PPO-Kalman - PPO", "G", orig),
        ("Kalman-Greedy - corrected Greedy", "kalman_greedy_vs_corrected_greedy", corrected),
        ("Lead - corrected Greedy (descriptive)", "lead_vs_corrected_greedy_descriptive", corrected),
        ("PN - corrected Greedy (descriptive)", "pn_vs_corrected_greedy_descriptive", corrected),
    ]
    for label, key, source in labels:
        r = source[source["comparison_name"] == key].iloc[0] if "comparison_name" in source.columns else source[source["comparison"] == key].iloc[0]
        rows.append({
            "comparison": label, "observed_diff_pp": r["observed_diff_pp"],
            "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"],
            "comparison_type": r["comparison_type"],
        })
    return pd.DataFrame(rows)


# =============================================================================
# 3. ACQUISITION ABLATION SUMMARY
# =============================================================================
def build_acquisition_summary() -> pd.DataFrame:
    cond = pd.read_csv(ACQUISITION_ABLATION / "condition_summary.csv")
    paired = pd.read_csv(ACQUISITION_ABLATION / "paired_comparisons.csv")
    direct = cond[cond["track"] == "direct"]

    rows = []
    for _, r in direct.iterrows():
        rows.append({
            "condition": r["condition"], "success_rate": r["success_rate"],
            "success_ci_low": r["success_ci_low"], "success_ci_high": r["success_ci_high"],
            "mean_detection_time_seconds": r["mean_detection_time_seconds"],
            "mean_episode_length_seconds": r["mean_episode_length_seconds"],
        })
    summary_df = pd.DataFrame(rows)

    comp_rows = []
    for name, comp_id in [("Full PPO - Escorted PPO", "A"), ("Escorted PPO - Lead", "B"), ("Full PPO - Lead", "E")]:
        r = paired[(paired["comparison_name"] == comp_id) & (paired["track"] == "direct")].iloc[0]
        comp_rows.append({"comparison": name, "observed_diff_pp": r["observed_diff_pp"],
                           "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"]})
    for name, comp_id in [("Full PPO-Kalman - Escorted PPO-Kalman", "C"), ("Escorted PPO-Kalman - Lead-Kalman", "D")]:
        r = paired[(paired["comparison_name"] == comp_id) & (paired["track"] == "kalman")].iloc[0]
        comp_rows.append({"comparison": name + " [supplemental, kalman track]", "observed_diff_pp": r["observed_diff_pp"],
                           "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"]})
    comparisons_df = pd.DataFrame(comp_rows)
    detection_time_row = paired[paired["comparison_name"] == "detection_time_full_minus_escorted_direct"].iloc[0]

    combined = pd.concat([
        summary_df.assign(row_type="condition_success"),
        comparisons_df.assign(row_type="comparison"),
        pd.DataFrame([{"comparison": "Full PPO - Escorted PPO detection time (s)",
                       "observed_diff_pp": float("nan"), "ci_low_pp": float("nan"), "ci_high_pp": float("nan"),
                       "observed_diff_seconds": detection_time_row["observed_diff_seconds"],
                       "ci_low_seconds": detection_time_row["ci_low_seconds"],
                       "ci_high_seconds": detection_time_row["ci_high_seconds"],
                       "row_type": "detection_time_comparison"}])
    ], ignore_index=True)
    return combined


# =============================================================================
# 4. DETECTION-RADIUS SUMMARY
# =============================================================================
def build_detection_radius_summary() -> pd.DataFrame:
    radius_summary = pd.read_csv(DETECTION_RADIUS / "radius_summary.csv")
    corrected_greedy = pd.read_csv(CORRECTED_GREEDY_DETECTION / "corrected_greedy_radius_summary.csv")
    pairwise = pd.read_csv(DETECTION_RADIUS / "pairwise_comparisons.csv")
    acquisition = pd.read_csv(DETECTION_RADIUS / "acquisition_summary.csv")

    rows = []
    for radius in sorted(radius_summary["detection_radius"].unique()):
        for m in SIX_METHODS:
            if m == "greedy":
                r = corrected_greedy[corrected_greedy["detection_radius"] == radius].iloc[0]
                success_rate, ci_low, ci_high = r["success_rate"], r["success_ci_low"], r["success_ci_high"]
                detection_rate = r["detection_rate"]
            else:
                r = radius_summary[(radius_summary["detection_radius"] == radius) & (radius_summary["method"] == m)].iloc[0]
                success_rate, ci_low, ci_high = r["success_rate"], r["success_ci_low"], r["success_ci_high"]
                detection_rate = r["detection_rate"]
            rows.append({
                "detection_radius": radius, "method": METHOD_DISPLAY_NAME[m], "method_key": m,
                "success_rate": success_rate, "success_ci_low": ci_low, "success_ci_high": ci_high,
                "detection_rate": detection_rate,
            })
    df = pd.DataFrame(rows)

    ppo_lead = pairwise[pairwise["comparison_name"] == "ppo_vs_lead"][["detection_radius", "observed_diff_pp", "ci_low_pp", "ci_high_pp"]]
    ppo_lead = ppo_lead.rename(columns={"observed_diff_pp": "ppo_minus_lead_pp", "ci_low_pp": "ppo_minus_lead_ci_low_pp", "ci_high_pp": "ppo_minus_lead_ci_high_pp"})
    df = df.merge(ppo_lead, on="detection_radius", how="left")

    acq = acquisition[["detection_radius", "ppo_detection_rate", "ppo_mean_detection_time_seconds_conditional",
                        "lead_detection_rate", "lead_mean_detection_time_seconds_conditional"]]
    df = df.merge(acq, on="detection_radius", how="left")
    return df


# =============================================================================
# 5. MEASUREMENT-NOISE SUMMARY
# =============================================================================
def build_measurement_noise_summary() -> pd.DataFrame:
    noise_summary = pd.read_csv(MEASUREMENT_NOISE / "noise_summary.csv")
    pairwise = pd.read_csv(MEASUREMENT_NOISE / "pairwise_comparisons.csv")
    estimator = pd.read_csv(CORRECTED_MEASUREMENT_ESTIMATOR / "corrected_estimator_summary.csv")

    rows = []
    for var in sorted(noise_summary["measurement_var"].unique()):
        for m in SIX_METHODS:
            r = noise_summary[(noise_summary["measurement_var"] == var) & (noise_summary["method"] == m)].iloc[0]
            row = {
                "measurement_var": var, "method": METHOD_DISPLAY_NAME[m], "method_key": m,
                "success_rate": r["success_rate"], "success_ci_low": r["success_ci_low"], "success_ci_high": r["success_ci_high"],
            }
            rows.append(row)
    df = pd.DataFrame(rows)

    for name, col_prefix in [("ppo_vs_lead", "ppo_minus_lead"), ("kalman_greedy_vs_greedy", "kalman_greedy_minus_greedy"),
                              ("ppo_kalman_vs_ppo", "ppo_kalman_minus_ppo")]:
        sub = pairwise[pairwise["comparison_name"] == name][["measurement_var", "observed_diff_pp", "ci_low_pp", "ci_high_pp"]]
        sub = sub.rename(columns={"observed_diff_pp": f"{col_prefix}_pp", "ci_low_pp": f"{col_prefix}_ci_low_pp", "ci_high_pp": f"{col_prefix}_ci_high_pp"})
        df = df.merge(sub, on="measurement_var", how="left")

    est_cols = ["measurement_var", "measurement_std", "method", "mean_error_sample_weighted", "rmse_sample_weighted",
                "theoretical_raw_mean_error", "theoretical_raw_rmse"]
    est = estimator[est_cols].rename(columns={"method": "method_key"})
    df = df.merge(est, on=["measurement_var", "method_key"], how="left", suffixes=("", "_estimator"))
    return df


# =============================================================================
# 6. MOBILITY-GRID SUMMARY + EQUAL-RATIO SUMMARY
# =============================================================================
def build_mobility_grid_summary() -> pd.DataFrame:
    grid_success = pd.read_csv(MOBILITY_GRID / "grid_success_summary.csv")
    envelope = pd.read_csv(MOBILITY_GRID / "ppo_vs_lead_envelope.csv")
    feasibility = pd.read_csv(MOBILITY_GRID / "feasibility_region_summary.csv")

    df = grid_success[grid_success["method"].isin(SIX_METHODS)].copy()
    df["method_display"] = df["method"].map(METHOD_DISPLAY_NAME)

    env = envelope[["defender_speed", "hostile_speed", "observed_diff_pp", "ci_low_pp", "ci_high_pp", "classification"]]
    env = env.rename(columns={"observed_diff_pp": "ppo_minus_lead_pp", "ci_low_pp": "ppo_minus_lead_ci_low_pp",
                                "ci_high_pp": "ppo_minus_lead_ci_high_pp"})
    df = df.merge(env, on=["defender_speed", "hostile_speed"], how="left")
    return df


def build_mobility_equal_ratio_summary() -> pd.DataFrame:
    return pd.read_csv(MOBILITY_GRID / "equal_ratio_summary.csv")


# =============================================================================
# 7. HOSTILE-EVASION SUMMARY
# =============================================================================
def build_hostile_evasion_summary() -> pd.DataFrame:
    evasion = pd.read_csv(HOSTILE_EVASION / "evasion_summary.csv")
    ppo_lead = pd.read_csv(HOSTILE_EVASION / "ppo_vs_lead_comparisons.csv")
    estimator_cmp = pd.read_csv(HOSTILE_EVASION / "estimator_success_comparisons.csv")
    failure = pd.read_csv(HOSTILE_EVASION / "failure_mode_summary.csv")
    acquisition = pd.read_csv(HOSTILE_EVASION / "acquisition_evasion_summary.csv")

    df = evasion[evasion["method"].isin(SIX_METHODS)].copy()
    df["method_display"] = df["method"].map(METHOD_DISPLAY_NAME)

    ppo_lead_sub = ppo_lead[ppo_lead["comparison_name"] == "ppo_vs_lead"][
        ["enemy_evasion_gain", "observed_diff_pp", "ci_low_pp", "ci_high_pp", "classification"]
    ].rename(columns={"observed_diff_pp": "ppo_minus_lead_pp", "ci_low_pp": "ppo_minus_lead_ci_low_pp", "ci_high_pp": "ppo_minus_lead_ci_high_pp"})
    df = df.merge(ppo_lead_sub, on="enemy_evasion_gain", how="left")

    kg = estimator_cmp[estimator_cmp["comparison_name"] == "kalman_greedy_vs_greedy"][
        ["enemy_evasion_gain", "observed_diff_pp", "ci_low_pp", "ci_high_pp"]
    ].rename(columns={"observed_diff_pp": "kalman_greedy_minus_greedy_pp", "ci_low_pp": "kg_minus_greedy_ci_low_pp", "ci_high_pp": "kg_minus_greedy_ci_high_pp"})
    df = df.merge(kg, on="enemy_evasion_gain", how="left")

    pk = estimator_cmp[estimator_cmp["comparison_name"] == "ppo_kalman_vs_ppo"][
        ["enemy_evasion_gain", "observed_diff_pp", "ci_low_pp", "ci_high_pp"]
    ].rename(columns={"observed_diff_pp": "ppo_kalman_minus_ppo_pp", "ci_low_pp": "pk_minus_ppo_ci_low_pp", "ci_high_pp": "pk_minus_ppo_ci_high_pp"})
    df = df.merge(pk, on="enemy_evasion_gain", how="left")

    fm = failure.rename(columns={"method": "method"})
    df = df.merge(fm[["enemy_evasion_gain", "method", "soldier_caught_rate", "unsafe_intercept_rate", "timeout_rate"]],
                   on=["enemy_evasion_gain", "method"], how="left")

    acq = acquisition[["enemy_evasion_gain", "method", "fraction_evasion_before_detection",
                        "mean_time_evasion_onset_to_detection_seconds"]]
    df = df.merge(acq, on=["enemy_evasion_gain", "method"], how="left")
    return df


# =============================================================================
# 8. MANEUVERABILITY SUMMARY
# =============================================================================
def build_maneuverability_summary() -> pd.DataFrame:
    success = pd.read_csv(MANEUVERABILITY / "maneuverability_success_summary.csv")
    ppo_lead = pd.read_csv(MANEUVERABILITY / "ppo_vs_lead_comparisons.csv")
    operational = pd.read_csv(MANEUVERABILITY / "operational_summary.csv")
    post_detection = pd.read_csv(MANEUVERABILITY / "post_detection_summary.csv")
    failure = pd.read_csv(MANEUVERABILITY / "failure_mode_summary.csv")

    df = success[success["method"].isin(SIX_METHODS)].copy()
    df["method_display"] = df["method"].map(METHOD_DISPLAY_NAME)

    ppo_lead_sub = ppo_lead[ppo_lead["comparison_name"] == "ppo_vs_lead"][
        ["defender_max_accel", "defender_max_turn_rate_deg", "observed_diff_pp", "ci_low_pp", "ci_high_pp", "classification"]
    ].rename(columns={"observed_diff_pp": "ppo_minus_lead_pp", "ci_low_pp": "ppo_minus_lead_ci_low_pp", "ci_high_pp": "ppo_minus_lead_ci_high_pp"})
    df = df.merge(ppo_lead_sub, on=["defender_max_accel", "defender_max_turn_rate_deg"], how="left")

    op_cols = ["defender_max_accel", "defender_max_turn_rate_deg", "method",
               "fraction_defender_accel_saturated", "fraction_defender_turn_saturated",
               "mean_accel_utilization", "mean_turn_rate_utilization"]
    df = df.merge(operational[op_cols], on=["defender_max_accel", "defender_max_turn_rate_deg", "method"], how="left")

    pd_cols = ["defender_max_accel", "defender_max_turn_rate_deg", "method",
               "mean_post_detection_intercept_seconds", "mean_post_detection_closing_efficiency"]
    df = df.merge(post_detection[pd_cols], on=["defender_max_accel", "defender_max_turn_rate_deg", "method"], how="left")

    fm_cols = ["defender_max_accel", "defender_max_turn_rate_deg", "method",
               "soldier_caught_rate", "unsafe_intercept_rate", "timeout_rate"]
    df = df.merge(failure[fm_cols], on=["defender_max_accel", "defender_max_turn_rate_deg", "method"], how="left")
    return df


# =============================================================================
# 9. KALMAN EVIDENCE CONSOLIDATION
# =============================================================================
def build_kalman_evidence_summary(final_nominal_comparisons, detection_df, measurement_df, mobility_df,
                                    hostile_evasion_df, maneuverability_df) -> pd.DataFrame:
    rows = []

    def sig_class(diff, lo, hi):
        if lo > 0:
            return "Kalman significantly HIGHER"
        if hi < 0:
            return "Kalman significantly LOWER"
        return "CI includes 0 (unresolved)"

    # Final nominal
    kg_row = final_nominal_comparisons[final_nominal_comparisons["comparison"] == "Kalman-Greedy - corrected Greedy"].iloc[0]
    rows.append({"experiment": "final_nominal", "condition": "nominal", "comparison": "Kalman-Greedy - Greedy",
                 "observed_diff_pp": kg_row["observed_diff_pp"], "ci_low_pp": kg_row["ci_low_pp"], "ci_high_pp": kg_row["ci_high_pp"],
                 "classification": sig_class(kg_row["observed_diff_pp"], kg_row["ci_low_pp"], kg_row["ci_high_pp"])})
    pk_row = final_nominal_comparisons[final_nominal_comparisons["comparison"] == "PPO-Kalman - PPO"].iloc[0]
    rows.append({"experiment": "final_nominal", "condition": "nominal", "comparison": "PPO-Kalman - PPO",
                 "observed_diff_pp": pk_row["observed_diff_pp"], "ci_low_pp": pk_row["ci_low_pp"], "ci_high_pp": pk_row["ci_high_pp"],
                 "classification": sig_class(pk_row["observed_diff_pp"], pk_row["ci_low_pp"], pk_row["ci_high_pp"])})

    # Detection radius (use corrected KG-Greedy)
    corrected_kg = pd.read_csv(CORRECTED_GREEDY_DETECTION / "corrected_greedy_pairwise_comparisons.csv")
    for _, r in corrected_kg[corrected_kg["comparison"] == "kalman_greedy_vs_corrected_greedy"].iterrows():
        rows.append({"experiment": "detection_radius", "condition": f"radius={r['detection_radius']}m",
                     "comparison": "Kalman-Greedy - Greedy", "observed_diff_pp": r["observed_diff_pp"],
                     "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"],
                     "classification": sig_class(r["observed_diff_pp"], r["ci_low_pp"], r["ci_high_pp"])})
    original_pairwise = pd.read_csv(DETECTION_RADIUS / "pairwise_comparisons.csv")
    for _, r in original_pairwise[original_pairwise["comparison_name"] == "ppo_kalman_vs_ppo"].iterrows():
        rows.append({"experiment": "detection_radius", "condition": f"radius={r['detection_radius']}m",
                     "comparison": "PPO-Kalman - PPO", "observed_diff_pp": r["observed_diff_pp"],
                     "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"],
                     "classification": sig_class(r["observed_diff_pp"], r["ci_low_pp"], r["ci_high_pp"])})

    # Measurement noise
    for _, r in measurement_df.drop_duplicates(subset=["measurement_var"]).iterrows():
        if not pd.isna(r.get("kalman_greedy_minus_greedy_pp")):
            rows.append({"experiment": "measurement_noise", "condition": f"var={r['measurement_var']}",
                         "comparison": "Kalman-Greedy - Greedy", "observed_diff_pp": r["kalman_greedy_minus_greedy_pp"],
                         "ci_low_pp": r["kalman_greedy_minus_greedy_ci_low_pp"], "ci_high_pp": r["kalman_greedy_minus_greedy_ci_high_pp"],
                         "classification": sig_class(r["kalman_greedy_minus_greedy_pp"], r["kalman_greedy_minus_greedy_ci_low_pp"], r["kalman_greedy_minus_greedy_ci_high_pp"])})
        if not pd.isna(r.get("ppo_kalman_minus_ppo_pp")):
            rows.append({"experiment": "measurement_noise", "condition": f"var={r['measurement_var']}",
                         "comparison": "PPO-Kalman - PPO", "observed_diff_pp": r["ppo_kalman_minus_ppo_pp"],
                         "ci_low_pp": r["ppo_kalman_minus_ppo_ci_low_pp"], "ci_high_pp": r["ppo_kalman_minus_ppo_ci_high_pp"],
                         "classification": sig_class(r["ppo_kalman_minus_ppo_pp"], r["ppo_kalman_minus_ppo_ci_low_pp"], r["ppo_kalman_minus_ppo_ci_high_pp"])})

    # Mobility grid (25 cells, both comparisons)
    mg_pairwise = pd.read_csv(MOBILITY_GRID / "pairwise_comparisons.csv")
    for _, r in mg_pairwise[mg_pairwise["comparison_name"] == "kalman_greedy_vs_greedy"].iterrows():
        rows.append({"experiment": "mobility_grid", "condition": f"v_d={r['defender_speed']},v_e={r['hostile_speed']}",
                     "comparison": "Kalman-Greedy - Greedy", "observed_diff_pp": r["observed_diff_pp"],
                     "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"],
                     "classification": sig_class(r["observed_diff_pp"], r["ci_low_pp"], r["ci_high_pp"])})
    for _, r in mg_pairwise[mg_pairwise["comparison_name"] == "ppo_kalman_vs_ppo"].iterrows():
        rows.append({"experiment": "mobility_grid", "condition": f"v_d={r['defender_speed']},v_e={r['hostile_speed']}",
                     "comparison": "PPO-Kalman - PPO", "observed_diff_pp": r["observed_diff_pp"],
                     "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"],
                     "classification": sig_class(r["observed_diff_pp"], r["ci_low_pp"], r["ci_high_pp"])})

    # Hostile evasion (5 gains, both comparisons)
    for _, r in hostile_evasion_df.drop_duplicates(subset=["enemy_evasion_gain"]).iterrows():
        if not pd.isna(r.get("kalman_greedy_minus_greedy_pp")):
            rows.append({"experiment": "hostile_evasion", "condition": f"gain={r['enemy_evasion_gain']}",
                         "comparison": "Kalman-Greedy - Greedy", "observed_diff_pp": r["kalman_greedy_minus_greedy_pp"],
                         "ci_low_pp": r["kg_minus_greedy_ci_low_pp"], "ci_high_pp": r["kg_minus_greedy_ci_high_pp"],
                         "classification": sig_class(r["kalman_greedy_minus_greedy_pp"], r["kg_minus_greedy_ci_low_pp"], r["kg_minus_greedy_ci_high_pp"])})
        if not pd.isna(r.get("ppo_kalman_minus_ppo_pp")):
            rows.append({"experiment": "hostile_evasion", "condition": f"gain={r['enemy_evasion_gain']}",
                         "comparison": "PPO-Kalman - PPO", "observed_diff_pp": r["ppo_kalman_minus_ppo_pp"],
                         "ci_low_pp": r["pk_minus_ppo_ci_low_pp"], "ci_high_pp": r["pk_minus_ppo_ci_high_pp"],
                         "classification": sig_class(r["ppo_kalman_minus_ppo_pp"], r["pk_minus_ppo_ci_low_pp"], r["pk_minus_ppo_ci_high_pp"])})

    # Maneuverability (9 cells, both comparisons)
    man_est = pd.read_csv(MANEUVERABILITY / "estimator_comparisons.csv")
    for _, r in man_est[man_est["comparison_name"] == "kalman_greedy_vs_greedy"].iterrows():
        rows.append({"experiment": "maneuverability", "condition": f"accel={r['defender_max_accel']},turn={r['defender_max_turn_rate_deg']}",
                     "comparison": "Kalman-Greedy - Greedy", "observed_diff_pp": r["observed_diff_pp"],
                     "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"],
                     "classification": sig_class(r["observed_diff_pp"], r["ci_low_pp"], r["ci_high_pp"])})
    for _, r in man_est[man_est["comparison_name"] == "ppo_kalman_vs_ppo"].iterrows():
        rows.append({"experiment": "maneuverability", "condition": f"accel={r['defender_max_accel']},turn={r['defender_max_turn_rate_deg']}",
                     "comparison": "PPO-Kalman - PPO", "observed_diff_pp": r["observed_diff_pp"],
                     "ci_low_pp": r["ci_low_pp"], "ci_high_pp": r["ci_high_pp"],
                     "classification": sig_class(r["observed_diff_pp"], r["ci_low_pp"], r["ci_high_pp"])})

    return pd.DataFrame(rows)


def summarize_kalman_evidence(kalman_df: pd.DataFrame) -> dict:
    summary = {}
    for comparison in ("Kalman-Greedy - Greedy", "PPO-Kalman - PPO"):
        sub = kalman_df[kalman_df["comparison"] == comparison]
        n_total = len(sub)
        n_sig_lower = int((sub["classification"] == "Kalman significantly LOWER").sum())
        n_sig_higher = int((sub["classification"] == "Kalman significantly HIGHER").sum())
        n_unresolved = int((sub["classification"] == "CI includes 0 (unresolved)").sum())
        summary[comparison] = {
            "n_conditions_evaluated": n_total, "n_significantly_lower": n_sig_lower,
            "n_significantly_higher": n_sig_higher, "n_unresolved": n_unresolved,
        }
    return summary


# =============================================================================
# 10. PAPER CLAIM LEDGER
# =============================================================================
def build_claim_ledger(final_nominal_table, final_nominal_comparisons, acquisition_summary,
                        detection_radius_summary, measurement_noise_summary, mobility_grid_summary,
                        hostile_evasion_summary, maneuverability_summary, kalman_totals) -> pd.DataFrame:
    def get_diff(df, name):
        return df[df["comparison"] == name].iloc[0]

    ppo_greedy = get_diff(final_nominal_comparisons, "PPO - corrected Greedy")
    ppo_pn = get_diff(final_nominal_comparisons, "PPO - PN")
    ppo_lead = get_diff(final_nominal_comparisons, "PPO - Lead")

    rows = [
        {
            "claim_id": "C1", "claim_text": "We evaluate a constrained 3-D point-mass engagement model with bounded "
            "acceleration, horizontal turn rate, speed, and vertical rate for the defender, and a reactive evasive "
            "hostile UAV under generic noisy 3-D position sensing.",
            "claim_strength": "primary", "supporting_experiment": "all", "supporting_file": "experiment_manifest.json (all experiments)",
            "primary_statistic": "n/a (model description)", "confidence_interval": "n/a",
            "contradictory_or_limiting_evidence": "none",
            "allowed_wording": "constrained 3-D point-mass engagement model; bounded acceleration, horizontal turn "
            "rate, speed, and vertical rate; reactive evasive hostile UAV; generic noisy 3-D position sensing",
            "prohibited_wording": "high-fidelity UAV model; realistic hardware model; six-DOF; radar-validated sensor",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C2", "claim_text": f"PPO outperforms corrected Greedy at the nominal operating point "
            f"(+{ppo_greedy['observed_diff_pp']:.2f}pp, 95% CI [{ppo_greedy['ci_low_pp']:.2f},{ppo_greedy['ci_high_pp']:.2f}]).",
            "claim_strength": "primary", "supporting_experiment": "final_nominal",
            "supporting_file": "final_nominal_comparisons.csv", "primary_statistic": f"{ppo_greedy['observed_diff_pp']:.2f}pp",
            "confidence_interval": f"[{ppo_greedy['ci_low_pp']:.2f},{ppo_greedy['ci_high_pp']:.2f}]",
            "contradictory_or_limiting_evidence": "None at nominal; Lead becomes competitive/superior off-nominal (C10).",
            "allowed_wording": "PPO outperforms Greedy nominally", "prohibited_wording": "PPO is universally superior",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C3", "claim_text": f"PPO outperforms PN at the nominal operating point "
            f"(+{ppo_pn['observed_diff_pp']:.2f}pp, 95% CI [{ppo_pn['ci_low_pp']:.2f},{ppo_pn['ci_high_pp']:.2f}]).",
            "claim_strength": "primary", "supporting_experiment": "final_nominal",
            "supporting_file": "final_nominal_comparisons.csv", "primary_statistic": f"{ppo_pn['observed_diff_pp']:.2f}pp",
            "confidence_interval": f"[{ppo_pn['ci_low_pp']:.2f},{ppo_pn['ci_high_pp']:.2f}]",
            "contradictory_or_limiting_evidence": "none",
            "allowed_wording": "PPO outperforms PN nominally", "prohibited_wording": "PPO is universally superior",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C4", "claim_text": f"PPO is statistically indistinguishable from Lead at the nominal operating "
            f"point ({ppo_lead['observed_diff_pp']:+.2f}pp, 95% CI [{ppo_lead['ci_low_pp']:.2f},{ppo_lead['ci_high_pp']:.2f}], "
            f"includes 0).", "claim_strength": "primary", "supporting_experiment": "final_nominal",
            "supporting_file": "final_nominal_comparisons.csv", "primary_statistic": f"{ppo_lead['observed_diff_pp']:.2f}pp",
            "confidence_interval": f"[{ppo_lead['ci_low_pp']:.2f},{ppo_lead['ci_high_pp']:.2f}]",
            "contradictory_or_limiting_evidence": "Lead significantly outperforms PPO under several off-nominal "
            "maneuverability and mobility conditions (C8, C10).",
            "allowed_wording": "PPO is statistically competitive with Lead nominally",
            "prohibited_wording": "PPO has superior post-detection guidance; PPO is universally superior",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C5", "claim_text": "A meaningful part of PPO's nominal advantage over scripted baselines "
            "arises from learned pre-acquisition positioning, not superior post-detection guidance: equalizing "
            "pre-detection behavior (Escorted PPO) reduces success by +3.63pp relative to Full PPO (CI excludes 0), "
            "and Escorted PPO becomes statistically indistinguishable from Lead.",
            "claim_strength": "primary", "supporting_experiment": "acquisition_ablation",
            "supporting_file": "acquisition_paper_summary.csv", "primary_statistic": "Full PPO - Escorted PPO = +3.63pp",
            "confidence_interval": "[+1.30,+5.93]",
            "contradictory_or_limiting_evidence": "Direct track only; kalman track effect is smaller and CI includes 0.",
            "allowed_wording": "PPO's performance benefit includes learned pre-acquisition positioning",
            "prohibited_wording": "PPO has superior post-detection guidance",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C6", "claim_text": "PPO's success rate is >= Lead's at every tested detection radius (5,10,15,"
            "20,25,30m), with PPO significantly higher at 5m and 15m; PPO's acquisition-time advantage over Lead is "
            "largest at short sensing radius (~37s at 5m) and shrinks markedly as sensing range increases (~4s at 30m).",
            "claim_strength": "primary", "supporting_experiment": "detection_radius",
            "supporting_file": "detection_radius_paper_summary.csv", "primary_statistic": "PPO-Lead detection-time gap: 37.1s (5m) to 4.4s (30m)",
            "confidence_interval": "per-radius CIs in detection_radius_paper_summary.csv",
            "contradictory_or_limiting_evidence": "PPO-Lead success difference is unresolved (CI includes 0) at 10/20/25/30m.",
            "allowed_wording": "PPO is more robust than Lead under constrained sensing range",
            "prohibited_wording": "PPO dominates at all radii",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C7", "claim_text": "PPO success remains approximately stable (~82-86%) across the tested "
            "measurement-noise range (var 0.05-4 m^2/axis), while Lead's success degrades from ~86% to ~69%; PPO-Lead "
            "becomes significantly positive at the higher tested noise levels (var>=1).",
            "claim_strength": "primary", "supporting_experiment": "measurement_noise",
            "supporting_file": "measurement_noise_paper_summary.csv", "primary_statistic": "PPO-Lead at var=4: +12.83pp",
            "confidence_interval": "[+9.63,+16.00] at var=4",
            "contradictory_or_limiting_evidence": "PPO-Lead CI includes 0 at var<=0.5.",
            "allowed_wording": "PPO is more robust than Lead to increasing measurement noise over the tested range",
            "prohibited_wording": "PPO is immune to sensor noise",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C8", "claim_text": "Absolute defender and hostile speeds materially affect success even at a "
            "fixed speed ratio (mu=v_d/v_e=1.5), demonstrating that v_d/v_e alone is not a sufficient similarity "
            "parameter under fixed acceleration, turn-rate, and vertical-rate constraints; PPO vs Lead is unresolved "
            "at 17/25 measured cells, with 4/25 favoring each controller.",
            "claim_strength": "primary", "supporting_experiment": "mobility_grid",
            "supporting_file": "mobility_grid_paper_summary.csv / mobility_equal_ratio_summary.csv",
            "primary_statistic": "Greedy success spread at mu=1.5: 63.0%-95.0% (32pp)", "confidence_interval": "see mobility_equal_ratio_summary.csv",
            "contradictory_or_limiting_evidence": "1-D mobility sweep is superseded for this claim; use only mobility_grid.",
            "allowed_wording": "absolute speeds materially affect success even at fixed mobility ratio",
            "prohibited_wording": "mobility ratio determines performance; PPO dominant operating region",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C9", "claim_text": "PPO's advantage over Lead exists at lower reactive-evasion gains "
            "(0,0.25,0.5) but erodes as gain increases; at the nominal gain (0.75) and at the maximum tested gain "
            "(1.0), PPO and Lead are statistically unresolved. PPO success degrades from ~87.3% (gain=0) to ~81.5% "
            "(gain=1).", "claim_strength": "primary", "supporting_experiment": "hostile_evasion",
            "supporting_file": "hostile_evasion_paper_summary.csv", "primary_statistic": "PPO-Lead at gain=0: +5.63pp; at gain=1: -0.90pp",
            "confidence_interval": "gain=0 [+3.73,+7.60]; gain=1 [-4.80,+3.10]",
            "contradictory_or_limiting_evidence": "none beyond the gain range tested",
            "allowed_wording": "gain=0 is 'no reactive evasion component' (not non-maneuvering hostile); gain=1 is "
            "'maximum tested reactive-evasion gain' (not optimal evasion)",
            "prohibited_wording": "gain=1 is optimal evasion; gain=0 is non-maneuvering hostile",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C10", "claim_text": "Lead significantly outperforms PPO at 6/9 tested defender-maneuverability "
            "cells; PPO is never significantly higher than Lead at any of the 9 cells (0/9); 3/9 are unresolved. At "
            "the most restrictive tested cell (accel=4,turn=45) Lead>Greedy>...>PPO; at the most permissive tested "
            "cell (accel=12,turn=135) Lead and scripted methods gain substantially while PPO's gain is marginal, "
            "revealing limited zero-shot dynamics generalization.",
            "claim_strength": "primary", "supporting_experiment": "maneuverability",
            "supporting_file": "maneuverability_paper_summary.csv", "primary_statistic": "6/9 Lead sig. higher, 0/9 PPO sig. higher, 3/9 unresolved",
            "confidence_interval": "per-cell CIs in maneuverability_paper_summary.csv",
            "contradictory_or_limiting_evidence": "PPO and Lead are statistically tied at the (8,90) nominal cell.",
            "allowed_wording": "Lead outperforms PPO in several off-nominal maneuverability conditions; the defender "
            "operates at or near its acceleration limit almost continuously under PPO",
            "prohibited_wording": "PPO commands maximum acceleration; PPO dominates classical methods",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C11", "claim_text": "Across every canonical evaluated maneuvering-target condition "
            f"({kalman_totals['Kalman-Greedy - Greedy']['n_conditions_evaluated']} conditions for Kalman-Greedy vs "
            f"Greedy; {kalman_totals['PPO-Kalman - PPO']['n_conditions_evaluated']} for PPO-Kalman vs PPO), the tested "
            "constant-velocity Kalman preprocessing never shows a statistically significant advantage for "
            "Kalman-Greedy vs Greedy "
            f"(0/{kalman_totals['Kalman-Greedy - Greedy']['n_conditions_evaluated']} significantly higher, "
            f"{kalman_totals['Kalman-Greedy - Greedy']['n_significantly_lower']}/"
            f"{kalman_totals['Kalman-Greedy - Greedy']['n_conditions_evaluated']} significantly lower), and only "
            f"1/{kalman_totals['PPO-Kalman - PPO']['n_conditions_evaluated']} isolated condition for PPO-Kalman vs "
            "PPO shows a significant advantage. Kalman RMSE is consistently ~1.8-2.3x the raw measurement RMSE.",
            "claim_strength": "primary", "supporting_experiment": "all (consolidated)",
            "supporting_file": "kalman_evidence_summary.csv",
            "primary_statistic": f"KG-Greedy: 0 sig-higher / {kalman_totals['Kalman-Greedy - Greedy']['n_significantly_lower']} sig-lower "
            f"of {kalman_totals['Kalman-Greedy - Greedy']['n_conditions_evaluated']}",
            "confidence_interval": "see kalman_evidence_summary.csv (per-condition)",
            "contradictory_or_limiting_evidence": "One isolated mobility-grid cell (v_d=18,v_e=8) shows PPO-Kalman "
            "significantly higher than PPO (+7.07pp) -- a single exception among 52+52 evaluated conditions.",
            "allowed_wording": "tested constant-velocity Kalman preprocessing does not consistently improve "
            "performance; the frozen constant-velocity Kalman model is poorly matched to the evasive, maneuvering "
            "hostile in this configuration",
            "prohibited_wording": "Kalman improves tracking; Kalman filtering is universally harmful; Kalman filters "
            "are bad (in general, outside this process/sensor/hostile model)",
            "main_or_supplement": "main",
        },
        {
            "claim_id": "C12", "claim_text": "Action semantics are unchanged across every robustness study: "
            "desired_velocity = v_d * normalize(action) for nonzero actions, with v_d fixed at 18 m/s; only the "
            "rate at which the defender's actual velocity can approach the commanded vector varies across the "
            "maneuverability grid. Maneuverability limits are not exposed to the PPO observation, so all non-nominal "
            "maneuverability/evasion-gain cells are zero-shot generalization tests, not retrained conditions. No "
            "hardware validation or six-DOF realism is claimed anywhere in this evidence freeze.",
            "claim_strength": "limitation", "supporting_experiment": "maneuverability, hostile_evasion",
            "supporting_file": "maneuverability_paper_summary.csv, hostile_evasion_paper_summary.csv",
            "primary_statistic": "n/a (methodological limitation)", "confidence_interval": "n/a",
            "contradictory_or_limiting_evidence": "n/a",
            "allowed_wording": "zero-shot dynamics/evasion-strength generalization; simulation parameters, not "
            "hardware-validated specifications",
            "prohibited_wording": "high-fidelity UAV model; realistic hardware model; six-DOF; radar-validated sensor",
            "main_or_supplement": "main",
        },
    ]
    return pd.DataFrame(rows)


# =============================================================================
# 11. EXPERIMENT PRIORITY
# =============================================================================
def build_experiment_priority() -> pd.DataFrame:
    rows = [
        {"experiment": "final_nominal (six-method comparison)", "main_paper_or_supplement": "main",
         "reason": "Establishes the core PPO-vs-classical result the entire paper narrative depends on.",
         "candidate_table_or_figure": "TABLE II, FIG 2", "must_show_numeric_result": "six-method success rate + 95% CI"},
        {"experiment": "acquisition_ablation", "main_paper_or_supplement": "main",
         "reason": "Directly explains the mechanism behind PPO's nominal advantage; addresses a likely reviewer question.",
         "candidate_table_or_figure": "TABLE III, FIG 3", "must_show_numeric_result": "Full PPO vs Escorted PPO vs Lead success + detection time"},
        {"experiment": "measurement_noise", "main_paper_or_supplement": "main",
         "reason": "Strong, clean monotonic robustness story (PPO stable, Lead degrades); directly answers a "
         "reviewer concern about sensing realism.",
         "candidate_table_or_figure": "FIG 4", "must_show_numeric_result": "success vs measurement_var, PPO-Lead diff"},
        {"experiment": "maneuverability (preferred over mobility_grid if only one fits)", "main_paper_or_supplement": "main",
         "reason": "Directly rebuts the reviewer criticism about unconstrained defender dynamics; produces the "
         "single most reviewer-relevant off-nominal finding (Lead>PPO under restrictive/permissive dynamics).",
         "candidate_table_or_figure": "FIG 6", "must_show_numeric_result": "3x3 success heatmap + PPO-Lead diff heatmap"},
        {"experiment": "hostile_evasion", "main_paper_or_supplement": "main (if space permits)",
         "reason": "Directly rebuts the reviewer criticism about insufficient hostile evasiveness; compact 1-D result.",
         "candidate_table_or_figure": "FIG 7 (compact)", "must_show_numeric_result": "success vs gain, PPO-Lead diff at gain 0/.75/1"},
        {"experiment": "mobility_grid", "main_paper_or_supplement": "supplement (unless maneuverability cannot fit)",
         "reason": "Reinforces the same 'v_d/v_e ratio insufficient' + 'no broad PPO-dominant region' finding as "
         "maneuverability's off-nominal message, with redundant narrative value; heatmap figure is expensive in "
         "two-column space.",
         "candidate_table_or_figure": "supplemental heatmap + equal-ratio table", "must_show_numeric_result": "25-cell success matrix + PPO-vs-Lead envelope counts"},
        {"experiment": "detection_radius", "main_paper_or_supplement": "supplement",
         "reason": "Acquisition-time-vs-radius curve is redundant with the acquisition ablation's mechanism story; "
         "detailed 6-radius curves add supporting evidence but are not essential in the main text.",
         "candidate_table_or_figure": "supplemental table/curve", "must_show_numeric_result": "success + detection time per radius"},
        {"experiment": "estimator/Kalman tables (detailed)", "main_paper_or_supplement": "supplement",
         "reason": "Kalman finding is stated as one sentence in the main narrative (C11); detailed per-condition "
         "RMSE tables belong in the supplement.",
         "candidate_table_or_figure": "supplemental table", "must_show_numeric_result": "raw vs Kalman RMSE per condition"},
        {"experiment": "mobility_1d (original 1-D sweep)", "main_paper_or_supplement": "supplement",
         "reason": "Superseded as primary mobility evidence by mobility_grid; retained only as a secondary "
         "zero-shot check.", "candidate_table_or_figure": "none / brief mention only",
         "must_show_numeric_result": "not required in main text"},
        {"experiment": "mobility_dynamics_audit", "main_paper_or_supplement": "supplement",
         "reason": "Diagnostic/model-verification evidence only; supports interpretation of the 1-D mobility "
         "results but is not an independent controller-performance experiment.",
         "candidate_table_or_figure": "supplemental appendix note", "must_show_numeric_result": "not required in main text"},
    ]
    return pd.DataFrame(rows)


# =============================================================================
# 12. STATIC MARKDOWN DOCS (table plan, figure plan, wording guardrails)
# =============================================================================
TABLE_PLAN_MD = """# Final Table Plan (Paper Evidence Freeze)

No more than 3 main-paper tables, per task instructions.

## TABLE I -- Nominal configuration / dynamics / sensor parameters
- **Columns**: parameter name, symbol, value, units
- **Source**: `results/final_nominal/experiment_manifest.json` (nominal_env_config block); cross-checked against
  every later experiment's manifest `note_only_*_swept` fields to confirm the same nominal values are held fixed.
- **Scope**: main paper (Table I)

## TABLE II -- Six-method nominal performance
- **Columns**: method, success rate, 95% CI, soldier-caught rate, unsafe-intercept rate, timeout rate, mean
  detection time, mean intercept time (PPO/PPO-Kalman also: training-seed SD, min/max)
- **Source**: `results/paper_evidence/final_nominal_paper_table.csv`
- **Scope**: main paper (Table II)

## TABLE III -- Key statistical comparisons / acquisition ablation
- **Columns**: comparison, observed difference (pp), 95% CI, interpretation
- **Source**: `results/paper_evidence/final_nominal_comparisons.csv` + `results/paper_evidence/acquisition_paper_summary.csv`
- **Scope**: main paper (Table III)

## Supplemental tables (not counted against the 3-table main-paper budget)
- Detection-radius per-radius success/detection-time table -- `detection_radius_paper_summary.csv`
- Measurement-noise per-variance success/estimator-RMSE table -- `measurement_noise_paper_summary.csv`
- Mobility-grid 25-cell success table + equal-ratio table -- `mobility_grid_paper_summary.csv`, `mobility_equal_ratio_summary.csv`
- Hostile-evasion per-gain table -- `hostile_evasion_paper_summary.csv`
- Maneuverability 9-cell table -- `maneuverability_paper_summary.csv`
- Consolidated Kalman-evidence table -- `kalman_evidence_summary.csv`
"""

FIGURE_PLAN_MD = """# Final Figure Plan (Paper Evidence Freeze)

**NO FIGURES ARE GENERATED YET.** This document defines the next graph-generation tasks only.

Recommended compact main-paper set: **FIG 2, FIG 3, FIG 4, FIG 6** (4 figures). FIG 1, FIG 5, FIG 7 are supplemental
or optional, per the "3-5 main-paper figures maximum" guidance.

## FIG 1 -- 3-D engagement/environment schematic or representative trajectory
- **Scientific question**: What does the engagement geometry look like?
- **Data source**: none (schematic) or one representative episode trajectory from `results/final_nominal/raw_episode_results.csv`
- **Methods shown**: one representative trajectory (e.g. PPO nominal)
- **Axes**: x/y/z spatial (3-D) or top-down + side projections
- **Error representation**: none (illustrative)
- **Panels**: 1 (or 2: schematic + example trajectory)
- **Paper priority**: supplemental / optional main (context-setting)
- **Caption message**: "Representative 3-D engagement geometry; illustrative only."
- **What NOT to imply**: physical/hardware realism.

## FIG 2 -- Nominal six-method success comparison with 95% CIs
- **Scientific question**: How do the six methods compare at the nominal operating point?
- **Data source**: `final_nominal_paper_table.csv`
- **Methods shown**: all six (Greedy, Kalman-Greedy, PN, Lead, PPO, PPO-Kalman)
- **Axes**: x=method, y=success rate (%)
- **Error representation**: 95% CI whiskers; hierarchical CI for PPO/PPO-Kalman, Wilson CI for scripted methods
- **Panels**: 1
- **Paper priority**: main (essential)
- **Caption message**: state exact seed range (20000-24999, 5000 episodes) and that Greedy uses the corrected contract.
- **What NOT to imply**: "PPO dominates all classical methods" -- PPO and Lead are statistically tied.

## FIG 3 -- Acquisition ablation
- **Scientific question**: How much of PPO's advantage comes from pre-acquisition positioning vs post-detection guidance?
- **Data source**: `acquisition_paper_summary.csv`
- **Methods shown**: Full PPO, Escorted PPO, Lead (direct track primary; kalman track optional secondary panel)
- **Axes**: x=condition, y=success rate (%); secondary panel: mean detection time (s)
- **Error representation**: 95% CI whiskers
- **Panels**: 2 (success; detection time)
- **Paper priority**: main
- **Caption message**: exact seed range (25000-25999, 1000 episodes); state Escorted PPO vs Lead CI includes 0.
- **What NOT to imply**: PPO has superior post-detection guidance.

## FIG 4 -- Measurement-noise robustness
- **Scientific question**: How does success degrade with increasing measurement noise, and how does PPO compare to Lead?
- **Data source**: `measurement_noise_paper_summary.csv`
- **Methods shown**: all six
- **Axes**: x=measurement_var (log scale optional), y=success rate (%)
- **Error representation**: 95% CI shaded band or whiskers per point
- **Panels**: 1 (success curves) + optional 1 (PPO-Lead diff with CI band)
- **Paper priority**: main
- **Caption message**: exact variance range (0.05-4 m^2/axis), seed range 31000-31999.
- **What NOT to imply**: an exact noise threshold beyond what the stored CIs support.

## FIG 5 -- Mobility 5x5 interaction heatmaps / PPO-vs-Lead difference
- **Scientific question**: How does joint (v_d, v_e) variation affect success and the PPO-vs-Lead gap?
- **Data source**: `mobility_grid_paper_summary.csv`
- **Methods shown**: heatmap per method (or a compact 2-panel: best-method success + PPO-Lead diff)
- **Axes**: x=v_e, y=v_d, color=success rate or PPO-Lead diff (pp)
- **Error representation**: significance overlay (hatch/marker) for PPO-Lead classification
- **Panels**: up to 6 (one per method) or 2 (compact)
- **Paper priority**: supplemental (unless maneuverability cannot fit in main text)
- **Caption message**: seed range 35000-35499, 500 episodes/cell; mark nominal cell (18,12) and mu=1.5 diagonal.
- **What NOT to imply**: a broad PPO-dominant operating region (17/25 cells are unresolved).

## FIG 6 -- Maneuverability 3x3 heatmaps / PPO-vs-Lead difference
- **Scientific question**: How does joint (accel, turn-rate) variation affect success and the PPO-vs-Lead gap?
- **Data source**: `maneuverability_paper_summary.csv`
- **Methods shown**: compact 2-panel (e.g. Lead success heatmap + PPO-Lead diff heatmap with significance markers)
- **Axes**: x=turn rate, y=acceleration, color=success rate or PPO-Lead diff (pp)
- **Error representation**: significance overlay for PPO-Lead classification (6/9 Lead-higher cells marked)
- **Panels**: 2
- **Paper priority**: main
- **Caption message**: seed range 34000-34999, 1000 episodes/cell; mark nominal cell (8,90); state PPO/PPO-Kalman
  trained only at (8,90).
- **What NOT to imply**: "PPO commands maximum acceleration" -- use "operates at or near its acceleration limit
  almost continuously."

## FIG 7 -- Hostile-evasion gain robustness
- **Scientific question**: How does reactive-evasion strength affect success and the PPO-vs-Lead gap?
- **Data source**: `hostile_evasion_paper_summary.csv`
- **Methods shown**: all six (or compact: PPO, Lead, Greedy)
- **Axes**: x=enemy_evasion_gain, y=success rate (%); secondary: PPO-Lead diff with CI band
- **Error representation**: 95% CI band/whiskers
- **Panels**: 1-2
- **Paper priority**: main (if space permits) or supplemental
- **Caption message**: seed range 33000-33999, 1000 episodes/gain; state gain=0 is "no reactive evasion component"
  (not non-maneuvering) and gain=1 is "maximum tested gain" (not optimal evasion).
- **What NOT to imply**: gain=1 is an adversarially optimal evasion strategy.

## Design principles reminder (do not generate yet)
Publication-quality vector PDF + PNG; no misleading truncated y-axis without justification; consistent percentage
formatting; CIs shown where appropriate; distinguish PPO training-seed uncertainty from environment-seed
uncertainty; consistent method colors/markers across all figures; six final method names only (no PN-K/Lead-K);
nominal operating point visibly marked in every sweep figure; sized for IEEE two-column format; captions must state
the exact sweep/seed range.
"""

WORDING_GUARDRAILS_MD = """# Manuscript Wording Guardrails (Paper Evidence Freeze)

## ALLOWED
- "constrained 3-D point-mass engagement model"
- "bounded acceleration, horizontal turn rate, speed, and vertical rate"
- "reactive evasive hostile UAV"
- "generic noisy 3-D position sensing"
- "PPO outperforms Greedy and PN nominally"
- "PPO is statistically competitive with Lead nominally"
- "PPO's performance benefit includes learned pre-acquisition positioning"
- "PPO is more robust than Lead to increasing measurement noise over the tested range"
- "Lead outperforms PPO in several off-nominal maneuverability conditions"
- "tested constant-velocity Kalman preprocessing does not consistently improve performance"
- "the defender operates at or near its acceleration limit almost continuously under PPO"
- "simulation parameters, not hardware-validated or claimed-realistic airframe specifications"
- "zero-shot dynamics / evasion-strength generalization"

## PROHIBITED / AVOID
- "high-fidelity UAV model"
- "realistic hardware model"
- "six-DOF"
- "radar-validated sensor"
- "PPO is universally superior"
- "PPO has superior post-detection guidance"
- "Kalman improves tracking"
- "Kalman filtering is universally harmful"
- "mobility ratio determines performance"
- "gain=1 is optimal evasion"
- "gain=0 is non-maneuvering hostile"
- "PPO commands maximum acceleration"
- "PPO dominates all classical methods" / "PPO dominant operating region"

## Rationale cross-references
See `paper_claim_ledger.csv` for the claim-by-claim allowed/prohibited wording and the exact supporting statistic
and CI for each. See `kalman_evidence_summary.csv` for the full per-condition Kalman evidence backing claim C11
(one isolated exception exists among 100+ evaluated conditions -- avoid absolute "never" language for that reason).
"""


def write_static_docs() -> None:
    (OUTPUT_ROOT / "table_plan.md").write_text(TABLE_PLAN_MD)
    (OUTPUT_ROOT / "figure_plan.md").write_text(FIGURE_PLAN_MD)
    (OUTPUT_ROOT / "wording_guardrails.md").write_text(WORDING_GUARDRAILS_MD)


def _df_to_simple_markdown(df: pd.DataFrame) -> str:
    """Minimal markdown table renderer (avoids requiring the optional 'tabulate' dependency)."""
    cols = list(df.columns)
    lines = ["| " + " | ".join(cols) + " |", "| " + " | ".join("---" for _ in cols) + " |"]
    for _, row in df.iterrows():
        lines.append("| " + " | ".join(str(row[c]) for c in cols) + " |")
    return "\n".join(lines)


def write_summary_md(manifest: dict, final_nominal_table: pd.DataFrame) -> None:
    lines = ["# Paper Evidence Freeze Summary\n"]
    lines.append(f"Generated at {manifest['generated_at_utc']} from commit `{manifest['source_git_commit']}` "
                  f"on branch `{manifest['source_git_branch']}`.\n")
    lines.append("## Six canonical methods\n")
    lines.append(", ".join(METHOD_DISPLAY_NAME[m] for m in SIX_METHODS) + "\n")
    lines.append("## Final nominal success rates (canonical)\n")
    lines.append(_df_to_simple_markdown(final_nominal_table[["method", "success_rate", "success_ci_low", "success_ci_high"]]) + "\n")
    lines.append("## Registry composition\n")
    lines.append(f"- canonical: {manifest['n_canonical_registry_rows']}\n")
    lines.append(f"- supplemental: {manifest['n_supplemental_registry_rows']}\n")
    lines.append(f"- diagnostic: {manifest['n_diagnostic_registry_rows']}\n")
    lines.append(f"- invalid (excluded from all canonical summaries): {manifest['n_invalid_registry_rows']}\n")
    lines.append(f"- superseded (excluded as primary evidence): {manifest['n_superseded_registry_rows']}\n")
    (OUTPUT_ROOT / "paper_evidence_summary.md").write_text("\n".join(lines))



def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    registry = build_registry()
    registry.to_csv(OUTPUT_ROOT / "paper_evidence_registry.csv", index=False)
    assert registry["evidence_id"].duplicated().sum() == 0, "duplicate evidence_id found in registry"

    final_nominal_table = build_final_nominal_table()
    discrepancies = _verify_final_nominal_table(final_nominal_table)
    if discrepancies:
        raise RuntimeError(f"Final nominal table discrepancies vs expected values: {discrepancies}")
    final_nominal_table.to_csv(OUTPUT_ROOT / "final_nominal_paper_table.csv", index=False)

    final_nominal_comparisons = build_final_nominal_comparisons()
    final_nominal_comparisons.to_csv(OUTPUT_ROOT / "final_nominal_comparisons.csv", index=False)

    acquisition_summary = build_acquisition_summary()
    acquisition_summary.to_csv(OUTPUT_ROOT / "acquisition_paper_summary.csv", index=False)

    detection_radius_summary = build_detection_radius_summary()
    detection_radius_summary.to_csv(OUTPUT_ROOT / "detection_radius_paper_summary.csv", index=False)

    measurement_noise_summary = build_measurement_noise_summary()
    measurement_noise_summary.to_csv(OUTPUT_ROOT / "measurement_noise_paper_summary.csv", index=False)

    mobility_grid_summary = build_mobility_grid_summary()
    mobility_grid_summary.to_csv(OUTPUT_ROOT / "mobility_grid_paper_summary.csv", index=False)

    mobility_equal_ratio_summary = build_mobility_equal_ratio_summary()
    mobility_equal_ratio_summary.to_csv(OUTPUT_ROOT / "mobility_equal_ratio_summary.csv", index=False)

    hostile_evasion_summary = build_hostile_evasion_summary()
    hostile_evasion_summary.to_csv(OUTPUT_ROOT / "hostile_evasion_paper_summary.csv", index=False)

    maneuverability_summary = build_maneuverability_summary()
    maneuverability_summary.to_csv(OUTPUT_ROOT / "maneuverability_paper_summary.csv", index=False)

    kalman_evidence_summary = build_kalman_evidence_summary(
        final_nominal_comparisons, detection_radius_summary, measurement_noise_summary,
        mobility_grid_summary, hostile_evasion_summary, maneuverability_summary,
    )
    kalman_evidence_summary.to_csv(OUTPUT_ROOT / "kalman_evidence_summary.csv", index=False)
    kalman_totals = summarize_kalman_evidence(kalman_evidence_summary)

    claim_ledger = build_claim_ledger(
        final_nominal_table, final_nominal_comparisons, acquisition_summary, detection_radius_summary,
        measurement_noise_summary, mobility_grid_summary, hostile_evasion_summary, maneuverability_summary,
        kalman_totals,
    )
    claim_ledger.to_csv(OUTPUT_ROOT / "paper_claim_ledger.csv", index=False)

    experiment_priority = build_experiment_priority()
    experiment_priority.to_csv(OUTPUT_ROOT / "experiment_priority.csv", index=False)

    write_static_docs()

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "no_execution_guarantee": {
            "simulations_run": 0, "episodes_run": 0, "new_evaluation_seeds_used": 0, "models_retrained": 0,
        },
        "six_canonical_methods": list(SIX_METHODS),
        "excluded_methods": list(EXCLUDED_METHODS),
        "final_nominal_verification_discrepancies": discrepancies,
        "kalman_evidence_totals": kalman_totals,
        "n_registry_rows": len(registry),
        "n_invalid_registry_rows": int((registry["status"] == "invalid").sum()),
        "n_superseded_registry_rows": int((registry["status"] == "superseded").sum()),
        "n_diagnostic_registry_rows": int((registry["status"] == "diagnostic").sum()),
        "n_canonical_registry_rows": int((registry["status"] == "canonical").sum()),
        "n_supplemental_registry_rows": int((registry["status"] == "supplemental").sum()),
    }
    (OUTPUT_ROOT / "paper_evidence_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    write_summary_md(manifest, final_nominal_table)

    print(json.dumps(manifest, indent=2, default=str))
    print(f"\nAll outputs written to {OUTPUT_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
