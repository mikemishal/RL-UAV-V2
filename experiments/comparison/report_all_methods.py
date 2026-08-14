"""
Generate advisor-ready four-method comparison report.

=============================================================================
PURPOSE: RESEARCH-STYLE SUMMARY FOR ADVISOR UPDATES
=============================================================================

Reads baseline, kalman baseline, RL, and RL-with-Kalman evaluation results and generates
a markdown report with interpretable conclusions.

Key Questions Addressed:
    1. Does Kalman improve the greedy baseline?
    2. Does RL beat both hand-designed baselines?
    3. Does RL-with-Kalman beat direct RL?
    4. Is performance gain driven more by estimation or by control?
    5. Where does each method help most in parameter sweeps?

Outputs:
    results/comparison/comparison_summary_four_methods.md

Usage:
    python experiments/comparison/report_all_methods.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Dict, Optional, Tuple

# Add project root to path
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd
from scipy import stats

from experiments.experiment_config import (
    get_eval_csv_filename,
    get_sweep_csv_filename,
)


# Default paths
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
RL_DIR = PROJECT_ROOT / "results" / "rl"
RL_KALMAN_DIR = PROJECT_ROOT / "results" / "rl_kalman"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


def load_all_data() -> Dict[str, Optional[pd.DataFrame]]:
    """Load all available result files from all four methods."""
    data = {}
    
    # Overall evaluation results
    paths = {
        "baseline_eval": [BASELINE_DIR / get_eval_csv_filename("baseline")],
        "kalman_baseline_eval": [BASELINE_DIR / get_eval_csv_filename("kalman_baseline")],
        "rl_eval": [RL_DIR / get_eval_csv_filename("rl")],
        "rl_kalman_eval": [RL_KALMAN_DIR / get_eval_csv_filename("rl_kalman")],
    }
    
    for key, path_list in paths.items():
        data[key] = None
        for path in path_list:
            if path.exists():
                data[key] = pd.read_csv(path)
                break
    
    # Sweep results
    sweeps = ["defender_speed", "enemy_speed", "detection_radius", "speed_grid"]
    method_dirs = {
        "baseline": BASELINE_DIR,
        "kalman_baseline": BASELINE_DIR,
        "rl": RL_DIR,
        "rl_kalman": RL_KALMAN_DIR,
    }
    
    for sweep in sweeps:
        for method, dir_path in method_dirs.items():
            key = f"{method}_{sweep}"
            path = dir_path / get_sweep_csv_filename(sweep, method)
            data[key] = pd.read_csv(path) if path.exists() else None

    # Noise-sweep results (fixed artifact names)
    noise_paths = {
        "baseline_noise": BASELINE_DIR / "sweep_noise_baseline.csv",
        "kalman_baseline_noise": BASELINE_DIR / "sweep_noise_kalman_baseline.csv",
        "rl_noise": RL_DIR / "sweep_noise_rl.csv",
        "rl_kalman_noise": RL_KALMAN_DIR / "sweep_noise_rl_kalman.csv",
        "noise_all_methods_summary": OUTPUT_DIR / "sweep_noise_all_methods_summary.csv",
    }
    for key, path in noise_paths.items():
        data[key] = pd.read_csv(path) if path.exists() else None
    
    return data


def compute_eval_stats(df: pd.DataFrame) -> Optional[dict]:
    """Compute summary statistics from episode results."""
    if df is None or len(df) == 0:
        return None
    
    n = len(df)
    n_success = int(df["success"].sum())
    
    success_rate = n_success / n
    success_se = np.sqrt(success_rate * (1 - success_rate) / n)
    
    # Wilson score CI (more accurate for proportions)
    z = 1.96
    denominator = 1 + z**2 / n
    center = (success_rate + z**2 / (2 * n)) / denominator
    margin = z * np.sqrt((success_rate * (1 - success_rate) + z**2 / (4 * n)) / n) / denominator
    ci_low = max(0, center - margin)
    ci_high = min(1, center + margin)
    
    # Tracking error (RL-Kalman only)
    mean_tracking_error = None
    if "mean_tracking_error" in df.columns:
        valid = df["mean_tracking_error"].dropna()
        if len(valid) > 0:
            mean_tracking_error = valid.mean()
    
    return {
        "n_episodes": n,
        "n_success": n_success,
        "success_rate": success_rate,
        "success_se": success_se,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "fail_rate": len(df[df["outcome"].isin(["soldier_caught", "unsafe_intercept"])]) / n,
        "timeout_rate": len(df[df["outcome"] == "timeout"]) / n,
        "mean_ep_len": df["episode_length"].mean(),
        "mean_tracking_error": mean_tracking_error,
    }


def two_proportion_z_test(
    n1: int, p1: float, n2: int, p2: float
) -> Tuple[float, float, str]:
    """
    Perform two-proportion z-test.
    
    Returns:
        Tuple of (z_statistic, p_value, significance_str).
    """
    # Pooled proportion
    p_pooled = (p1 * n1 + p2 * n2) / (n1 + n2)
    se_pooled = np.sqrt(p_pooled * (1 - p_pooled) * (1/n1 + 1/n2))
    
    if se_pooled == 0:
        return 0.0, 1.0, "n.s."
    
    z = (p2 - p1) / se_pooled
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))
    
    if p_value < 0.001:
        sig = "***"
    elif p_value < 0.01:
        sig = "**"
    elif p_value < 0.05:
        sig = "*"
    else:
        sig = "n.s."
    
    return z, p_value, sig


def analyze_tracking_error_correlation(df: pd.DataFrame) -> Optional[dict]:
    """
    Analyze correlation between tracking error and success.
    
    Returns:
        Dictionary with correlation analysis results.
    """
    if df is None or "mean_tracking_error" not in df.columns:
        return None
    
    # Filter to episodes with valid tracking error
    valid = df.dropna(subset=["mean_tracking_error"])
    if len(valid) < 10:
        return None
    
    # Point-biserial correlation (binary success vs continuous tracking error)
    success = valid["success"].values
    tracking_error = valid["mean_tracking_error"].values
    
    # Compute correlation
    r, p_value = stats.pointbiserialr(success, tracking_error)
    
    # Compare tracking error between success and failure
    success_errors = tracking_error[success == 1]
    failure_errors = tracking_error[success == 0]
    
    if len(success_errors) > 5 and len(failure_errors) > 5:
        t_stat, t_pval = stats.ttest_ind(success_errors, failure_errors)
        mean_success = np.mean(success_errors)
        mean_failure = np.mean(failure_errors)
    else:
        t_stat, t_pval = np.nan, np.nan
        mean_success = np.mean(success_errors) if len(success_errors) > 0 else np.nan
        mean_failure = np.mean(failure_errors) if len(failure_errors) > 0 else np.nan
    
    return {
        "correlation_r": r,
        "correlation_p": p_value,
        "mean_error_success": mean_success,
        "mean_error_failure": mean_failure,
        "t_statistic": t_stat,
        "t_pvalue": t_pval,
        "n_success": len(success_errors),
        "n_failure": len(failure_errors),
    }


def analyze_robustness(
    baseline_sweep: pd.DataFrame,
    rl_sweep: pd.DataFrame,
    rl_kalman_sweep: pd.DataFrame,
    x_col: str,
    challenging_condition: str,  # "low" or "high"
) -> Optional[dict]:
    """
    Analyze whether Kalman improves robustness at challenging conditions.
    
    Args:
        baseline_sweep, rl_sweep, rl_kalman_sweep: Sweep DataFrames.
        x_col: Column name for the swept parameter.
        challenging_condition: "low" for low values being challenging (e.g., detection_radius),
                              "high" for high values being challenging (e.g., enemy_speed).
    
    Returns:
        Dictionary with robustness analysis.
    """
    if any(df is None for df in [baseline_sweep, rl_sweep, rl_kalman_sweep]):
        return None
    
    # Merge all three
    merged = baseline_sweep[[x_col, "success_rate"]].copy()
    merged = merged.rename(columns={"success_rate": "baseline"})
    merged = merged.merge(
        rl_sweep[[x_col, "success_rate"]].rename(columns={"success_rate": "rl"}),
        on=x_col
    )
    merged = merged.merge(
        rl_kalman_sweep[[x_col, "success_rate"]].rename(columns={"success_rate": "rl_kalman"}),
        on=x_col
    )
    
    if len(merged) == 0:
        return None
    
    # Split into easy and challenging conditions
    median_val = merged[x_col].median()
    
    if challenging_condition == "low":
        challenging = merged[merged[x_col] <= median_val]
        easy = merged[merged[x_col] > median_val]
    else:  # "high"
        challenging = merged[merged[x_col] >= median_val]
        easy = merged[merged[x_col] < median_val]
    
    # Compute RL-Kalman advantage over RL
    kalman_advantage_challenging = (challenging["rl_kalman"] - challenging["rl"]).mean()
    kalman_advantage_easy = (easy["rl_kalman"] - easy["rl"]).mean()
    
    return {
        "parameter": x_col,
        "challenging_condition": challenging_condition,
        "kalman_advantage_challenging": kalman_advantage_challenging,
        "kalman_advantage_easy": kalman_advantage_easy,
        "improvement_ratio": kalman_advantage_challenging / kalman_advantage_easy if kalman_advantage_easy != 0 else np.nan,
        "n_challenging": len(challenging),
        "n_easy": len(easy),
    }


def format_pct(val: float, decimals: int = 1) -> str:
    """Format value as percentage."""
    return f"{val * 100:.{decimals}f}%"


def format_ci(stats: dict) -> str:
    """Format confidence interval string."""
    return f"[{format_pct(stats['ci_95_low'])}, {format_pct(stats['ci_95_high'])}]"


def _noise_trend(df: Optional[pd.DataFrame]) -> Optional[dict]:
    """Estimate degradation trend for success_rate vs noise_value."""
    if df is None or len(df) < 2:
        return None
    if "noise_value" not in df.columns or "success_rate" not in df.columns:
        return None

    work = df[["noise_value", "success_rate"]].dropna().sort_values("noise_value")
    if len(work) < 2:
        return None

    x = work["noise_value"].to_numpy(dtype=float)
    y = work["success_rate"].to_numpy(dtype=float)
    slope = float(np.polyfit(x, y, 1)[0])
    delta = float(y[-1] - y[0])
    return {
        "slope": slope,
        "delta": delta,
        "x_min": float(x[0]),
        "x_max": float(x[-1]),
        "y_start": float(y[0]),
        "y_end": float(y[-1]),
        "n": len(work),
    }


def _tracking_degradation_link(df: Optional[pd.DataFrame]) -> Optional[dict]:
    """Estimate correlation between tracking error and success degradation across noise levels."""
    if df is None or len(df) < 2:
        return None
    required = {"noise_value", "success_rate", "mean_tracking_error"}
    if not required.issubset(df.columns):
        return None

    work = df[list(required)].dropna().sort_values("noise_value")
    if len(work) < 2:
        return None

    r, p = stats.pearsonr(work["mean_tracking_error"].values, work["success_rate"].values)
    return {
        "r": float(r),
        "p": float(p),
        "n": len(work),
        "min_track": float(work["mean_tracking_error"].min()),
        "max_track": float(work["mean_tracking_error"].max()),
    }


def generate_report(data: Dict[str, Optional[pd.DataFrame]]) -> str:
    """Generate comprehensive markdown report for four methods."""
    lines = []

    method_meta = [
        ("Greedy Baseline", "baseline_eval"),
        ("Kalman Baseline", "kalman_baseline_eval"),
        ("PPO (Direct RL)", "rl_eval"),
        ("PPO (RL-Kalman)", "rl_kalman_eval"),
    ]

    stats_by_method = {
        name: compute_eval_stats(data.get(key))
        for name, key in method_meta
    }

    baseline_stats = stats_by_method["Greedy Baseline"]
    kalman_baseline_stats = stats_by_method["Kalman Baseline"]
    rl_stats = stats_by_method["PPO (Direct RL)"]
    rl_kalman_stats = stats_by_method["PPO (RL-Kalman)"]

    def comp(a: dict, b: dict) -> dict:
        delta = b["success_rate"] - a["success_rate"]
        z, p, sig = two_proportion_z_test(
            a["n_episodes"], a["success_rate"],
            b["n_episodes"], b["success_rate"],
        )
        return {"delta": delta, "z": z, "p": p, "sig": sig}

    # Header
    lines.append("# Four-Method Comparison Report")
    lines.append("")

    # Noise sensitivity section
    lines.append("## Noise Sensitivity Analysis")
    lines.append("")

    baseline_noise = data.get("baseline_noise")
    kalman_baseline_noise = data.get("kalman_baseline_noise")
    rl_noise = data.get("rl_noise")
    rl_kalman_noise = data.get("rl_kalman_noise")
    noise_summary = data.get("noise_all_methods_summary")

    noise_trends = {
        "Greedy Baseline": _noise_trend(baseline_noise),
        "Kalman Baseline": _noise_trend(kalman_baseline_noise),
        "PPO (Direct RL)": _noise_trend(rl_noise),
        "PPO (RL-Kalman)": _noise_trend(rl_kalman_noise),
    }

    available_trends = {k: v for k, v in noise_trends.items() if v is not None}
    if not available_trends:
        lines.append("Noise sweep data unavailable. Run all four noise sweep scripts before interpreting noise sensitivity.")
        lines.append("")
    else:
        lines.append("### Quantitative Trend Summary")
        lines.append("")
        lines.append("| Method | Noise Range | Success at Low Noise | Success at High Noise | Delta (high-low) | Linear Slope |")
        lines.append("|--------|-------------|----------------------|-----------------------|------------------|--------------|")
        for method_name in ["Greedy Baseline", "Kalman Baseline", "PPO (Direct RL)", "PPO (RL-Kalman)"]:
            tr = available_trends.get(method_name)
            if tr is None:
                lines.append(f"| {method_name} | — | — | — | — | — |")
                continue
            lines.append(
                f"| {method_name} | {tr['x_min']:.3f} to {tr['x_max']:.3f} | "
                f"{format_pct(tr['y_start'])} | {format_pct(tr['y_end'])} | "
                f"{format_pct(tr['delta'])} | {tr['slope']:.4f} |"
            )
        lines.append("")

        # Which method degrades fastest as noise increases?
        fastest_method = min(available_trends.items(), key=lambda kv: kv[1]["slope"])[0]
        fastest_slope = available_trends[fastest_method]["slope"]
        lines.append("### Interpretation")
        lines.append("")
        lines.append(
            f"- Method degrading fastest with increasing noise: **{fastest_method}** "
            f"(most negative slope = {fastest_slope:.4f})."
        )

        # Kalman helps greedy baseline at higher noise?
        if noise_summary is not None and "diff_kalman_baseline_vs_baseline" in noise_summary.columns:
            valid = noise_summary[["noise_value", "diff_kalman_baseline_vs_baseline"]].dropna()
            if len(valid) > 0:
                high_thr = valid["noise_value"].median()
                high = valid[valid["noise_value"] >= high_thr]
                low = valid[valid["noise_value"] < high_thr]
                high_mean = float(high["diff_kalman_baseline_vs_baseline"].mean()) if len(high) > 0 else np.nan
                low_mean = float(low["diff_kalman_baseline_vs_baseline"].mean()) if len(low) > 0 else np.nan
                lines.append(
                    f"- Kalman effect on greedy baseline (Kalman Baseline - Baseline): "
                    f"low-noise mean = {format_pct(low_mean) if not np.isnan(low_mean) else 'N/A'}, "
                    f"high-noise mean = {format_pct(high_mean) if not np.isnan(high_mean) else 'N/A'}."
                )
                if not np.isnan(high_mean):
                    if high_mean > 0:
                        lines.append("  Research interpretation: Kalman filtering helps the hand-designed baseline under higher sensing noise.")
                    elif high_mean < 0:
                        lines.append("  Research interpretation: Kalman filtering does not rescue the greedy baseline at higher noise under current settings.")
                    else:
                        lines.append("  Research interpretation: No measurable high-noise advantage from Kalman filtering for the greedy baseline.")

        # RL-Kalman competitiveness at higher noise
        if noise_summary is not None and "diff_rl_kalman_vs_rl" in noise_summary.columns:
            valid = noise_summary[["noise_value", "diff_rl_kalman_vs_rl"]].dropna()
            if len(valid) > 0:
                high_thr = valid["noise_value"].median()
                high = valid[valid["noise_value"] >= high_thr]
                high_mean = float(high["diff_rl_kalman_vs_rl"].mean()) if len(high) > 0 else np.nan
                lines.append(
                    f"- RL-Kalman competitiveness at higher noise (RL-Kalman - Direct RL): "
                    f"high-noise mean delta = {format_pct(high_mean) if not np.isnan(high_mean) else 'N/A'}."
                )
                if not np.isnan(high_mean):
                    if high_mean > 0:
                        lines.append("  Research interpretation: RL-Kalman becomes competitive/superior as noise increases.")
                    elif high_mean < 0:
                        lines.append("  Research interpretation: Direct RL remains stronger than RL-Kalman at higher noise in this dataset.")
                    else:
                        lines.append("  Research interpretation: RL-Kalman and Direct RL are effectively tied at higher noise.")
            else:
                lines.append("- RL-Kalman competitiveness at higher noise: insufficient overlapping RL and RL-Kalman noise points for a stable conclusion.")
        else:
            lines.append("- RL-Kalman competitiveness at higher noise: required pairwise noise summary columns are unavailable.")

        # Direct RL low/moderate robustness
        rl_tr = available_trends.get("PPO (Direct RL)")
        if rl_tr is not None:
            if rl_tr["slope"] > -0.03:
                lines.append("- Direct RL robustness: slope indicates relatively mild degradation, consistent with robustness to low/moderate noise.")
            else:
                lines.append("- Direct RL robustness: slope indicates notable degradation, suggesting limited robustness as noise rises.")
        else:
            lines.append("- Direct RL robustness to low/moderate noise: insufficient multi-point RL noise sweep data for a trend estimate.")

        # Tracking-error correlation with degradation
        kb_link = _tracking_degradation_link(kalman_baseline_noise)
        rlk_link = _tracking_degradation_link(rl_kalman_noise)
        if kb_link or rlk_link:
            lines.append("- Tracking error vs performance degradation:")
            if kb_link:
                lines.append(
                    f"  Kalman Baseline correlation(mean_tracking_error, success_rate) = "
                    f"{kb_link['r']:.3f} (p={kb_link['p']:.4f}, n={kb_link['n']})."
                )
            if rlk_link:
                lines.append(
                    f"  RL-Kalman correlation(mean_tracking_error, success_rate) = "
                    f"{rlk_link['r']:.3f} (p={rlk_link['p']:.4f}, n={rlk_link['n']})."
                )
            lines.append(
                "  Research interpretation: more negative correlation indicates that increasing tracking error is associated with success degradation under noisy sensing."
            )
        else:
            lines.append("- Tracking error vs performance degradation: insufficient multi-point tracking-error noise data to estimate correlation reliably.")
        lines.append("")
    lines.append("## UAV Defender: Baseline vs Kalman Baseline vs Direct RL vs RL-Kalman")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    lines.append("---")
    lines.append("")

    # Overall table
    lines.append("## Overall Performance Comparison")
    lines.append("")
    lines.append("| Method | Episodes | Success Rate | 95% CI | Failure Rate | Timeout Rate | Mean Episode Length | Mean Tracking Error |")
    lines.append("|--------|----------|--------------|--------|--------------|--------------|---------------------|---------------------|")

    for name, _ in method_meta:
        st = stats_by_method[name]
        if st is None:
            lines.append(f"| {name} | — | — | — | — | — | — | — |")
            continue
        mte = f"{st['mean_tracking_error']:.4f}" if st.get("mean_tracking_error") is not None else "N/A"
        lines.append(
            f"| {name} | {st['n_episodes']} | "
            f"{format_pct(st['success_rate'])} ± {format_pct(st['success_se'])} | "
            f"{format_ci(st)} | {format_pct(st['fail_rate'])} | {format_pct(st['timeout_rate'])} | "
            f"{st['mean_ep_len']:.1f} | {mte} |"
        )
    lines.append("")

    # Core causal questions
    lines.append("## Advisor Questions and Answers")
    lines.append("")

    all_present = all(st is not None for st in [baseline_stats, kalman_baseline_stats, rl_stats, rl_kalman_stats])
    if not all_present:
        lines.append("Insufficient data to answer all four-method questions. Ensure all four evaluation CSV files exist.")
        lines.append("")
    else:
        kalman_vs_baseline = comp(baseline_stats, kalman_baseline_stats)
        rl_vs_baseline = comp(baseline_stats, rl_stats)
        rl_vs_kalman_baseline = comp(kalman_baseline_stats, rl_stats)
        rl_kalman_vs_rl = comp(rl_stats, rl_kalman_stats)

        estimation_only_gain = kalman_baseline_stats["success_rate"] - baseline_stats["success_rate"]
        control_only_gain = rl_stats["success_rate"] - baseline_stats["success_rate"]
        combined_gain = rl_kalman_stats["success_rate"] - baseline_stats["success_rate"]
        within_rl_estimation_gain = rl_kalman_stats["success_rate"] - rl_stats["success_rate"]

        lines.append("### 1. Does Kalman improve the greedy baseline?")
        lines.append("")
        lines.append(
            f"Kalman Baseline vs Greedy Baseline: {format_pct(kalman_vs_baseline['delta'])} "
            f"(p={kalman_vs_baseline['p']:.4f}, {kalman_vs_baseline['sig']})."
        )
        if kalman_vs_baseline["p"] < 0.05 and kalman_vs_baseline["delta"] > 0:
            lines.append("Result: Kalman state estimation provides a statistically significant improvement for a hand-designed controller.")
        elif kalman_vs_baseline["p"] < 0.05 and kalman_vs_baseline["delta"] < 0:
            lines.append("Result: Kalman state estimation significantly degrades the hand-designed controller.")
        else:
            lines.append("Result: No statistically significant controller-independent gain from Kalman filtering.")
        lines.append("")

        lines.append("### 2. Does RL improve over both baselines?")
        lines.append("")
        lines.append(
            f"Direct RL vs Greedy Baseline: {format_pct(rl_vs_baseline['delta'])} "
            f"(p={rl_vs_baseline['p']:.4f}, {rl_vs_baseline['sig']})."
        )
        lines.append(
            f"Direct RL vs Kalman Baseline: {format_pct(rl_vs_kalman_baseline['delta'])} "
            f"(p={rl_vs_kalman_baseline['p']:.4f}, {rl_vs_kalman_baseline['sig']})."
        )
        if rl_vs_baseline["delta"] > 0 and rl_vs_kalman_baseline["delta"] > 0:
            lines.append("Result: Learned control outperforms both hand-designed baselines.")
        else:
            lines.append("Result: RL does not consistently dominate both baselines under current settings.")
        lines.append("")

        lines.append("### 3. Does RL-Kalman improve over Direct RL?")
        lines.append("")
        lines.append(
            f"RL-Kalman vs Direct RL: {format_pct(rl_kalman_vs_rl['delta'])} "
            f"(p={rl_kalman_vs_rl['p']:.4f}, {rl_kalman_vs_rl['sig']})."
        )
        if rl_kalman_vs_rl["p"] < 0.05 and rl_kalman_vs_rl["delta"] > 0:
            lines.append("Result: Estimation on top of learned control yields a significant additional gain.")
        elif rl_kalman_vs_rl["p"] < 0.05 and rl_kalman_vs_rl["delta"] < 0:
            lines.append("Result: Kalman estimation significantly hurts the learned controller in this environment.")
        else:
            lines.append("Result: RL-Kalman and Direct RL are statistically similar at current sample size.")
        lines.append("")

        lines.append("### 4. Which contributes more: estimation or control?")
        lines.append("")
        lines.append(f"Estimation-only gain (Greedy → Kalman Baseline): {format_pct(estimation_only_gain)}")
        lines.append(f"Control-only gain (Greedy → Direct RL): {format_pct(control_only_gain)}")
        lines.append(f"Combined gain (Greedy → RL-Kalman): {format_pct(combined_gain)}")
        lines.append(f"Within-RL estimation effect (Direct RL → RL-Kalman): {format_pct(within_rl_estimation_gain)}")
        lines.append("")
        if abs(control_only_gain) > abs(estimation_only_gain):
            lines.append("Interpretation: Performance gains are dominated by the control component (learning), not by estimation alone.")
        elif abs(control_only_gain) < abs(estimation_only_gain):
            lines.append("Interpretation: Estimation contributes more than controller learning in this setup.")
        else:
            lines.append("Interpretation: Estimation and control have comparable impact in this setup.")
        lines.append("")

    # Sweep-level summary
    lines.append("---")
    lines.append("")
    lines.append("## Parameter-Sweep Comparison (Method Ranking)")
    lines.append("")

    def sweep_wins(sweep_name: str, x_col: str) -> Optional[pd.DataFrame]:
        pieces = []
        for method_key, col_name in [
            ("baseline", "Greedy Baseline"),
            ("kalman_baseline", "Kalman Baseline"),
            ("rl", "PPO (Direct RL)"),
            ("rl_kalman", "PPO (RL-Kalman)"),
        ]:
            df = data.get(f"{method_key}_{sweep_name}")
            if df is None or x_col not in df.columns or "success_rate" not in df.columns:
                return None
            pieces.append(df[[x_col, "success_rate"]].rename(columns={"success_rate": col_name}))

        merged = pieces[0]
        for p in pieces[1:]:
            merged = merged.merge(p, on=x_col)

        score_cols = [c for c in merged.columns if c != x_col]
        merged["best_method"] = merged[score_cols].idxmax(axis=1)
        return merged

    for sweep_name, x_col, title in [
        ("defender_speed", "defender_speed", "Defender speed sweep"),
        ("enemy_speed", "enemy_speed", "Enemy speed sweep"),
        ("detection_radius", "detection_radius", "Detection radius sweep"),
    ]:
        merged = sweep_wins(sweep_name, x_col)
        if merged is None:
            lines.append(f"### {title}")
            lines.append("")
            lines.append("Data unavailable.")
            lines.append("")
            continue

        lines.append(f"### {title}")
        lines.append("")
        wins = merged["best_method"].value_counts().to_dict()
        total = len(merged)
        for method_name in ["Greedy Baseline", "Kalman Baseline", "PPO (Direct RL)", "PPO (RL-Kalman)"]:
            w = wins.get(method_name, 0)
            lines.append(f"- {method_name}: {w}/{total} best settings")
        lines.append("")

    # Speed grid summary
    grid_methods = {
        "Greedy Baseline": data.get("baseline_speed_grid"),
        "Kalman Baseline": data.get("kalman_baseline_speed_grid"),
        "PPO (Direct RL)": data.get("rl_speed_grid"),
        "PPO (RL-Kalman)": data.get("rl_kalman_speed_grid"),
    }

    lines.append("### Speed-grid sweep")
    lines.append("")
    if all(df is not None for df in grid_methods.values()):
        lines.append("Mean success rate across all defender/enemy speed grid points:")
        lines.append("")
        for method_name, df in grid_methods.items():
            lines.append(f"- {method_name}: {format_pct(df['success_rate'].mean())}")
    else:
        lines.append("Data unavailable.")
    lines.append("")

    # Conclusion
    lines.append("---")
    lines.append("")
    lines.append("## Research Conclusion")
    lines.append("")
    if all_present:
        if kalman_vs_baseline["delta"] > 0 and kalman_vs_baseline["p"] < 0.05:
            kalman_statement = "Kalman provides a measurable controller-independent gain for the hand-designed policy."
        elif kalman_vs_baseline["delta"] > 0:
            kalman_statement = "Kalman baseline is numerically better, but the gain is not statistically significant."
        elif kalman_vs_baseline["delta"] < 0 and kalman_vs_baseline["p"] < 0.05:
            kalman_statement = "Kalman significantly degrades the hand-designed baseline."
        else:
            kalman_statement = "Kalman does not provide a meaningful controller-independent baseline gain."

        lines.append(kalman_statement)
        lines.append("")
        lines.append(
            "Across methods, the dominant source of performance gain is the controller design (learned policy), "
            "while estimation contributes a smaller and context-dependent effect."
        )
    else:
        lines.append("The report is incomplete because one or more required method results are missing.")
    lines.append("")

    lines.append("---")
    lines.append("")
    lines.append("*Report generated automatically by `experiments/comparison/report_all_methods.py`*")
    lines.append("")

    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate advisor-ready four-method comparison report"
    )
    parser.add_argument(
        "--output", type=str, 
        default=str(OUTPUT_DIR / "comparison_summary_four_methods.md"),
        help="Output markdown file path"
    )
    args = parser.parse_args()
    
    print("=" * 70)
    print("GENERATING FOUR-METHOD COMPARISON REPORT")
    print("=" * 70)
    
    # Load data
    print("\nLoading data...")
    data = load_all_data()
    
    # Count available datasets
    n_available = sum(1 for v in data.values() if v is not None)
    print(f"Loaded {n_available} datasets")
    
    # Generate report
    print("\nGenerating report...")
    report = generate_report(data)
    
    # Save report
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")
    print("=" * 70)
    
    # Print preview
    print("\n--- REPORT PREVIEW (first 50 lines) ---\n")
    preview_lines = report.split("\n")[:50]
    print("\n".join(preview_lines))
    if len(report.split("\n")) > 50:
        print("\n... (truncated)")


if __name__ == "__main__":
    main()
