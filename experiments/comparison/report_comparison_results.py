"""
Generate advisor-ready comparison report.

Reads baseline and RL evaluation results and generates a markdown summary
with interpretable conclusions suitable for research updates.

Outputs:
    results/comparison/comparison_summary.md

Usage:
    python experiments/comparison/report_comparison_results.py
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
RL_DIR = PROJECT_ROOT / "results" / "rl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


def load_data() -> dict:
    """Load all available result files."""
    data = {}
    
    # Overall evaluation
    baseline_eval = BASELINE_DIR / "1k_episodes" / "baseline_results.csv"
    rl_eval = RL_DIR / "rl_results.csv"
    
    if baseline_eval.exists():
        data["baseline_eval"] = pd.read_csv(baseline_eval)
    if rl_eval.exists():
        data["rl_eval"] = pd.read_csv(rl_eval)
    
    # Sweep results
    for sweep in ["defender_speed", "enemy_speed", "detection_radius", "speed_grid"]:
        baseline_path = BASELINE_DIR / f"sweep_{sweep}.csv"
        rl_path = RL_DIR / f"sweep_{sweep}.csv"
        
        if baseline_path.exists():
            data[f"baseline_{sweep}"] = pd.read_csv(baseline_path)
        if rl_path.exists():
            data[f"rl_{sweep}"] = pd.read_csv(rl_path)
    
    return data


def compute_eval_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics from episode results."""
    n = len(df)
    n_success = df["success"].sum()
    
    success_rate = n_success / n
    success_se = np.sqrt(success_rate * (1 - success_rate) / n)
    
    # 95% CI
    ci_low = max(0, success_rate - 1.96 * success_se)
    ci_high = min(1, success_rate + 1.96 * success_se)
    
    return {
        "n_episodes": n,
        "success_rate": success_rate,
        "success_se": success_se,
        "ci_95_low": ci_low,
        "ci_95_high": ci_high,
        "fail_rate": len(df[df["outcome"].isin(["soldier_caught", "unsafe_intercept"])]) / n,
        "mean_ep_len": df["episode_length"].mean(),
    }


def compare_sweep(baseline_df: pd.DataFrame, rl_df: pd.DataFrame, x_col: str) -> dict:
    """Compare sweep results point-by-point."""
    # Merge on x_col
    merged = pd.merge(
        baseline_df[[x_col, "success_rate", "success_se"]],
        rl_df[[x_col, "success_rate", "success_se"]],
        on=x_col,
        suffixes=("_baseline", "_rl"),
        how="inner"
    )
    
    if len(merged) == 0:
        return None
    
    # Compute differences
    merged["diff"] = merged["success_rate_rl"] - merged["success_rate_baseline"]
    merged["diff_pct"] = merged["diff"] * 100
    
    # Find where RL is best/worst
    best_idx = merged["diff"].idxmax()
    worst_idx = merged["diff"].idxmin()
    
    return {
        "n_points": len(merged),
        "mean_diff": merged["diff"].mean(),
        "best_point": {
            "x": merged.loc[best_idx, x_col],
            "diff": merged.loc[best_idx, "diff"],
            "baseline": merged.loc[best_idx, "success_rate_baseline"],
            "rl": merged.loc[best_idx, "success_rate_rl"],
        },
        "worst_point": {
            "x": merged.loc[worst_idx, x_col],
            "diff": merged.loc[worst_idx, "diff"],
            "baseline": merged.loc[worst_idx, "success_rate_baseline"],
            "rl": merged.loc[worst_idx, "success_rate_rl"],
        },
        "merged": merged,
    }


def generate_report(data: dict) -> str:
    """Generate markdown report."""
    lines = []
    
    # Header
    lines.append("# Baseline vs RL Comparison Report")
    lines.append("")
    lines.append(f"**Generated:** {datetime.now().strftime('%Y-%m-%d %H:%M')}")
    lines.append("")
    
    # Executive summary
    lines.append("## Executive Summary")
    lines.append("")
    
    # Overall comparison
    if "baseline_eval" in data and "rl_eval" in data:
        baseline_stats = compute_eval_stats(data["baseline_eval"])
        rl_stats = compute_eval_stats(data["rl_eval"])
        
        diff = rl_stats["success_rate"] - baseline_stats["success_rate"]
        diff_pct = diff * 100
        
        # Determine if difference is statistically significant
        # Pooled SE for two-sample comparison
        pooled_se = np.sqrt(baseline_stats["success_se"]**2 + rl_stats["success_se"]**2)
        z_score = diff / pooled_se if pooled_se > 0 else 0
        significant = abs(z_score) > 1.96
        
        if diff > 0:
            direction = "outperforms"
            symbol = "↑"
        elif diff < 0:
            direction = "underperforms"
            symbol = "↓"
        else:
            direction = "matches"
            symbol = "="
        
        sig_str = "statistically significant" if significant else "not statistically significant"
        
        lines.append(f"**Overall Finding:** The PPO (RL) policy {direction} the greedy baseline "
                    f"by {abs(diff_pct):.1f} percentage points ({symbol}). "
                    f"This difference is {sig_str} at α=0.05 (z={z_score:.2f}).")
        lines.append("")
        
        lines.append("| Metric | Greedy Baseline | PPO (RL) | Difference |")
        lines.append("|--------|-----------------|----------|------------|")
        lines.append(f"| Success Rate | {baseline_stats['success_rate']*100:.1f}% "
                    f"(95% CI: [{baseline_stats['ci_95_low']*100:.1f}%, {baseline_stats['ci_95_high']*100:.1f}%]) | "
                    f"{rl_stats['success_rate']*100:.1f}% "
                    f"(95% CI: [{rl_stats['ci_95_low']*100:.1f}%, {rl_stats['ci_95_high']*100:.1f}%]) | "
                    f"{diff_pct:+.1f}% |")
        lines.append(f"| Failure Rate | {baseline_stats['fail_rate']*100:.1f}% | "
                    f"{rl_stats['fail_rate']*100:.1f}% | "
                    f"{(rl_stats['fail_rate'] - baseline_stats['fail_rate'])*100:+.1f}% |")
        lines.append(f"| Mean Episode Length | {baseline_stats['mean_ep_len']:.1f} | "
                    f"{rl_stats['mean_ep_len']:.1f} | "
                    f"{rl_stats['mean_ep_len'] - baseline_stats['mean_ep_len']:+.1f} |")
        lines.append(f"| Episodes Evaluated | {baseline_stats['n_episodes']} | "
                    f"{rl_stats['n_episodes']} | - |")
        lines.append("")
    else:
        lines.append("*Overall evaluation data not available for both policies.*")
        lines.append("")
    
    # Defender speed analysis
    lines.append("## Analysis: Defender Speed Sensitivity")
    lines.append("")
    
    if "baseline_defender_speed" in data and "rl_defender_speed" in data:
        comparison = compare_sweep(
            data["baseline_defender_speed"],
            data["rl_defender_speed"],
            "defender_speed"
        )
        
        if comparison:
            mean_diff_pct = comparison["mean_diff"] * 100
            best = comparison["best_point"]
            worst = comparison["worst_point"]
            
            lines.append(f"Across the defender speed range, RL achieves an average "
                        f"{mean_diff_pct:+.1f}% difference in success rate.")
            lines.append("")
            lines.append(f"- **Largest RL gain:** At v_d={best['x']:.0f}, RL achieves "
                        f"{best['rl']*100:.1f}% vs baseline {best['baseline']*100:.1f}% "
                        f"(+{best['diff']*100:.1f}%)")
            lines.append(f"- **Smallest RL gain:** At v_d={worst['x']:.0f}, RL achieves "
                        f"{worst['rl']*100:.1f}% vs baseline {worst['baseline']*100:.1f}% "
                        f"({worst['diff']*100:+.1f}%)")
            lines.append("")
        else:
            lines.append("*Insufficient overlapping data points for comparison.*")
            lines.append("")
    elif "baseline_defender_speed" in data:
        lines.append("*RL defender speed sweep data not available.*")
        lines.append("")
    else:
        lines.append("*Defender speed sweep data not available.*")
        lines.append("")
    
    # Enemy speed analysis (robustness)
    lines.append("## Analysis: Robustness to Faster Enemies")
    lines.append("")
    
    if "baseline_enemy_speed" in data and "rl_enemy_speed" in data:
        comparison = compare_sweep(
            data["baseline_enemy_speed"],
            data["rl_enemy_speed"],
            "enemy_speed"
        )
        
        if comparison:
            merged = comparison["merged"]
            
            # Check if RL degrades less at higher enemy speeds
            high_speed = merged[merged["enemy_speed"] >= 14]
            low_speed = merged[merged["enemy_speed"] < 14]
            
            if len(high_speed) > 0 and len(low_speed) > 0:
                low_diff = low_speed["diff"].mean() * 100
                high_diff = high_speed["diff"].mean() * 100
                
                if high_diff > low_diff:
                    robustness = "**more robust**"
                    interpretation = "RL's advantage grows as enemies become faster"
                elif high_diff < low_diff:
                    robustness = "**less robust**"
                    interpretation = "baseline maintains better performance at high enemy speeds"
                else:
                    robustness = "**similarly robust**"
                    interpretation = "both policies degrade similarly"
                
                lines.append(f"The RL policy is {robustness} to faster enemies than baseline.")
                lines.append(f"At low enemy speeds (v_e < 14): RL advantage = {low_diff:+.1f}%")
                lines.append(f"At high enemy speeds (v_e ≥ 14): RL advantage = {high_diff:+.1f}%")
                lines.append(f"*Interpretation:* {interpretation}.")
                lines.append("")
            else:
                lines.append("*Insufficient data across enemy speed ranges.*")
                lines.append("")
        else:
            lines.append("*Insufficient overlapping data points for comparison.*")
            lines.append("")
    elif "baseline_enemy_speed" in data:
        lines.append("*RL enemy speed sweep data not available.*")
        lines.append("")
    else:
        lines.append("*Enemy speed sweep data not available.*")
        lines.append("")
    
    # Detection radius analysis
    lines.append("## Analysis: Detection Radius Sensitivity")
    lines.append("")
    
    if "baseline_detection_radius" in data and "rl_detection_radius" in data:
        comparison = compare_sweep(
            data["baseline_detection_radius"],
            data["rl_detection_radius"],
            "detection_radius"
        )
        
        if comparison:
            merged = comparison["merged"]
            
            # Check if RL benefits more from larger detection radius
            large_radius = merged[merged["detection_radius"] >= 12]
            small_radius = merged[merged["detection_radius"] < 12]
            
            if len(large_radius) > 0 and len(small_radius) > 0:
                small_diff = small_radius["diff"].mean() * 100
                large_diff = large_radius["diff"].mean() * 100
                
                if large_diff > small_diff:
                    benefit = "benefits more"
                    interpretation = "RL leverages additional sensing range more effectively"
                elif large_diff < small_diff:
                    benefit = "benefits less"
                    interpretation = "baseline captures more value from larger detection radius"
                else:
                    benefit = "benefits similarly"
                    interpretation = "both policies scale similarly with detection range"
                
                lines.append(f"The RL policy {benefit} from larger detection radius compared to baseline.")
                lines.append(f"At small radius (r < 12): RL advantage = {small_diff:+.1f}%")
                lines.append(f"At large radius (r ≥ 12): RL advantage = {large_diff:+.1f}%")
                lines.append(f"*Interpretation:* {interpretation}.")
                lines.append("")
            else:
                lines.append("*Insufficient data across detection radius ranges.*")
                lines.append("")
        else:
            lines.append("*Insufficient overlapping data points for comparison.*")
            lines.append("")
    elif "baseline_detection_radius" in data:
        lines.append("*RL detection radius sweep data not available.*")
        lines.append("")
    else:
        lines.append("*Detection radius sweep data not available.*")
        lines.append("")
    
    # Speed grid analysis
    lines.append("## Analysis: 2D Speed Configuration Space")
    lines.append("")
    
    if "baseline_speed_grid" in data and "rl_speed_grid" in data:
        baseline = data["baseline_speed_grid"]
        rl = data["rl_speed_grid"]
        
        # Merge grids
        merged = pd.merge(
            baseline[["defender_speed", "enemy_speed", "success_rate"]],
            rl[["defender_speed", "enemy_speed", "success_rate"]],
            on=["defender_speed", "enemy_speed"],
            suffixes=("_baseline", "_rl"),
            how="inner"
        )
        
        if len(merged) > 0:
            merged["diff"] = merged["success_rate_rl"] - merged["success_rate_baseline"]
            
            # Count wins/losses
            rl_wins = (merged["diff"] > 0.01).sum()
            baseline_wins = (merged["diff"] < -0.01).sum()
            ties = len(merged) - rl_wins - baseline_wins
            
            # Best/worst configurations
            best_idx = merged["diff"].idxmax()
            worst_idx = merged["diff"].idxmin()
            best = merged.loc[best_idx]
            worst = merged.loc[worst_idx]
            
            lines.append(f"Across {len(merged)} speed configurations:")
            lines.append(f"- RL outperforms baseline: **{rl_wins}** configurations")
            lines.append(f"- Baseline outperforms RL: **{baseline_wins}** configurations")
            lines.append(f"- Roughly equal (±1%): **{ties}** configurations")
            lines.append("")
            lines.append(f"**Best RL configuration:** v_d={best['defender_speed']:.0f}, "
                        f"v_e={best['enemy_speed']:.0f} "
                        f"(RL: {best['success_rate_rl']*100:.0f}%, "
                        f"Baseline: {best['success_rate_baseline']*100:.0f}%, "
                        f"Δ={best['diff']*100:+.0f}%)")
            lines.append(f"**Worst RL configuration:** v_d={worst['defender_speed']:.0f}, "
                        f"v_e={worst['enemy_speed']:.0f} "
                        f"(RL: {worst['success_rate_rl']*100:.0f}%, "
                        f"Baseline: {worst['success_rate_baseline']*100:.0f}%, "
                        f"Δ={worst['diff']*100:+.0f}%)")
            lines.append("")
        else:
            lines.append("*No overlapping grid points for comparison.*")
            lines.append("")
    elif "baseline_speed_grid" in data:
        lines.append("*RL speed grid data not available.*")
        lines.append("")
    else:
        lines.append("*Speed grid data not available.*")
        lines.append("")
    
    # Key takeaways
    lines.append("## Key Takeaways")
    lines.append("")
    
    takeaways = []
    
    if "baseline_eval" in data and "rl_eval" in data:
        baseline_stats = compute_eval_stats(data["baseline_eval"])
        rl_stats = compute_eval_stats(data["rl_eval"])
        diff = rl_stats["success_rate"] - baseline_stats["success_rate"]
        
        if diff > 0.05:
            takeaways.append("✅ RL achieves meaningfully higher success rate than greedy baseline")
        elif diff > 0:
            takeaways.append("⚖️ RL shows marginal improvement over greedy baseline")
        elif diff > -0.05:
            takeaways.append("⚖️ RL performs comparably to greedy baseline")
        else:
            takeaways.append("⚠️ RL underperforms greedy baseline - investigate reward shaping")
    
    if "rl_defender_speed" not in data:
        takeaways.append("📝 Run RL sweeps to enable full comparison")
    
    if len(takeaways) == 0:
        takeaways.append("📝 More data needed for conclusions")
    
    for t in takeaways:
        lines.append(f"- {t}")
    
    lines.append("")
    
    # Next steps
    lines.append("## Recommended Next Steps")
    lines.append("")
    
    next_steps = []
    
    if "rl_eval" in data:
        rl_stats = compute_eval_stats(data["rl_eval"])
        if rl_stats["n_episodes"] < 1000:
            next_steps.append(f"Run full 1000-episode RL evaluation (currently {rl_stats['n_episodes']} episodes)")
    
    if "rl_defender_speed" not in data:
        next_steps.append("Run RL defender speed sweep for full comparison")
    if "rl_enemy_speed" not in data:
        next_steps.append("Run RL enemy speed sweep for robustness analysis")
    if "rl_detection_radius" not in data:
        next_steps.append("Run RL detection radius sweep")
    if "rl_speed_grid" not in data:
        next_steps.append("Run RL 2D speed grid sweep for configuration space analysis")
    
    if len(next_steps) == 0:
        next_steps.append("All comparison data available - ready for publication")
    
    for i, step in enumerate(next_steps, 1):
        lines.append(f"{i}. {step}")
    
    lines.append("")
    
    # Data sources
    lines.append("---")
    lines.append("")
    lines.append("*Data sources:*")
    for key in sorted(data.keys()):
        df = data[key]
        lines.append(f"- `{key}`: {len(df)} rows")
    
    return "\n".join(lines)


def main():
    parser = argparse.ArgumentParser(
        description="Generate advisor-ready comparison report"
    )
    parser.add_argument(
        "--output", type=str, default=None,
        help="Output path (default: results/comparison/comparison_summary.md)"
    )
    
    args = parser.parse_args()
    
    # Load data
    print("Loading data...")
    data = load_data()
    print(f"Loaded {len(data)} datasets")
    
    # Generate report
    print("Generating report...")
    report = generate_report(data)
    
    # Save
    output_path = Path(args.output) if args.output else OUTPUT_DIR / "comparison_summary.md"
    os.makedirs(output_path.parent, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        f.write(report)
    
    print(f"\nReport saved to: {output_path}")
    print("\n" + "=" * 60)
    print(report)


if __name__ == "__main__":
    main()
