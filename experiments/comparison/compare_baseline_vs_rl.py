"""
Baseline vs RL Comparison Script.

Loads baseline and RL evaluation results and generates side-by-side
comparison plots and summary tables.

Comparisons:
    1. Overall evaluation (1k episodes)
    2. Defender speed sweep
    3. Enemy speed sweep
    4. Detection radius sweep
    5. 2D speed grid (heatmaps)

Outputs:
    - results/comparison/baseline_vs_rl_summary.csv
    - results/comparison/compare_defender_speed.png
    - results/comparison/compare_enemy_speed.png
    - results/comparison/compare_detection_radius.png
    - results/comparison/compare_speed_grid_heatmap.png
    - results/comparison/compare_speed_grid_diff.png

Usage:
    python experiments/comparison/compare_baseline_vs_rl.py
    python experiments/comparison/compare_baseline_vs_rl.py --show-plots
"""

from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), '..', '..'))

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap


# Default paths
PROJECT_ROOT = Path(__file__).parent.parent.parent
BASELINE_DIR = PROJECT_ROOT / "results" / "baseline"
RL_DIR = PROJECT_ROOT / "results" / "rl"
OUTPUT_DIR = PROJECT_ROOT / "results" / "comparison"


def load_evaluation_results() -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load overall evaluation results for baseline and RL.
    
    Returns:
        Tuple of (baseline_df, rl_df). Either may be None if file doesn't exist.
    """
    baseline_path = BASELINE_DIR / "1k_episodes" / "baseline_results.csv"
    rl_path = RL_DIR / "rl_results.csv"
    
    baseline_df = None
    rl_df = None
    
    if baseline_path.exists():
        baseline_df = pd.read_csv(baseline_path)
        print(f"Loaded baseline: {len(baseline_df)} episodes")
    else:
        print(f"Warning: Baseline results not found at {baseline_path}")
    
    if rl_path.exists():
        rl_df = pd.read_csv(rl_path)
        print(f"Loaded RL: {len(rl_df)} episodes")
    else:
        print(f"Warning: RL results not found at {rl_path}")
    
    return baseline_df, rl_df


def compute_summary_stats(df: pd.DataFrame) -> dict:
    """Compute summary statistics from episode results."""
    if df is None or len(df) == 0:
        return None
    
    n = len(df)
    n_success = df["success"].sum()
    n_fail = len(df[df["outcome"].isin(["soldier_caught", "unsafe_intercept"])])
    n_timeout = len(df[df["outcome"] == "timeout"])
    
    success_rate = n_success / n
    fail_rate = n_fail / n
    timeout_rate = n_timeout / n
    
    # Standard error for success rate
    success_se = np.sqrt(success_rate * (1 - success_rate) / n)
    
    # Episode length
    mean_ep_len = df["episode_length"].mean()
    
    # Detection time (only for detected episodes)
    detected = df[df["detected"] == 1]
    valid_det = detected[detected["detection_time"] > 0]
    mean_det_time = valid_det["detection_time"].mean() if len(valid_det) > 0 else np.nan
    
    # Intercept time (only for successful episodes)
    success_df = df[df["success"] == 1]
    valid_int = success_df[success_df["intercept_time"] > 0]
    mean_int_time = valid_int["intercept_time"].mean() if len(valid_int) > 0 else np.nan
    
    return {
        "n_episodes": n,
        "success_rate": success_rate,
        "success_se": success_se,
        "fail_rate": fail_rate,
        "timeout_rate": timeout_rate,
        "mean_episode_length": mean_ep_len,
        "mean_detection_time": mean_det_time,
        "mean_intercept_time": mean_int_time,
    }


def create_summary_table(
    baseline_df: pd.DataFrame | None,
    rl_df: pd.DataFrame | None,
    output_path: Path,
) -> pd.DataFrame:
    """
    Create summary comparison table.
    
    Args:
        baseline_df: Baseline episode results.
        rl_df: RL episode results.
        output_path: Path to save CSV.
    
    Returns:
        Summary DataFrame.
    """
    rows = []
    
    if baseline_df is not None:
        stats = compute_summary_stats(baseline_df)
        if stats:
            rows.append({
                "policy": "greedy_baseline",
                **stats,
            })
    
    if rl_df is not None:
        stats = compute_summary_stats(rl_df)
        if stats:
            rows.append({
                "policy": "ppo_rl",
                **stats,
            })
    
    if not rows:
        print("No data available for summary table")
        return None
    
    df = pd.DataFrame(rows)
    
    # Reorder columns
    column_order = [
        "policy",
        "n_episodes",
        "success_rate",
        "success_se",
        "fail_rate",
        "timeout_rate",
        "mean_episode_length",
        "mean_detection_time",
        "mean_intercept_time",
    ]
    df = df[[c for c in column_order if c in df.columns]]
    
    # Save
    os.makedirs(output_path.parent, exist_ok=True)
    df.to_csv(output_path, index=False)
    print(f"\nSummary table saved to: {output_path}")
    
    # Print formatted table
    print("\n" + "=" * 80)
    print("BASELINE VS RL SUMMARY")
    print("=" * 80)
    print(f"{'Policy':<20} {'Success':>12} {'Failure':>12} {'Timeout':>12} {'Ep Len':>10}")
    print("-" * 80)
    for _, row in df.iterrows():
        se_str = f"±{row['success_se']*100:.1f}%" if 'success_se' in row else ""
        print(f"{row['policy']:<20} "
              f"{row['success_rate']*100:>7.1f}%{se_str:>4} "
              f"{row['fail_rate']*100:>11.1f}% "
              f"{row['timeout_rate']*100:>11.1f}% "
              f"{row['mean_episode_length']:>10.1f}")
    print("=" * 80)
    
    return df


def load_sweep_results(
    sweep_name: str,
) -> tuple[pd.DataFrame | None, pd.DataFrame | None]:
    """
    Load sweep results for baseline and RL.
    
    Args:
        sweep_name: "defender_speed", "enemy_speed", "detection_radius", or "speed_grid"
    
    Returns:
        Tuple of (baseline_df, rl_df).
    """
    baseline_path = BASELINE_DIR / f"sweep_{sweep_name}.csv"
    rl_path = RL_DIR / f"sweep_{sweep_name}.csv"
    
    baseline_df = pd.read_csv(baseline_path) if baseline_path.exists() else None
    rl_df = pd.read_csv(rl_path) if rl_path.exists() else None
    
    if baseline_df is not None:
        print(f"Loaded baseline sweep_{sweep_name}: {len(baseline_df)} rows")
    if rl_df is not None:
        print(f"Loaded RL sweep_{sweep_name}: {len(rl_df)} rows")
    
    return baseline_df, rl_df


def plot_1d_comparison(
    baseline_df: pd.DataFrame | None,
    rl_df: pd.DataFrame | None,
    x_col: str,
    x_label: str,
    title: str,
    output_path: Path,
    show: bool = False,
    add_reference_line: float | None = None,
    reference_label: str = "",
) -> None:
    """
    Create comparison plot for 1D sweep.
    
    Args:
        baseline_df: Baseline sweep results.
        rl_df: RL sweep results.
        x_col: Column name for x-axis.
        x_label: Label for x-axis.
        title: Plot title.
        output_path: Path to save plot.
        show: If True, display plot.
        add_reference_line: Optional vertical reference line.
        reference_label: Label for reference line.
    """
    if baseline_df is None and rl_df is None:
        print(f"No data for {title}")
        return
    
    fig, ax = plt.subplots(figsize=(10, 7))
    
    # Determine SE column name (varies between sweeps)
    se_col = "success_se" if "success_se" in (baseline_df.columns if baseline_df is not None else []) else "standard_error"
    
    # Plot baseline
    if baseline_df is not None:
        if se_col in baseline_df.columns:
            ax.errorbar(
                baseline_df[x_col],
                baseline_df["success_rate"] * 100,
                yerr=baseline_df[se_col] * 100,
                fmt='o-',
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8,
                color='#2E86AB',
                ecolor='#1A5276',
                label='Greedy Baseline'
            )
        else:
            ax.plot(
                baseline_df[x_col],
                baseline_df["success_rate"] * 100,
                'o-',
                linewidth=2,
                markersize=8,
                color='#2E86AB',
                label='Greedy Baseline'
            )
    
    # Plot RL
    if rl_df is not None:
        if se_col in rl_df.columns:
            ax.errorbar(
                rl_df[x_col],
                rl_df["success_rate"] * 100,
                yerr=rl_df[se_col] * 100,
                fmt='s--',
                capsize=5,
                capthick=2,
                linewidth=2,
                markersize=8,
                color='#9B59B6',
                ecolor='#6C3483',
                label='PPO (RL)'
            )
        else:
            ax.plot(
                rl_df[x_col],
                rl_df["success_rate"] * 100,
                's--',
                linewidth=2,
                markersize=8,
                color='#9B59B6',
                label='PPO (RL)'
            )
    
    # Reference line
    if add_reference_line is not None:
        ax.axvline(x=add_reference_line, color='red', linestyle=':', alpha=0.7,
                   linewidth=2, label=reference_label)
    
    # Formatting
    ax.set_xlabel(x_label, fontsize=12)
    ax.set_ylabel('Success Rate (%)', fontsize=12)
    ax.set_title(title, fontsize=14)
    ax.set_ylim(0, 100)
    ax.grid(True, alpha=0.3)
    ax.legend(loc='best', fontsize=11)
    
    plt.tight_layout()
    
    # Save
    os.makedirs(output_path.parent, exist_ok=True)
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()


def plot_speed_grid_comparison(
    baseline_df: pd.DataFrame | None,
    rl_df: pd.DataFrame | None,
    output_dir: Path,
    show: bool = False,
) -> None:
    """
    Create comparison plots for 2D speed grid sweep.
    
    Creates:
        1. Side-by-side heatmaps
        2. Difference heatmap (RL - Baseline)
    
    Args:
        baseline_df: Baseline grid results.
        rl_df: RL grid results.
        output_dir: Directory to save plots.
        show: If True, display plots.
    """
    if baseline_df is None and rl_df is None:
        print("No speed grid data available")
        return
    
    os.makedirs(output_dir, exist_ok=True)
    
    # Create custom colormap
    colors = ['#E94F37', '#F2C14E', '#44AF69']
    cmap = LinearSegmentedColormap.from_list('success', colors)
    
    # --- Side-by-side heatmaps ---
    fig, axes = plt.subplots(1, 2, figsize=(16, 7))
    
    for idx, (df, name, ax) in enumerate([
        (baseline_df, "Greedy Baseline", axes[0]),
        (rl_df, "PPO (RL)", axes[1]),
    ]):
        if df is None:
            ax.text(0.5, 0.5, f"No {name} data", ha='center', va='center', fontsize=14)
            ax.set_title(name)
            continue
        
        # Pivot data
        pivot = df.pivot(
            index="defender_speed",
            columns="enemy_speed",
            values="success_rate"
        ).sort_index(ascending=False)
        
        # Plot heatmap
        im = ax.imshow(
            pivot.values * 100,
            aspect='auto',
            cmap=cmap,
            vmin=0,
            vmax=100,
        )
        
        # Set tick labels
        ax.set_xticks(range(len(pivot.columns)))
        ax.set_xticklabels([f'{v:.0f}' for v in pivot.columns])
        ax.set_yticks(range(len(pivot.index)))
        ax.set_yticklabels([f'{v:.0f}' for v in pivot.index])
        
        # Add value annotations
        for i, v_d in enumerate(pivot.index):
            for j, v_e in enumerate(pivot.columns):
                value = pivot.loc[v_d, v_e] * 100
                text_color = 'white' if value < 40 or value > 70 else 'black'
                ax.text(j, i, f'{value:.0f}%',
                       ha='center', va='center',
                       fontsize=9, fontweight='bold',
                       color=text_color)
        
        ax.set_xlabel('Enemy Speed (v_e)', fontsize=11)
        ax.set_ylabel('Defender Speed (v_d)', fontsize=11)
        ax.set_title(f'{name}: Success Rate', fontsize=12)
    
    # Add colorbar
    cbar = fig.colorbar(im, ax=axes, label='Success Rate (%)', shrink=0.8)
    
    plt.suptitle('Baseline vs RL: Success Rate by Speed Configuration', fontsize=14, y=1.02)
    plt.tight_layout()
    
    output_path = output_dir / "compare_speed_grid_heatmap.png"
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"Plot saved to: {output_path}")
    
    if show:
        plt.show()
    else:
        plt.close()
    
    # --- Difference heatmap ---
    if baseline_df is not None and rl_df is not None:
        fig, ax = plt.subplots(figsize=(10, 8))
        
        # Pivot both
        pivot_baseline = baseline_df.pivot(
            index="defender_speed",
            columns="enemy_speed",
            values="success_rate"
        ).sort_index(ascending=False)
        
        pivot_rl = rl_df.pivot(
            index="defender_speed",
            columns="enemy_speed",
            values="success_rate"
        ).sort_index(ascending=False)
        
        # Compute difference (RL - Baseline)
        # Need to align indices
        common_defenders = sorted(set(pivot_baseline.index) & set(pivot_rl.index), reverse=True)
        common_enemies = sorted(set(pivot_baseline.columns) & set(pivot_rl.columns))
        
        if len(common_defenders) > 0 and len(common_enemies) > 0:
            diff = (pivot_rl.loc[common_defenders, common_enemies] - 
                   pivot_baseline.loc[common_defenders, common_enemies]) * 100
            
            # Diverging colormap for difference
            im = ax.imshow(
                diff.values,
                aspect='auto',
                cmap='RdBu',
                vmin=-30,
                vmax=30,
            )
            
            # Set tick labels
            ax.set_xticks(range(len(common_enemies)))
            ax.set_xticklabels([f'{v:.0f}' for v in common_enemies])
            ax.set_yticks(range(len(common_defenders)))
            ax.set_yticklabels([f'{v:.0f}' for v in common_defenders])
            
            # Add value annotations
            for i, v_d in enumerate(common_defenders):
                for j, v_e in enumerate(common_enemies):
                    value = diff.loc[v_d, v_e]
                    sign = "+" if value > 0 else ""
                    text_color = 'black' if abs(value) < 15 else 'white'
                    ax.text(j, i, f'{sign}{value:.0f}%',
                           ha='center', va='center',
                           fontsize=9, fontweight='bold',
                           color=text_color)
            
            ax.set_xlabel('Enemy Speed (v_e)', fontsize=12)
            ax.set_ylabel('Defender Speed (v_d)', fontsize=12)
            ax.set_title('RL - Baseline: Success Rate Difference\n(Blue = RL better, Red = Baseline better)', fontsize=12)
            
            cbar = plt.colorbar(im, ax=ax, label='Difference (%)')
            
            plt.tight_layout()
            
            output_path = output_dir / "compare_speed_grid_diff.png"
            plt.savefig(output_path, dpi=150, bbox_inches='tight')
            print(f"Plot saved to: {output_path}")
            
            if show:
                plt.show()
            else:
                plt.close()


def main():
    parser = argparse.ArgumentParser(
        description="Compare baseline vs RL evaluation results"
    )
    parser.add_argument(
        "--baseline-dir", type=str, default=None,
        help=f"Baseline results directory (default: {BASELINE_DIR})"
    )
    parser.add_argument(
        "--rl-dir", type=str, default=None,
        help=f"RL results directory (default: {RL_DIR})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help=f"Output directory (default: {OUTPUT_DIR})"
    )
    parser.add_argument(
        "--show-plots", action="store_true",
        help="Display plots interactively"
    )
    
    args = parser.parse_args()
    
    # Set paths
    baseline_dir = Path(args.baseline_dir) if args.baseline_dir else BASELINE_DIR
    rl_dir = Path(args.rl_dir) if args.rl_dir else RL_DIR
    output_dir = Path(args.output_dir) if args.output_dir else OUTPUT_DIR
    
    os.makedirs(output_dir, exist_ok=True)
    
    print("=" * 60)
    print("BASELINE VS RL COMPARISON")
    print("=" * 60)
    print(f"Baseline dir: {baseline_dir}")
    print(f"RL dir:       {rl_dir}")
    print(f"Output dir:   {output_dir}")
    print("-" * 60)
    
    # 1. Overall evaluation comparison
    print("\n>>> Loading overall evaluation results...")
    
    # Load evaluation results
    baseline_eval_path = baseline_dir / "1k_episodes" / "baseline_results.csv"
    rl_eval_path = rl_dir / "rl_results.csv"
    
    baseline_df = pd.read_csv(baseline_eval_path) if baseline_eval_path.exists() else None
    rl_df = pd.read_csv(rl_eval_path) if rl_eval_path.exists() else None
    
    if baseline_df is not None:
        print(f"Loaded baseline: {len(baseline_df)} episodes")
    else:
        print(f"Warning: Baseline results not found at {baseline_eval_path}")
    
    if rl_df is not None:
        print(f"Loaded RL: {len(rl_df)} episodes")
    else:
        print(f"Warning: RL results not found at {rl_eval_path}")
    
    summary_path = output_dir / "baseline_vs_rl_summary.csv"
    create_summary_table(baseline_df, rl_df, summary_path)
    
    # Helper to load sweep results
    def load_sweep(sweep_name):
        baseline_path = baseline_dir / f"sweep_{sweep_name}.csv"
        rl_path = rl_dir / f"sweep_{sweep_name}.csv"
        baseline = pd.read_csv(baseline_path) if baseline_path.exists() else None
        rl = pd.read_csv(rl_path) if rl_path.exists() else None
        if baseline is not None:
            print(f"Loaded baseline sweep_{sweep_name}: {len(baseline)} rows")
        if rl is not None:
            print(f"Loaded RL sweep_{sweep_name}: {len(rl)} rows")
        return baseline, rl
    
    # 2. Defender speed sweep
    print("\n>>> Comparing defender speed sweep...")
    baseline_sweep, rl_sweep = load_sweep("defender_speed")
    if baseline_sweep is not None or rl_sweep is not None:
        ref_speed = 12.0
        if baseline_sweep is not None and "enemy_speed" in baseline_sweep.columns:
            ref_speed = baseline_sweep["enemy_speed"].iloc[0]
        
        plot_1d_comparison(
            baseline_sweep, rl_sweep,
            x_col="defender_speed",
            x_label="Defender Speed (v_d)",
            title="Baseline vs RL: Success Rate vs Defender Speed",
            output_path=output_dir / "compare_defender_speed.png",
            show=args.show_plots,
            add_reference_line=ref_speed,
            reference_label=f"Enemy speed (v_e = {ref_speed})",
        )
    
    # 3. Enemy speed sweep
    print("\n>>> Comparing enemy speed sweep...")
    baseline_sweep, rl_sweep = load_sweep("enemy_speed")
    if baseline_sweep is not None or rl_sweep is not None:
        ref_speed = 18.0
        if baseline_sweep is not None and "defender_speed" in baseline_sweep.columns:
            ref_speed = baseline_sweep["defender_speed"].iloc[0]
        
        plot_1d_comparison(
            baseline_sweep, rl_sweep,
            x_col="enemy_speed",
            x_label="Enemy Speed (v_e)",
            title="Baseline vs RL: Success Rate vs Enemy Speed",
            output_path=output_dir / "compare_enemy_speed.png",
            show=args.show_plots,
            add_reference_line=ref_speed,
            reference_label=f"Defender speed (v_d = {ref_speed})",
        )
    
    # 4. Detection radius sweep
    print("\n>>> Comparing detection radius sweep...")
    baseline_sweep, rl_sweep = load_sweep("detection_radius")
    if baseline_sweep is not None or rl_sweep is not None:
        plot_1d_comparison(
            baseline_sweep, rl_sweep,
            x_col="detection_radius",
            x_label="Detection Radius",
            title="Baseline vs RL: Success Rate vs Detection Radius",
            output_path=output_dir / "compare_detection_radius.png",
            show=args.show_plots,
        )
    
    # 5. Speed grid
    print("\n>>> Comparing speed grid sweep...")
    baseline_sweep, rl_sweep = load_sweep("speed_grid")
    if baseline_sweep is not None or rl_sweep is not None:
        plot_speed_grid_comparison(
            baseline_sweep, rl_sweep,
            output_dir=output_dir,
            show=args.show_plots,
        )
    
    print("\n" + "=" * 60)
    print("COMPARISON COMPLETE")
    print("=" * 60)
    print(f"Results saved to: {output_dir}")
    print("=" * 60)


if __name__ == "__main__":
    main()
