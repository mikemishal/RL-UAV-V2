"""
Automated RL Training and Evaluation Pipeline.

This script automates the full RL experiment workflow:
    1. Train PPO for specified timesteps
    2. Run Monte Carlo evaluation (1000 episodes)
    3. Run all parameter sweeps
    4. Generate comparison plots and reports

The baseline results are NOT modified - only RL results are regenerated.

Usage:
    # Full pipeline with 100k timesteps (default)
    python experiments/rl/run_full_pipeline.py
    
    # Custom timesteps
    python experiments/rl/run_full_pipeline.py --timesteps 500000
    
    # Skip training (use existing model)
    python experiments/rl/run_full_pipeline.py --skip-training
    
    # Fewer evaluation episodes (quick test)
    python experiments/rl/run_full_pipeline.py --eval-episodes 100

Output:
    - results/rl/models/ppo_defender_final.zip
    - results/rl/rl_results.csv
    - results/rl/defender_speed_sweep_rl.csv
    - results/rl/enemy_speed_sweep_rl.csv
    - results/rl/detection_radius_sweep_rl.csv
    - results/rl/speed_grid_sweep_rl.csv
    - results/comparison/*.png (comparison plots)
    - results/comparison/comparison_summary.md (full report)
"""

from __future__ import annotations

import argparse
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

# Add project root to path for imports
PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.experiment_config import CONFIG, EVAL_CONFIG, SWEEP_CONFIG

# =============================================================================
# CONFIG
# =============================================================================
PYTHON = sys.executable

# Default settings from shared config
DEFAULT_TIMESTEPS = 100_000
DEFAULT_EVAL_EPISODES = EVAL_CONFIG["n_episodes"]
DEFAULT_SWEEP_EPISODES = SWEEP_CONFIG["defender_speed"]["n_episodes"]

# Script paths
TRAIN_SCRIPT = PROJECT_ROOT / "experiments" / "rl" / "train_ppo.py"
EVAL_SCRIPT = PROJECT_ROOT / "experiments" / "rl" / "evaluate_rl.py"
SWEEP_DEFENDER_SPEED = PROJECT_ROOT / "experiments" / "rl" / "sweep_defender_speed_rl.py"
SWEEP_ENEMY_SPEED = PROJECT_ROOT / "experiments" / "rl" / "sweep_enemy_speed_rl.py"
SWEEP_DETECTION_RADIUS = PROJECT_ROOT / "experiments" / "rl" / "sweep_detection_radius_rl.py"
SWEEP_SPEED_GRID = PROJECT_ROOT / "experiments" / "rl" / "sweep_speed_grid_rl.py"
COMPARE_SCRIPT = PROJECT_ROOT / "experiments" / "comparison" / "compare_baseline_vs_rl.py"
REPORT_SCRIPT = PROJECT_ROOT / "experiments" / "comparison" / "report_comparison_results.py"

# Model path
MODEL_PATH = PROJECT_ROOT / "results" / "rl" / "models" / "ppo_defender_final.zip"


def log(msg: str) -> None:
    """Print timestamped log message."""
    timestamp = datetime.now().strftime("%H:%M:%S")
    print(f"[{timestamp}] {msg}")


def run_command(cmd: list[str], description: str) -> bool:
    """
    Run a command and return success status.
    
    Args:
        cmd: Command and arguments.
        description: Human-readable description for logging.
    
    Returns:
        True if command succeeded, False otherwise.
    """
    log(f"Starting: {description}")
    start_time = time.time()
    
    try:
        result = subprocess.run(
            cmd,
            cwd=PROJECT_ROOT,
            check=True,
            text=True,
        )
        elapsed = time.time() - start_time
        log(f"Completed: {description} ({elapsed:.1f}s)")
        return True
    except subprocess.CalledProcessError as e:
        elapsed = time.time() - start_time
        log(f"FAILED: {description} ({elapsed:.1f}s)")
        print(f"  Command: {' '.join(cmd)}")
        print(f"  Return code: {e.returncode}")
        return False
    except Exception as e:
        log(f"ERROR: {description}")
        print(f"  Exception: {e}")
        return False


def run_pipeline(
    timesteps: int,
    eval_episodes: int,
    sweep_episodes: int,
    skip_training: bool,
    skip_sweeps: bool,
    skip_comparison: bool,
) -> dict:
    """
    Run the full RL pipeline.
    
    Args:
        timesteps: Total training timesteps.
        eval_episodes: Episodes for main evaluation.
        sweep_episodes: Episodes per sweep data point.
        skip_training: Skip training, use existing model.
        skip_sweeps: Skip parameter sweeps.
        skip_comparison: Skip comparison report generation.
    
    Returns:
        Dict with results and timings.
    """
    results = {
        "start_time": datetime.now().isoformat(),
        "timesteps": timesteps,
        "eval_episodes": eval_episodes,
        "sweep_episodes": sweep_episodes,
        "stages": {},
    }
    
    pipeline_start = time.time()
    
    print("=" * 60)
    print("UAV Defend - RL Training and Evaluation Pipeline")
    print("=" * 60)
    print(f"  Timesteps:       {timesteps:,}")
    print(f"  Eval episodes:   {eval_episodes:,}")
    print(f"  Sweep episodes:  {sweep_episodes:,}")
    print(f"  Skip training:   {skip_training}")
    print(f"  Skip sweeps:     {skip_sweeps}")
    print(f"  Skip comparison: {skip_comparison}")
    print("=" * 60)
    print()
    
    # =========================================================================
    # Stage 1: Training
    # =========================================================================
    if not skip_training:
        log("=" * 40)
        log("STAGE 1: PPO Training")
        log("=" * 40)
        
        cmd = [
            PYTHON, str(TRAIN_SCRIPT),
            "--total-timesteps", str(timesteps),
        ]
        success = run_command(cmd, f"Train PPO for {timesteps:,} timesteps")
        results["stages"]["training"] = success
        
        if not success:
            log("Training failed. Aborting pipeline.")
            return results
    else:
        log("Skipping training (using existing model)")
        if not MODEL_PATH.exists():
            log(f"ERROR: Model not found at {MODEL_PATH}")
            log("Remove --skip-training or train a model first.")
            results["stages"]["training"] = False
            return results
        results["stages"]["training"] = "skipped"
    
    print()
    
    # =========================================================================
    # Stage 2: Main Evaluation
    # =========================================================================
    log("=" * 40)
    log("STAGE 2: Monte Carlo Evaluation")
    log("=" * 40)
    
    cmd = [
        PYTHON, str(EVAL_SCRIPT),
        "--model", str(MODEL_PATH),
        "--n-episodes", str(eval_episodes),
    ]
    success = run_command(cmd, f"Evaluate PPO ({eval_episodes:,} episodes)")
    results["stages"]["evaluation"] = success
    
    print()
    
    # =========================================================================
    # Stage 3: Parameter Sweeps
    # =========================================================================
    if not skip_sweeps:
        log("=" * 40)
        log("STAGE 3: Parameter Sweeps")
        log("=" * 40)
        
        sweeps = [
            (SWEEP_DEFENDER_SPEED, "Defender speed sweep"),
            (SWEEP_ENEMY_SPEED, "Enemy speed sweep"),
            (SWEEP_DETECTION_RADIUS, "Detection radius sweep"),
            (SWEEP_SPEED_GRID, "2D speed grid sweep"),
        ]
        
        sweep_results = {}
        for script, name in sweeps:
            cmd = [
                PYTHON, str(script),
                "--model", str(MODEL_PATH),
                "--n-episodes", str(sweep_episodes),
            ]
            success = run_command(cmd, name)
            sweep_results[name] = success
        
        results["stages"]["sweeps"] = sweep_results
    else:
        log("Skipping parameter sweeps")
        results["stages"]["sweeps"] = "skipped"
    
    print()
    
    # =========================================================================
    # Stage 4: Comparison and Reporting
    # =========================================================================
    if not skip_comparison:
        log("=" * 40)
        log("STAGE 4: Comparison and Reporting")
        log("=" * 40)
        
        # Generate comparison plots
        cmd = [PYTHON, str(COMPARE_SCRIPT)]
        success = run_command(cmd, "Generate comparison plots")
        results["stages"]["comparison_plots"] = success
        
        # Generate markdown report
        cmd = [PYTHON, str(REPORT_SCRIPT)]
        success = run_command(cmd, "Generate comparison report")
        results["stages"]["comparison_report"] = success
    else:
        log("Skipping comparison reports")
        results["stages"]["comparison_plots"] = "skipped"
        results["stages"]["comparison_report"] = "skipped"
    
    # =========================================================================
    # Summary
    # =========================================================================
    pipeline_elapsed = time.time() - pipeline_start
    
    print()
    print("=" * 60)
    print("PIPELINE COMPLETE")
    print("=" * 60)
    print(f"  Total time: {pipeline_elapsed / 60:.1f} minutes")
    print()
    print("Results:")
    for stage, status in results["stages"].items():
        if isinstance(status, dict):
            print(f"  {stage}:")
            for sub, sub_status in status.items():
                icon = "✓" if sub_status else "✗"
                print(f"    {icon} {sub}")
        else:
            icon = "✓" if status == True else ("–" if status == "skipped" else "✗")
            print(f"  {icon} {stage}")
    print()
    print("Output files:")
    print(f"  - {MODEL_PATH}")
    print(f"  - {PROJECT_ROOT / 'results' / 'rl' / 'rl_results.csv'}")
    print(f"  - {PROJECT_ROOT / 'results' / 'comparison' / 'comparison_summary.md'}")
    print("=" * 60)
    
    results["end_time"] = datetime.now().isoformat()
    results["elapsed_minutes"] = pipeline_elapsed / 60
    
    return results


def main():
    parser = argparse.ArgumentParser(
        description="Run full RL training and evaluation pipeline.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Standard run with 100k timesteps
    python experiments/rl/run_full_pipeline.py
    
    # Longer training
    python experiments/rl/run_full_pipeline.py --timesteps 500000
    
    # Quick test (skip training, fewer episodes)
    python experiments/rl/run_full_pipeline.py --skip-training --eval-episodes 100
    
    # Re-run comparison only (after manual changes)
    python experiments/rl/run_full_pipeline.py --skip-training --skip-sweeps
        """,
    )
    
    parser.add_argument(
        "--timesteps", "-t",
        type=int,
        default=DEFAULT_TIMESTEPS,
        help=f"Total training timesteps (default: {DEFAULT_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--eval-episodes", "-e",
        type=int,
        default=DEFAULT_EVAL_EPISODES,
        help=f"Episodes for main evaluation (default: {DEFAULT_EVAL_EPISODES:,})",
    )
    parser.add_argument(
        "--sweep-episodes", "-s",
        type=int,
        default=DEFAULT_SWEEP_EPISODES,
        help=f"Episodes per sweep data point (default: {DEFAULT_SWEEP_EPISODES})",
    )
    parser.add_argument(
        "--skip-training",
        action="store_true",
        help="Skip training, use existing model",
    )
    parser.add_argument(
        "--skip-sweeps",
        action="store_true",
        help="Skip parameter sweeps",
    )
    parser.add_argument(
        "--skip-comparison",
        action="store_true",
        help="Skip comparison report generation",
    )
    
    args = parser.parse_args()
    
    run_pipeline(
        timesteps=args.timesteps,
        eval_episodes=args.eval_episodes,
        sweep_episodes=args.sweep_episodes,
        skip_training=args.skip_training,
        skip_sweeps=args.skip_sweeps,
        skip_comparison=args.skip_comparison,
    )


if __name__ == "__main__":
    main()
