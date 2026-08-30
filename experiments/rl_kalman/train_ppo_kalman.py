"""
PPO Training Script for UAV Defender -- Kalman track.

This is a thin CLI wrapper around experiments/training_runner.run_training(),
the single shared implementation used by BOTH tracks (Direct and Kalman) and
by experiments/train_ppo_comparison.py (the multi-seed orchestrator). All PPO
algorithm hyperparameters are fixed in experiments/ppo_hyperparams.py and are
NOT exposed here as tunable CLI flags -- this script exists for single-run
ad-hoc training, not hyperparameter search. Kalman filter parameters
(process_var, measurement_var, lead_time) come from EnvConfig defaults,
identical to the Direct track except use_kalman_tracking.

Usage:
    python experiments/rl_kalman/train_ppo_kalman.py
    python experiments/rl_kalman/train_ppo_kalman.py --seed 43 --total-timesteps 50000
    python experiments/rl_kalman/train_ppo_kalman.py --output-dir results/training/kalman/seed_42
    python experiments/rl_kalman/train_ppo_kalman.py --overwrite

For the full multi-seed training campaign (3 seeds x 2 tracks), use
experiments/train_ppo_comparison.py instead.

Output (default): results/training/kalman/seed_<seed>/{best_model.zip,
    final_model.zip, checkpoints/, training_logs/, monitor/,
    training_manifest.json}
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ppo_hyperparams import TOTAL_TIMESTEPS
from experiments.training_protocol import VALIDATION_SEED_OFFSET, VALIDATION_EPISODES, TRAINING_SEEDS, run_output_dir
from experiments.training_runner import run_training

DEFAULT_SEED = TRAINING_SEEDS[0]


def train_ppo_kalman(
    total_timesteps: int = TOTAL_TIMESTEPS,
    seed: int = DEFAULT_SEED,
    output_dir: str | Path | None = None,
    validation_seed_offset: int = VALIDATION_SEED_OFFSET,
    validation_episodes: int = VALIDATION_EPISODES,
    overwrite: bool = False,
    verbose: int = 1,
    progress_bar: bool = True,
) -> dict:
    """
    Train one Kalman-track PPO run. Thin wrapper around
    experiments.training_runner.run_training(track="kalman", ...).

    Returns the training manifest dict (see training_runner.run_training).
    """
    if output_dir is None:
        output_dir = run_output_dir(PROJECT_ROOT / "results" / "training", "kalman", seed)
    return run_training(
        track="kalman",
        seed=seed,
        output_dir=Path(output_dir),
        total_timesteps=total_timesteps,
        validation_seed_offset=validation_seed_offset,
        validation_episodes=validation_episodes,
        overwrite=overwrite,
        verbose=verbose,
        progress_bar=progress_bar,
    )


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Train one Kalman-track PPO run on UAV Defender"
    )
    parser.add_argument(
        "--total-timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps (default: {TOTAL_TIMESTEPS:,})"
    )
    parser.add_argument(
        "--seed", type=int, default=DEFAULT_SEED,
        help=f"Training seed (default: {DEFAULT_SEED})"
    )
    parser.add_argument(
        "--output-dir", type=str, default=None,
        help="Output directory (default: results/training/kalman/seed_<seed>/)"
    )
    parser.add_argument(
        "--validation-seed-offset", type=int, default=VALIDATION_SEED_OFFSET,
        help=f"First validation seed (default: {VALIDATION_SEED_OFFSET})"
    )
    parser.add_argument(
        "--validation-episodes", type=int, default=VALIDATION_EPISODES,
        help=f"Validation episodes per evaluation (default: {VALIDATION_EPISODES})"
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite a completed run instead of skipping it"
    )
    parser.add_argument(
        "--verbose", type=int, default=1,
        help="Verbosity level (0=none, 1=info, 2=debug)"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    manifest = train_ppo_kalman(
        total_timesteps=args.total_timesteps,
        seed=args.seed,
        output_dir=args.output_dir,
        validation_seed_offset=args.validation_seed_offset,
        validation_episodes=args.validation_episodes,
        overwrite=args.overwrite,
        verbose=args.verbose,
    )
    print(f"\nbest_validation_success_rate={manifest.get('best_validation_success_rate')}")
    return manifest


if __name__ == "__main__":
    main()

