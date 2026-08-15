"""
Multi-seed PPO training orchestrator (Direct + Kalman tracks).

Trains the default six runs (Direct seeds 42/43/44, Kalman seeds 42/43/44)
for TOTAL_TIMESTEPS each, using the shared experiments/training_runner.py
implementation so both tracks share identical PPO hyperparameters,
callbacks, and model-selection criterion -- see
experiments/ppo_hyperparams.py and experiments/training_protocol.py.

Usage:
    python experiments/train_ppo_comparison.py
    python experiments/train_ppo_comparison.py --seeds 42
    python experiments/train_ppo_comparison.py --tracks direct
    python experiments/train_ppo_comparison.py --tracks kalman
    python experiments/train_ppo_comparison.py --timesteps 50000
    python experiments/train_ppo_comparison.py --overwrite

    # Separate 400k replication campaign (own output root, never touches
    # the original 200k campaign under results/training/):
    python experiments/train_ppo_comparison.py \
        --timesteps 400000 \
        --output-root results/training_400k \
        --campaign 400k_replication

Output layout (default, 200k campaign):
    results/training/direct/seed_<N>/{best_model.zip, final_model.zip,
        checkpoints/, training_logs/, monitor/, training_manifest.json}
    results/training/kalman/seed_<N>/{...}

Existing completed runs (same git commit, track, seed, timesteps, and PPO
hyperparameters) are SKIPPED unless --overwrite is passed -- see
experiments.training_runner.is_run_complete().
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.ppo_hyperparams import TOTAL_TIMESTEPS
from experiments.training_protocol import TRAINING_SEEDS, VALIDATION_SEED_OFFSET, VALIDATION_EPISODES, run_output_dir
from experiments.training_runner import run_training

ALL_TRACKS = ("direct", "kalman")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Train the multi-seed PPO comparison (Direct + Kalman)")
    parser.add_argument(
        "--seeds", type=int, nargs="+", default=list(TRAINING_SEEDS),
        help=f"Training seeds to run (default: {list(TRAINING_SEEDS)})",
    )
    parser.add_argument(
        "--tracks", type=str, nargs="+", default=list(ALL_TRACKS), choices=list(ALL_TRACKS),
        help=f"Tracks to run (default: {list(ALL_TRACKS)})",
    )
    parser.add_argument(
        "--timesteps", type=int, default=TOTAL_TIMESTEPS,
        help=f"Total training timesteps per run (default: {TOTAL_TIMESTEPS:,})",
    )
    parser.add_argument(
        "--validation-seed-offset", type=int, default=VALIDATION_SEED_OFFSET,
        help=f"First validation seed for best-model selection (default: {VALIDATION_SEED_OFFSET})",
    )
    parser.add_argument(
        "--validation-episodes", type=int, default=VALIDATION_EPISODES,
        help=f"Number of validation episodes per evaluation (default: {VALIDATION_EPISODES})",
    )
    parser.add_argument(
        "--output-root", type=str, default=str(PROJECT_ROOT / "results" / "training"),
        help="Campaign output root, containing <track>/seed_<N>/ (default: <repo>/results/training)",
    )
    parser.add_argument(
        "--campaign", type=str, default="200k_initial",
        help="Free-text campaign label recorded in each run's manifest (default: 200k_initial)",
    )
    parser.add_argument(
        "--overwrite", action="store_true",
        help="Overwrite a completed run instead of skipping it (protects expensive runs by default)",
    )
    parser.add_argument(
        "--progress-bar", action="store_true",
        help="Show SB3's tqdm progress bar during training",
    )
    return parser.parse_args()


def main() -> int:
    args = parse_args()
    output_root = Path(args.output_root)

    runs = [(track, seed) for track in args.tracks for seed in args.seeds]
    print("=" * 70)
    print("PPO MULTI-SEED TRAINING ORCHESTRATOR")
    print("=" * 70)
    print(f"Tracks:              {args.tracks}")
    print(f"Seeds:               {args.seeds}")
    print(f"Total timesteps/run: {args.timesteps:,}")
    print(f"Validation seeds:    {args.validation_seed_offset}..{args.validation_seed_offset + args.validation_episodes - 1}")
    print(f"Runs to process:     {len(runs)}")
    print(f"Overwrite existing:  {args.overwrite}")
    print("=" * 70)

    results = []
    for i, (track, seed) in enumerate(runs, start=1):
        output_dir = run_output_dir(output_root, track, seed)
        print(f"\n[{i}/{len(runs)}] track={track} seed={seed} -> {output_dir}")
        manifest = run_training(
            track=track,
            seed=seed,
            output_dir=output_dir,
            total_timesteps=args.timesteps,
            validation_seed_offset=args.validation_seed_offset,
            validation_episodes=args.validation_episodes,
            overwrite=args.overwrite,
            progress_bar=args.progress_bar,
            campaign=args.campaign,
        )
        results.append((track, seed, manifest))
        print(f"  best_validation_success_rate={manifest.get('best_validation_success_rate')} "
              f"best_checkpoint_timestep={manifest.get('best_checkpoint_timestep')}")

    print("\n" + "=" * 70)
    print("ALL RUNS PROCESSED")
    print("=" * 70)
    for track, seed, manifest in results:
        print(f"  {track:8s} seed={seed:<4d} status={manifest.get('status')} "
              f"best_success={manifest.get('best_validation_success_rate')}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
