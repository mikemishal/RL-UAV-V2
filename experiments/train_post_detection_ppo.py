"""
Post-detection PPO training entry point (revised canonical protocol).

Trains PPO/PPO-Kalman using PostDetectionTrainingEnv (PPO's episode begins at
the first detected observation; standby transitions/rewards are never seen
by the learner) with a deterministic per-episode seed sequence derived from
--training-seed (see experiments/post_detection_training_protocol.py).

Model selection uses the FULL canonical SoldierEnv (defender_standby_until_detection
=True, NOT the training wrapper) via SuccessRateEvalCallback, on the fixed
validation seed range 40000-40099, success rate as the sole criterion.

Usage:
    python experiments/train_post_detection_ppo.py --track direct --training-seed 45 \
        --total-timesteps 400000 --output-dir results/training_post_detection_400k

    # Small smoke test (development seed, NOT a protocol seed):
    python experiments/train_post_detection_ppo.py --track direct --training-seed 12345 \
        --total-timesteps 4096 --output-dir results/_smoke_test_post_detection --smoke-test
"""

from __future__ import annotations

import argparse
import dataclasses
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed
from stable_baselines3.common.callbacks import CheckpointCallback, CallbackList

from uav_defend.config import EnvConfig

from experiments import ppo_hyperparams as H
from experiments.post_detection_training_env import PostDetectionTrainingEnv
from experiments.post_detection_training_protocol import (
    TRAINING_SEEDS,
    VALIDATION_SEED_OFFSET,
    VALIDATION_EPISODES,
    CAMPAIGN_DIRNAME,
    CAMPAIGN_TIMESTEPS,
    MODEL_SELECTION_METRIC,
    MODEL_SELECTION_TIE_BREAK_RULE,
    POST_DETECTION_TIMESTEP_DEFINITION,
    build_env_config,
    assert_only_estimator_differs,
    assert_not_reserved_seed,
    run_output_dir,
)
from experiments.success_rate_eval_callback import SuccessRateEvalCallback


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


def _sb3_architecture_info(model: PPO) -> dict:
    policy = model.policy
    mlp_extractor = getattr(policy, "mlp_extractor", None)
    pi_layers, vf_layers, activation = [], [], None
    if mlp_extractor is not None:
        pi_layers = [str(m) for m in mlp_extractor.policy_net]
        vf_layers = [str(m) for m in mlp_extractor.value_net]
        for m in mlp_extractor.policy_net:
            cls_name = type(m).__name__
            if cls_name != "Linear":
                activation = cls_name
                break
    n_params = sum(p.numel() for p in policy.parameters())
    return {
        "policy_class": type(policy).__name__,
        "net_arch": getattr(policy, "net_arch", None),
        "policy_net_layers": pi_layers,
        "value_net_layers": vf_layers,
        "activation_fn": activation,
        "total_parameters": int(n_params),
        "expected_total_parameters": 10759,
    }


def _make_train_env_fn(track: str, root_seed: int, monitor_dir: Path):
    def _init():
        config = build_env_config(track)
        env = PostDetectionTrainingEnv(config=config, root_seed=root_seed)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, str(monitor_dir / "monitor"))
    return _init


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-detection PPO training (revised canonical protocol)")
    parser.add_argument("--track", choices=["direct", "kalman"], required=True)
    parser.add_argument("--training-seed", type=int, required=True,
                         help=f"Training root seed. Protocol roots: {TRAINING_SEEDS}. "
                              "Development/smoke-test seeds (e.g. 12345) are allowed but must "
                              "never be confused with a protocol root.")
    parser.add_argument("--total-timesteps", type=int, default=CAMPAIGN_TIMESTEPS)
    parser.add_argument("--output-dir", type=str, required=True)
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--checkpoint-freq", type=int, default=H.CHECKPOINT_FREQ)
    parser.add_argument("--eval-freq", type=int, default=H.EVAL_FREQ)
    parser.add_argument("--validation-episodes", type=int, default=VALIDATION_EPISODES)
    parser.add_argument("--smoke-test", action="store_true",
                         help="Marks this run as a SMOKE TEST -- NOT a paper model. "
                              "Requires --training-seed to be outside the reserved protocol ranges.")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        if args.training_seed in TRAINING_SEEDS:
            raise ValueError(
                f"--smoke-test requires a development seed, not a protocol training root "
                f"{TRAINING_SEEDS}; got {args.training_seed}."
            )
    assert_not_reserved_seed(VALIDATION_SEED_OFFSET)  # sanity: validation range itself must not be reserved

    output_dir = Path(args.output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    training_logs_dir = output_dir / "training_logs"
    monitor_dir = output_dir / "monitor"
    for d in (output_dir, checkpoints_dir, training_logs_dir, monitor_dir):
        d.mkdir(parents=True, exist_ok=True)

    config_direct = build_env_config("direct")
    config_kalman = build_env_config("kalman")
    assert_only_estimator_differs(config_direct, config_kalman)

    hyperparams = {
        "learning_rate": H.LEARNING_RATE, "gamma": H.GAMMA, "batch_size": H.BATCH_SIZE,
        "n_steps": H.N_STEPS, "n_epochs": H.N_EPOCHS, "clip_range": H.CLIP_RANGE,
        "ent_coef": H.ENT_COEF, "vf_coef": H.VF_COEF, "max_grad_norm": H.MAX_GRAD_NORM,
        "gae_lambda": H.GAE_LAMBDA,
    }

    print("=" * 70)
    print(f"POST-DETECTION PPO TRAINING {'[SMOKE TEST -- NOT A PAPER MODEL]' if args.smoke_test else ''}")
    print("=" * 70)
    print(f"Track: {args.track} | Training root seed: {args.training_seed} | "
          f"Total timesteps (post-detection only): {args.total_timesteps}")
    print(POST_DETECTION_TIMESTEP_DEFINITION)
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    set_random_seed(args.training_seed)
    train_env = DummyVecEnv([_make_train_env_fn(args.track, args.training_seed, monitor_dir)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=H.LEARNING_RATE,
        gamma=H.GAMMA,
        batch_size=H.BATCH_SIZE,
        n_steps=H.N_STEPS,
        n_epochs=H.N_EPOCHS,
        clip_range=H.CLIP_RANGE,
        ent_coef=H.ENT_COEF,
        vf_coef=H.VF_COEF,
        max_grad_norm=H.MAX_GRAD_NORM,
        gae_lambda=H.GAE_LAMBDA,
        seed=args.training_seed,
        device=args.device,
        verbose=1,
    )
    arch_info = _sb3_architecture_info(model)
    print(f"Network architecture: {arch_info['policy_net_layers']} / "
          f"{arch_info['total_parameters']} parameters (expected ~10,759)")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix=f"ppo_{args.track}_seed{args.training_seed}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    # Model selection uses the FULL canonical SoldierEnv (defender_standby_
    # until_detection=True is the EnvConfig default and is set explicitly by
    # build_env_config()) -- NOT the PostDetectionTrainingEnv wrapper. See
    # test_post_detection_training_env.py test J for the proof that wrapper
    # and full-canonical-mission continuations agree given identical
    # post-detection actions.
    eval_callback = SuccessRateEvalCallback(
        use_kalman_tracking=(args.track == "kalman"),
        eval_freq=args.eval_freq,
        n_eval_episodes=args.validation_episodes,
        eval_seed=VALIDATION_SEED_OFFSET,
        log_dir=str(training_logs_dir),
        best_model_save_path=str(output_dir),
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    commit = _get_git_commit()
    branch = _get_git_branch()
    start_time = datetime.now(timezone.utc)
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=args.progress_bar)
    end_time = datetime.now(timezone.utc)

    final_model_path = output_dir / "final_model.zip"
    model.save(str(final_model_path))

    eval_history = eval_callback.eval_history
    best_success_rate = eval_callback.best_success_rate
    best_timestep = next(
        (e["timesteps"] for e in eval_history if e["success_rate"] == best_success_rate), None,
    )

    import stable_baselines3
    import torch
    import gymnasium

    manifest = {
        "status": "completed",
        "smoke_test": bool(args.smoke_test),
        "git_commit": commit,
        "git_branch": branch,
        "protocol": "post_detection_v1",
        "track": args.track,
        "estimator": "kalman" if args.track == "kalman" else "measurement",
        "training_seed": args.training_seed,
        "training_timesteps": args.total_timesteps,
        "campaign": CAMPAIGN_DIRNAME,
        "timestep_definition": POST_DETECTION_TIMESTEP_DEFINITION,
        "validation_seed_offset": VALIDATION_SEED_OFFSET,
        "validation_episodes": args.validation_episodes,
        "validation_seed_range": [VALIDATION_SEED_OFFSET, VALIDATION_SEED_OFFSET + args.validation_episodes - 1],
        "model_selection_metric": MODEL_SELECTION_METRIC,
        "model_selection_tie_break_rule": MODEL_SELECTION_TIE_BREAK_RULE,
        "ppo_hyperparameters": hyperparams,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "env_config": dataclasses.asdict(build_env_config(args.track)),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "device": str(model.device),
        "start_timestamp": start_time.isoformat(),
        "end_timestamp": end_time.isoformat(),
        "network_architecture": arch_info,
        "best_validation_success_rate": best_success_rate,
        "best_checkpoint_timestep": best_timestep,
        "final_validation_success_rate": eval_history[-1]["success_rate"] if eval_history else None,
        "validation_history": eval_history,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if args.smoke_test:
        (output_dir / "SMOKE_TEST_NOT_PAPER_MODEL.txt").write_text(
            "SMOKE TEST -- NOT A PAPER MODEL.\n"
            f"training_seed={args.training_seed}, total_timesteps={args.total_timesteps}.\n"
            "Generated only to verify SB3 compatibility, reset/step behavior, callback "
            "operation, and model saving for the post-detection training protocol. "
            "Never use for publication.\n"
        )

    print(f"\nTraining complete. best_validation_success_rate={best_success_rate}")
    print(f"Manifest written to {output_dir / 'training_manifest.json'}")

    train_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
