"""
Shared PPO training-run implementation for the multi-seed training protocol.

This module is the SINGLE implementation of "run one PPO training run" used
by BOTH the Direct and Kalman tracks. experiments/train_ppo_comparison.py
(the orchestrator) calls run_training() once per (track, seed) combination.
This avoids duplicating the SB3 wiring (environment construction, shared
hyperparameters, shared callbacks, manifest generation, completed-run
protection) across two near-identical scripts.

Nothing here tunes PPO hyperparameters, reward, sensing, or dynamics --
all algorithm hyperparameters come from experiments/ppo_hyperparams.py
(unchanged in this task except the shared eval-reliability constants).
"""

from __future__ import annotations

import dataclasses
import json
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from stable_baselines3 import PPO
from stable_baselines3.common.callbacks import BaseCallback, CheckpointCallback, CallbackList
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.vec_env import DummyVecEnv
from stable_baselines3.common.utils import set_random_seed

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from experiments import ppo_hyperparams as H
from experiments.success_rate_eval_callback import SuccessRateEvalCallback
from experiments.training_protocol import (
    VALIDATION_SEED_OFFSET,
    VALIDATION_EPISODES,
    assert_not_final_test_seed,
)

PROJECT_ROOT = Path(__file__).parent.parent


def build_env_config(track: str) -> EnvConfig:
    """
    Return the EnvConfig for the given track. The ONLY difference between
    tracks is use_kalman_tracking -- every other field uses the current
    EnvConfig defaults, identical for both (see test_training_protocol.py
    for the programmatic equality check).
    """
    if track == "direct":
        return EnvConfig(use_kalman_tracking=False)
    elif track == "kalman":
        return EnvConfig(use_kalman_tracking=True)
    raise ValueError(f"Unknown track '{track}'; expected 'direct' or 'kalman'")


def _make_env_fn(track: str, seed: int, monitor_dir: Path):
    def _init():
        env = SoldierEnv(config=build_env_config(track))
        env.reset(seed=seed)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, str(monitor_dir / "monitor"))
    return _init


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(
            ["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT
        ).decode().strip()
    except Exception:
        return "unknown"


def _sb3_architecture_info(model: PPO) -> dict:
    """Extract the ACTUAL SB3 policy/value network architecture, for
    reporting and for the Direct-vs-Kalman architecture-equality test."""
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
    }


class TrainingStatsCallback(BaseCallback):
    """
    Best-effort capture of SB3's own training-loss diagnostics (policy loss,
    value loss, entropy loss, approx KL, clip fraction, explained variance,
    learning rate) at each rollout boundary, for post-hoc diagnosis only --
    NEVER used to tune hyperparameters within this task.
    """

    _KEYS = {
        "train/policy_gradient_loss": "policy_loss",
        "train/value_loss": "value_loss",
        "train/entropy_loss": "entropy_loss",
        "train/approx_kl": "approx_kl",
        "train/clip_fraction": "clip_fraction",
        "train/explained_variance": "explained_variance",
        "train/learning_rate": "learning_rate",
    }

    def __init__(self, verbose: int = 0):
        super().__init__(verbose)
        self.history: list[dict] = []

    def _on_step(self) -> bool:
        return True

    def _on_rollout_start(self) -> None:
        values = getattr(self.model.logger, "name_to_value", None)
        if not values:
            return
        snapshot: dict[str, Any] = {"timesteps": self.num_timesteps}
        for raw_key, short_key in self._KEYS.items():
            if raw_key in values:
                snapshot[short_key] = float(values[raw_key])
        if len(snapshot) > 1:
            self.history.append(snapshot)


def _env_config_to_dict(config: EnvConfig) -> dict:
    return dataclasses.asdict(config)


def expected_run_signature(
    track: str, seed: int, total_timesteps: int, hyperparams: dict, commit: str
) -> dict:
    """The fields a completed run's manifest must match to be considered
    'the same run' for completed-run protection purposes."""
    return {
        "git_commit": commit,
        "track": track,
        "training_seed": seed,
        "training_timesteps": total_timesteps,
        "ppo_hyperparameters": hyperparams,
    }


def is_run_complete(output_dir: Path, expected: dict) -> bool:
    """
    True if output_dir already contains a completed run whose manifest
    matches `expected` (commit/track/seed/timesteps/hyperparams) AND both
    best_model.zip and final_model.zip are present.
    """
    manifest_path = output_dir / "training_manifest.json"
    best_path = output_dir / "best_model.zip"
    final_path = output_dir / "final_model.zip"
    if not (manifest_path.exists() and best_path.exists() and final_path.exists()):
        return False
    try:
        manifest = json.loads(manifest_path.read_text())
    except Exception:
        return False
    if manifest.get("status") != "completed":
        return False
    return all(manifest.get(k) == v for k, v in expected.items())


def run_training(
    track: str,
    seed: int,
    output_dir: Path,
    total_timesteps: int = H.TOTAL_TIMESTEPS,
    validation_seed_offset: int = VALIDATION_SEED_OFFSET,
    validation_episodes: int = VALIDATION_EPISODES,
    checkpoint_freq: int = H.CHECKPOINT_FREQ,
    eval_freq: int = H.EVAL_FREQ,
    overwrite: bool = False,
    verbose: int = 1,
    progress_bar: bool = False,
    campaign: str = "200k_initial",
) -> dict:
    """
    Run ONE full PPO training run for the given track/seed, writing all
    artifacts to output_dir. Returns the training manifest dict (also
    written to output_dir/training_manifest.json).

    Guards against accidentally re-running an already-completed, identical
    run unless overwrite=True (see is_run_complete()).

    `campaign` is a free-text label recorded in the manifest to distinguish
    training campaigns that share this same implementation (e.g. the
    original 200k campaign vs. the 400k replication campaign). It does NOT
    affect training behavior -- output_dir alone determines where a run's
    artifacts live, so campaigns are kept separate by using distinct
    output_dir roots (see experiments/training_protocol.py).
    """
    if track not in ("direct", "kalman"):
        raise ValueError(f"Unknown track '{track}'; expected 'direct' or 'kalman'")
    assert_not_final_test_seed(validation_seed_offset)

    output_dir = Path(output_dir)
    checkpoints_dir = output_dir / "checkpoints"
    training_logs_dir = output_dir / "training_logs"
    monitor_dir = output_dir / "monitor"

    hyperparams = {
        "total_timesteps": total_timesteps,
        "learning_rate": H.LEARNING_RATE,
        "gamma": H.GAMMA,
        "batch_size": H.BATCH_SIZE,
        "n_steps": H.N_STEPS,
        "n_epochs": H.N_EPOCHS,
        "clip_range": H.CLIP_RANGE,
        "ent_coef": H.ENT_COEF,
        "vf_coef": H.VF_COEF,
        "max_grad_norm": H.MAX_GRAD_NORM,
        "gae_lambda": H.GAE_LAMBDA,
    }
    commit = _get_git_commit()
    expected = expected_run_signature(track, seed, total_timesteps, hyperparams, commit)

    if not overwrite and is_run_complete(output_dir, expected):
        print(f"[SKIP] Completed run already exists at {output_dir} "
              f"(matching commit/track/seed/timesteps/hyperparams). Pass overwrite=True to rerun.")
        return json.loads((output_dir / "training_manifest.json").read_text())

    if output_dir.exists() and overwrite:
        shutil.rmtree(output_dir)

    for d in (output_dir, checkpoints_dir, training_logs_dir, monitor_dir):
        d.mkdir(parents=True, exist_ok=True)
    (output_dir / "training_manifest.json").write_text(json.dumps({"status": "in_progress"}, indent=2))

    env_config = build_env_config(track)
    set_random_seed(seed)
    train_env = DummyVecEnv([_make_env_fn(track, seed, monitor_dir)])

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
        seed=seed,
        verbose=verbose,
    )

    checkpoint_callback = CheckpointCallback(
        save_freq=checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix=f"ppo_{track}_seed{seed}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    # Shared, IDENTICAL model-selection criterion for both tracks: maximum
    # success rate on the FIXED validation seed range (never offset by
    # training seed -- see experiments/training_protocol.py).
    eval_callback = SuccessRateEvalCallback(
        use_kalman_tracking=(track == "kalman"),
        eval_freq=eval_freq,
        n_eval_episodes=validation_episodes,
        eval_seed=validation_seed_offset,
        log_dir=str(training_logs_dir),
        best_model_save_path=str(output_dir),
        verbose=1,
    )
    stats_callback = TrainingStatsCallback(verbose=0)
    callbacks = CallbackList([checkpoint_callback, eval_callback, stats_callback])

    start_time = datetime.now(timezone.utc)
    model.learn(total_timesteps=total_timesteps, callback=callbacks, progress_bar=progress_bar)
    end_time = datetime.now(timezone.utc)

    final_model_path = output_dir / "final_model.zip"
    model.save(str(final_model_path))

    eval_history = eval_callback.eval_history
    best_success_rate = eval_callback.best_success_rate
    best_timestep = next(
        (e["timesteps"] for e in eval_history if e["success_rate"] == best_success_rate),
        None,
    )
    final_validation_success = eval_history[-1]["success_rate"] if eval_history else None

    import stable_baselines3
    import torch
    import gymnasium

    manifest = {
        "status": "completed",
        "git_commit": commit,
        "git_branch": _get_git_branch(),
        "track": track,
        "estimator": "kalman" if track == "kalman" else "measurement",
        "training_seed": seed,
        "training_timesteps": total_timesteps,
        "campaign": campaign,
        "training_budget_timesteps": total_timesteps,
        "training_mode": "fresh_from_scratch",
        "previous_200k_campaign_preserved": True,
        "validation_seed_offset": validation_seed_offset,
        "validation_episodes": validation_episodes,
        "validation_seed_range": [validation_seed_offset, validation_seed_offset + validation_episodes - 1],
        "validation_seed_start": validation_seed_offset,
        "validation_seed_end": validation_seed_offset + validation_episodes - 1,
        "ppo_hyperparameters": hyperparams,
        "checkpoint_freq": checkpoint_freq,
        "eval_freq": eval_freq,
        "env_config": _env_config_to_dict(env_config),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "torch_version": torch.__version__,
        "gymnasium_version": gymnasium.__version__,
        "stable_baselines3_version": stable_baselines3.__version__,
        "device": str(model.device),
        "start_timestamp": start_time.isoformat(),
        "end_timestamp": end_time.isoformat(),
        "network_architecture": _sb3_architecture_info(model),
        "best_validation_success_rate": best_success_rate,
        "best_checkpoint_timestep": best_timestep,
        "final_validation_success_rate": final_validation_success,
        "validation_history": eval_history,
        "training_loss_history": stats_callback.history,
    }
    (output_dir / "training_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    train_env.close()
    return manifest
