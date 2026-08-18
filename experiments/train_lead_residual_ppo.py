"""
Lead-Residual PPO (LR-PPO) training entry point -- analogous to
experiments/train_post_detection_ppo.py, adapted for the residual interface.

PPO's observation is the 19-D residual observation (concat(direct_obs,
lead_direction)); its raw action is composed with the canonical Lead
direction via compose_lead_residual() (through LeadResidualTrainingEnv)
before being applied to the canonical SoldierEnv. Model selection uses the
FULL canonical SoldierEnv via LeadResidualSuccessRateEvalCallback, success
rate as the sole criterion.

DO NOT run a full (400,000-step, training-root 55/56/57) campaign with this
script without explicit authorization -- see
experiments/lead_residual_training_protocol.py. This script performs NO
training on import; it must be invoked explicitly.

Usage (FUTURE full campaign -- NOT run in this task):
    python experiments/train_lead_residual_ppo.py --training-seed 55 \\
        --total-timesteps 400000 --output-dir results/lead_residual_training/seed_55

    # Small smoke test (development seed, NOT a protocol root):
    python experiments/train_lead_residual_ppo.py --training-seed 12345 \\
        --total-timesteps 4096 --output-dir results/_smoke_test_lead_residual --smoke-test
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

from experiments.lead_residual_training_env import LeadResidualTrainingEnv
from experiments.lead_residual_success_rate_eval_callback import LeadResidualSuccessRateEvalCallback
from experiments.lead_residual_training_protocol import (
    TRAINING_ROOTS,
    RESIDUAL_VALIDATION_SEED_START,
    VALIDATION_EPISODES,
    CAMPAIGN_DIRNAME,
    CAMPAIGN_TIMESTEPS,
    VALIDATION_FREQUENCY_TIMESTEPS,
    CHECKPOINT_FREQUENCY_TIMESTEPS,
    RESIDUAL_MAX_ANGLE_DEG,
    CHECKPOINT_SELECTION_METRIC,
    CHECKPOINT_TIE_BREAK_RULE,
    PPO_HYPERPARAMS,
    METHOD_NAME,
    METHOD_ABBREVIATION,
    MODEL_OUTPUT_ROOT,
    build_env_config,
    assert_not_opened_yet,
)


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


def _get_git_status_clean() -> bool:
    try:
        status = subprocess.check_output(["git", "status", "--porcelain"], cwd=PROJECT_ROOT).decode()
        return not status.strip()
    except Exception:
        return False


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
        "observation_dim": int(model.observation_space.shape[0]),
        "action_dim": int(model.action_space.shape[0]),
    }


def _make_train_env_fn(root_seed: int, monitor_dir: Path):
    def _init():
        config = build_env_config()
        env = LeadResidualTrainingEnv(config=config, root_seed=root_seed, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG)
        monitor_dir.mkdir(parents=True, exist_ok=True)
        return Monitor(env, str(monitor_dir / "monitor"))
    return _init


def _enforce_official_run_lock(args: argparse.Namespace) -> None:
    """
    HARD GUARDRAIL (pre-launch protocol lock): for an OFFICIAL protocol
    training root (55/56/57) run WITHOUT --smoke-test, every
    scientific/protocol CLI value is MANDATORY and must exactly equal the
    frozen value -- never silently coerced. Also enforces the source-commit
    lock (git HEAD must equal the caller-supplied --expected-source-commit,
    with a clean working tree) so all three official runs demonstrably come
    from the identical, immutable source. Raises ValueError/RuntimeError
    BEFORE any directory creation, environment construction, or reserved
    validation-seed use.
    """
    if args.training_seed not in TRAINING_ROOTS or args.smoke_test:
        return  # development/smoke roots keep full CLI override freedom

    required = {
        "--total-timesteps": (args.total_timesteps, CAMPAIGN_TIMESTEPS),
        "--eval-freq": (args.eval_freq, VALIDATION_FREQUENCY_TIMESTEPS),
        "--validation-episodes": (args.validation_episodes, VALIDATION_EPISODES),
        "--residual-max-angle-deg": (args.residual_max_angle_deg, RESIDUAL_MAX_ANGLE_DEG),
        "--checkpoint-freq": (args.checkpoint_freq, CHECKPOINT_FREQUENCY_TIMESTEPS),
    }
    mismatches = [f"{flag}={got!r} (frozen value={expected!r})" for flag, (got, expected) in required.items() if got != expected]
    if mismatches:
        raise ValueError(
            f"OFFICIAL training root {args.training_seed} requires every frozen protocol value "
            f"(no override permitted without --smoke-test). Mismatched: {'; '.join(mismatches)}"
        )

    if not args.expected_source_commit:
        raise ValueError(
            f"OFFICIAL training root {args.training_seed} requires --expected-source-commit "
            "(the frozen LR_PPO_FINAL_TRAINING_SOURCE_COMMIT) to be supplied explicitly."
        )
    actual_commit = _get_git_commit()
    if actual_commit != args.expected_source_commit:
        raise RuntimeError(
            f"OFFICIAL training root {args.training_seed}: git HEAD ({actual_commit}) does not match "
            f"--expected-source-commit ({args.expected_source_commit}). Refusing to launch."
        )
    if not _get_git_status_clean():
        raise RuntimeError(
            f"OFFICIAL training root {args.training_seed}: working tree is not clean at commit "
            f"{actual_commit}. Refusing to launch an official run against an unclean/uncommitted tree."
        )


def _check_output_collision_guard(output_dir: Path, model_dir: Path) -> None:
    """
    Refuses to launch into a directory that already contains a COMPLETED
    official training run (never silently overwritten). Partial files from a
    prior failed/interrupted attempt are reported (not deleted) so they can
    be inspected before reuse.
    """
    completed_marker = output_dir / "COMPLETED.txt"
    manifest_path = output_dir / "run_manifest.json"
    is_smoke = None
    if manifest_path.exists():
        try:
            is_smoke = bool(json.loads(manifest_path.read_text()).get("smoke_test", False))
        except Exception:
            is_smoke = None

    if completed_marker.exists() and is_smoke is False:
        raise RuntimeError(
            f"Output directory {output_dir} already contains a COMPLETED official training run "
            f"(COMPLETED.txt present, run_manifest.json smoke_test=False). Refusing to overwrite. "
            "Move/rename the existing directory first if retraining is genuinely intended."
        )
    best_model = model_dir / "best_model.zip"
    final_model = model_dir / "final_model.zip"
    if (best_model.exists() or final_model.exists()) and is_smoke is False:
        raise RuntimeError(
            f"Model directory {model_dir} already contains a completed official model "
            f"(best_model.zip/final_model.zip present, run_manifest.json smoke_test=False). "
            "Refusing to overwrite."
        )

    partial_files = []
    for d in (output_dir, model_dir):
        if d.exists():
            partial_files.extend(p for p in d.rglob("*") if p.is_file())
    if partial_files:
        print(f"[output-collision-guard] {len(partial_files)} pre-existing file(s) found under "
              f"{output_dir} / {model_dir} (NOT a completed official run) -- inspect before reuse:")
        for p in partial_files[:30]:
            print(f"    {p}")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=f"{METHOD_NAME} ({METHOD_ABBREVIATION}) training")
    parser.add_argument("--training-seed", type=int, required=True,
                         help=f"Training root seed. Protocol roots: {TRAINING_ROOTS}. "
                              "Development/smoke-test seeds (e.g. 12345) are allowed but must "
                              "never be confused with a protocol root.")
    parser.add_argument("--total-timesteps", type=int, default=CAMPAIGN_TIMESTEPS)
    parser.add_argument("--output-dir", type=str, required=True,
                         help="Results/logs/manifest directory (e.g. results/lead_residual_training/seed_55).")
    parser.add_argument("--model-dir", type=str, default=None,
                         help="Model checkpoint directory. Defaults to "
                              "models/lead_residual_post_detection/seed_<training-seed>/ -- "
                              "a namespace separate from existing PPO/PPO-Kalman models.")
    parser.add_argument("--device", choices=["cpu", "auto"], default="cpu")
    parser.add_argument("--checkpoint-freq", type=int, default=CHECKPOINT_FREQUENCY_TIMESTEPS)
    parser.add_argument("--eval-freq", type=int, default=VALIDATION_FREQUENCY_TIMESTEPS)
    parser.add_argument("--validation-episodes", type=int, default=VALIDATION_EPISODES)
    parser.add_argument("--residual-max-angle-deg", type=float, default=RESIDUAL_MAX_ANGLE_DEG)
    parser.add_argument("--expected-source-commit", type=str, default=None,
                         help="REQUIRED for official roots 55/56/57 (without --smoke-test): the frozen "
                              "LR_PPO_FINAL_TRAINING_SOURCE_COMMIT. The launcher aborts unless "
                              "`git rev-parse HEAD` matches this value exactly and the tree is clean.")
    parser.add_argument("--smoke-test", action="store_true",
                         help="Marks this run as a SMOKE TEST -- NOT a paper model. "
                              "Requires --training-seed to be outside the reserved protocol roots.")
    parser.add_argument("--progress-bar", action="store_true")
    return parser.parse_args()


def main() -> int:
    args = parse_args()

    if args.smoke_test:
        if args.training_seed in TRAINING_ROOTS:
            raise ValueError(
                f"--smoke-test requires a development seed, not a protocol training root "
                f"{TRAINING_ROOTS}; got {args.training_seed}."
            )
    # HARD GUARDRAIL: official roots (55/56/57, non-smoke) must use every
    # frozen protocol value and the exact frozen source commit -- checked
    # BEFORE any directory creation, environment construction, or reserved
    # validation-seed use.
    _enforce_official_run_lock(args)

    # Sanity: assert_not_opened_yet must reject the expanded-final range (fails
    # loudly here if the protocol constants were ever miswired).
    try:
        assert_not_opened_yet(RESIDUAL_VALIDATION_SEED_START + 1000)  # 61000 -- lands inside the expanded-final range
        raise AssertionError("expected assert_not_opened_yet to reject the expanded-final range")
    except ValueError:
        pass

    output_dir = Path(args.output_dir)
    model_dir = Path(args.model_dir) if args.model_dir else (MODEL_OUTPUT_ROOT / f"seed_{args.training_seed}")
    _check_output_collision_guard(output_dir, model_dir)
    checkpoints_dir = model_dir / "checkpoints"
    training_logs_dir = output_dir / "training_logs"
    monitor_dir = output_dir / "monitor"
    for d in (output_dir, model_dir, checkpoints_dir, training_logs_dir, monitor_dir):
        d.mkdir(parents=True, exist_ok=True)

    config = build_env_config()

    print("=" * 70)
    print(f"{METHOD_NAME} ({METHOD_ABBREVIATION}) TRAINING {'[SMOKE TEST -- NOT A PAPER MODEL]' if args.smoke_test else ''}")
    print("=" * 70)
    print(f"Training root seed: {args.training_seed} | Total timesteps (post-detection only): {args.total_timesteps} "
          f"| residual_max_angle_deg: {args.residual_max_angle_deg}")
    print(f"Output dir: {output_dir}")
    print("=" * 70)

    set_random_seed(args.training_seed)
    train_env = DummyVecEnv([_make_train_env_fn(args.training_seed, monitor_dir)])

    model = PPO(
        policy="MlpPolicy",
        env=train_env,
        learning_rate=PPO_HYPERPARAMS["learning_rate"],
        gamma=PPO_HYPERPARAMS["gamma"],
        batch_size=PPO_HYPERPARAMS["batch_size"],
        n_steps=PPO_HYPERPARAMS["n_steps"],
        n_epochs=PPO_HYPERPARAMS["n_epochs"],
        clip_range=PPO_HYPERPARAMS["clip_range"],
        ent_coef=PPO_HYPERPARAMS["ent_coef"],
        vf_coef=PPO_HYPERPARAMS["vf_coef"],
        max_grad_norm=PPO_HYPERPARAMS["max_grad_norm"],
        gae_lambda=PPO_HYPERPARAMS["gae_lambda"],
        seed=args.training_seed,
        device=args.device,
        verbose=1,
    )
    arch_info = _sb3_architecture_info(model)
    print(f"Network architecture: {arch_info['policy_net_layers']} / "
          f"{arch_info['total_parameters']} parameters (obs_dim={arch_info['observation_dim']}, "
          f"action_dim={arch_info['action_dim']})")

    checkpoint_callback = CheckpointCallback(
        save_freq=args.checkpoint_freq,
        save_path=str(checkpoints_dir),
        name_prefix=f"lr_ppo_seed{args.training_seed}",
        save_replay_buffer=False,
        save_vecnormalize=False,
        verbose=1,
    )
    eval_callback = LeadResidualSuccessRateEvalCallback(
        residual_max_angle_deg=args.residual_max_angle_deg,
        eval_freq=args.eval_freq,
        n_eval_episodes=args.validation_episodes,
        eval_seed=RESIDUAL_VALIDATION_SEED_START,
        log_dir=str(training_logs_dir),
        best_model_save_path=str(model_dir),
        verbose=1,
    )
    callbacks = CallbackList([checkpoint_callback, eval_callback])

    commit = _get_git_commit()
    branch = _get_git_branch()
    start_time = datetime.now(timezone.utc)
    model.learn(total_timesteps=args.total_timesteps, callback=callbacks, progress_bar=args.progress_bar)
    end_time = datetime.now(timezone.utc)

    final_model_path = model_dir / "final_model.zip"
    model.save(str(final_model_path))

    eval_history = eval_callback.eval_history
    best_success_rate = eval_callback.best_success_rate
    best_timestep = next(
        (e["timesteps"] for e in eval_history if e["success_rate"] == best_success_rate), None,
    ) if eval_history else None
    remaining = [e for e in eval_history if e["timesteps"] != best_timestep]
    second_best = max(remaining, key=lambda e: (e["success_rate"], -e["timesteps"])) if remaining else None

    import stable_baselines3
    import torch
    import gymnasium
    from experiments.final_eval_utils import sha256_of_file

    best_model_path = model_dir / "best_model.zip"
    model_hashes = {
        "final_model.zip": sha256_of_file(final_model_path) if final_model_path.exists() else None,
        "best_model.zip": sha256_of_file(best_model_path) if best_model_path.exists() else None,
    }

    manifest = {
        "status": "completed",
        "smoke_test": bool(args.smoke_test),
        "is_official_protocol_root": bool(args.training_seed in TRAINING_ROOTS and not args.smoke_test),
        "expected_source_commit": args.expected_source_commit,
        "method_name": METHOD_NAME,
        "method_abbreviation": METHOD_ABBREVIATION,
        "git_commit": commit,
        "git_branch": branch,
        "protocol": "lead_residual_post_detection_v1",
        "estimator": "measurement",
        "training_seed": args.training_seed,
        "requested_timesteps": args.total_timesteps,
        "actual_model_num_timesteps": int(model.num_timesteps),
        "campaign": CAMPAIGN_DIRNAME,
        "output_dir": str(output_dir),
        "model_dir": str(model_dir),
        "residual_max_angle_deg": args.residual_max_angle_deg,
        "validation_seed_offset": RESIDUAL_VALIDATION_SEED_START,
        "validation_episodes": args.validation_episodes,
        "validation_seed_range": [RESIDUAL_VALIDATION_SEED_START, RESIDUAL_VALIDATION_SEED_START + args.validation_episodes - 1],
        "model_selection_metric": CHECKPOINT_SELECTION_METRIC,
        "model_selection_tie_break_rule": CHECKPOINT_TIE_BREAK_RULE,
        "ppo_hyperparameters": PPO_HYPERPARAMS,
        "checkpoint_freq": args.checkpoint_freq,
        "eval_freq": args.eval_freq,
        "env_config": dataclasses.asdict(config),
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
        "second_best_checkpoint_timestep": second_best["timesteps"] if second_best else None,
        "second_best_validation_success_rate": second_best["success_rate"] if second_best else None,
        "final_validation_success_rate": eval_history[-1]["success_rate"] if eval_history else None,
        "validation_history": eval_history,
        "model_hashes_sha256": model_hashes,
    }
    (output_dir / "run_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    if eval_history:
        import pandas as pd
        pd.DataFrame(eval_history).to_csv(output_dir / "validation_history.csv", index=False)
    (output_dir / "COMPLETED.txt").write_text(
        f"Run completed at {end_time.isoformat()}. training_seed={args.training_seed} "
        f"requested_timesteps={args.total_timesteps} actual_model_num_timesteps={model.num_timesteps}\n"
    )
    if args.smoke_test:
        (output_dir / "SMOKE_TEST_NOT_PAPER_MODEL.txt").write_text(
            "SMOKE TEST -- NOT A PAPER MODEL. DEVELOPMENT ONLY -- NEVER PUBLICATION EVIDENCE.\n"
            f"training_seed={args.training_seed}, total_timesteps={args.total_timesteps}.\n"
            "Generated only to verify SB3 compatibility, 19-D observation / 3-D residual action "
            "acceptance, checkpoint saving, model reload, and deterministic inference for the "
            "Lead-Residual PPO training protocol. Never use for publication.\n"
        )

    print(f"\nTraining complete. best_validation_success_rate={best_success_rate}")
    print(f"Manifest written to {output_dir / 'run_manifest.json'}")

    train_env.close()
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
