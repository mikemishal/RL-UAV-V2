"""
Shared success-rate-based model-selection callback for PPO training.

Both experiment tracks (Direct/measurement and Kalman) must use the SAME
best-model-selection criterion so that neither track is given an unfair
advantage by a different notion of "best". This module generalizes the
Kalman track's original custom evaluation callback (which selected the best
model by success rate) into a single shared class parameterized by
`use_kalman_tracking`, used by BOTH experiments/rl/train_ppo.py and
experiments/rl_kalman/train_ppo_kalman.py.

Rationale for success-rate (not mean reward) as the selection criterion:
mean reward is dominated by dense shaping terms (progress, time penalty,
proximity warning) that do not necessarily track the task's actual
win condition (safe interception). Success rate is the metric that is
actually reported in the paper, so selecting on it directly avoids a
train/reporting metric mismatch.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv


class SuccessRateEvalCallback(BaseCallback):
    """
    Custom evaluation callback shared by the Direct and Kalman PPO training
    scripts. Tracks detailed task metrics and saves the best model based on
    success rate (fraction of evaluation episodes ending in outcome
    "intercepted") -- IDENTICAL selection criterion for both tracks.

    Args:
        use_kalman_tracking: Whether the evaluation environment should be
            configured with Kalman tracking (True) or the Direct/measurement
            track (False). Must match the environment the model was trained
            on.
        eval_freq: Evaluate every N training steps (in `num_timesteps`).
        n_eval_episodes: Number of episodes per evaluation.
        process_var, measurement_var, lead_time: Kalman-filter/environment
            parameters, applied regardless of track (measurement_var also
            affects the Direct track's sensor noise).
        eval_seed: Base seed for evaluation episodes (offset by episode index).
        log_dir: Directory to write the JSON evaluation log to.
        best_model_save_path: Directory to save "best_model.zip" to.
        verbose: SB3 verbosity level.
    """

    def __init__(
        self,
        use_kalman_tracking: bool,
        eval_freq: int = 10000,
        n_eval_episodes: int = 20,
        process_var: float = 1.0,
        measurement_var: float = 0.5,
        lead_time: float = 0.0,
        eval_seed: int = 42,
        log_dir: str = ".",
        best_model_save_path: str = ".",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.use_kalman_tracking = use_kalman_tracking
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.process_var = process_var
        self.measurement_var = measurement_var
        self.lead_time = lead_time
        self.eval_seed = eval_seed
        self.log_dir = Path(log_dir)
        self.best_model_save_path = Path(best_model_save_path)

        # Track best success rate (identical criterion for both tracks)
        self.best_success_rate = -1.0

        # Evaluation history
        self.eval_history = []

    def _on_step(self) -> bool:
        """Called after each training step."""
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        """Run evaluation episodes and log metrics."""
        if self.verbose > 0:
            track = "Kalman" if self.use_kalman_tracking else "Direct"
            print(f"\n[Eval @ {self.num_timesteps} timesteps, {track} track]")

        # Create fresh evaluation environment with fixed seeds. Explicit
        # use_kalman_tracking (never relying on the environment default).
        config = EnvConfig(
            use_kalman_tracking=self.use_kalman_tracking,
            process_var=self.process_var,
            measurement_var=self.measurement_var,
            lead_time=self.lead_time,
        )

        # Collect episode statistics
        successes = 0
        failures = 0
        timeouts = 0
        episode_lengths = []
        tracking_errors = []

        for ep_idx in range(self.n_eval_episodes):
            env = SoldierEnv(config=config)
            obs, info = env.reset(seed=self.eval_seed + ep_idx)

            done = False
            ep_tracking_errors = []

            while not done:
                # Get action from model (deterministic)
                action, _ = self.model.predict(obs, deterministic=True)
                obs, reward, terminated, truncated, info = env.step(action)
                done = terminated or truncated

                # Track tracking error (an EVALUATION metric only, meaningful
                # for both tracks -- Kalman-filter error or raw measurement
                # error respectively).
                if info.get("tracking_error") is not None:
                    ep_tracking_errors.append(info["tracking_error"])

            # Record outcome
            outcome = info.get("outcome", "unknown")
            if outcome == "intercepted":
                successes += 1
            elif outcome == "timeout":
                timeouts += 1
            else:
                failures += 1

            episode_lengths.append(info.get("step_count", 0))
            if ep_tracking_errors:
                tracking_errors.append(np.mean(ep_tracking_errors))

            env.close()

        # Compute metrics
        n = self.n_eval_episodes
        success_rate = successes / n
        failure_rate = failures / n
        timeout_rate = timeouts / n
        mean_ep_length = np.mean(episode_lengths) if episode_lengths else 0.0
        mean_tracking_error = np.mean(tracking_errors) if tracking_errors else 0.0

        # Log metrics
        eval_result = {
            "timesteps": self.num_timesteps,
            "success_rate": success_rate,
            "failure_rate": failure_rate,
            "timeout_rate": timeout_rate,
            "mean_episode_length": mean_ep_length,
            "mean_tracking_error": mean_tracking_error,
            "n_episodes": n,
            "successes": successes,
            "failures": failures,
            "timeouts": timeouts,
        }
        self.eval_history.append(eval_result)

        # Print summary
        if self.verbose > 0:
            print(f"  Success: {success_rate:.1%} | Fail: {failure_rate:.1%} | "
                  f"Timeout: {timeout_rate:.1%}")
            print(f"  Mean ep len: {mean_ep_length:.1f} | "
                  f"Mean tracking error: {mean_tracking_error:.3f}")

        # Save best model based on success rate (IDENTICAL criterion for
        # both Direct and Kalman tracks -- see module docstring).
        if success_rate > self.best_success_rate:
            self.best_success_rate = success_rate
            best_path = self.best_model_save_path / "best_model"
            self.model.save(str(best_path))
            if self.verbose > 0:
                print(f"  New best model! Success rate: {success_rate:.1%}")

        # Save evaluation log
        self._save_eval_log()

    def _save_eval_log(self) -> None:
        """Save evaluation history to JSON file."""
        self.log_dir.mkdir(parents=True, exist_ok=True)
        log_path = self.log_dir / "eval_log.json"

        with open(log_path, "w") as f:
            json.dump({
                "eval_history": self.eval_history,
                "best_success_rate": self.best_success_rate,
                "config": {
                    "use_kalman_tracking": self.use_kalman_tracking,
                    "process_var": self.process_var,
                    "measurement_var": self.measurement_var,
                    "lead_time": self.lead_time,
                    "n_eval_episodes": self.n_eval_episodes,
                    "eval_freq": self.eval_freq,
                }
            }, f, indent=2)

    def _on_training_end(self) -> None:
        """Called at the end of training."""
        self._save_eval_log()
        if self.verbose > 0:
            print(f"\n[Training complete] Best success rate: {self.best_success_rate:.1%}")
