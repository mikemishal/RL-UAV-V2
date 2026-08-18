"""
Success-rate validation callback for LR-PPO training -- analogous to
experiments/success_rate_eval_callback.py (used by standalone PPO/PPO-Kalman)
but adapted for the Lead-Residual interface: the model's raw prediction is a
19-D-observation-conditioned residual, which must be composed with the
canonical Lead direction (via LeadResidualPPOPolicyWrapper, the SAME wrapper
used for future evaluation -- no duplicated composition logic) before being
passed to SoldierEnv.step().

Uses the FULL canonical SoldierEnv (defender_standby_until_detection=True),
NOT the LeadResidualTrainingEnv wrapper, for validation -- mirroring the
existing standalone-PPO validation design (train on the wrapper, validate on
the full canonical mission).
"""

from __future__ import annotations

from pathlib import Path

from stable_baselines3.common.callbacks import BaseCallback

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper


class LeadResidualSuccessRateEvalCallback(BaseCallback):
    """
    Validates the in-training LR-PPO model on the full canonical mission
    every `eval_freq` training steps, selecting the best checkpoint by
    SAFE INTERCEPTION SUCCESS RATE ONLY (ties broken by earliest step --
    enforced by the caller only overwriting best_model on a STRICT `>`
    improvement, mirroring SuccessRateEvalCallback).
    """

    def __init__(
        self,
        residual_max_angle_deg: float,
        eval_freq: int = 25_000,
        n_eval_episodes: int = 100,
        eval_seed: int = 0,
        log_dir: str = ".",
        best_model_save_path: str = ".",
        verbose: int = 1,
    ):
        super().__init__(verbose)
        self.residual_max_angle_deg = residual_max_angle_deg
        self.eval_freq = eval_freq
        self.n_eval_episodes = n_eval_episodes
        self.eval_seed = eval_seed
        self.log_dir = Path(log_dir)
        self.best_model_save_path = Path(best_model_save_path)
        self.best_success_rate = -1.0
        self.eval_history = []

    def _on_step(self) -> bool:
        if self.n_calls % self.eval_freq == 0:
            self._evaluate()
        return True

    def _evaluate(self) -> None:
        if self.verbose > 0:
            print(f"\n[LR-PPO Eval @ {self.num_timesteps} timesteps]")

        config = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)
        policy = LeadResidualPPOPolicyWrapper(
            self.model, residual_max_angle_deg=self.residual_max_angle_deg, config=config, deterministic=True,
        )

        successes = failures = timeouts = 0
        episode_lengths = []

        for ep_idx in range(self.n_eval_episodes):
            env = SoldierEnv(config=config)
            obs, info = env.reset(seed=self.eval_seed + ep_idx)
            policy.reset()
            done = False
            steps = 0
            while not done:
                policy_info = build_policy_info(info, "measurement")
                action = policy.act(obs, policy_info)
                obs, reward, terminated, truncated, info = env.step(action)
                steps += 1
                done = terminated or truncated
            env.close()

            outcome = info.get("outcome", "unknown")
            if outcome == "intercepted":
                successes += 1
            elif outcome == "timeout":
                timeouts += 1
            else:
                failures += 1
            episode_lengths.append(steps)

        success_rate = successes / self.n_eval_episodes
        entry = {
            "timesteps": int(self.num_timesteps),
            "success_rate": success_rate,
            "successes": successes,
            "failures": failures,
            "timeouts": timeouts,
            "mean_episode_length": float(sum(episode_lengths) / len(episode_lengths)) if episode_lengths else None,
        }
        self.eval_history.append(entry)

        if self.verbose > 0:
            print(f"  Success: {success_rate:.1%} | Fail: {failures/self.n_eval_episodes:.1%} "
                  f"| Timeout: {timeouts/self.n_eval_episodes:.1%}")

        if success_rate > self.best_success_rate:  # STRICT '>' -- earliest-step tie-break
            self.best_success_rate = success_rate
            self.best_model_save_path.mkdir(parents=True, exist_ok=True)
            self.model.save(str(self.best_model_save_path / "best_model.zip"))
