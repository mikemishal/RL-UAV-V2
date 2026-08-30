"""
Training-time Gymnasium wrapper for the Lead-Residual PPO controller (see
docs/lead_residual_ppo_design.md). IMPLEMENTATION ONLY -- this module does
NOT train anything; no training script exists yet and NONE has been run as
part of this task.

Wraps the existing PostDetectionTrainingEnv (which already implements the
corrected post-detection-only training protocol -- reused verbatim, not
duplicated) and layers the residual composition on top:

    - PPO's action space becomes the bounded raw residual r_raw in [-1,1]^3.
    - PPO's observation space becomes the 19-D concat(direct_obs, lead_direction).
    - The action actually executed against the canonical SoldierEnv (via
      PostDetectionTrainingEnv -> SoldierEnv) is the composed hybrid
      direction u_hybrid = compose_lead_residual(lead_direction, r_raw, theta_max).

The canonical Lead controller (LeadInterceptPolicy(state_source="measurement"))
is reused verbatim -- no duplicate analytical Lead solver. The canonical
SoldierEnv dynamics (_advance_velocity, reward, hostile behavior) are never
modified; this wrapper only composes the ACTION passed in.

A future (not-yet-written) training script would construct this env and run
stable_baselines3.PPO(MlpPolicy, policy_kwargs=dict(net_arch=[64, 64]),
activation_fn=torch.nn.Tanh, ...) against it -- the SAME PPO architecture
family as the existing PPO/PPO-Kalman models -- but no such script exists or
has been run in this task.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym
from gymnasium import spaces

from uav_defend.config import EnvConfig
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.sanitize import build_policy_info
from uav_defend.policies.residual.lead_residual_composition import (
    DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
    compose_lead_residual,
)
from uav_defend.policies.residual.lead_residual_observation import (
    build_residual_observation,
    residual_observation_dim,
)

from experiments.post_detection_training_env import PostDetectionTrainingEnv
from experiments.post_detection_training_protocol import DEFAULT_EPISODE_SEED_POOL_SIZE
from experiments.lead_residual_training_protocol import (
    TRAINING_ROOT_NAMESPACE_BASE,
    TRAINING_NAMESPACE_SIZE,
    build_lr_ppo_episode_seed_sequence,
)


class LeadResidualTrainingEnv(gym.Env):
    """
    Training-time wrapper: PPO controls only the bounded directional residual
    around the canonical Lead direction, starting from the same first-detected
    post-detection observation as PostDetectionTrainingEnv (co-located,
    at-rest defender; controller authority begins the FOLLOWING step -- the
    corrected protocol is preserved exactly, unchanged, by delegating reset()/
    step() timing entirely to PostDetectionTrainingEnv).

    Args:
        config: EnvConfig for the wrapped canonical SoldierEnv. Must have
            defender_standby_until_detection=True and
            use_kalman_tracking=False -- this first minimal implementation
            targets the measurement/Direct track only (a Kalman variant is
            deferred to a later ablation, per the task's minimality constraint).
        root_seed: Deterministic training-episode-seed root, forwarded to
            PostDetectionTrainingEnv unchanged.
        residual_max_angle_deg: theta_max, the maximum angular authority (deg)
            the residual may rotate the Lead direction by. See
            uav_defend.policies.residual.lead_residual_composition.
            DEFAULT_RESIDUAL_MAX_ANGLE_DEG -- a PROVISIONAL, NOT
            publication-locked, development-only default.
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig,
        root_seed: int,
        residual_max_angle_deg: float = DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
        max_standby_steps: int | None = None,
        episode_seed_pool_size: int = DEFAULT_EPISODE_SEED_POOL_SIZE,
    ):
        super().__init__()
        assert config.use_kalman_tracking is False, (
            "LeadResidualTrainingEnv (this first minimal implementation) targets the "
            "measurement/Direct track only; a Kalman variant is deferred to a later ablation."
        )
        self._inner = PostDetectionTrainingEnv(
            config=config,
            root_seed=root_seed,
            max_standby_steps=max_standby_steps,
            episode_seed_pool_size=episode_seed_pool_size,
        )
        # For a recognized LR-PPO protocol training root (55/56/57), REPLACE
        # PostDetectionTrainingEnv's internal episode-seed pool (built above
        # via the OLD generic, non-namespaced mechanism) with the corrected,
        # collision-free per-root sequence -- WITHOUT modifying
        # post_detection_training_env.py or post_detection_training_protocol.py
        # (both remain exactly as used by the already-frozen 6-model 400k
        # campaign). This only substitutes the DATA a pre-existing, unchanged
        # consumption mechanism (_next_episode_seed) reads from -- the class
        # itself is untouched. The FULL 1,000,000-seed namespace is generated
        # up front (fast: plain Python ints) so the pool can never exhaust
        # and silently fall back to the old, non-namespaced extension logic
        # (a 400k-timestep campaign needs far fewer than 1,000,000 episodes
        # even in the extreme one-step-per-episode case). Ordinary
        # development/smoke-test roots (e.g. 12345) are NOT in
        # TRAINING_ROOT_NAMESPACE_BASE and continue using the untouched OLD
        # generic mechanism unchanged.
        if root_seed in TRAINING_ROOT_NAMESPACE_BASE:
            self._inner._episode_seeds = build_lr_ppo_episode_seed_sequence(root_seed, TRAINING_NAMESPACE_SIZE)
            self._inner._pool_size = TRAINING_NAMESPACE_SIZE
            self._inner._episode_seed_index = 0
        self._lead_policy = LeadInterceptPolicy(state_source="measurement", config=config)
        self._estimator_mode = "measurement"
        self.residual_max_angle_deg = float(residual_max_angle_deg)
        self.config = config

        obs_dim = residual_observation_dim(self._inner.observation_space)
        self.observation_space = spaces.Box(low=-np.inf, high=np.inf, shape=(obs_dim,), dtype=np.float32)
        self.action_space = spaces.Box(low=-1.0, high=1.0, shape=(3,), dtype=np.float32)

        self._last_lead_direction: np.ndarray | None = None
        self.last_reset_info: dict | None = None
        self.last_episode_seed: int | None = None

    def _compute_lead_direction(self, obs: np.ndarray, info: dict) -> np.ndarray:
        policy_info = build_policy_info(info, self._estimator_mode)  # never enemy_pos/enemy_vel
        return np.asarray(self._lead_policy.act(obs, policy_info), dtype=np.float32)

    def reset(self, *, seed=None, options=None):
        obs, info = self._inner.reset(seed=seed, options=options)
        self._lead_policy.reset()
        lead_direction = self._compute_lead_direction(obs, info)
        self._last_lead_direction = lead_direction
        self.last_reset_info = info
        self.last_episode_seed = self._inner.last_episode_seed
        residual_obs = build_residual_observation(obs, lead_direction)
        return residual_obs, info

    def step(self, action):
        lead_direction = self._last_lead_direction
        hybrid_action = compose_lead_residual(lead_direction, action, self.residual_max_angle_deg)
        obs, reward, terminated, truncated, info = self._inner.step(hybrid_action)
        next_lead_direction = self._compute_lead_direction(obs, info)
        self._last_lead_direction = next_lead_direction
        self.last_reset_info = info
        residual_obs = build_residual_observation(obs, next_lead_direction)
        return residual_obs, reward, terminated, truncated, info

    def close(self):
        self._inner.close()

    def render(self):
        return self._inner.render()
