"""
Gymnasium-compatible training wrapper implementing the revised post-detection
PPO training protocol (see docs/revised_predetection_protocol.md and
experiments/post_detection_training_protocol.py).

PostDetectionTrainingEnv WRAPS a canonical SoldierEnv (never a separate
physics simulator) and re-exposes reset()/step() so that:

  - reset() internally drives the canonical standby fast-forward (a
    placeholder zero action, which standby guarantees has NO effect) until
    the environment's own info["just_detected"] becomes True, then returns
    THAT exact observation/info as the wrapper's reset() result. The
    underlying state (soldier/defender/hostile positions, velocities,
    hostile weave/evasion state, measurement RNG, Kalman state) is whatever
    the canonical mission naturally produced -- nothing is resampled,
    respawned, or reset a second time.
  - All standby-phase rewards are discarded (never returned to the caller);
    the wrapper's episode reward begins at the first post-detection,
    PPO-controlled transition.
  - step() executes the supplied action through the SAME canonical
    SoldierEnv.step() (the underlying environment is never modified),
    beginning immediately with the caller's very first action.

Per-episode environment seeds are derived deterministically from a training
root seed via experiments.post_detection_training_protocol.build_episode_seed_sequence
-- SoldierEnv.reset(seed=None) is NEVER relied upon for reproducibility.
"""

from __future__ import annotations

import numpy as np
import gymnasium as gym

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv

from experiments.post_detection_training_protocol import (
    build_episode_seed_sequence,
    DEFAULT_EPISODE_SEED_POOL_SIZE,
)


class PostDetectionTrainingEnv(gym.Env):
    """
    Training-time wrapper: PPO's episode begins at the first detected
    observation (defender co-located with soldier, at rest) and PPO controls
    ONLY the post-detection interception phase.

    Args:
        config: EnvConfig for the wrapped canonical SoldierEnv. MUST have
            defender_standby_until_detection=True (asserted in __init__) --
            this wrapper's entire purpose depends on the canonical standby
            mechanism, not a separate/legacy behavior.
        root_seed: Deterministic training-episode-seed root (see
            experiments.post_detection_training_protocol.build_episode_seed_sequence).
            Every call to reset() consumes the NEXT seed in the sequence
            derived from this root -- never OS entropy.
        max_standby_steps: Safety cap on the number of fast-forward standby
            steps per attempt, guarding against a pathological configuration
            where detection never occurs (falls back to retrying with a
            fresh episode seed; see _reset_until_detected()).
        episode_seed_pool_size: How many episode seeds to precompute from
            root_seed up front (extended automatically if exhausted).
    """

    metadata = {"render_modes": []}

    def __init__(
        self,
        config: EnvConfig,
        root_seed: int,
        max_standby_steps: int | None = None,
        episode_seed_pool_size: int = DEFAULT_EPISODE_SEED_POOL_SIZE,
    ):
        super().__init__()
        assert config.defender_standby_until_detection is True, (
            "PostDetectionTrainingEnv requires defender_standby_until_detection=True; "
            "the wrapper's entire fast-forward mechanism depends on the canonical "
            "standby behavior, not a legacy action-driven pre-detection phase."
        )
        self._env = SoldierEnv(config=config)
        self.observation_space = self._env.observation_space
        self.action_space = self._env.action_space
        self.config = config

        self.root_seed = root_seed
        self._max_standby_steps = max_standby_steps or config.max_steps
        self._episode_seeds = build_episode_seed_sequence(root_seed, episode_seed_pool_size)
        self._pool_size = episode_seed_pool_size
        self._episode_seed_index = 0

        self._placeholder_action = np.zeros(self.action_space.shape, dtype=np.float32)
        self.last_reset_info: dict | None = None
        self.last_episode_seed: int | None = None

    def _next_episode_seed(self) -> int:
        if self._episode_seed_index >= len(self._episode_seeds):
            # Defensive extension -- should not normally trigger given
            # DEFAULT_EPISODE_SEED_POOL_SIZE headroom, but never falls back
            # to non-deterministic seeding.
            more = build_episode_seed_sequence(
                int(self.root_seed) * 2_147_483_647 + len(self._episode_seeds),
                self._pool_size,
            )
            self._episode_seeds.extend(more)
        seed = self._episode_seeds[self._episode_seed_index]
        self._episode_seed_index += 1
        return seed

    def _reset_until_detected(self):
        """
        Reset the canonical env and fast-forward the standby phase with a
        placeholder (zero) action -- which standby guarantees has NO effect
        on the resulting state -- until info["just_detected"] is True. If an
        episode terminates during standby without ever reaching detection
        (only possible via timeout, since detection_radius > threat_radius
        makes soldier_caught-before-detection geometrically impossible under
        standby), retry with the NEXT deterministic episode seed rather than
        exposing a never-detected episode to PPO training.
        """
        while True:
            env_seed = self._next_episode_seed()
            obs, info = self._env.reset(seed=env_seed)
            steps = 0
            while not info["just_detected"]:
                obs, reward, terminated, truncated, info = self._env.step(self._placeholder_action)
                steps += 1
                if terminated or truncated:
                    break
                if steps > self._max_standby_steps:
                    break
            if info["just_detected"] and info["enemy_detected"]:
                self.last_episode_seed = env_seed
                return obs, info
            # else: retry with a fresh deterministic seed (rare/degenerate case).

    def reset(self, *, seed=None, options=None):
        # `seed`/`options` accepted for Gymnasium API compatibility but
        # intentionally IGNORED -- per-episode environment seeds are driven
        # entirely by the deterministic root_seed-derived sequence, never by
        # a caller-supplied seed (see class docstring / protocol module).
        obs, info = self._reset_until_detected()
        self.last_reset_info = info
        return obs, info

    def step(self, action):
        obs, reward, terminated, truncated, info = self._env.step(action)
        self.last_reset_info = info
        return obs, reward, terminated, truncated, info

    def close(self):
        self._env.close()

    def render(self):
        return self._env.render()
