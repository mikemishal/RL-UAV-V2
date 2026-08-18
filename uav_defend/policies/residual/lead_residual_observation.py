"""
Residual observation construction -- the SINGLE function shared by both the
training environment and the evaluation policy wrapper (see
docs/lead_residual_ppo_design.md, item 3).

    residual_obs = concat(direct_obs, lead_direction)

Direct observation dimension is verified programmatically (never blindly
hardcoded) by callers via SoldierEnv's own observation_space.
"""

from __future__ import annotations

import numpy as np

# Lead direction is always a 3-D unit (or zero) vector.
LEAD_DIRECTION_DIM = 3


def build_residual_observation(direct_obs: np.ndarray, lead_direction: np.ndarray) -> np.ndarray:
    """
    Concatenate the existing Direct observation with the canonical Lead
    direction to form the Lead-Residual PPO's observation.

    Args:
        direct_obs: shape (N,) -- the environment's existing Direct
            observation (N=16 for the current canonical SoldierEnv, but this
            function does not hardcode that; see residual_observation_dim()).
        lead_direction: shape (3,) -- the canonical Lead controller's action
            for the SAME state (never a duplicate/independent computation).

    Returns:
        residual_obs: shape (N+3,), float32. Contains NO true hostile
        position/velocity, future state, terminal information, tracking
        error, or reward components -- only direct_obs (itself already
        ground-truth-free) and the Lead direction (itself computed only from
        legitimate measurement-mode information).
    """
    direct_obs = np.asarray(direct_obs, dtype=np.float32).reshape(-1)
    lead_direction = np.asarray(lead_direction, dtype=np.float32).reshape(-1)
    if lead_direction.shape[0] != LEAD_DIRECTION_DIM:
        raise ValueError(f"lead_direction must have shape ({LEAD_DIRECTION_DIM},), got {lead_direction.shape}")
    return np.concatenate([direct_obs, lead_direction]).astype(np.float32)


def residual_observation_dim(direct_observation_space) -> int:
    """Programmatically derive the residual observation dimension from the
    environment's actual Direct observation_space -- never hardcode 16+3=19."""
    return int(direct_observation_space.shape[0]) + LEAD_DIRECTION_DIM
