"""
Experiment-level policy wrappers for behavioral ablations.

These wrappers COMPOSE existing policies without modifying any trained
weights, PPOPolicyWrapper/PPOKalmanPolicyWrapper, or SoldierEnv. They exist
only in experiments/ (never in uav_defend/policies/) because they are
ablation-specific evaluation constructs, not new trained controllers.

HISTORICAL NOTE (see docs/revised_predetection_protocol.md): the revised
canonical pre-detection engagement semantics (defender held co-located with
the soldier until hostile detection) are now enforced directly by
SoldierEnv/EnvConfig.defender_standby_until_detection -- NOT by this module.
EscortUntilDetectionPolicy below is retained only as a HISTORICAL/ABLATION
construct for the acquisition-ablation study (results/acquisition_ablation/),
which intentionally used policy-side escort behavior under the ORIGINAL
(pre-revision) environment. It is not part of, and must not be used for, the
revised canonical engagement protocol.
"""

from __future__ import annotations

import numpy as np


class EscortUntilDetectionPolicy:
    """
    HISTORICAL/ABLATION CONSTRUCT ONLY (see module docstring) -- superseded
    by SoldierEnv/EnvConfig.defender_standby_until_detection as the revised
    canonical pre-detection mechanism. Retained solely to reproduce the
    original acquisition-ablation study.

    Wraps any policy implementing act(obs, info) / reset(). Before hostile
    detection, uses the fixed escort-toward-soldier law -- BIT-IDENTICAL to
    the scripted controllers' shared pre-detection fallback (see
    uav_defend.policies.baseline.lead_intercept_policy.LeadInterceptPolicy._pursue):

        r_s = soldier_pos - defender_pos
        u = r_s / ||r_s||   if ||r_s|| >= eps else [0, 0, 0]

    Once `info["enemy_detected"]` is True, delegates completely to the
    wrapped policy for the rest of the episode -- no blending interval, no
    delayed takeover.

    Ground-truth safety: only reads the sanitized `soldier_pos`,
    `defender_pos`, `enemy_detected` fields before detection; never reads
    `enemy_pos`/`enemy_vel`. The wrapped policy receives the SAME
    (sanitized) `info` the caller passes in -- this wrapper never adds or
    removes fields from it.
    """

    def __init__(self, wrapped_policy, eps: float = 1e-8):
        self.wrapped_policy = wrapped_policy
        self.eps = eps

    def reset(self) -> None:
        self.wrapped_policy.reset()

    def act(self, obs, info: dict) -> np.ndarray:
        if not bool(info.get("enemy_detected", False)):
            return self._escort(info)
        return self.wrapped_policy.act(obs, info)

    def _escort(self, info: dict) -> np.ndarray:
        soldier_pos = np.asarray(info.get("soldier_pos"), dtype=np.float64)
        defender_pos = np.asarray(info.get("defender_pos"), dtype=np.float64)
        direction = soldier_pos - defender_pos
        dist = float(np.linalg.norm(direction))
        if dist < self.eps:
            return np.zeros(3, dtype=np.float32)
        return (direction / dist).astype(np.float32)
