"""
Scenario-based Robust Model Predictive Control (RMPC) policy.

Status: IMPLEMENTED, NOT YET TUNED, NOT YET PUBLICATION-EVALUATED. See
`docs/rmpc_baseline_design.md` for the full design rationale, information-
contract argument, and hyperparameter-selection plan (Section 15 -- the
`RMPCConfig` defaults below are an UNTUNED "smallest reasonable" TEST
configuration, not a selected final configuration).

GROUND-TRUTH INDEPENDENCE: this policy NEVER reads `info['enemy_pos']`,
`info['enemy_vel']`, `EnvConfig.enemy_evasion_gain`, or
`EnvConfig.enemy_evasion_radius`. It reads ONLY the measurement-track
fields (`soldier_pos`, `defender_pos`, `defender_vel`, `enemy_detected`,
`enemy_measurement`, `enemy_measurement_velocity`,
`enemy_measurement_velocity_valid`) plus PUBLIC `EnvConfig` scalar fields
(vehicle physical limits and safety radii only).
"""

from __future__ import annotations

import time
from dataclasses import dataclass

import numpy as np

from uav_defend.config.env_config import EnvConfig
from uav_defend.policies.mpc.cem_optimizer import cem_optimize
from uav_defend.policies.mpc.rmpc_cost import evaluate_candidate

VALID_GUIDANCE_MODES = (
    "standby",
    "rmpc",
    "first_measurement",
    "optimizer_fallback",
    "measurement_fallback",
)


@dataclass
class RMPCConfig:
    """
    RMPC hyperparameters. These are an UNTUNED, midpoint-of-the-predeclared-
    candidate-grid TEST configuration (see `docs/rmpc_baseline_design.md`
    Section 11's candidate grids and Section 15's staged selection plan --
    Stage A/B/C hyperparameter selection has NOT been performed). Do not
    treat these defaults as a final, evaluated configuration.

    Fixed numerical safeguards (`cem_initial_std`, `cem_min_std`,
    `cem_sample_clip`) are NOT tunable performance hyperparameters -- they
    are conventional CEM numerical-stability constants and are not subject
    to the staged selection plan.
    """

    horizon: int = 6
    population_size: int = 128
    elite_fraction: float = 0.2
    cem_iterations: int = 5
    controller_seed: int = 0
    cem_initial_std: float = 1.0
    cem_min_std: float = 0.05
    cem_sample_clip: float = 3.0
    enable_timing: bool = True


class RMPCPolicy:
    """
    Scenario-based robust model predictive controller (see module docstring
    and `docs/rmpc_baseline_design.md`).

    Implements the shared `act(obs, info) -> np.ndarray(shape=(3,))` /
    `reset() -> None` interface used by every policy in this repository.

    RNG architecture (Section 13): owns a single dedicated
    `numpy.random.Generator`, used EXCLUSIVELY for CEM's Gaussian sampling.
    It is (re)seeded from `rmpc_config.controller_seed` both in `__init__`
    and in every `reset()` call, so that repeated full episodes (each
    preceded by the evaluator's own `policy.reset()` call, per this
    repository's existing convention) are independently, bit-for-bit
    reproducible regardless of how many decisions a PRIOR episode consumed.
    It never reads from, seeds, or is seeded by any environment-owned RNG
    stream.

    Diagnostics (NOT part of the returned action):
        last_guidance_mode: one of `VALID_GUIDANCE_MODES`.
        last_scenario_scores: dict[str, float] | None, per-scenario robust
            score of the finally selected control sequence.
        last_robust_score: float | None, `min` over `last_scenario_scores`.
        last_best_sequence: np.ndarray | None, shape (horizon, 3).
        last_objective_evaluations: int, number of `rmpc_cost.evaluate_candidate`
            calls made by the most recent `act()` (0 during standby).
        last_decision_time_s: float | None, wall-clock seconds for the most
            recent `act()` call (only recorded if `enable_timing=True`).
        decision_times_s: list[float], history of recorded decision times.
    """

    def __init__(self, config: EnvConfig | None = None, rmpc_config: RMPCConfig | None = None):
        self.config = config if config is not None else EnvConfig()
        self.rmpc_config = rmpc_config if rmpc_config is not None else RMPCConfig()
        self.decision_times_s: list[float] = []
        self._rng = self._new_rng()
        self._init_diagnostics()

    def _new_rng(self) -> np.random.Generator:
        return np.random.Generator(np.random.PCG64(np.random.SeedSequence(self.rmpc_config.controller_seed)))

    def _init_diagnostics(self) -> None:
        self.last_guidance_mode: str | None = None
        self.last_scenario_scores: dict[str, float] | None = None
        self.last_robust_score: float | None = None
        self.last_best_sequence: np.ndarray | None = None
        self.last_objective_evaluations: int = 0
        self.last_decision_time_s: float | None = None

    def reset(self) -> None:
        """
        Reset all per-episode state: re-seed the dedicated RNG (so every new
        episode's CEM sampling is independently reproducible from the same
        `controller_seed`, regardless of prior-episode call count) and
        clear diagnostics. Runtime history (`decision_times_s`) is
        intentionally NOT cleared, matching the project's convention of
        accumulating cross-episode runtime diagnostics unless explicitly
        reset by the caller.
        """
        self._rng = self._new_rng()
        self._init_diagnostics()

    def _pursue(self, target: np.ndarray | None, defender_pos: np.ndarray) -> np.ndarray:
        """Pure-pursuit fallback direction toward `target` (never ground truth)."""
        eps = self.config.eps
        if target is None:
            return np.zeros(3, dtype=np.float32)
        target = np.asarray(target, dtype=np.float64)
        direction = target - defender_pos
        dist = float(np.linalg.norm(direction))
        if dist < eps:
            return np.zeros(3, dtype=np.float32)
        return (direction / dist).astype(np.float32)

    def act(self, obs: np.ndarray, info: dict) -> np.ndarray:
        """
        Compute the RMPC action.

        Args:
            obs: Environment observation (unused directly; all state is
                read from `info`, matching the existing baseline-policy
                convention).
            info: Environment info dict. Only measurement-track fields are
                read (see module docstring); `info['enemy_pos']` and
                `info['enemy_vel']` are NEVER read.

        Returns:
            action: 3-D desired-velocity-direction vector, shape (3,),
                dtype float32, with `norm(action) <= 1 + 1e-6`.
        """
        start = time.perf_counter() if self.rmpc_config.enable_timing else None

        self._init_diagnostics()

        defender_pos = np.asarray(info.get("defender_pos"), dtype=np.float64)
        enemy_detected = bool(info.get("enemy_detected", False))

        if not enemy_detected:
            self.last_guidance_mode = "standby"
            action = np.zeros(3, dtype=np.float32)
            self._record_time(start)
            return action

        enemy_measurement = info.get("enemy_measurement")
        if enemy_measurement is None:
            # Defensive fallback: detected but no legitimate target field
            # available (should not occur under the standard sanitizer
            # contract) -- escort the soldier rather than crash.
            self.last_guidance_mode = "measurement_fallback"
            action = self._pursue(info.get("soldier_pos"), defender_pos)
            self._record_time(start)
            return action

        soldier_pos = np.asarray(info.get("soldier_pos"), dtype=np.float64)
        defender_vel = np.asarray(info.get("defender_vel"), dtype=np.float64)
        hostile_pos0 = np.asarray(enemy_measurement, dtype=np.float64)

        if info.get("enemy_measurement_velocity_valid", False):
            hostile_vel0 = np.asarray(info.get("enemy_measurement_velocity"), dtype=np.float64)
            provisional_mode = "rmpc"
        else:
            hostile_vel0 = None
            provisional_mode = "first_measurement"

        def objective(directions: np.ndarray) -> float:
            self.last_objective_evaluations += 1
            evaluation = evaluate_candidate(
                directions, defender_pos, defender_vel, hostile_pos0, hostile_vel0, soldier_pos, self.config
            )
            return evaluation.robust_score

        result = cem_optimize(
            objective=objective,
            horizon=self.rmpc_config.horizon,
            population_size=self.rmpc_config.population_size,
            elite_fraction=self.rmpc_config.elite_fraction,
            n_iterations=self.rmpc_config.cem_iterations,
            rng=self._rng,
            init_mean=None,
            initial_std=self.rmpc_config.cem_initial_std,
            min_std=self.rmpc_config.cem_min_std,
            sample_clip=self.rmpc_config.cem_sample_clip,
            eps=self.config.eps,
        )

        if not result.success:
            self.last_guidance_mode = "optimizer_fallback"
            action = self._pursue(enemy_measurement, defender_pos)
            self._record_time(start)
            return action

        final_evaluation = evaluate_candidate(
            result.best_sequence, defender_pos, defender_vel, hostile_pos0, hostile_vel0, soldier_pos, self.config
        )
        self.last_objective_evaluations += 1

        self.last_guidance_mode = provisional_mode
        self.last_scenario_scores = dict(final_evaluation.scenario_scores)
        self.last_robust_score = final_evaluation.robust_score
        self.last_best_sequence = result.best_sequence.copy()

        action = result.best_sequence[0].astype(np.float32)
        self._record_time(start)
        return action

    def _record_time(self, start: float | None) -> None:
        if start is None:
            return
        elapsed = time.perf_counter() - start
        self.last_decision_time_s = elapsed
        self.decision_times_s.append(elapsed)

    def decision_time_stats(self) -> dict:
        """Summary statistics (count/mean/median/p95/max) over
        `decision_times_s`. Purely a reporting convenience -- never
        consulted when computing an action."""
        if not self.decision_times_s:
            return {"count": 0, "mean": None, "median": None, "p95": None, "max": None}
        arr = np.asarray(self.decision_times_s, dtype=np.float64)
        return {
            "count": int(arr.size),
            "mean": float(np.mean(arr)),
            "median": float(np.median(arr)),
            "p95": float(np.percentile(arr, 95)),
            "max": float(np.max(arr)),
        }

    def __repr__(self) -> str:
        return f"RMPCPolicy(rmpc_config={self.rmpc_config!r})"
