"""
RMPC predeclared development/validation tuning campaign.

See `docs/rmpc_tuning_protocol.md` for the full predeclared protocol (which
this module implements verbatim -- candidate set, fixed constants, seed
ranges, selection metric, tie-break rule, artifact layout). This module
MUST NOT be modified once seed 71000 has been opened (protocol Section 17).

THIS MODULE ONLY DEFINES/GUARDS THE PROTOCOL AND RUNS EPISODES. Importing it
does not by itself open any seed -- seeds are only consumed when
`run_stage_*` (or `main()`) is actually invoked with an authorized seed.

Uses the CANONICAL episode runner
(`experiments.post_detection_diagnostic_eval_utils.run_diagnostic_episode`)
unmodified for every episode, which already calls `build_policy_info`
before `policy.act(...)`. No simplified RMPC-only environment is created;
the legacy `experiments/eval_utils.py` / `experiments/evaluate_for_comparison.py`
path is not used.

DEVELOPMENT / MODEL-SELECTION DATA -- NOT FINAL PUBLICATION EVIDENCE.
"""

from __future__ import annotations

import dataclasses
import json
import os

# Cap per-process BLAS/OpenMP thread pools to 1 BEFORE numpy is imported --
# this campaign parallelizes across EPISODES (one process per episode) via
# ProcessPoolExecutor, not across linear-algebra ops within an episode.
# Without this, each of up to os.cpu_count() worker processes would ALSO
# try to spawn a full per-process BLAS thread pool, causing severe CPU
# oversubscription that (empirically) made even a 14-episode smoke batch
# hang for many minutes. This is pure execution infrastructure and does
# not alter any episode's inputs, RNG streams, or outputs.
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

import platform
import subprocess
import sys
from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.mpc.rmpc_policy import RMPCConfig, RMPCPolicy

from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode, sha256_of_file
from experiments.final_stats import wilson_interval

RESULTS_ROOT = PROJECT_ROOT / "results" / "rmpc_tuning"

RMPC_DESIGN_COMMIT = "61483f6c448ff2c3b3dfa3bd4c3fa25f18fecf07"
RMPC_IMPLEMENTATION_COMMIT = "80ea7bd"
RMPC_HARDENING_COMMIT = "cb06b04ed39d17ad64b4243c2fe84407bc95e6e4"
RMPC_TUNING_BASE_COMMIT = "cb06b04ed39d17ad64b4243c2fe84407bc95e6e4"

# =============================================================================
# Section 2: fixed (never tuned) RMPC constants
# =============================================================================
FIXED_RMPC_CONSTANTS: dict = dict(
    controller_seed=0,
    cem_initial_std=1.0,
    cem_min_std=0.05,
    cem_sample_clip=3.0,
    enable_timing=True,
)

# =============================================================================
# Section 9: authorized seed ranges (exact, disjoint, hard-guarded)
# =============================================================================
STAGE_A_RANGE = (71_000, 71_029)
STAGE_B_RANGE = (71_030, 71_099)
VALIDATION_RANGE = (71_100, 71_199)
DIAGNOSTIC_RANGE = (71_200, 71_299)

STAGE_SEED_RANGES: dict[str, tuple[int, int]] = {
    "stage_a": STAGE_A_RANGE,
    "stage_b": STAGE_B_RANGE,
    "validation": VALIDATION_RANGE,
    "diagnostic": DIAGNOSTIC_RANGE,
}

RESERVED_UNOPENED_RANGES = ((71_300, 71_999), (72_000, 99_999))

STAGE_A_N = 30
STAGE_B_N = 70
VALIDATION_N = 100
DIAGNOSTIC_N = 100

assert STAGE_A_RANGE[1] - STAGE_A_RANGE[0] + 1 == STAGE_A_N
assert STAGE_B_RANGE[1] - STAGE_B_RANGE[0] + 1 == STAGE_B_N
assert VALIDATION_RANGE[1] - VALIDATION_RANGE[0] + 1 == VALIDATION_N
assert DIAGNOSTIC_RANGE[1] - DIAGNOSTIC_RANGE[0] + 1 == DIAGNOSTIC_N


def seeds_for_stage(stage: str) -> list[int]:
    start, end = STAGE_SEED_RANGES[stage]
    return list(range(start, end + 1))


def assert_seed_in_stage(seed: int, stage: str) -> None:
    """Positive-membership check: seed must fall exactly within the given
    stage's authorized range."""
    if stage not in STAGE_SEED_RANGES:
        raise ValueError(f"unknown stage '{stage}'")
    start, end = STAGE_SEED_RANGES[stage]
    if not (start <= seed <= end):
        raise ValueError(f"seed {seed} is not in the '{stage}' range [{start}, {end}]")


def assert_seed_authorized(seed: int) -> None:
    """Reject any seed not in ANY of the four authorized stage ranges above
    -- in particular, all of 71300-71999 (reserved, unopened) and every
    seed >= 72000 (final-campaign namespace, out of scope for this prompt)."""
    for start, end in STAGE_SEED_RANGES.values():
        if start <= seed <= end:
            return
    raise ValueError(f"seed {seed} is not within any authorized RMPC-tuning stage range")


# =============================================================================
# Section 3: predeclared candidate configurations (C1-C7) -- fixed, no
# post-hoc additions, no exhaustive Cartesian grid.
# =============================================================================
CANDIDATE_IDS: tuple[str, ...] = (
    "C1_FAST", "C2_SHORT", "C3_BALANCED", "C4_LONG",
    "C5_LARGE_POP", "C6_TIGHT_ELITE", "C7_MORE_ITER",
)

CANDIDATES: dict[str, dict] = {
    "C1_FAST":        dict(horizon=4, population_size=64,  elite_fraction=0.20, cem_iterations=3),
    "C2_SHORT":       dict(horizon=4, population_size=128, elite_fraction=0.20, cem_iterations=5),
    "C3_BALANCED":    dict(horizon=6, population_size=128, elite_fraction=0.20, cem_iterations=5),
    "C4_LONG":        dict(horizon=8, population_size=128, elite_fraction=0.20, cem_iterations=5),
    "C5_LARGE_POP":   dict(horizon=6, population_size=256, elite_fraction=0.20, cem_iterations=5),
    "C6_TIGHT_ELITE": dict(horizon=6, population_size=128, elite_fraction=0.10, cem_iterations=5),
    "C7_MORE_ITER":   dict(horizon=6, population_size=128, elite_fraction=0.20, cem_iterations=8),
}
assert set(CANDIDATES) == set(CANDIDATE_IDS)

# Predeclared BEFORE any seed is opened -- avoids post-hoc cherry-picking of
# the fairness-audit representative (protocol Section 11).
REPRESENTATIVE_FAIRNESS_CANDIDATE = "C3_BALANCED"


def build_rmpc_config(candidate_id: str) -> RMPCConfig:
    if candidate_id not in CANDIDATES:
        raise ValueError(f"unknown candidate_id '{candidate_id}'")
    params = CANDIDATES[candidate_id]
    return RMPCConfig(
        horizon=params["horizon"],
        population_size=params["population_size"],
        elite_fraction=params["elite_fraction"],
        cem_iterations=params["cem_iterations"],
        **FIXED_RMPC_CONSTANTS,
    )


def theoretical_scenario_step_cost(candidate_id: str) -> int:
    """N * I * 6 scenarios * H (Section 10's first tie-break criterion)."""
    p = CANDIDATES[candidate_id]
    return p["population_size"] * p["cem_iterations"] * 6 * p["horizon"]


def theoretical_candidate_evaluations(candidate_id: str) -> int:
    p = CANDIDATES[candidate_id]
    return p["population_size"] * p["cem_iterations"]


def theoretical_defender_rollout_steps(candidate_id: str) -> int:
    p = CANDIDATES[candidate_id]
    return p["population_size"] * p["cem_iterations"] * p["horizon"]


# =============================================================================
# Section 8/10: predeclared exact-tie-break selection rule
# =============================================================================
def rank_candidates(success_counts: dict[str, int],
                     median_planning_times: dict[str, float] | None = None) -> list[str]:
    """
    Descending-success ranking of candidates with the predeclared exact-tie
    rule: (1) lower theoretical_scenario_step_cost, (2) lower median
    post-detection planning time (if provided), (3) lexicographically
    smaller candidate_id. No new tie-break is invented after results are
    observed.
    """
    times = median_planning_times or {}

    def sort_key(cid: str):
        return (
            -success_counts[cid],
            theoretical_scenario_step_cost(cid),
            times.get(cid, float("inf")),
            cid,
        )

    return sorted(success_counts.keys(), key=sort_key)


def assert_candidates_allowed(candidate_ids, allowed_ids, stage_name: str) -> None:
    disallowed = [c for c in candidate_ids if c not in allowed_ids]
    if disallowed:
        raise ValueError(f"{stage_name}: candidate(s) {disallowed} not in allowed set {list(allowed_ids)}")


def assert_diagnostic_candidate(candidate_id: str, selected_candidate_id: str) -> None:
    if candidate_id != selected_candidate_id:
        raise ValueError(
            f"diagnostic stage may only evaluate the frozen SELECTED_RMPC_CONFIG "
            f"'{selected_candidate_id}', got '{candidate_id}'"
        )


# =============================================================================
# Nominal environment (Section 4)
# =============================================================================
def build_nominal_env_config() -> EnvConfig:
    config = EnvConfig()
    assert config.defender_standby_until_detection is True
    assert config.use_kalman_tracking is False
    return config


# =============================================================================
# Section 5: thin pass-through diagnostics observer (does NOT alter the
# returned action; only reads RMPCPolicy's own public diagnostic attributes
# after each forwarded act() call).
# =============================================================================
class _RMPCDiagnosticsWrapper:
    def __init__(self, policy: RMPCPolicy):
        self._policy = policy
        self.guidance_mode_counts: dict[str, int] = {}
        self.initialization_mode_counts: dict[str, int] = {}
        self.n_calls = 0

    def reset(self) -> None:
        self._policy.reset()
        self.guidance_mode_counts = {}
        self.initialization_mode_counts = {}
        self.n_calls = 0

    def act(self, obs, info):
        action = self._policy.act(obs, info)
        self.n_calls += 1
        mode = self._policy.last_guidance_mode
        if mode is not None:
            self.guidance_mode_counts[mode] = self.guidance_mode_counts.get(mode, 0) + 1
        init_mode = self._policy.last_initialization_mode
        if init_mode is not None:
            self.initialization_mode_counts[init_mode] = self.initialization_mode_counts.get(init_mode, 0) + 1
        return action

    @property
    def decision_times_s(self) -> list[float]:
        return self._policy.decision_times_s


def _post_detection_times(all_times_before: int, all_times_after: list[float], detection_step: int | None) -> list[float]:
    """Section 13/18: isolate THIS episode's new timing entries, then
    exclude the first `detection_step` entries (pre-detection/transition
    standby calls) -- only post-detection planning-call times remain."""
    new_times = all_times_after[all_times_before:]
    if detection_step is None:
        return []
    return new_times[detection_step:]


# =============================================================================
# Episode runners (Section 5/6) -- canonical run_diagnostic_episode, unmodified
# =============================================================================
def run_rmpc_episode(env_config: EnvConfig, candidate_id: str, seed: int,
                      record_trajectory: bool = False) -> tuple[dict, list[float]]:
    env = SoldierEnv(config=env_config)
    policy = RMPCPolicy(config=env_config, rmpc_config=build_rmpc_config(candidate_id))
    wrapped = _RMPCDiagnosticsWrapper(policy)
    n_before = len(policy.decision_times_s)
    row = run_diagnostic_episode(env, wrapped, seed=seed, dt=env_config.dt, record_trajectory=record_trajectory)
    env.close()

    row["candidate_id"] = candidate_id
    row["optimizer_fallback_count"] = wrapped.guidance_mode_counts.get("optimizer_fallback", 0)
    row["measurement_fallback_count"] = wrapped.guidance_mode_counts.get("measurement_fallback", 0)
    row["lead_init_count"] = wrapped.initialization_mode_counts.get("lead", 0)
    row["pursuit_init_count"] = wrapped.initialization_mode_counts.get("pursuit", 0)
    row["n_policy_calls"] = wrapped.n_calls

    post_detection_times = _post_detection_times(n_before, policy.decision_times_s, row["detection_step"])
    row["n_post_detection_decisions"] = len(post_detection_times)
    return row, post_detection_times


def run_lead_episode(env_config: EnvConfig, seed: int, record_trajectory: bool = False) -> dict:
    env = SoldierEnv(config=env_config)
    policy = LeadInterceptPolicy(state_source="measurement", config=env_config)
    row = run_diagnostic_episode(env, policy, seed=seed, dt=env_config.dt, record_trajectory=record_trajectory)
    env.close()
    row["candidate_id"] = "LEAD"
    return row


def assert_detection_step_fairness(lead_row: dict, rmpc_row: dict) -> None:
    """Section 11/16: pre-detection history is controller-independent under
    the canonical standby protocol -- detection_step (and
    detection_time_seconds) must match exactly for the same environment
    seed regardless of which controller is driving the defender."""
    if lead_row["environment_seed"] != rmpc_row["environment_seed"]:
        raise ValueError("assert_detection_step_fairness: mismatched environment_seed")
    if lead_row["detection_step"] != rmpc_row["detection_step"]:
        raise ValueError(
            f"PRE-DETECTION FAIRNESS VIOLATION at seed {lead_row['environment_seed']}: "
            f"Lead detection_step={lead_row['detection_step']} != "
            f"RMPC({rmpc_row['candidate_id']}) detection_step={rmpc_row['detection_step']}"
        )
    if lead_row["detection_time_seconds"] != rmpc_row["detection_time_seconds"]:
        raise ValueError(
            f"PRE-DETECTION FAIRNESS VIOLATION at seed {lead_row['environment_seed']}: "
            f"detection_time_seconds differs"
        )


_TRAJECTORY_FIELDS = ("soldier_pos", "defender_pos", "defender_vel", "enemy_pos", "enemy_vel", "enemy_detected")


def assert_pre_detection_trajectory_matches(lead_trajectory: list[dict], rmpc_trajectory: list[dict], seed: int) -> None:
    if len(lead_trajectory) != len(rmpc_trajectory):
        raise ValueError(f"PRE-DETECTION TRAJECTORY LENGTH MISMATCH at seed {seed}")
    for lead_step, rmpc_step in zip(lead_trajectory, rmpc_trajectory):
        for field in _TRAJECTORY_FIELDS:
            lead_val, rmpc_val = lead_step[field], rmpc_step[field]
            if isinstance(lead_val, np.ndarray):
                if not np.array_equal(lead_val, rmpc_val):
                    raise ValueError(f"PRE-DETECTION TRAJECTORY MISMATCH at seed {seed}, field '{field}'")
            elif lead_val != rmpc_val:
                raise ValueError(f"PRE-DETECTION TRAJECTORY MISMATCH at seed {seed}, field '{field}'")


# =============================================================================
# Parallel execution (pure infrastructure -- does not alter any episode's
# inputs/outputs; a single-process run and a parallel run are equivalent
# since each episode is an independent, deterministically seeded rollout).
# Predeclared BEFORE any seed >= 71000 is opened, per protocol Section 17.
# =============================================================================
DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)


def _rmpc_task(args):
    env_config, candidate_id, seed = args
    return run_rmpc_episode(env_config, candidate_id, seed)


def _lead_task(args):
    env_config, seed = args
    return run_lead_episode(env_config, seed)


def _run_rmpc_episodes_parallel(env_config: EnvConfig, tasks: list[tuple[str, int]],
                                 max_workers: int | None = None) -> list[tuple[dict, list[float]]]:
    """tasks: list of (candidate_id, seed). Returns results in the SAME
    order as `tasks` (executor.map preserves input order)."""
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    args = [(env_config, cid, seed) for cid, seed in tasks]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_rmpc_task, args))


def _run_lead_episodes_parallel(env_config: EnvConfig, seeds: list[int],
                                 max_workers: int | None = None) -> list[dict]:
    max_workers = max_workers or DEFAULT_MAX_WORKERS
    args = [(env_config, seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=max_workers) as executor:
        return list(executor.map(_lead_task, args))


# =============================================================================
# Statistics (Section 14)
# =============================================================================
def summarize_success(rows: list[dict]) -> dict:
    n = len(rows)
    successes = sum(r["success"] for r in rows)
    lo, hi = wilson_interval(successes, n) if n else (float("nan"), float("nan"))
    return {
        "n": n,
        "success_count": successes,
        "success_rate": successes / n if n else float("nan"),
        "wilson_ci_low": lo,
        "wilson_ci_high": hi,
        "soldier_caught_count": sum(r["soldier_caught"] for r in rows),
        "soldier_caught_rate": sum(r["soldier_caught"] for r in rows) / n if n else float("nan"),
        "unsafe_intercept_count": sum(r["unsafe_intercept"] for r in rows),
        "unsafe_intercept_rate": sum(r["unsafe_intercept"] for r in rows) / n if n else float("nan"),
        "timeout_count": sum(r["timeout"] for r in rows),
        "timeout_rate": sum(r["timeout"] for r in rows) / n if n else float("nan"),
    }


def summarize_planning_times(times: list[float]) -> dict:
    if not times:
        return {"n_planning_decisions": 0, "mean": None, "median": None, "p95": None, "max": None}
    arr = np.asarray(times, dtype=np.float64)
    return {
        "n_planning_decisions": int(arr.size),
        "mean": float(np.mean(arr)),
        "median": float(np.median(arr)),
        "p95": float(np.percentile(arr, 95)),
        "max": float(np.max(arr)),
    }


# =============================================================================
# Platform / provenance metadata (Section 16/21)
# =============================================================================
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


RMPC_SOURCE_FILES = (
    PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "rmpc_policy.py",
    PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "rmpc_cost.py",
    PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "hostile_scenarios.py",
    PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "cem_optimizer.py",
    PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "lead_init.py",
    PROJECT_ROOT / "uav_defend" / "dynamics" / "constrained_point_mass.py",
)


def platform_metadata() -> dict:
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }


def source_file_hashes() -> dict[str, str]:
    return {p.name: sha256_of_file(p) for p in RMPC_SOURCE_FILES}


def build_protocol_manifest(env_config: EnvConfig) -> dict:
    return {
        "repository": "mikemishal/RL-UAV-V2",
        "branch": _git_branch(),
        "rmpc_tuning_protocol_commit": _git_commit(),
        "rmpc_design_commit": RMPC_DESIGN_COMMIT,
        "rmpc_implementation_commit": RMPC_IMPLEMENTATION_COMMIT,
        "rmpc_hardening_commit": RMPC_HARDENING_COMMIT,
        "rmpc_tuning_base_commit": RMPC_TUNING_BASE_COMMIT,
        "env_config_snapshot": dataclasses.asdict(env_config),
        "candidate_configs": CANDIDATES,
        "fixed_cem_constants": FIXED_RMPC_CONSTANTS,
        "controller_seed": FIXED_RMPC_CONSTANTS["controller_seed"],
        "scenario_count": 6,
        "seed_ranges_opened": {k: list(v) for k, v in STAGE_SEED_RANGES.items()},
        "seed_ranges_unopened_reserve": [list(r) for r in RESERVED_UNOPENED_RANGES],
        "selection_metric": "safe_interception_success_count (outcome == 'intercepted')",
        "tie_break_rule": [
            "lower theoretical scenario-step cost (population_size * cem_iterations * 6 * horizon)",
            "lower median post-detection RMPC planning time",
            "lexicographically smaller candidate_id",
        ],
        "canonical_episode_runner": "experiments.post_detection_diagnostic_eval_utils.run_diagnostic_episode",
        "sanitizer_function": "uav_defend.policies.sanitize.build_policy_info",
        "representative_fairness_candidate": REPRESENTATIVE_FAIRNESS_CANDIDATE,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "platform": platform_metadata(),
        "rmpc_source_file_hashes": source_file_hashes(),
        "development_data_disclaimer": "DEVELOPMENT / MODEL-SELECTION DATA -- NOT FINAL PUBLICATION EVIDENCE",
    }


# =============================================================================
# Stage orchestration (Section 10)
# =============================================================================
def run_stage_a(env_config: EnvConfig, max_workers: int | None = None) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    seeds = seeds_for_stage("stage_a")
    for s in seeds:
        assert_seed_in_stage(s, "stage_a")
        assert_seed_authorized(s)

    lead_rows = _run_lead_episodes_parallel(env_config, seeds, max_workers)
    lead_by_seed = {row["environment_seed"]: row for row in lead_rows}

    tasks = [(cid, seed) for cid in CANDIDATE_IDS for seed in seeds]
    results = _run_rmpc_episodes_parallel(env_config, tasks, max_workers)

    rmpc_rows: list[dict] = []
    planning_times: dict[str, list[float]] = {cid: [] for cid in CANDIDATE_IDS}
    for (cid, seed), (row, times) in zip(tasks, results):
        assert_detection_step_fairness(lead_by_seed[seed], row)
        rmpc_rows.append(row)
        planning_times[cid].extend(times)

    return rmpc_rows, lead_rows, planning_times


def run_stage_b(env_config: EnvConfig, stage_a_top3: list[str],
                max_workers: int | None = None) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    assert_candidates_allowed(stage_a_top3, CANDIDATE_IDS, "stage_b")
    if len(stage_a_top3) != 3:
        raise ValueError(f"stage_b requires exactly 3 candidates, got {len(stage_a_top3)}")

    seeds = seeds_for_stage("stage_b")
    for s in seeds:
        assert_seed_in_stage(s, "stage_b")
        assert_seed_authorized(s)

    lead_rows = _run_lead_episodes_parallel(env_config, seeds, max_workers)
    lead_by_seed = {row["environment_seed"]: row for row in lead_rows}

    tasks = [(cid, seed) for cid in stage_a_top3 for seed in seeds]
    results = _run_rmpc_episodes_parallel(env_config, tasks, max_workers)

    rmpc_rows: list[dict] = []
    planning_times: dict[str, list[float]] = {cid: [] for cid in stage_a_top3}
    for (cid, seed), (row, times) in zip(tasks, results):
        assert_detection_step_fairness(lead_by_seed[seed], row)
        rmpc_rows.append(row)
        planning_times[cid].extend(times)

    return rmpc_rows, lead_rows, planning_times


def run_stage_c_validation(env_config: EnvConfig, development_finalists: list[str],
                           max_workers: int | None = None) -> tuple[list[dict], list[dict], dict[str, list[float]]]:
    if len(development_finalists) != 2:
        raise ValueError(f"validation requires exactly 2 candidates, got {len(development_finalists)}")

    seeds = seeds_for_stage("validation")
    for s in seeds:
        assert_seed_in_stage(s, "validation")
        assert_seed_authorized(s)

    lead_rows = _run_lead_episodes_parallel(env_config, seeds, max_workers)
    lead_by_seed = {row["environment_seed"]: row for row in lead_rows}

    tasks = [(cid, seed) for cid in development_finalists for seed in seeds]
    results = _run_rmpc_episodes_parallel(env_config, tasks, max_workers)

    rmpc_rows: list[dict] = []
    planning_times: dict[str, list[float]] = {cid: [] for cid in development_finalists}
    for (cid, seed), (row, times) in zip(tasks, results):
        assert_detection_step_fairness(lead_by_seed[seed], row)
        rmpc_rows.append(row)
        planning_times[cid].extend(times)

    return rmpc_rows, lead_rows, planning_times


def run_stage_d_diagnostic(env_config: EnvConfig, selected_candidate_id: str,
                           max_workers: int | None = None) -> tuple[list[dict], list[dict], list[float]]:
    seeds = seeds_for_stage("diagnostic")
    for s in seeds:
        assert_seed_in_stage(s, "diagnostic")
        assert_seed_authorized(s)

    lead_rows = _run_lead_episodes_parallel(env_config, seeds, max_workers)
    lead_by_seed = {row["environment_seed"]: row for row in lead_rows}

    tasks = [(selected_candidate_id, seed) for seed in seeds]
    results = _run_rmpc_episodes_parallel(env_config, tasks, max_workers)

    rmpc_rows: list[dict] = []
    planning_times: list[float] = []
    for (cid, seed), (row, times) in zip(tasks, results):
        assert_detection_step_fairness(lead_by_seed[seed], row)
        rmpc_rows.append(row)
        planning_times.extend(times)

    return rmpc_rows, lead_rows, planning_times



def run_fairness_trajectory_audit(env_config: EnvConfig, seeds: list[int], representative_candidate_id: str) -> None:
    """Section 11: full pre-detection trajectory comparison for a small
    deterministic subset (first 3 seeds of Stage A)."""
    for seed in seeds:
        lead_row = run_lead_episode(env_config, seed, record_trajectory=True)
        rmpc_row, _times = run_rmpc_episode(env_config, representative_candidate_id, seed, record_trajectory=True)
        assert_detection_step_fairness(lead_row, rmpc_row)
        assert_pre_detection_trajectory_matches(
            lead_row["pre_detection_trajectory"], rmpc_row["pre_detection_trajectory"], seed
        )


# =============================================================================
# Result-writing helpers (Section 20)
# =============================================================================
_EPISODE_COLUMNS_LEAD = [
    "candidate_id", "environment_seed", "outcome", "success", "soldier_caught", "unsafe_intercept", "timeout",
    "total_steps", "detection_step", "detection_time_seconds", "post_detection_steps", "intercept_step",
    "intercept_time_seconds", "min_enemy_soldier_dist", "physical_limit_violations", "mean_tracking_error",
    "final_tracking_error", "tracking_rmse", "n_tracking_samples",
]
_EPISODE_COLUMNS_RMPC = _EPISODE_COLUMNS_LEAD + [
    "optimizer_fallback_count", "measurement_fallback_count", "lead_init_count", "pursuit_init_count",
    "n_policy_calls", "n_post_detection_decisions",
]


def _rows_to_dataframe(rows: list[dict], columns: list[str]) -> pd.DataFrame:
    return pd.DataFrame(rows)[columns]


def write_episode_csv(rows: list[dict], path: Path, rmpc: bool) -> None:
    columns = _EPISODE_COLUMNS_RMPC if rmpc else _EPISODE_COLUMNS_LEAD
    path.parent.mkdir(parents=True, exist_ok=True)
    _rows_to_dataframe(rows, columns).to_csv(path, index=False)


def write_summary_csv(summaries: dict[str, dict], path: Path) -> None:
    records = []
    for cid, summary in summaries.items():
        record = {"candidate_id": cid}
        record.update(summary)
        records.append(record)
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(records).to_csv(path, index=False)


def build_selected_config_json(selected_candidate_id: str, validation_summary: dict, tie_break_used: bool) -> dict:
    params = CANDIDATES[selected_candidate_id]
    return {
        "candidate_id": selected_candidate_id,
        "horizon": params["horizon"],
        "population_size": params["population_size"],
        "elite_fraction": params["elite_fraction"],
        "cem_iterations": params["cem_iterations"],
        **FIXED_RMPC_CONSTANTS,
        "validation_success_count": validation_summary["success_count"],
        "validation_n": validation_summary["n"],
        "validation_success_rate": validation_summary["success_rate"],
        "selection_rule": "validation safe-interception success count on seeds 71100-71199 only",
        "tie_break_used": tie_break_used,
        "source_commit": _git_commit(),
        "selected_at_timestamp": datetime.now(timezone.utc).isoformat(),
        "frozen_for_final_evaluation": True,
    }


def _summary_for_rows(rows: list[dict], times: list[float], candidate_id: str) -> dict:
    summary = summarize_success(rows)
    summary.update({f"planning_time_{k}": v for k, v in summarize_planning_times(times).items()})
    summary["theoretical_scenario_step_cost"] = theoretical_scenario_step_cost(candidate_id)
    summary["theoretical_candidate_evaluations"] = theoretical_candidate_evaluations(candidate_id)
    summary["theoretical_defender_rollout_steps"] = theoretical_defender_rollout_steps(candidate_id)
    return summary


def write_tuning_report(report_path: Path, *, stage_a_summaries, stage_a_top3, stage_b_summaries,
                         development_finalists, validation_summaries, selected_candidate_id, tie_break_used,
                         diagnostic_summary, lead_summary_by_stage) -> None:
    lines = [
        "# RMPC Predeclared Tuning Campaign Report",
        "",
        "**DEVELOPMENT / MODEL-SELECTION DATA -- NOT FINAL PUBLICATION EVIDENCE.**",
        "This report must never be copied into `paper_artifacts/` or used to",
        "update publication figures. See `docs/rmpc_tuning_protocol.md` for the",
        "full predeclared protocol.",
        "",
        "## Stage A -- 30-seed development screen (all 7 candidates)",
        "",
        "| Candidate | Success/30 | Success % | Wilson 95% CI |",
        "|---|---|---|---|",
    ]
    for cid in CANDIDATE_IDS:
        s = stage_a_summaries[cid]
        lines.append(f"| {cid} | {s['success_count']}/{s['n']} | {s['success_rate']*100:.1f}% | "
                      f"[{s['wilson_ci_low']:.3f}, {s['wilson_ci_high']:.3f}] |")
    lines += ["", f"**STAGE_A_TOP3 = {stage_a_top3}**", "", "## Stage B -- 100-seed development total (top 3)", "",
              "| Candidate | Combined Success/100 | Success % | Wilson 95% CI |", "|---|---|---|---|"]
    for cid in stage_a_top3:
        s = stage_b_summaries[cid]
        lines.append(f"| {cid} | {s['success_count']}/{s['n']} | {s['success_rate']*100:.1f}% | "
                      f"[{s['wilson_ci_low']:.3f}, {s['wilson_ci_high']:.3f}] |")
    lines += ["", f"**DEVELOPMENT_FINALISTS = {development_finalists}**", "",
              "## Validation -- 100 independent seeds (2 finalists)", "",
              "| Candidate | Success/100 | Success % | Wilson 95% CI |", "|---|---|---|---|"]
    for cid in development_finalists:
        s = validation_summaries[cid]
        lines.append(f"| {cid} | {s['success_count']}/{s['n']} | {s['success_rate']*100:.1f}% | "
                      f"[{s['wilson_ci_low']:.3f}, {s['wilson_ci_high']:.3f}] |")
    lines += ["", f"**SELECTED_RMPC_CONFIG = {selected_candidate_id}** (tie-break used: {tie_break_used})", "",
              "## Stage D -- 100-seed post-selection diagnostic (frozen config, informational only)", "",
              f"Success: {diagnostic_summary['success_count']}/{diagnostic_summary['n']} "
              f"({diagnostic_summary['success_rate']*100:.1f}%), "
              f"Wilson 95% CI [{diagnostic_summary['wilson_ci_low']:.3f}, {diagnostic_summary['wilson_ci_high']:.3f}]",
              "", "Diagnostic performance did NOT alter the selected configuration.", "",
              "## Lead reference (contextual only)", ""]
    for stage_name, s in lead_summary_by_stage.items():
        lines.append(f"- {stage_name}: {s['success_count']}/{s['n']} ({s['success_rate']*100:.1f}%)")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    env_config = build_nominal_env_config()

    # Fairness trajectory audit (Section 11) -- first 3 Stage-A seeds, one
    # predeclared representative candidate.
    fairness_seeds = seeds_for_stage("stage_a")[:3]
    run_fairness_trajectory_audit(env_config, fairness_seeds, REPRESENTATIVE_FAIRNESS_CANDIDATE)

    # ---- Stage A ----
    stage_a_rmpc_rows, stage_a_lead_rows, stage_a_times = run_stage_a(env_config)
    write_episode_csv(stage_a_rmpc_rows, RESULTS_ROOT / "stage_a_episodes.csv", rmpc=True)

    stage_a_success = {cid: sum(r["success"] for r in stage_a_rmpc_rows if r["candidate_id"] == cid)
                        for cid in CANDIDATE_IDS}
    stage_a_median_times = {cid: (summarize_planning_times(stage_a_times[cid])["median"] or float("inf"))
                             for cid in CANDIDATE_IDS}
    stage_a_ranked = rank_candidates(stage_a_success, stage_a_median_times)
    stage_a_top3 = stage_a_ranked[:3]

    stage_a_summaries = {
        cid: _summary_for_rows([r for r in stage_a_rmpc_rows if r["candidate_id"] == cid], stage_a_times[cid], cid)
        for cid in CANDIDATE_IDS
    }
    write_summary_csv(stage_a_summaries, RESULTS_ROOT / "stage_a_summary.csv")

    # ---- Stage B ----
    stage_b_rmpc_rows, stage_b_lead_rows, stage_b_times = run_stage_b(env_config, stage_a_top3)
    write_episode_csv(stage_b_rmpc_rows, RESULTS_ROOT / "stage_b_episodes.csv", rmpc=True)

    combined_success, combined_times, combined_rows = {}, {}, {}
    for cid in stage_a_top3:
        a_rows = [r for r in stage_a_rmpc_rows if r["candidate_id"] == cid]
        b_rows = [r for r in stage_b_rmpc_rows if r["candidate_id"] == cid]
        combined_rows[cid] = a_rows + b_rows
        combined_times[cid] = stage_a_times[cid] + stage_b_times[cid]
        combined_success[cid] = sum(r["success"] for r in combined_rows[cid])
    combined_median_times = {cid: (summarize_planning_times(combined_times[cid])["median"] or float("inf"))
                              for cid in stage_a_top3}
    stage_b_ranked = rank_candidates(combined_success, combined_median_times)
    development_finalists = stage_b_ranked[:2]

    stage_b_summaries = {cid: _summary_for_rows(combined_rows[cid], combined_times[cid], cid) for cid in stage_a_top3}
    write_summary_csv(stage_b_summaries, RESULTS_ROOT / "stage_b_summary.csv")

    # ---- Stage C: validation ----
    val_rmpc_rows, val_lead_rows, val_times = run_stage_c_validation(env_config, development_finalists)
    write_episode_csv(val_rmpc_rows, RESULTS_ROOT / "validation_episodes.csv", rmpc=True)

    val_success = {cid: sum(r["success"] for r in val_rmpc_rows if r["candidate_id"] == cid)
                   for cid in development_finalists}
    val_median_times = {cid: (summarize_planning_times(val_times[cid])["median"] or float("inf"))
                         for cid in development_finalists}
    val_ranked = rank_candidates(val_success, val_median_times)
    selected_candidate_id = val_ranked[0]
    tie_break_used = len(val_ranked) > 1 and val_success[val_ranked[0]] == val_success[val_ranked[1]]

    val_summaries = {
        cid: _summary_for_rows([r for r in val_rmpc_rows if r["candidate_id"] == cid], val_times[cid], cid)
        for cid in development_finalists
    }
    write_summary_csv(val_summaries, RESULTS_ROOT / "validation_summary.csv")

    selected_config = build_selected_config_json(selected_candidate_id, val_summaries[selected_candidate_id],
                                                  tie_break_used)
    (RESULTS_ROOT / "selected_rmpc_config.json").write_text(json.dumps(selected_config, indent=2, default=str))

    # ---- Stage D: post-selection diagnostic ----
    diag_rmpc_rows, diag_lead_rows, diag_times = run_stage_d_diagnostic(env_config, selected_candidate_id)
    write_episode_csv(diag_rmpc_rows, RESULTS_ROOT / "diagnostic_episodes.csv", rmpc=True)
    diagnostic_summary = _summary_for_rows(diag_rmpc_rows, diag_times, selected_candidate_id)
    write_summary_csv({selected_candidate_id: diagnostic_summary}, RESULTS_ROOT / "diagnostic_summary.csv")

    # ---- Lead reference (all stages) ----
    all_lead_rows = stage_a_lead_rows + stage_b_lead_rows + val_lead_rows + diag_lead_rows
    write_episode_csv(all_lead_rows, RESULTS_ROOT / "lead_reference_episodes.csv", rmpc=False)
    lead_summary_by_stage = {
        "stage_a_71000_71029": summarize_success(stage_a_lead_rows),
        "stage_b_71030_71099": summarize_success(stage_b_lead_rows),
        "development_combined_71000_71099": summarize_success(stage_a_lead_rows + stage_b_lead_rows),
        "validation_71100_71199": summarize_success(val_lead_rows),
        "diagnostic_71200_71299": summarize_success(diag_lead_rows),
    }
    write_summary_csv(lead_summary_by_stage, RESULTS_ROOT / "lead_reference_summary.csv")

    # ---- Manifest ----
    manifest = build_protocol_manifest(env_config)
    manifest.update({
        "stage_a_top3": stage_a_top3,
        "development_finalists": development_finalists,
        "selected_rmpc_config": selected_candidate_id,
        "tie_break_used_at_validation": tie_break_used,
    })
    (RESULTS_ROOT / "protocol_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    (RESULTS_ROOT / "candidate_configs.json").write_text(json.dumps(CANDIDATES, indent=2))

    # ---- Report ----
    write_tuning_report(
        RESULTS_ROOT / "tuning_report.md",
        stage_a_summaries=stage_a_summaries, stage_a_top3=stage_a_top3,
        stage_b_summaries=stage_b_summaries, development_finalists=development_finalists,
        validation_summaries=val_summaries, selected_candidate_id=selected_candidate_id,
        tie_break_used=tie_break_used, diagnostic_summary=diagnostic_summary,
        lead_summary_by_stage=lead_summary_by_stage,
    )

    print(f"STAGE_A_TOP3 = {stage_a_top3}")
    print(f"DEVELOPMENT_FINALISTS = {development_finalists}")
    print(f"SELECTED_RMPC_CONFIG = {selected_candidate_id} (tie_break_used={tie_break_used})")
    print(f"Artifacts written to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
