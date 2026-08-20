"""
RMPC final nominal publication evaluation: a fresh matched-seed comparison
of all 13 method instances (Greedy, PN, Lead, RMPC, PPO x3, PPO-Kalman x3,
LR-PPO x3) on seeds 72000-76999.

See `docs/rmpc_final_nominal_protocol.md` for the full predeclared protocol.
This module MUST NOT be modified once seed 72000 has been opened (protocol
Section 17).

Reuses existing canonical infrastructure verbatim wherever possible:
    - MethodInstance / build_instances (experiments.run_lead_residual_diagnostic)
    - FROZEN_MODELS / TRAINING_SEEDS / LR_TRAINING_ROOTS / bootstrap constants
      (experiments.lead_residual_expanded_final_protocol)
    - run_diagnostic_episode / run_lr_ppo_diagnostic_episode (canonical runners)
    - final_stats.py statistical helpers (wilson_interval, hierarchical
      bootstrap family, hierarchical paired bootstrap, etc.)
    - _RMPCDiagnosticsWrapper (experiments.tune_rmpc)

THIS PROMPT PRODUCES FINAL RAW NOMINAL EVIDENCE ONLY -- no manuscript file,
paper_artifacts/ entry, or publication figure is updated here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import platform
import subprocess
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd
from stable_baselines3 import PPO

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.registry import get_policy
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper
from uav_defend.policies.mpc.rmpc_policy import RMPCConfig, RMPCPolicy

from experiments.run_lead_residual_diagnostic import MethodInstance, build_instances, verify_model_hashes
from experiments.lead_residual_expanded_final_protocol import (
    FROZEN_MODELS,
    TRAINING_SEEDS,
    LR_TRAINING_ROOTS,
    RESIDUAL_MAX_ANGLE_DEG,
    BOOTSTRAP_REPLICATES,
    BOOTSTRAP_SEED,
    CONFIDENCE_LEVEL,
)
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode, sha256_of_file
from experiments.lead_residual_diagnostic_eval_utils import run_lr_ppo_diagnostic_episode
from experiments.tune_rmpc import _RMPCDiagnosticsWrapper, _post_detection_times
from experiments.final_stats import (
    wilson_interval,
    paired_bootstrap_ci,
    hierarchical_bootstrap_track,
    hierarchical_paired_bootstrap_vs_scripted,
    hierarchical_paired_bootstrap_matched_seeds,
    hierarchical_bootstrap_independent_families,
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "rmpc_final_nominal"
SELECTED_RMPC_CONFIG_PATH = PROJECT_ROOT / "results" / "rmpc_tuning" / "selected_rmpc_config.json"

RMPC_DESIGN_COMMIT = "61483f6c448ff2c3b3dfa3bd4c3fa25f18fecf07"
RMPC_IMPLEMENTATION_COMMIT = "80ea7bd"
RMPC_HARDENING_COMMIT = "cb06b04ed39d17ad64b4243c2fe84407bc95e6e4"
RMPC_TUNING_PROTOCOL_COMMIT = "dc6e29ac755e51e5d545a9ddc5bcb5d609eac864"
RMPC_TUNING_RESULTS_COMMIT = "ed4124c"
FINAL_NOMINAL_BASE_COMMIT = "ed4124c898e095f88db16906712b9e4d48c6b8c6"

# =============================================================================
# Section 9/13: seed guards -- exact, hard-enforced
# =============================================================================
FINAL_SEED_START = 72_000
FINAL_SEED_END = 76_999
FINAL_EPISODES = FINAL_SEED_END - FINAL_SEED_START + 1  # 5000
assert FINAL_EPISODES == 5000


def final_seeds() -> list[int]:
    return list(range(FINAL_SEED_START, FINAL_SEED_END + 1))


def assert_final_seed(seed: int) -> None:
    """Hard reject any seed outside [72000, 76999] -- no command-line
    override is permitted outside this exact block."""
    if not (FINAL_SEED_START <= seed <= FINAL_SEED_END):
        raise ValueError(f"seed {seed} is outside the authorized final-nominal range "
                          f"[{FINAL_SEED_START}, {FINAL_SEED_END}]")


# =============================================================================
# Section 3: frozen RMPC configuration -- loaded from disk, never duplicated
# =============================================================================
_REQUIRED_RMPC_FIELDS = dict(
    candidate_id="C1_FAST", horizon=4, population_size=64, elite_fraction=0.20,
    cem_iterations=3, controller_seed=0, cem_initial_std=1.0, cem_min_std=0.05, cem_sample_clip=3.0,
)


def load_frozen_rmpc_config() -> tuple[RMPCConfig, dict]:
    """Load results/rmpc_tuning/selected_rmpc_config.json, verify it is
    frozen and matches the exact predeclared values, and construct the
    corresponding RMPCConfig FROM THIS FILE (never a second, independently
    editable hardcoded copy)."""
    if not SELECTED_RMPC_CONFIG_PATH.exists():
        raise FileNotFoundError(f"frozen RMPC config not found: {SELECTED_RMPC_CONFIG_PATH}")
    data = json.loads(SELECTED_RMPC_CONFIG_PATH.read_text())

    if data.get("frozen_for_final_evaluation") is not True:
        raise RuntimeError("selected_rmpc_config.json: frozen_for_final_evaluation is not true -- STOP")

    for key, expected in _REQUIRED_RMPC_FIELDS.items():
        actual = data.get(key)
        if actual != expected:
            raise RuntimeError(
                f"selected_rmpc_config.json field '{key}' = {actual!r}, expected {expected!r} -- STOP"
            )

    rmpc_config = RMPCConfig(
        horizon=data["horizon"],
        population_size=data["population_size"],
        elite_fraction=data["elite_fraction"],
        cem_iterations=data["cem_iterations"],
        controller_seed=data["controller_seed"],
        cem_initial_std=data["cem_initial_std"],
        cem_min_std=data["cem_min_std"],
        cem_sample_clip=data["cem_sample_clip"],
        enable_timing=True,
    )
    return rmpc_config, data


# =============================================================================
# Section 5: thirteen method instances
# =============================================================================
def build_final_instances() -> tuple[MethodInstance, ...]:
    """The existing 12-instance builder, verbatim, plus exactly one new RMPC
    instance. RMPC is ONE fixed deterministic configuration -- never treated
    as having multiple training roots."""
    instances = list(build_instances())
    assert len(instances) == 12
    instances.append(MethodInstance("rmpc", "RMPC", "rmpc", None))
    return tuple(instances)


def instance_id(instance: MethodInstance) -> str:
    """Stable per-instance identifier used for episode CSV filenames and
    instance_summary.csv rows, e.g. 'ppo_seed45', 'rmpc', 'lead'."""
    if instance.training_seed is not None:
        return f"{instance.name}_seed{instance.training_seed}"
    return instance.name


REPORT_FAMILY_OF_INSTANCE_NAME = {
    "greedy": "Greedy", "pn": "PN", "lead": "Lead", "rmpc": "RMPC",
    "ppo": "PPO", "ppo_kalman": "PPO-Kalman", "lr_ppo": "LR-PPO",
}
DETERMINISTIC_FAMILIES = ("Greedy", "PN", "Lead", "RMPC")
LEARNED_FAMILIES = ("PPO", "PPO-Kalman", "LR-PPO")

# =============================================================================
# Section 6/8: nominal environment configs (identical to the frozen paper campaign)
# =============================================================================
_CFG_DIRECT = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)
_CFG_KALMAN = EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True)


def env_config_for_instance(instance: MethodInstance) -> EnvConfig:
    return _CFG_KALMAN if instance.family == "ppo_kalman" else _CFG_DIRECT


# =============================================================================
# Section 5/10: policy + environment construction per instance
# =============================================================================
_INSTANCES_BY_ID: dict[str, MethodInstance] = {}
_FROZEN_RMPC_CONFIG: RMPCConfig | None = None


def _register_instances(instances: tuple[MethodInstance, ...], rmpc_config: RMPCConfig) -> None:
    global _FROZEN_RMPC_CONFIG
    _FROZEN_RMPC_CONFIG = rmpc_config
    for inst in instances:
        _INSTANCES_BY_ID[instance_id(inst)] = inst


def build_policy_and_env(instance: MethodInstance):
    if instance.family == "scripted":
        env = SoldierEnv(config=_CFG_DIRECT)
        policy = get_policy(instance.name)
    elif instance.family == "rmpc":
        env = SoldierEnv(config=_CFG_DIRECT)
        policy = RMPCPolicy(config=_CFG_DIRECT, rmpc_config=_FROZEN_RMPC_CONFIG)
    elif instance.family == "ppo":
        env = SoldierEnv(config=_CFG_DIRECT)
        model_path = FROZEN_MODELS[("direct", instance.training_seed)]["path"]
        policy = get_policy("ppo", model_path=str(model_path), deterministic=True)
    elif instance.family == "ppo_kalman":
        env = SoldierEnv(config=_CFG_KALMAN)
        model_path = FROZEN_MODELS[("kalman", instance.training_seed)]["path"]
        policy = get_policy("ppo_kalman", model_path=str(model_path), deterministic=True)
    elif instance.family == "lr_ppo":
        env = SoldierEnv(config=_CFG_DIRECT)
        model_path = FROZEN_MODELS[("lr_ppo", instance.training_seed)]["path"]
        model = PPO.load(str(model_path), device="cpu")
        policy = LeadResidualPPOPolicyWrapper(
            model, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG, config=_CFG_DIRECT, deterministic=True,
        )
    else:
        raise ValueError(instance.family)
    return env, policy


# =============================================================================
# Episode execution (single episode, one instance, one seed)
# =============================================================================
def run_final_nominal_episode(instance: MethodInstance, seed: int, record_trajectory: bool = False) -> dict:
    env, policy = build_policy_and_env(instance)
    dt = env.config.dt

    if instance.family == "rmpc":
        wrapped = _RMPCDiagnosticsWrapper(policy)
        n_before = len(policy.decision_times_s)
        row = run_diagnostic_episode(env, wrapped, seed=seed, dt=dt, record_trajectory=record_trajectory)
        env.close()
        row["optimizer_fallback_count"] = wrapped.guidance_mode_counts.get("optimizer_fallback", 0)
        row["measurement_fallback_count"] = wrapped.guidance_mode_counts.get("measurement_fallback", 0)
        row["lead_init_count"] = wrapped.initialization_mode_counts.get("lead", 0)
        row["pursuit_init_count"] = wrapped.initialization_mode_counts.get("pursuit", 0)
        row["n_policy_calls"] = wrapped.n_calls
        post_times = _post_detection_times(n_before, policy.decision_times_s, row["detection_step"])
        row["n_post_detection_decisions"] = len(post_times)
        # NOT VALID FOR PUBLICATION RUNTIME COMPARISON (Section 16/22) -- diagnostic only.
        row["_diagnostic_planning_times_not_for_publication"] = post_times
    elif instance.family == "lr_ppo":
        row = run_lr_ppo_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=record_trajectory)
        env.close()
    else:
        row = run_diagnostic_episode(env, policy, seed=seed, dt=dt, record_trajectory=record_trajectory)
        env.close()

    row["instance_id"] = instance_id(instance)
    row["instance_name"] = instance.name
    row["family"] = REPORT_FAMILY_OF_INSTANCE_NAME[instance.name]
    row["training_seed"] = instance.training_seed
    return row


# =============================================================================
# Parallel execution (Section 16/21) -- controlled, fixed worker count
# =============================================================================
def _episode_task(args):
    instance_id_str, seed = args
    instance = _INSTANCES_BY_ID[instance_id_str]
    return run_final_nominal_episode(instance, seed)


def run_instance_parallel(instance: MethodInstance, seeds: list[int], max_workers: int) -> list[dict]:
    for s in seeds:
        assert_final_seed(s)
    args = [(instance_id(instance), seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init,
                              initargs=(_FROZEN_RMPC_CONFIG,)) as executor:
        return list(executor.map(_episode_task, args))


def _worker_init(rmpc_config: RMPCConfig) -> None:
    """Each worker process needs _INSTANCES_BY_ID/_FROZEN_RMPC_CONFIG populated
    independently (fresh interpreter under spawn)."""
    global _FROZEN_RMPC_CONFIG
    _FROZEN_RMPC_CONFIG = rmpc_config
    for inst in build_final_instances():
        _INSTANCES_BY_ID[instance_id(inst)] = inst


# =============================================================================
# Fairness audit (Section 14)
# =============================================================================
_TRAJECTORY_FIELDS = ("soldier_pos", "defender_pos", "defender_vel", "enemy_pos", "enemy_vel", "enemy_detected")
FAIRNESS_AUDIT_SEEDS = (72_000, 72_001, 72_002)
FAIRNESS_AUDIT_REPRESENTATIVE_IDS = (
    "greedy", "pn", "lead", "rmpc", "ppo_seed45", "ppo_kalman_seed45", "lr_ppo_seed55",
)


def _trajectories_equal(traj_a: list[dict], traj_b: list[dict]) -> bool:
    if len(traj_a) != len(traj_b):
        return False
    for step_a, step_b in zip(traj_a, traj_b):
        for field in _TRAJECTORY_FIELDS:
            va, vb = step_a[field], step_b[field]
            if isinstance(va, np.ndarray):
                if not np.array_equal(va, vb):
                    return False
            elif va != vb:
                return False
    return True


def run_fairness_trajectory_audit() -> dict:
    """Section 14: full pre-detection trajectory equality for one
    representative per controller family on seeds 72000-72002."""
    results: dict[str, dict] = {}
    reference_id = FAIRNESS_AUDIT_REPRESENTATIVE_IDS[0]
    for seed in FAIRNESS_AUDIT_SEEDS:
        rows = {}
        for inst_id in FAIRNESS_AUDIT_REPRESENTATIVE_IDS:
            instance = _INSTANCES_BY_ID[inst_id]
            rows[inst_id] = run_final_nominal_episode(instance, seed, record_trajectory=True)
        ref_traj = rows[reference_id]["pre_detection_trajectory"]
        ref_detection_step = rows[reference_id]["detection_step"]
        for inst_id, row in rows.items():
            match_traj = _trajectories_equal(ref_traj, row["pre_detection_trajectory"])
            match_detect = row["detection_step"] == ref_detection_step
            results[f"seed_{seed}_{inst_id}"] = {
                "trajectory_matches_reference": match_traj,
                "detection_step_matches_reference": match_detect,
            }
            if not match_traj or not match_detect:
                raise RuntimeError(f"FAIRNESS AUDIT VIOLATION at seed {seed}, instance {inst_id}")
    return results


def assert_detection_step_equality_all_instances(episodes_by_instance: dict[str, pd.DataFrame]) -> int:
    """Section 14: detection_step must be identical across ALL 13 instances
    for every environment seed. Returns the number of comparisons made."""
    reference_id = list(episodes_by_instance)[0]
    reference = episodes_by_instance[reference_id].set_index("environment_seed")["detection_step"]
    n_checked = 0
    for inst_id, df in episodes_by_instance.items():
        other = df.set_index("environment_seed")["detection_step"]
        aligned = reference.reindex(other.index)
        mismatches = (aligned != other) & ~(aligned.isna() & other.isna())
        if mismatches.any():
            bad_seeds = other.index[mismatches].tolist()[:5]
            raise RuntimeError(f"DETECTION-STEP FAIRNESS VIOLATION: {inst_id} vs {reference_id} at seeds {bad_seeds}")
        n_checked += len(other)
    return n_checked


# =============================================================================
# Statistics (Section 11-13)
# =============================================================================
def resolution_label(ci_low: float, ci_high: float) -> str:
    if ci_low > 0:
        return "resolved positive"
    if ci_high < 0:
        return "resolved negative"
    return "unresolved"


def summarize_instance(df: pd.DataFrame) -> dict:
    n = len(df)
    successes = int(df["success"].sum())
    lo, hi = wilson_interval(successes, n)
    post_detect = df["post_detection_steps"].dropna()
    intercept_time = df.loc[df["outcome"] == "intercepted", "intercept_time_seconds"]
    return {
        "n": n,
        "successes": successes,
        "success_rate": successes / n,
        "success_ci_low": lo,
        "success_ci_high": hi,
        "soldier_caught_rate": float(df["soldier_caught"].mean()),
        "unsafe_intercept_rate": float(df["unsafe_intercept"].mean()),
        "timeout_rate": float(df["timeout"].mean()),
        "mean_post_detection_steps": float(post_detect.mean()) if len(post_detect) else None,
        "mean_intercept_time_if_success": float(intercept_time.mean()) if len(intercept_time) else None,
    }


# =============================================================================
# Platform / provenance metadata (Section 24)
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


SOURCE_FILES_TO_HASH = {
    "soldier_env.py": PROJECT_ROOT / "uav_defend" / "envs" / "soldier_env.py",
    "constrained_point_mass.py": PROJECT_ROOT / "uav_defend" / "dynamics" / "constrained_point_mass.py",
    "sanitize.py": PROJECT_ROOT / "uav_defend" / "policies" / "sanitize.py",
    "greedy_intercept_policy.py": PROJECT_ROOT / "uav_defend" / "policies" / "baseline" / "greedy_intercept_policy.py",
    "proportional_navigation_policy.py": PROJECT_ROOT / "uav_defend" / "policies" / "baseline" / "proportional_navigation_policy.py",
    "lead_intercept_policy.py": PROJECT_ROOT / "uav_defend" / "policies" / "baseline" / "lead_intercept_policy.py",
    "ppo_policy_wrapper.py": PROJECT_ROOT / "uav_defend" / "policies" / "rl" / "ppo_policy_wrapper.py",
    "ppo_kalman_policy_wrapper.py": PROJECT_ROOT / "uav_defend" / "policies" / "rl_kalman" / "ppo_kalman_policy_wrapper.py",
    "lead_residual_ppo_policy_wrapper.py": PROJECT_ROOT / "uav_defend" / "policies" / "residual" / "lead_residual_ppo_policy_wrapper.py",
    "rmpc_policy.py": PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "rmpc_policy.py",
    "rmpc_cost.py": PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "rmpc_cost.py",
    "hostile_scenarios.py": PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "hostile_scenarios.py",
    "cem_optimizer.py": PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "cem_optimizer.py",
    "lead_init.py": PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "lead_init.py",
}


def source_file_hashes() -> dict[str, str]:
    return {name: sha256_of_file(path) for name, path in SOURCE_FILES_TO_HASH.items()}


def platform_metadata() -> dict:
    return {
        "os": platform.platform(),
        "cpu": platform.processor() or platform.machine(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
    }


# =============================================================================
# Completeness check (Section 19)
# =============================================================================
_VALID_OUTCOMES = {"intercepted", "soldier_caught", "unsafe_intercept", "timeout"}


def assert_instance_completeness(inst_id: str, df: pd.DataFrame) -> None:
    if len(df) != FINAL_EPISODES:
        raise RuntimeError(f"{inst_id}: expected {FINAL_EPISODES} rows, got {len(df)}")
    seeds = set(df["environment_seed"])
    expected_seeds = set(final_seeds())
    if seeds != expected_seeds:
        missing = expected_seeds - seeds
        extra = seeds - expected_seeds
        raise RuntimeError(f"{inst_id}: seed set mismatch (missing={list(missing)[:5]}, extra={list(extra)[:5]})")
    if df["environment_seed"].duplicated().any():
        raise RuntimeError(f"{inst_id}: duplicate seed(s) present")
    unknown = set(df["outcome"]) - _VALID_OUTCOMES
    if unknown:
        raise RuntimeError(f"{inst_id}: unknown outcome categor{'y' if len(unknown)==1 else 'ies'} {unknown}")
    outcome_sum = df["success"].sum() + df["soldier_caught"].sum() + df["unsafe_intercept"].sum() + df["timeout"].sum()
    if outcome_sum != FINAL_EPISODES:
        raise RuntimeError(f"{inst_id}: terminal-outcome counts sum to {outcome_sum}, expected {FINAL_EPISODES}")


# =============================================================================
# Statistics helpers for family/paired-difference computation
# =============================================================================
def success_vector(df: pd.DataFrame) -> np.ndarray:
    return df.sort_values("environment_seed")["success"].to_numpy(dtype=np.float64)


def success_matrix(dfs_in_root_order: list[pd.DataFrame]) -> np.ndarray:
    return np.stack([success_vector(df) for df in dfs_in_root_order], axis=0)


def build_family_summary(episodes: dict[str, pd.DataFrame]) -> list[dict]:
    rows = []
    for family in DETERMINISTIC_FAMILIES:
        inst_id = {"Greedy": "greedy", "PN": "pn", "Lead": "lead", "RMPC": "rmpc"}[family]
        rate = float(episodes[inst_id]["success"].mean())
        lo, hi = wilson_interval(int(episodes[inst_id]["success"].sum()), len(episodes[inst_id]))
        rows.append(dict(family=family, n_roots=1, per_root_rates=json.dumps([rate]),
                          family_mean_success_rate=rate, root_sd=0.0,
                          hierarchical_ci_low=lo, hierarchical_ci_high=hi))
    for family, name, roots in (("PPO", "ppo", TRAINING_SEEDS), ("PPO-Kalman", "ppo_kalman", TRAINING_SEEDS),
                                 ("LR-PPO", "lr_ppo", LR_TRAINING_ROOTS)):
        dfs = [episodes[f"{name}_seed{r}"] for r in roots]
        mat = success_matrix(dfs)
        per_root = mat.mean(axis=1).tolist()
        observed, lo, hi = hierarchical_bootstrap_track(mat, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
        rows.append(dict(family=family, n_roots=len(roots), per_root_rates=json.dumps(per_root),
                          family_mean_success_rate=observed, root_sd=float(np.std(per_root, ddof=1)),
                          hierarchical_ci_low=lo, hierarchical_ci_high=hi))
    return rows


def build_paired_differences(episodes: dict[str, pd.DataFrame]) -> list[dict]:
    lead_vec = success_vector(episodes["lead"])
    greedy_vec = success_vector(episodes["greedy"])
    pn_vec = success_vector(episodes["pn"])
    rmpc_vec = success_vector(episodes["rmpc"])
    ppo_mat = success_matrix([episodes[f"ppo_seed{s}"] for s in TRAINING_SEEDS])
    ppo_kalman_mat = success_matrix([episodes[f"ppo_kalman_seed{s}"] for s in TRAINING_SEEDS])
    lr_ppo_mat = success_matrix([episodes[f"lr_ppo_seed{s}"] for s in LR_TRAINING_ROOTS])

    rows = []

    def _add(label, observed, lo, hi):
        diff_pp = observed * 100.0
        lo_pp, hi_pp = lo * 100.0, hi * 100.0
        rows.append(dict(comparison=label, difference_pp=diff_pp, ci_low_pp=lo_pp, ci_high_pp=hi_pp,
                          statistical_resolution=resolution_label(lo_pp, hi_pp)))

    # Primary (Section 13/16/27)
    obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_ppo_mat, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("LR-PPO - Lead", obs, lo, hi)

    obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(lr_ppo_mat, rmpc_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("LR-PPO - RMPC", obs, lo, hi)

    obs, lo, hi = hierarchical_bootstrap_independent_families(lr_ppo_mat, ppo_mat, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("LR-PPO - PPO", obs, lo, hi)

    obs, lo, hi = paired_bootstrap_ci(rmpc_vec, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("RMPC - Lead", obs, lo, hi)

    obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_mat, rmpc_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("RMPC - PPO", -obs, -hi, -lo)  # flip sign: helper computes (PPO - RMPC)

    # Secondary (Section 16/27)
    obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_mat, greedy_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("PPO - Greedy", obs, lo, hi)

    obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_mat, pn_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("PPO - PN", obs, lo, hi)

    obs, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_mat, lead_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("PPO - Lead", obs, lo, hi)

    obs, lo, hi, _per_seed = hierarchical_paired_bootstrap_matched_seeds(ppo_kalman_mat, ppo_mat, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED, CONFIDENCE_LEVEL)
    _add("PPO-Kalman - PPO", obs, lo, hi)

    return rows


# =============================================================================
# Manifest builders (Section 24)
# =============================================================================
def build_model_manifest() -> dict:
    hash_results = verify_model_hashes()
    manifest = {}
    for (track, seed), spec in FROZEN_MODELS.items():
        family = {"direct": "PPO", "kalman": "PPO-Kalman", "lr_ppo": "LR-PPO"}[track]
        manifest[f"{track}_{seed}"] = {
            "family": family,
            "training_root": seed,
            "model_path": str(spec["path"]),
            "sha256": spec["sha256"],
            "verified_match": hash_results[f"{track}_{seed}"]["match"],
        }
    return manifest


def build_protocol_manifest(rmpc_config: RMPCConfig, rmpc_config_data: dict, worker_count: int,
                             start_time: str, end_time: str) -> dict:
    return {
        "repository": "mikemishal/RL-UAV-V2",
        "branch": _git_branch(),
        "rmpc_final_nominal_protocol_commit": _git_commit(),
        "rmpc_design_commit": RMPC_DESIGN_COMMIT,
        "rmpc_implementation_commit": RMPC_IMPLEMENTATION_COMMIT,
        "rmpc_hardening_commit": RMPC_HARDENING_COMMIT,
        "rmpc_tuning_protocol_commit": RMPC_TUNING_PROTOCOL_COMMIT,
        "rmpc_tuning_results_commit": RMPC_TUNING_RESULTS_COMMIT,
        "final_nominal_base_commit": FINAL_NOMINAL_BASE_COMMIT,
        "selected_rmpc_config_hash": sha256_of_file(SELECTED_RMPC_CONFIG_PATH),
        "selected_rmpc_config": rmpc_config_data,
        "rmpc_config_constructed": dataclasses.asdict(rmpc_config),
        "source_file_hashes": source_file_hashes(),
        "env_config_direct": dataclasses.asdict(_CFG_DIRECT),
        "env_config_kalman": dataclasses.asdict(_CFG_KALMAN),
        "seed_range": [FINAL_SEED_START, FINAL_SEED_END],
        "n_unique_seeds": FINAL_EPISODES,
        "method_instances": [
            {"instance_id": instance_id(i), "name": i.name, "display_name": i.display_name,
             "family": i.family, "training_seed": i.training_seed,
             "estimator_mode": "kalman" if i.family == "ppo_kalman" else "measurement"}
            for i in build_final_instances()
        ],
        "primary_endpoint": "safe_interception_success (outcome == 'intercepted')",
        "statistical_methods": {
            "single_instance_ci": "wilson_interval",
            "family_ci": "hierarchical_bootstrap_track",
            "family_vs_scripted_diff": "hierarchical_paired_bootstrap_vs_scripted",
            "independent_families_diff": "hierarchical_bootstrap_independent_families",
            "matched_seeds_diff": "hierarchical_paired_bootstrap_matched_seeds",
            "deterministic_pair_diff": "paired_bootstrap_ci",
            "bootstrap_replicates": BOOTSTRAP_REPLICATES,
            "bootstrap_seed": BOOTSTRAP_SEED,
            "confidence_level": CONFIDENCE_LEVEL,
        },
        "worker_count": worker_count,
        "run_start_utc": start_time,
        "run_end_utc": end_time,
        "platform": platform_metadata(),
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "disclaimer": "FINAL RAW NOMINAL EVIDENCE -- statistical results only; RMPC decision "
                       "times from this parallel run are diagnostic-only and NOT VALID FOR "
                       "PUBLICATION RUNTIME COMPARISON.",
    }


# =============================================================================
# CSV writing / numerical audit / RMPC diagnostics
# =============================================================================
_SCALAR_DROP_KEYS = ("step_angles", "pre_detection_trajectory", "_diagnostic_planning_times_not_for_publication")


def _clean_rows_for_csv(rows: list[dict]) -> pd.DataFrame:
    cleaned = [{k: v for k, v in row.items() if k not in _SCALAR_DROP_KEYS} for row in rows]
    return pd.DataFrame(cleaned)


def numerical_audit(episodes: dict[str, pd.DataFrame]) -> dict:
    audit = {}
    total_violations = 0
    total_nan_unexpected = 0
    total_inf = 0
    for inst_id, df in episodes.items():
        numeric = df.select_dtypes(include=[np.number])
        inf_count = int(np.isinf(numeric.to_numpy()).sum())
        violations = int(df["physical_limit_violations"].sum())
        # Expected NaNs: intercept_step/intercept_time_seconds for non-"intercepted"
        # outcomes, and tracking columns when n_tracking_samples == 0.
        non_intercepted = (df["outcome"] != "intercepted").sum()
        expected_nan = non_intercepted * 2  # intercept_step + intercept_time_seconds
        no_tracking = (df["n_tracking_samples"] == 0).sum()
        expected_nan += no_tracking * 3  # mean/final_tracking_error + tracking_rmse
        actual_nan = int(numeric.isna().sum().sum())
        unexpected_nan = max(0, actual_nan - expected_nan)
        audit[inst_id] = {
            "physical_limit_violations": violations,
            "inf_count": inf_count,
            "nan_count": actual_nan,
            "expected_nan_count": int(expected_nan),
            "unexpected_nan_count": int(unexpected_nan),
        }
        total_violations += violations
        total_nan_unexpected += unexpected_nan
        total_inf += inf_count
    audit["_totals"] = {
        "physical_limit_violations": total_violations,
        "unexpected_nan_count": total_nan_unexpected,
        "inf_count": total_inf,
    }
    if total_violations > 0 or total_nan_unexpected > 0 or total_inf > 0:
        raise RuntimeError(f"NUMERICAL AUDIT FAILURE: {audit['_totals']}")
    return audit


def rmpc_diagnostics_summary(rmpc_df: pd.DataFrame, rmpc_raw_rows: list[dict]) -> dict:
    all_times = []
    for row in rmpc_raw_rows:
        all_times.extend(row.get("_diagnostic_planning_times_not_for_publication", []))
    arr = np.asarray(all_times, dtype=np.float64) if all_times else np.array([])
    return {
        "optimizer_fallback_count_total": int(rmpc_df["optimizer_fallback_count"].sum()),
        "measurement_fallback_count_total": int(rmpc_df["measurement_fallback_count"].sum()),
        "lead_init_count_total": int(rmpc_df["lead_init_count"].sum()),
        "pursuit_init_count_total": int(rmpc_df["pursuit_init_count"].sum()),
        "n_policy_calls_total": int(rmpc_df["n_policy_calls"].sum()),
        "n_post_detection_decisions_total": int(rmpc_df["n_post_detection_decisions"].sum()),
        "planning_time_diagnostic_NOT_FOR_PUBLICATION": {
            "n": int(arr.size),
            "mean": float(np.mean(arr)) if arr.size else None,
            "median": float(np.median(arr)) if arr.size else None,
            "p95": float(np.percentile(arr, 95)) if arr.size else None,
            "max": float(np.max(arr)) if arr.size else None,
        },
    }


def write_final_nominal_report(report_path: Path, *, instance_summary: pd.DataFrame,
                                family_summary: pd.DataFrame, paired_differences: pd.DataFrame) -> None:
    lines = [
        "# RMPC Final Nominal Publication Evaluation Report",
        "",
        "**FINAL RAW NOMINAL EVIDENCE -- not yet integrated into the manuscript,",
        "paper_artifacts/, or publication figures.**",
        "",
        "## Instance results", "",
        "| Instance | Success/n | Success % | 95% CI |", "|---|---|---|---|",
    ]
    for _, r in instance_summary.iterrows():
        lines.append(f"| {r['instance']} | {int(r['successes'])}/{int(r['n'])} | {r['success_rate']*100:.2f}% | "
                      f"[{r['success_ci_low']*100:.2f}%, {r['success_ci_high']*100:.2f}%] |")
    lines += ["", "## Family results", "", "| Family | Success % | Root SD | 95% interval |", "|---|---|---|---|"]
    for _, r in family_summary.iterrows():
        lines.append(f"| {r['family']} | {r['family_mean_success_rate']*100:.2f}% | {r['root_sd']*100:.2f}pp | "
                      f"[{r['hierarchical_ci_low']*100:.2f}%, {r['hierarchical_ci_high']*100:.2f}%] |")
    lines += ["", "## Paired differences", "", "| Comparison | Diff (pp) | 95% CI (pp) | Resolution |", "|---|---|---|---|"]
    for _, r in paired_differences.iterrows():
        lines.append(f"| {r['comparison']} | {r['difference_pp']:.2f} | "
                      f"[{r['ci_low_pp']:.2f}, {r['ci_high_pp']:.2f}] | {r['statistical_resolution']} |")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)
    (RESULTS_ROOT / "episodes").mkdir(parents=True, exist_ok=True)

    rmpc_config, rmpc_config_data = load_frozen_rmpc_config()
    instances = build_final_instances()
    _register_instances(instances, rmpc_config)

    model_hash_results = verify_model_hashes()  # raises on mismatch
    print("Model hashes verified OK for all 9 frozen checkpoints.", flush=True)

    print("Running pre-detection fairness trajectory audit (seeds 72000-72002)...", flush=True)
    fairness_trajectory_results = run_fairness_trajectory_audit()
    print("Fairness trajectory audit passed.", flush=True)

    seeds = final_seeds()
    worker_count = DEFAULT_MAX_WORKERS
    start_time = datetime.now(timezone.utc).isoformat()

    episodes: dict[str, pd.DataFrame] = {}
    raw_rows_by_instance: dict[str, list[dict]] = {}
    for i, instance in enumerate(instances):
        inst_id = instance_id(instance)
        print(f"[{i+1}/{len(instances)}] {inst_id} ({instance.family}) -- {len(seeds)} episodes "
              f"(workers={worker_count})...", flush=True)
        rows = run_instance_parallel(instance, seeds, worker_count)
        raw_rows_by_instance[inst_id] = rows
        df = _clean_rows_for_csv(rows)
        assert_instance_completeness(inst_id, df)
        df.to_csv(RESULTS_ROOT / "episodes" / f"{inst_id}.csv", index=False)
        episodes[inst_id] = df
        print(f"    {inst_id}: {int(df['success'].sum())}/{len(df)} intercepted", flush=True)

    end_time = datetime.now(timezone.utc).isoformat()

    print("Running detection-step fairness audit across all 13 instances...", flush=True)
    n_fairness_checks = assert_detection_step_equality_all_instances(episodes)
    print(f"Detection-step fairness OK across {n_fairness_checks} seed x instance comparisons.", flush=True)

    print("Running numerical audit...", flush=True)
    num_audit = numerical_audit(episodes)
    (RESULTS_ROOT / "numerical_audit.json").write_text(json.dumps(num_audit, indent=2, default=str))

    fairness_audit = {
        "detection_step_equality_all_instances": True,
        "n_detection_step_comparisons": n_fairness_checks,
        "trajectory_audit_seeds": list(FAIRNESS_AUDIT_SEEDS),
        "trajectory_audit_representative_instances": list(FAIRNESS_AUDIT_REPRESENTATIVE_IDS),
        "trajectory_audit_results": fairness_trajectory_results,
    }
    (RESULTS_ROOT / "fairness_audit.json").write_text(json.dumps(fairness_audit, indent=2, default=str))

    print("Computing statistical summaries...", flush=True)
    instance_rows = []
    for instance in instances:
        inst_id = instance_id(instance)
        summary = summarize_instance(episodes[inst_id])
        instance_rows.append(dict(instance=inst_id, family=REPORT_FAMILY_OF_INSTANCE_NAME[instance.name],
                                   training_seed=instance.training_seed, **summary))
    instance_summary = pd.DataFrame(instance_rows)
    instance_summary.to_csv(RESULTS_ROOT / "instance_summary.csv", index=False)

    family_summary = pd.DataFrame(build_family_summary(episodes))
    family_summary.to_csv(RESULTS_ROOT / "family_summary.csv", index=False)

    paired_differences = pd.DataFrame(build_paired_differences(episodes))
    paired_differences.to_csv(RESULTS_ROOT / "paired_differences.csv", index=False)

    rmpc_diag = rmpc_diagnostics_summary(episodes["rmpc"], raw_rows_by_instance["rmpc"])
    (RESULTS_ROOT / "rmpc_diagnostics_NOT_FOR_PUBLICATION.json").write_text(json.dumps(rmpc_diag, indent=2, default=str))

    model_manifest = build_model_manifest()
    (RESULTS_ROOT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, default=str))

    (RESULTS_ROOT / "selected_rmpc_config_snapshot.json").write_text(json.dumps(rmpc_config_data, indent=2, default=str))

    protocol_manifest = build_protocol_manifest(rmpc_config, rmpc_config_data, worker_count, start_time, end_time)
    (RESULTS_ROOT / "protocol_manifest.json").write_text(json.dumps(protocol_manifest, indent=2, default=str))

    write_final_nominal_report(RESULTS_ROOT / "final_nominal_report.md", instance_summary=instance_summary,
                                family_summary=family_summary, paired_differences=paired_differences)

    print("\n=== FINAL NOMINAL CAMPAIGN COMPLETE ===")
    print(instance_summary[["instance", "family", "n", "successes", "success_rate"]].to_string(index=False))
    print(f"\nArtifacts written to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()

