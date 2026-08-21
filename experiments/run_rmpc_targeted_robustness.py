"""
RMPC targeted robustness publication evaluation: five predeclared
off-nominal conditions (strong reactive evasion, restrictive defender
maneuverability, high measurement noise, hard mobility, short detection
range), each evaluated on the SAME 1000 environment seeds (77000-77999),
for all 13 method instances (Greedy, PN, Lead, RMPC, PPO x3, PPO-Kalman x3,
LR-PPO x3).

See `docs/rmpc_targeted_robustness_protocol.md` for the full predeclared
protocol. This module MUST NOT be modified once seed 77000 has been opened
(protocol Section 18).

Reuses existing canonical infrastructure verbatim wherever possible,
including helpers from the completed final-nominal campaign
(`experiments.run_rmpc_final_nominal`) and the expanded-robustness
config-isolation helper (`experiments.expanded_robustness_protocol`).

THIS PROMPT PRODUCES FINAL RAW ROBUSTNESS EVIDENCE ONLY -- no manuscript
file, paper_artifacts/ entry, or publication figure is updated here.
"""

from __future__ import annotations

import dataclasses
import json
import os
import sys

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")

from concurrent.futures import ProcessPoolExecutor
from datetime import datetime, timezone
from pathlib import Path
from typing import Callable

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

from experiments.run_lead_residual_diagnostic import MethodInstance, verify_model_hashes
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
from experiments.expanded_robustness_protocol import assert_only_fields_differ

# Reused verbatim from the completed, source-frozen final-nominal campaign.
from experiments.run_rmpc_final_nominal import (
    build_final_instances,
    instance_id,
    REPORT_FAMILY_OF_INSTANCE_NAME,
    DETERMINISTIC_FAMILIES,
    LEARNED_FAMILIES,
    _CFG_DIRECT as NOMINAL_CFG_DIRECT,
    _CFG_KALMAN as NOMINAL_CFG_KALMAN,
    load_frozen_rmpc_config,
    _REQUIRED_RMPC_FIELDS,
    SELECTED_RMPC_CONFIG_PATH,
    resolution_label,
    summarize_instance,
    success_vector,
    success_matrix,
    build_family_summary,
    build_paired_differences,
    build_model_manifest,
    source_file_hashes,
    platform_metadata,
    _git_commit,
    _git_branch,
    _clean_rows_for_csv,
    numerical_audit,
    rmpc_diagnostics_summary,
    assert_detection_step_equality_all_instances,
    _trajectories_equal,
    RMPC_DESIGN_COMMIT,
    RMPC_IMPLEMENTATION_COMMIT,
    RMPC_HARDENING_COMMIT,
    RMPC_TUNING_PROTOCOL_COMMIT,
    RMPC_TUNING_RESULTS_COMMIT,
    FINAL_NOMINAL_BASE_COMMIT,
    RESULTS_ROOT as FINAL_NOMINAL_RESULTS_ROOT,
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "rmpc_targeted_robustness"
FINAL_NOMINAL_PROTOCOL_COMMIT = "d017d32"
FINAL_NOMINAL_RESULTS_COMMIT = "b0f952f"
ROBUSTNESS_BASE_COMMIT = "b0f952fbff4a0596239f178222f398c7600f9526"

# =============================================================================
# Section 9/10: seed guards -- exact, hard-enforced
# =============================================================================
ROBUSTNESS_SEED_START = 77_000
ROBUSTNESS_SEED_END = 77_999
ROBUSTNESS_EPISODES = ROBUSTNESS_SEED_END - ROBUSTNESS_SEED_START + 1  # 1000
assert ROBUSTNESS_EPISODES == 1000


def robustness_seeds() -> list[int]:
    return list(range(ROBUSTNESS_SEED_START, ROBUSTNESS_SEED_END + 1))


def assert_robustness_seed(seed: int) -> None:
    """Hard reject any seed outside [77000, 77999] -- no command-line
    override is permitted outside this exact block."""
    if not (ROBUSTNESS_SEED_START <= seed <= ROBUSTNESS_SEED_END):
        raise ValueError(f"seed {seed} is outside the authorized robustness range "
                          f"[{ROBUSTNESS_SEED_START}, {ROBUSTNESS_SEED_END}]")


# =============================================================================
# Section 8: five predeclared robustness conditions
# =============================================================================
def _diff_keys(cfg: EnvConfig, nominal: EnvConfig) -> set[str]:
    """Programmatically derive the set of EnvConfig fields that differ from
    the matching (Direct/Kalman) canonical nominal reference."""
    cfg_d = dataclasses.asdict(cfg)
    nom_d = dataclasses.asdict(nominal)
    return {k for k in cfg_d if cfg_d[k] != nom_d[k]}


def _build_condition_configs(overrides: dict) -> tuple[EnvConfig, EnvConfig, set[str]]:
    direct = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True, **overrides)
    kalman = EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True, **overrides)
    diff_d = _diff_keys(direct, NOMINAL_CFG_DIRECT)
    diff_k = _diff_keys(kalman, NOMINAL_CFG_KALMAN)
    assert diff_d == diff_k, f"direct/kalman changed-field-set mismatch: {diff_d} vs {diff_k}"
    assert_only_fields_differ(direct, diff_d)
    assert_only_fields_differ(kalman, diff_k)
    return direct, kalman, diff_d


def build_strong_evasion_configs() -> tuple[EnvConfig, EnvConfig, set[str]]:
    direct, kalman, diff = _build_condition_configs({"enemy_evasion_gain": 1.00})
    assert diff == {"enemy_evasion_gain"}, f"R1 changed-field set unexpected: {diff}"
    return direct, kalman, diff


def build_restrictive_dynamics_configs() -> tuple[EnvConfig, EnvConfig, set[str]]:
    direct, kalman, diff = _build_condition_configs(
        {"defender_max_accel": 4.0, "defender_max_turn_rate_deg": 45.0}
    )
    assert diff == {"defender_max_accel", "defender_max_turn_rate_deg"}, f"R2 changed-field set unexpected: {diff}"
    return direct, kalman, diff


def build_high_measurement_noise_configs() -> tuple[EnvConfig, EnvConfig, set[str]]:
    direct, kalman, diff = _build_condition_configs({"measurement_var": 4.0})
    assert diff == {"measurement_var"}, f"R3 changed-field set unexpected: {diff}"
    # Not retuning Kalman: process_var must remain identical to the nominal value.
    assert direct.process_var == NOMINAL_CFG_DIRECT.process_var
    assert kalman.process_var == NOMINAL_CFG_KALMAN.process_var
    return direct, kalman, diff


def build_hard_mobility_configs() -> tuple[EnvConfig, EnvConfig, set[str]]:
    # v_d=18.0 may already equal the nominal value -- the effective changed
    # field set is derived programmatically, never hardcoded as {"v_d","v_e"}.
    direct, kalman, diff = _build_condition_configs({"v_d": 18.0, "v_e": 8.0})
    return direct, kalman, diff


def build_short_detection_configs() -> tuple[EnvConfig, EnvConfig, set[str]]:
    direct, kalman, diff = _build_condition_configs({"detection_radius": 5.0})
    assert diff == {"detection_radius"}, f"R5 changed-field set unexpected: {diff}"
    return direct, kalman, diff


CONDITION_DISPLAY_NAMES: dict[str, str] = {
    "strong_evasion": "Strong evasion",
    "restrictive_dynamics": "Restrictive dynamics",
    "high_measurement_noise": "High measurement noise",
    "hard_mobility": "Hard mobility",
    "short_detection": "Short detection",
}

CONDITIONS: tuple[tuple[str, str, Callable[[], tuple[EnvConfig, EnvConfig, set[str]]]], ...] = (
    ("strong_evasion", "Strong Reactive Evasion", build_strong_evasion_configs),
    ("restrictive_dynamics", "Restrictive Defender Maneuverability", build_restrictive_dynamics_configs),
    ("high_measurement_noise", "High Measurement Noise", build_high_measurement_noise_configs),
    ("hard_mobility", "Hard Mobility Condition", build_hard_mobility_configs),
    ("short_detection", "Short Detection Range", build_short_detection_configs),
)
assert len(CONDITIONS) == 5
assert set(CONDITION_DISPLAY_NAMES) == {key for key, _, _ in CONDITIONS}


# =============================================================================
# Policy + environment construction and episode execution (parameterized on
# the active condition's Direct/Kalman EnvConfig and the frozen RMPC config)
# =============================================================================
_INSTANCES_BY_ID: dict[str, MethodInstance] = {}


def build_policy_and_env(instance: MethodInstance, direct_cfg: EnvConfig, kalman_cfg: EnvConfig,
                          rmpc_config: RMPCConfig):
    if instance.family == "scripted":
        env = SoldierEnv(config=direct_cfg)
        policy = get_policy(instance.name)
    elif instance.family == "rmpc":
        env = SoldierEnv(config=direct_cfg)
        policy = RMPCPolicy(config=direct_cfg, rmpc_config=rmpc_config)
    elif instance.family == "ppo":
        env = SoldierEnv(config=direct_cfg)
        model_path = FROZEN_MODELS[("direct", instance.training_seed)]["path"]
        policy = get_policy("ppo", model_path=str(model_path), deterministic=True)
    elif instance.family == "ppo_kalman":
        env = SoldierEnv(config=kalman_cfg)
        model_path = FROZEN_MODELS[("kalman", instance.training_seed)]["path"]
        policy = get_policy("ppo_kalman", model_path=str(model_path), deterministic=True)
    elif instance.family == "lr_ppo":
        env = SoldierEnv(config=direct_cfg)
        model_path = FROZEN_MODELS[("lr_ppo", instance.training_seed)]["path"]
        model = PPO.load(str(model_path), device="cpu")
        policy = LeadResidualPPOPolicyWrapper(
            model, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG, config=direct_cfg, deterministic=True,
        )
    else:
        raise ValueError(instance.family)
    return env, policy


def run_robustness_episode(instance: MethodInstance, seed: int, direct_cfg: EnvConfig, kalman_cfg: EnvConfig,
                            rmpc_config: RMPCConfig, record_trajectory: bool = False) -> dict:
    env, policy = build_policy_and_env(instance, direct_cfg, kalman_cfg, rmpc_config)
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
        # NOT VALID FOR PUBLICATION RUNTIME COMPARISON -- diagnostic only.
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
# Parallel execution -- controlled, fixed worker count, condition-parameterized
# =============================================================================
_ACTIVE_RMPC_CONFIG: RMPCConfig | None = None
_ACTIVE_DIRECT_CFG: EnvConfig | None = None
_ACTIVE_KALMAN_CFG: EnvConfig | None = None


def _worker_init(rmpc_config: RMPCConfig, direct_cfg: EnvConfig, kalman_cfg: EnvConfig) -> None:
    """Each worker process needs its condition-specific configs populated
    independently (fresh interpreter under spawn)."""
    global _ACTIVE_RMPC_CONFIG, _ACTIVE_DIRECT_CFG, _ACTIVE_KALMAN_CFG
    _ACTIVE_RMPC_CONFIG = rmpc_config
    _ACTIVE_DIRECT_CFG = direct_cfg
    _ACTIVE_KALMAN_CFG = kalman_cfg
    for inst in build_final_instances():
        _INSTANCES_BY_ID[instance_id(inst)] = inst


def _episode_task(args):
    instance_id_str, seed = args
    instance = _INSTANCES_BY_ID[instance_id_str]
    return run_robustness_episode(instance, seed, _ACTIVE_DIRECT_CFG, _ACTIVE_KALMAN_CFG, _ACTIVE_RMPC_CONFIG)


def run_condition_instance_parallel(instance: MethodInstance, seeds: list[int], direct_cfg: EnvConfig,
                                     kalman_cfg: EnvConfig, rmpc_config: RMPCConfig, max_workers: int) -> list[dict]:
    for s in seeds:
        assert_robustness_seed(s)
    args = [(instance_id(instance), seed) for seed in seeds]
    with ProcessPoolExecutor(max_workers=max_workers, initializer=_worker_init,
                              initargs=(rmpc_config, direct_cfg, kalman_cfg)) as executor:
        return list(executor.map(_episode_task, args))


# =============================================================================
# Fairness audit (per condition)
# =============================================================================
FAIRNESS_AUDIT_SEEDS = (77_000, 77_001, 77_002)
FAIRNESS_AUDIT_REPRESENTATIVE_IDS = (
    "greedy", "pn", "lead", "rmpc", "ppo_seed45", "ppo_kalman_seed45", "lr_ppo_seed55",
)


def run_condition_fairness_trajectory_audit(direct_cfg: EnvConfig, kalman_cfg: EnvConfig,
                                             rmpc_config: RMPCConfig) -> dict:
    """Full pre-detection trajectory equality for one representative per
    controller family on seeds 77000-77002, WITHIN a single condition."""
    results: dict[str, dict] = {}
    reference_id = FAIRNESS_AUDIT_REPRESENTATIVE_IDS[0]
    for seed in FAIRNESS_AUDIT_SEEDS:
        rows = {}
        for inst_id in FAIRNESS_AUDIT_REPRESENTATIVE_IDS:
            instance = _INSTANCES_BY_ID[inst_id]
            rows[inst_id] = run_robustness_episode(
                instance, seed, direct_cfg, kalman_cfg, rmpc_config, record_trajectory=True
            )
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


# =============================================================================
# Completeness check -- 1000 rows per instance per condition
# =============================================================================
_VALID_OUTCOMES = {"intercepted", "soldier_caught", "unsafe_intercept", "timeout"}


def assert_condition_completeness(inst_id: str, df: pd.DataFrame) -> None:
    if len(df) != ROBUSTNESS_EPISODES:
        raise RuntimeError(f"{inst_id}: expected {ROBUSTNESS_EPISODES} rows, got {len(df)}")
    seeds = set(df["environment_seed"])
    expected_seeds = set(robustness_seeds())
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
    if outcome_sum != ROBUSTNESS_EPISODES:
        raise RuntimeError(f"{inst_id}: terminal-outcome counts sum to {outcome_sum}, expected {ROBUSTNESS_EPISODES}")


# =============================================================================
# Root-level manifests / summary tables
# =============================================================================
FAMILY_DISPLAY_ORDER = ("Greedy", "PN", "Lead", "RMPC", "PPO", "PPO-Kalman", "LR-PPO")
PRIMARY_COMPARISON_LABELS = ("LR-PPO - Lead", "LR-PPO - RMPC", "LR-PPO - PPO", "RMPC - Lead", "RMPC - PPO")


def build_robustness_summary_table(condition_family_summaries: dict[str, pd.DataFrame]) -> pd.DataFrame:
    nominal_fs = pd.read_csv(FINAL_NOMINAL_RESULTS_ROOT / "family_summary.csv").set_index("family")
    rows = [{"condition": "Nominal",
             **{fam: nominal_fs.loc[fam, "family_mean_success_rate"] * 100.0 for fam in FAMILY_DISPLAY_ORDER}}]
    for condition_key, _display, _builder in CONDITIONS:
        fs = condition_family_summaries[condition_key].set_index("family")
        rows.append({"condition": CONDITION_DISPLAY_NAMES[condition_key],
                     **{fam: fs.loc[fam, "family_mean_success_rate"] * 100.0 for fam in FAMILY_DISPLAY_ORDER}})
    return pd.DataFrame(rows)


def build_primary_differences_summary(condition_paired_diffs: dict[str, pd.DataFrame]) -> pd.DataFrame:
    rows = []
    for condition_key, _display, _builder in CONDITIONS:
        diffs = condition_paired_diffs[condition_key].set_index("comparison")
        for label in PRIMARY_COMPARISON_LABELS:
            r = diffs.loc[label]
            rows.append(dict(condition=CONDITION_DISPLAY_NAMES[condition_key], comparison=label,
                              difference_pp=float(r["difference_pp"]), ci_low_pp=float(r["ci_low_pp"]),
                              ci_high_pp=float(r["ci_high_pp"]), resolution=r["statistical_resolution"]))
    return pd.DataFrame(rows)


def build_robustness_protocol_manifest(rmpc_config: RMPCConfig, rmpc_config_data: dict, worker_count: int,
                                        start_time: str, end_time: str) -> dict:
    return {
        "repository": "mikemishal/RL-UAV-V2",
        "branch": _git_branch(),
        "rmpc_targeted_robustness_protocol_commit": _git_commit(),
        "rmpc_design_commit": RMPC_DESIGN_COMMIT,
        "rmpc_implementation_commit": RMPC_IMPLEMENTATION_COMMIT,
        "rmpc_hardening_commit": RMPC_HARDENING_COMMIT,
        "rmpc_tuning_protocol_commit": RMPC_TUNING_PROTOCOL_COMMIT,
        "rmpc_tuning_results_commit": RMPC_TUNING_RESULTS_COMMIT,
        "final_nominal_base_commit": FINAL_NOMINAL_BASE_COMMIT,
        "final_nominal_protocol_commit": FINAL_NOMINAL_PROTOCOL_COMMIT,
        "final_nominal_results_commit": FINAL_NOMINAL_RESULTS_COMMIT,
        "robustness_base_commit": ROBUSTNESS_BASE_COMMIT,
        "selected_rmpc_config_hash": sha256_of_file(SELECTED_RMPC_CONFIG_PATH),
        "selected_rmpc_config": rmpc_config_data,
        "rmpc_config_constructed": dataclasses.asdict(rmpc_config),
        "source_file_hashes": source_file_hashes(),
        "seed_range": [ROBUSTNESS_SEED_START, ROBUSTNESS_SEED_END],
        "n_unique_seeds": ROBUSTNESS_EPISODES,
        "reused_seed_across_conditions": True,
        "conditions_are_not_statistically_independent": True,
        "conditions": [{"condition_key": key, "display_name": name} for key, name, _ in CONDITIONS],
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
        "disclaimer": "FINAL RAW ROBUSTNESS EVIDENCE -- statistical results only; RMPC decision "
                       "times from this parallel run are diagnostic-only and NOT VALID FOR "
                       "PUBLICATION RUNTIME COMPARISON. The five conditions reuse the SAME 1000 "
                       "seeds and are NOT statistically independent of one another.",
    }


def write_targeted_robustness_report(report_path: Path, *, robustness_summary: pd.DataFrame,
                                      primary_diff_summary: pd.DataFrame,
                                      condition_instance_summaries: dict[str, pd.DataFrame]) -> None:
    lines = [
        "# RMPC Targeted Robustness Publication Evaluation Report",
        "",
        "**FINAL RAW ROBUSTNESS EVIDENCE -- not yet integrated into the manuscript,",
        "paper_artifacts/, or publication figures. The five conditions reuse the SAME",
        "1000 seeds (77000-77999) and are NOT statistically independent of one another.**",
        "",
        "## Family success-rate summary (%)", "",
        "| Condition | " + " | ".join(FAMILY_DISPLAY_ORDER) + " |",
        "|---" * (len(FAMILY_DISPLAY_ORDER) + 1) + "|",
    ]
    for _, r in robustness_summary.iterrows():
        lines.append("| " + r["condition"] + " | " + " | ".join(f"{r[fam]:.2f}" for fam in FAMILY_DISPLAY_ORDER) + " |")
    lines += ["", "## Primary comparisons by condition", "",
              "| Condition | Comparison | Diff (pp) | 95% CI (pp) | Resolution |", "|---|---|---|---|---|"]
    for _, r in primary_diff_summary.iterrows():
        lines.append(f"| {r['condition']} | {r['comparison']} | {r['difference_pp']:.2f} | "
                      f"[{r['ci_low_pp']:.2f}, {r['ci_high_pp']:.2f}] | {r['resolution']} |")
    lines += ["", "## Instance results by condition", ""]
    for condition_key, display_name, _ in CONDITIONS:
        lines += [f"### {display_name} (`{condition_key}`)", "",
                  "| Instance | Success/n | Success % | 95% CI |", "|---|---|---|---|"]
        for _, r in condition_instance_summaries[condition_key].iterrows():
            lines.append(f"| {r['instance']} | {int(r['successes'])}/{int(r['n'])} | {r['success_rate']*100:.2f}% | "
                          f"[{r['success_ci_low']*100:.2f}%, {r['success_ci_high']*100:.2f}%] |")
        lines.append("")
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text("\n".join(lines) + "\n")


DEFAULT_MAX_WORKERS = max(1, (os.cpu_count() or 1) - 1)


def main() -> None:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    rmpc_config, rmpc_config_data = load_frozen_rmpc_config()
    instances = build_final_instances()
    for inst in instances:
        _INSTANCES_BY_ID[instance_id(inst)] = inst

    verify_model_hashes()  # raises on mismatch
    print("Model hashes verified OK for all 9 frozen checkpoints.", flush=True)

    seeds = robustness_seeds()
    worker_count = DEFAULT_MAX_WORKERS
    start_time = datetime.now(timezone.utc).isoformat()

    condition_family_summaries: dict[str, pd.DataFrame] = {}
    condition_paired_diffs: dict[str, pd.DataFrame] = {}
    condition_instance_summaries: dict[str, pd.DataFrame] = {}
    all_fairness: dict[str, dict] = {}
    all_numerical: dict[str, dict] = {}

    for condition_key, display_name, builder in CONDITIONS:
        print(f"\n=== Condition: {condition_key} ({display_name}) ===", flush=True)
        direct_cfg, kalman_cfg, diff_keys = builder()
        condition_dir = RESULTS_ROOT / condition_key
        (condition_dir / "episodes").mkdir(parents=True, exist_ok=True)

        print("  Running pre-detection fairness trajectory audit (seeds 77000-77002)...", flush=True)
        trajectory_results = run_condition_fairness_trajectory_audit(direct_cfg, kalman_cfg, rmpc_config)
        print("  Fairness trajectory audit passed.", flush=True)

        episodes: dict[str, pd.DataFrame] = {}
        raw_rows_by_instance: dict[str, list[dict]] = {}
        for i, instance in enumerate(instances):
            inst_id = instance_id(instance)
            print(f"  [{i+1}/{len(instances)}] {inst_id} ({instance.family}) -- {len(seeds)} episodes "
                  f"(workers={worker_count})...", flush=True)
            rows = run_condition_instance_parallel(instance, seeds, direct_cfg, kalman_cfg, rmpc_config, worker_count)
            raw_rows_by_instance[inst_id] = rows
            df = _clean_rows_for_csv(rows)
            assert_condition_completeness(inst_id, df)
            df.to_csv(condition_dir / "episodes" / f"{inst_id}.csv", index=False)
            episodes[inst_id] = df
            print(f"      {inst_id}: {int(df['success'].sum())}/{len(df)} intercepted", flush=True)

        print("  Running detection-step fairness audit across all 13 instances...", flush=True)
        n_fairness_checks = assert_detection_step_equality_all_instances(episodes)
        print(f"  Detection-step fairness OK across {n_fairness_checks} seed x instance comparisons.", flush=True)

        num_audit = numerical_audit(episodes)

        instance_rows = []
        for instance in instances:
            inst_id = instance_id(instance)
            summary = summarize_instance(episodes[inst_id])
            instance_rows.append(dict(instance=inst_id, family=REPORT_FAMILY_OF_INSTANCE_NAME[instance.name],
                                       training_seed=instance.training_seed, **summary))
        instance_summary = pd.DataFrame(instance_rows)
        instance_summary.to_csv(condition_dir / "instance_summary.csv", index=False)
        condition_instance_summaries[condition_key] = instance_summary

        family_summary = pd.DataFrame(build_family_summary(episodes))
        family_summary.to_csv(condition_dir / "family_summary.csv", index=False)
        condition_family_summaries[condition_key] = family_summary

        paired_differences = pd.DataFrame(build_paired_differences(episodes))
        paired_differences.to_csv(condition_dir / "paired_differences.csv", index=False)
        condition_paired_diffs[condition_key] = paired_differences

        rmpc_diag = rmpc_diagnostics_summary(episodes["rmpc"], raw_rows_by_instance["rmpc"])

        condition_audit = {
            "condition": condition_key,
            "display_name": display_name,
            "env_config_direct": dataclasses.asdict(direct_cfg),
            "env_config_kalman": dataclasses.asdict(kalman_cfg),
            "changed_fields": sorted(diff_keys),
            "seed_range": [ROBUSTNESS_SEED_START, ROBUSTNESS_SEED_END],
            "n_seeds": ROBUSTNESS_EPISODES,
            "n_instances": len(instances),
            "detection_step_equality_all_instances": True,
            "n_detection_step_comparisons": n_fairness_checks,
            "trajectory_audit_seeds": list(FAIRNESS_AUDIT_SEEDS),
            "trajectory_audit_representative_instances": list(FAIRNESS_AUDIT_REPRESENTATIVE_IDS),
            "trajectory_audit_results": trajectory_results,
            "numerical_audit": num_audit,
            "rmpc_diagnostics": rmpc_diag,
        }
        (condition_dir / "condition_audit.json").write_text(json.dumps(condition_audit, indent=2, default=str))

        all_fairness[condition_key] = {
            "detection_step_equality_all_instances": True,
            "n_detection_step_comparisons": n_fairness_checks,
            "trajectory_audit_results": trajectory_results,
        }
        all_numerical[condition_key] = num_audit

    end_time = datetime.now(timezone.utc).isoformat()

    print("\nWriting root-level manifests and summaries...", flush=True)
    model_manifest = build_model_manifest()
    (RESULTS_ROOT / "model_manifest.json").write_text(json.dumps(model_manifest, indent=2, default=str))
    (RESULTS_ROOT / "selected_rmpc_config_snapshot.json").write_text(json.dumps(rmpc_config_data, indent=2, default=str))

    protocol_manifest = build_robustness_protocol_manifest(rmpc_config, rmpc_config_data, worker_count, start_time, end_time)
    (RESULTS_ROOT / "protocol_manifest.json").write_text(json.dumps(protocol_manifest, indent=2, default=str))

    total_violations = sum(v["_totals"]["physical_limit_violations"] for v in all_numerical.values())
    total_nan = sum(v["_totals"]["unexpected_nan_count"] for v in all_numerical.values())
    total_inf = sum(v["_totals"]["inf_count"] for v in all_numerical.values())
    combined_fairness = {"conditions": all_fairness}
    combined_numerical = {"conditions": all_numerical,
                           "_grand_totals": {"physical_limit_violations": total_violations,
                                              "unexpected_nan_count": total_nan, "inf_count": total_inf}}
    (RESULTS_ROOT / "fairness_audit.json").write_text(json.dumps(combined_fairness, indent=2, default=str))
    (RESULTS_ROOT / "numerical_audit.json").write_text(json.dumps(combined_numerical, indent=2, default=str))

    robustness_summary = build_robustness_summary_table(condition_family_summaries)
    robustness_summary.to_csv(RESULTS_ROOT / "robustness_summary.csv", index=False)

    primary_diff_summary = build_primary_differences_summary(condition_paired_diffs)
    primary_diff_summary.to_csv(RESULTS_ROOT / "primary_differences_summary.csv", index=False)

    write_targeted_robustness_report(RESULTS_ROOT / "targeted_robustness_report.md",
                                      robustness_summary=robustness_summary,
                                      primary_diff_summary=primary_diff_summary,
                                      condition_instance_summaries=condition_instance_summaries)

    print("\n=== TARGETED ROBUSTNESS CAMPAIGN COMPLETE ===")
    print(robustness_summary.to_string(index=False))
    print(f"\nArtifacts written to {RESULTS_ROOT}")


if __name__ == "__main__":
    main()
