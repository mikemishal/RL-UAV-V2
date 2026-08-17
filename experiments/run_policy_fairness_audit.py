"""
Pre-training paper-facing policy fairness audit (revised canonical protocol).

Proves that the five REVISED paper-facing controllers (Greedy, PN, Lead, PPO,
PPO-Kalman) are physically identical through the hostile-detection transition
under defender_standby_until_detection=True, using ONLY non-publication
development seeds. Kalman-Greedy is excluded (not part of the revised
paper-facing method set).

This script does NOT train, does NOT touch any reserved seed (>=40000), and
does NOT evaluate paper-facing baseline performance -- it stops at/just after
the detection transition only.

Usage:
    python experiments/run_policy_fairness_audit.py
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

AUDIT_SEEDS = (123, 124, 125, 126, 127)
OUTPUT_DIR = PROJECT_ROOT / "results" / "training_post_detection_400k" / "fairness_audit"

# Loadable checkpoints used ONLY for this fairness audit -- post-detection
# performance is irrelevant here; any loadable frozen checkpoint suffices.
_PPO_MODEL_PATH = PROJECT_ROOT / "results" / "_smoke_test_post_detection" / "direct" / "final_model.zip"
_PPO_KALMAN_MODEL_PATH = PROJECT_ROOT / "results" / "_smoke_test_post_detection" / "kalman" / "final_model.zip"

METHODS = [
    {"name": "greedy", "policy_name": "greedy", "use_kalman_tracking": False, "model_path": None},
    {"name": "pn", "policy_name": "pn", "use_kalman_tracking": False, "model_path": None},
    {"name": "lead", "policy_name": "lead", "use_kalman_tracking": False, "model_path": None},
    {"name": "ppo", "policy_name": "ppo", "use_kalman_tracking": False, "model_path": _PPO_MODEL_PATH},
    {"name": "ppo_kalman", "policy_name": "ppo_kalman", "use_kalman_tracking": True, "model_path": _PPO_KALMAN_MODEL_PATH},
]


def _build_policy(method: dict):
    if method["model_path"] is not None:
        return get_policy(method["policy_name"], model_path=str(method["model_path"]), deterministic=True)
    return get_policy(method["policy_name"])


def run_one(method: dict, seed: int) -> dict:
    config = EnvConfig(
        defender_standby_until_detection=True,
        use_kalman_tracking=method["use_kalman_tracking"],
    )
    env = SoldierEnv(config=config)
    policy = _build_policy(method)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)

    obs, info = env.reset(seed=seed)
    dt = config.dt
    step = 0

    pre_detection_states = [dict(
        step=step, soldier_pos=info["soldier_pos"].copy(), defender_pos=info["defender_pos"].copy(),
        defender_vel=info["defender_vel"].copy(), enemy_pos=info["enemy_pos"].copy(),
        enemy_vel=info["enemy_vel"].copy(), enemy_detected=info["enemy_detected"],
        controller_action_executed=info["controller_action_executed"],
    )]

    detection_step = None
    detection_info = None
    post_detection_info = None

    while not info["enemy_detected"]:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        obs, reward, term, trunc, info = env.step(action)
        step += 1
        pre_detection_states.append(dict(
            step=step, soldier_pos=info["soldier_pos"].copy(), defender_pos=info["defender_pos"].copy(),
            defender_vel=info["defender_vel"].copy(), enemy_pos=info["enemy_pos"].copy(),
            enemy_vel=info["enemy_vel"].copy(), enemy_detected=info["enemy_detected"],
            controller_action_executed=info["controller_action_executed"],
        ))
        if info["just_detected"]:
            detection_step = step
            detection_info = dict(info)
        if term or trunc:
            break

    if detection_step is not None:
        # First post-detection controlled step.
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        obs, reward, term, trunc, info = env.step(action)
        post_detection_info = dict(info)

    env.close()

    return {
        "method": method["name"], "seed": seed,
        "pre_detection_states": pre_detection_states,
        "detection_step": detection_step,
        "detection_time_seconds": detection_step * dt if detection_step is not None else None,
        "detection_info": detection_info,
        "post_detection_info": post_detection_info,
    }


def compare_methods_for_seed(results_for_seed: list[dict]) -> dict:
    """Compare all methods' pre-detection state histories for one seed."""
    ref = results_for_seed[0]
    max_discrepancy = 0.0
    detection_steps = {r["method"]: r["detection_step"] for r in results_for_seed}
    all_same_detection_step = len(set(detection_steps.values())) == 1

    n_states = len(ref["pre_detection_states"])
    all_same_length = all(len(r["pre_detection_states"]) == n_states for r in results_for_seed)

    identical_through_detection = all_same_detection_step and all_same_length
    if identical_through_detection:
        for i in range(n_states):
            ref_state = ref["pre_detection_states"][i]
            for r in results_for_seed[1:]:
                s = r["pre_detection_states"][i]
                for field in ("soldier_pos", "defender_pos", "defender_vel", "enemy_pos", "enemy_vel"):
                    diff = float(np.max(np.abs(ref_state[field] - s[field])))
                    max_discrepancy = max(max_discrepancy, diff)
                if ref_state["enemy_detected"] != s["enemy_detected"]:
                    identical_through_detection = False
                if ref_state["controller_action_executed"] != s["controller_action_executed"]:
                    identical_through_detection = False

    if max_discrepancy > 1e-6:
        identical_through_detection = False

    defender_soldier_seps = {}
    for r in results_for_seed:
        di = r["detection_info"]
        if di is not None:
            defender_soldier_seps[r["method"]] = float(np.linalg.norm(di["defender_pos"] - di["soldier_pos"]))

    return {
        "seed": ref["seed"],
        "detection_steps": detection_steps,
        "all_same_detection_step": all_same_detection_step,
        "max_pre_detection_discrepancy": max_discrepancy,
        "identical_through_detection": identical_through_detection,
        "defender_soldier_separation_at_detection": defender_soldier_seps,
    }


def main() -> int:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = []
    seed_comparisons = []

    for seed in AUDIT_SEEDS:
        results_for_seed = [run_one(m, seed) for m in METHODS]
        comparison = compare_methods_for_seed(results_for_seed)
        seed_comparisons.append(comparison)

        for r in results_for_seed:
            di = r["detection_info"]
            pdi = r["post_detection_info"]
            rows.append({
                "seed": seed, "method": r["method"],
                "detection_step": r["detection_step"],
                "detection_time_seconds": r["detection_time_seconds"],
                "defender_pos_at_detection": di["defender_pos"].tolist() if di is not None else None,
                "soldier_pos_at_detection": di["soldier_pos"].tolist() if di is not None else None,
                "defender_vel_at_detection": di["defender_vel"].tolist() if di is not None else None,
                "enemy_pos_at_detection": di["enemy_pos"].tolist() if di is not None else None,
                "enemy_vel_at_detection": di["enemy_vel"].tolist() if di is not None else None,
                "defender_soldier_separation_at_detection": (
                    float(np.linalg.norm(di["defender_pos"] - di["soldier_pos"])) if di is not None else None
                ),
                "controller_action_executed_at_detection": di["controller_action_executed"] if di is not None else None,
                "controller_action_executed_post_detection": pdi["controller_action_executed"] if pdi is not None else None,
                "defender_accel_post_detection": pdi["defender_accel"] if pdi is not None else None,
            })

    df = pd.DataFrame(rows)
    df.to_csv(OUTPUT_DIR / "policy_fairness_audit.csv", index=False)

    all_pass = all(c["identical_through_detection"] for c in seed_comparisons)
    summary = {
        "audit_seeds": list(AUDIT_SEEDS),
        "methods_audited": [m["name"] for m in METHODS],
        "kalman_greedy_excluded": True,
        "per_seed_comparison": seed_comparisons,
        "all_seeds_identical_through_detection": all_pass,
        "max_discrepancy_overall": max(c["max_pre_detection_discrepancy"] for c in seed_comparisons),
        "pass_criterion": "identical soldier/defender/defender_vel/enemy_pos/enemy_vel/enemy_detected trajectories "
                           "and identical detection_step across all 5 methods, for every seed",
        "audit_result": "PASS" if all_pass else "FAIL -- DO NOT TRAIN",
    }
    (OUTPUT_DIR / "policy_fairness_summary.json").write_text(json.dumps(summary, indent=2, default=str))

    print(json.dumps(summary, indent=2, default=str))
    return 0 if all_pass else 1


if __name__ == "__main__":
    raise SystemExit(main())
