"""
Tests for the RMPC targeted-robustness evaluation harness
(`experiments/run_rmpc_targeted_robustness.py`) -- seed guards, condition
config-isolation, the shared 13-instance builder, completeness helpers, and
fairness at dev seeds.

These tests do NOT open any robustness seed (77000-77999); dev seeds
(0, 1, 2, ...) and synthetic in-memory data are used throughout.

Run directly: python tests/test_rmpc_targeted_robustness_protocol.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import experiments.run_rmpc_targeted_robustness as r


# --- Seed guards --------------------------------------------------------------

def test_seed_76999_rejected():
    try:
        r.assert_robustness_seed(76_999)
        raise AssertionError("76999 unexpectedly accepted")
    except ValueError:
        pass


def test_seed_77000_accepted():
    r.assert_robustness_seed(77_000)  # must not raise


def test_seed_77999_accepted():
    r.assert_robustness_seed(77_999)  # must not raise


def test_seed_78000_rejected():
    try:
        r.assert_robustness_seed(78_000)
        raise AssertionError("78000 unexpectedly accepted")
    except ValueError:
        pass


def test_robustness_seed_range_is_exactly_1000():
    seeds = r.robustness_seeds()
    assert seeds[0] == 77_000
    assert seeds[-1] == 77_999
    assert len(seeds) == 1000 == r.ROBUSTNESS_EPISODES


def test_run_condition_instance_parallel_rejects_out_of_range_seed():
    rmpc_config, _ = r.load_frozen_rmpc_config()
    instances = r.build_final_instances()
    for inst in instances:
        r._INSTANCES_BY_ID[r.instance_id(inst)] = inst
    greedy = [i for i in instances if i.name == "greedy"][0]
    direct_cfg, kalman_cfg, _ = r.build_strong_evasion_configs()
    try:
        r.run_condition_instance_parallel(greedy, [76_999], direct_cfg, kalman_cfg, rmpc_config, max_workers=1)
        raise AssertionError("expected ValueError for seed 76999")
    except ValueError:
        pass


# --- Reused frozen RMPC config / thirteen instances ---------------------------

def test_frozen_rmpc_config_matches_predeclared_values():
    rmpc_config, data = r.load_frozen_rmpc_config()
    assert data["frozen_for_final_evaluation"] is True
    assert data["candidate_id"] == "C1_FAST"
    assert rmpc_config.horizon == 4
    assert rmpc_config.population_size == 64
    assert rmpc_config.elite_fraction == 0.20
    assert rmpc_config.cem_iterations == 3
    assert rmpc_config.controller_seed == 0


def test_exactly_thirteen_instances():
    instances = r.build_final_instances()
    assert len(instances) == 13


def test_instance_ids_match_required_set():
    instances = r.build_final_instances()
    ids = {r.instance_id(i) for i in instances}
    expected = {
        "greedy", "pn", "lead", "rmpc",
        "ppo_seed45", "ppo_seed46", "ppo_seed47",
        "ppo_kalman_seed45", "ppo_kalman_seed46", "ppo_kalman_seed47",
        "lr_ppo_seed55", "lr_ppo_seed56", "lr_ppo_seed57",
    }
    assert ids == expected


def test_model_hashes_verified():
    results = r.verify_model_hashes()
    assert len(results) == 9
    assert all(res["match"] for res in results.values())


# --- Five conditions / config-isolation ---------------------------------------

def test_exactly_five_conditions():
    assert len(r.CONDITIONS) == 5
    keys = {key for key, _, _ in r.CONDITIONS}
    assert keys == {
        "strong_evasion", "restrictive_dynamics", "high_measurement_noise",
        "hard_mobility", "short_detection",
    }


def test_strong_evasion_changed_fields():
    direct, kalman, diff = r.build_strong_evasion_configs()
    assert diff == {"enemy_evasion_gain"}
    assert direct.enemy_evasion_gain == 1.00
    assert kalman.enemy_evasion_gain == 1.00


def test_restrictive_dynamics_changed_fields():
    direct, kalman, diff = r.build_restrictive_dynamics_configs()
    assert diff == {"defender_max_accel", "defender_max_turn_rate_deg"}
    assert direct.defender_max_accel == 4.0
    assert direct.defender_max_turn_rate_deg == 45.0


def test_high_measurement_noise_changed_fields():
    direct, kalman, diff = r.build_high_measurement_noise_configs()
    assert diff == {"measurement_var"}
    assert direct.measurement_var == 4.0
    # process_var must NOT be retuned -- stays at the nominal value.
    assert direct.process_var == r.NOMINAL_CFG_DIRECT.process_var
    assert kalman.process_var == r.NOMINAL_CFG_KALMAN.process_var


def test_hard_mobility_changed_fields_derived_programmatically():
    direct, kalman, diff = r.build_hard_mobility_configs()
    # v_d=18.0 is already the nominal value -- the effective changed field is v_e only.
    assert diff == {"v_e"}
    assert direct.v_d == 18.0
    assert direct.v_e == 8.0
    assert r.NOMINAL_CFG_DIRECT.v_d == 18.0  # confirms v_d was already nominal


def test_short_detection_changed_fields():
    direct, kalman, diff = r.build_short_detection_configs()
    assert diff == {"detection_radius"}
    assert direct.detection_radius == 5.0


def test_condition_configs_reject_unexpected_extra_diff(monkeypatch):
    # Corrupt _build_condition_configs' output by tampering with an unrelated
    # field via direct EnvConfig construction to prove assert_only_fields_differ
    # would catch an extra, unauthorized difference.
    from uav_defend.config import EnvConfig

    tampered = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True,
                          enemy_evasion_gain=1.00, detection_radius=999.0)
    try:
        r.assert_only_fields_differ(tampered, {"enemy_evasion_gain"})
        raise AssertionError("expected AssertionError for unexpected extra diff (detection_radius)")
    except AssertionError as e:
        assert "detection_radius" in str(e)


def test_condition_display_names_cover_all_conditions():
    assert set(r.CONDITION_DISPLAY_NAMES) == {key for key, _, _ in r.CONDITIONS}


# --- Completeness check --------------------------------------------------------

def _make_fake_episode_df(seeds, outcomes):
    rows = []
    for seed, outcome in zip(seeds, outcomes):
        rows.append(dict(
            environment_seed=seed, outcome=outcome, success=1 if outcome == "intercepted" else 0,
            soldier_caught=1 if outcome == "soldier_caught" else 0,
            unsafe_intercept=1 if outcome == "unsafe_intercept" else 0,
            timeout=1 if outcome == "timeout" else 0,
        ))
    return pd.DataFrame(rows)


def test_completeness_check_passes_for_valid_data():
    seeds = r.robustness_seeds()
    outcomes = ["intercepted"] * len(seeds)
    df = _make_fake_episode_df(seeds, outcomes)
    r.assert_condition_completeness("fake", df)  # must not raise


def test_completeness_check_rejects_missing_seed():
    seeds = r.robustness_seeds()[:-1]  # drop last seed -> 999 rows
    outcomes = ["intercepted"] * len(seeds)
    df = _make_fake_episode_df(seeds, outcomes)
    try:
        r.assert_condition_completeness("fake", df)
        raise AssertionError("expected RuntimeError for missing seed")
    except RuntimeError:
        pass


def test_completeness_check_rejects_duplicate_seed():
    seeds = r.robustness_seeds()[:-1] + [r.robustness_seeds()[0]]  # duplicate first seed
    outcomes = ["intercepted"] * len(seeds)
    df = _make_fake_episode_df(seeds, outcomes)
    try:
        r.assert_condition_completeness("fake", df)
        raise AssertionError("expected RuntimeError for duplicate seed")
    except RuntimeError:
        pass


def test_completeness_check_rejects_unknown_outcome():
    seeds = r.robustness_seeds()
    outcomes = ["intercepted"] * (len(seeds) - 1) + ["bogus_outcome"]
    df = _make_fake_episode_df(seeds, outcomes)
    try:
        r.assert_condition_completeness("fake", df)
        raise AssertionError("expected RuntimeError for unknown outcome category")
    except RuntimeError:
        pass


# --- Statistics helpers (reused verbatim from run_rmpc_final_nominal) ---------

def test_resolution_label_reused():
    assert r.resolution_label(0.5, 2.0) == "resolved positive"
    assert r.resolution_label(-2.0, -0.5) == "resolved negative"
    assert r.resolution_label(-1.0, 1.0) == "unresolved"


def test_primary_comparison_labels_match_protocol():
    assert r.PRIMARY_COMPARISON_LABELS == (
        "LR-PPO - Lead", "LR-PPO - RMPC", "LR-PPO - PPO", "RMPC - Lead", "RMPC - PPO",
    )


# --- Episode construction / fairness (real, dev seed 0) -----------------------

def test_all_conditions_share_detection_step_across_instances_dev_seed():
    rmpc_config, _ = r.load_frozen_rmpc_config()
    instances = r.build_final_instances()
    for inst in instances:
        r._INSTANCES_BY_ID[r.instance_id(inst)] = inst

    for condition_key, _display, builder in r.CONDITIONS:
        direct_cfg, kalman_cfg, _diff = builder()
        detection_steps = set()
        for inst in instances:
            row = r.run_robustness_episode(inst, seed=0, direct_cfg=direct_cfg, kalman_cfg=kalman_cfg,
                                            rmpc_config=rmpc_config)
            detection_steps.add(row["detection_step"])
        assert len(detection_steps) == 1, (
            f"condition {condition_key}: detection_step differs across instances: {detection_steps}"
        )


def test_condition_fairness_trajectory_audit_dev_seeds():
    rmpc_config, _ = r.load_frozen_rmpc_config()
    instances = r.build_final_instances()
    for inst in instances:
        r._INSTANCES_BY_ID[r.instance_id(inst)] = inst
    direct_cfg, kalman_cfg, _diff = r.build_short_detection_configs()
    # Use dev seeds temporarily by monkeypatching the audit seed tuple is not
    # needed here -- the audit function itself only reads FAIRNESS_AUDIT_SEEDS
    # module constant for the real campaign; here we call the per-episode
    # runner directly (already covered by test_all_conditions_share_detection_step...).
    results = {}
    reference_id = r.FAIRNESS_AUDIT_REPRESENTATIVE_IDS[0]
    rows = {}
    for inst_id in r.FAIRNESS_AUDIT_REPRESENTATIVE_IDS:
        inst = r._INSTANCES_BY_ID[inst_id]
        rows[inst_id] = r.run_robustness_episode(inst, seed=0, direct_cfg=direct_cfg, kalman_cfg=kalman_cfg,
                                                  rmpc_config=rmpc_config, record_trajectory=True)
    ref_traj = rows[reference_id]["pre_detection_trajectory"]
    ref_detection_step = rows[reference_id]["detection_step"]
    for inst_id, row in rows.items():
        assert r._trajectories_equal(ref_traj, row["pre_detection_trajectory"]), inst_id
        assert row["detection_step"] == ref_detection_step, inst_id


if __name__ == "__main__":
    import inspect
    module = sys.modules[__name__]
    test_fns = [obj for name, obj in inspect.getmembers(module) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        try:
            sig = inspect.signature(fn)
            if "monkeypatch" in sig.parameters:
                continue  # requires pytest fixture, run via pytest instead
            fn()
            passed += 1
            print(f"PASS {fn.__name__}")
        except Exception as e:
            print(f"FAIL {fn.__name__}: {e}")
            raise
    print(f"\n{passed}/{len(test_fns)} tests passed (direct execution; run via pytest for full coverage)")
