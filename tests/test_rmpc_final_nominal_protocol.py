"""
Tests for the RMPC final-nominal evaluation harness
(`experiments/run_rmpc_final_nominal.py`) -- seed guards, frozen-config
verification, the 13-instance builder, and completeness/statistics helpers.

These tests do NOT open any final-nominal seed (72000-76999); dev seeds
(0, 1, 2, ...) and synthetic in-memory data are used throughout.

Run directly: python tests/test_rmpc_final_nominal_protocol.py
"""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import numpy as np
import pandas as pd

import experiments.run_rmpc_final_nominal as f


# --- Section 13: seed guards -------------------------------------------------

def test_seed_71999_rejected():
    try:
        f.assert_final_seed(71_999)
        raise AssertionError("71999 unexpectedly accepted")
    except ValueError:
        pass


def test_seed_72000_accepted():
    f.assert_final_seed(72_000)  # must not raise


def test_seed_76999_accepted():
    f.assert_final_seed(76_999)  # must not raise


def test_seed_77000_rejected():
    try:
        f.assert_final_seed(77_000)
        raise AssertionError("77000 unexpectedly accepted")
    except ValueError:
        pass


def test_final_seed_range_is_exactly_5000():
    seeds = f.final_seeds()
    assert seeds[0] == 72_000
    assert seeds[-1] == 76_999
    assert len(seeds) == 5000 == f.FINAL_EPISODES


def test_run_instance_parallel_rejects_out_of_range_seed():
    rmpc_config, _ = f.load_frozen_rmpc_config()
    instances = f.build_final_instances()
    f._register_instances(instances, rmpc_config)
    greedy = [i for i in instances if i.name == "greedy"][0]
    try:
        f.run_instance_parallel(greedy, [71_999], max_workers=1)
        raise AssertionError("expected ValueError for seed 71999")
    except ValueError:
        pass


# --- Section 3: frozen RMPC configuration ------------------------------------

def test_frozen_rmpc_config_matches_predeclared_values():
    rmpc_config, data = f.load_frozen_rmpc_config()
    assert data["frozen_for_final_evaluation"] is True
    assert data["candidate_id"] == "C1_FAST"
    assert rmpc_config.horizon == 4
    assert rmpc_config.population_size == 64
    assert rmpc_config.elite_fraction == 0.20
    assert rmpc_config.cem_iterations == 3
    assert rmpc_config.controller_seed == 0
    assert rmpc_config.cem_initial_std == 1.0
    assert rmpc_config.cem_min_std == 0.05
    assert rmpc_config.cem_sample_clip == 3.0
    assert rmpc_config.enable_timing is True


def test_frozen_rmpc_config_rejects_tampered_file(tmp_path, monkeypatch):
    import json

    bad_path = tmp_path / "selected_rmpc_config.json"
    bad_data = dict(frozen_for_final_evaluation=True, candidate_id="C1_FAST", horizon=8,  # tampered
                    population_size=64, elite_fraction=0.20, cem_iterations=3, controller_seed=0,
                    cem_initial_std=1.0, cem_min_std=0.05, cem_sample_clip=3.0)
    bad_path.write_text(json.dumps(bad_data))
    monkeypatch.setattr(f, "SELECTED_RMPC_CONFIG_PATH", bad_path)
    try:
        f.load_frozen_rmpc_config()
        raise AssertionError("expected RuntimeError for tampered horizon field")
    except RuntimeError:
        pass


def test_frozen_rmpc_config_rejects_unfrozen_file(tmp_path, monkeypatch):
    import json

    bad_path = tmp_path / "selected_rmpc_config.json"
    bad_data = dict(frozen_for_final_evaluation=False, candidate_id="C1_FAST", horizon=4,
                    population_size=64, elite_fraction=0.20, cem_iterations=3, controller_seed=0,
                    cem_initial_std=1.0, cem_min_std=0.05, cem_sample_clip=3.0)
    bad_path.write_text(json.dumps(bad_data))
    monkeypatch.setattr(f, "SELECTED_RMPC_CONFIG_PATH", bad_path)
    try:
        f.load_frozen_rmpc_config()
        raise AssertionError("expected RuntimeError for frozen_for_final_evaluation=False")
    except RuntimeError:
        pass


# --- Section 5: thirteen method instances ------------------------------------

def test_exactly_thirteen_instances():
    instances = f.build_final_instances()
    assert len(instances) == 13


def test_instance_ids_match_required_set():
    instances = f.build_final_instances()
    ids = {f.instance_id(i) for i in instances}
    expected = {
        "greedy", "pn", "lead", "rmpc",
        "ppo_seed45", "ppo_seed46", "ppo_seed47",
        "ppo_kalman_seed45", "ppo_kalman_seed46", "ppo_kalman_seed47",
        "lr_ppo_seed55", "lr_ppo_seed56", "lr_ppo_seed57",
    }
    assert ids == expected


def test_no_kalman_greedy_or_lead_kalman():
    instances = f.build_final_instances()
    names = {i.name for i in instances}
    assert "kalman_greedy" not in names
    assert "lead_kalman" not in names
    assert "pn_kalman" not in names


def test_rmpc_is_single_instance_not_multiple_roots():
    instances = f.build_final_instances()
    rmpc_instances = [i for i in instances if i.family == "rmpc"]
    assert len(rmpc_instances) == 1
    assert rmpc_instances[0].training_seed is None


def test_estimator_mode_mapping():
    instances = f.build_final_instances()
    for i in instances:
        mode = "kalman" if i.family == "ppo_kalman" else "measurement"
        env_cfg = f.env_config_for_instance(i)
        expected_kalman_flag = (mode == "kalman")
        assert env_cfg.use_kalman_tracking == expected_kalman_flag


# --- Section 4: frozen learned models ----------------------------------------

def test_model_hashes_verified():
    results = f.verify_model_hashes()
    assert len(results) == 9
    assert all(r["match"] for r in results.values())


# --- Section 19: completeness check ------------------------------------------

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
    seeds = f.final_seeds()
    outcomes = ["intercepted"] * len(seeds)
    df = _make_fake_episode_df(seeds, outcomes)
    f.assert_instance_completeness("fake", df)  # must not raise


def test_completeness_check_rejects_missing_seed():
    seeds = f.final_seeds()[:-1]  # drop last seed -> 4999 rows
    outcomes = ["intercepted"] * len(seeds)
    df = _make_fake_episode_df(seeds, outcomes)
    try:
        f.assert_instance_completeness("fake", df)
        raise AssertionError("expected RuntimeError for missing seed")
    except RuntimeError:
        pass


def test_completeness_check_rejects_duplicate_seed():
    seeds = f.final_seeds()[:-1] + [f.final_seeds()[0]]  # duplicate first seed
    outcomes = ["intercepted"] * len(seeds)
    df = _make_fake_episode_df(seeds, outcomes)
    try:
        f.assert_instance_completeness("fake", df)
        raise AssertionError("expected RuntimeError for duplicate seed")
    except RuntimeError:
        pass


def test_completeness_check_rejects_unknown_outcome():
    seeds = f.final_seeds()
    outcomes = ["intercepted"] * (len(seeds) - 1) + ["bogus_outcome"]
    df = _make_fake_episode_df(seeds, outcomes)
    try:
        f.assert_instance_completeness("fake", df)
        raise AssertionError("expected RuntimeError for unknown outcome category")
    except RuntimeError:
        pass


# --- Statistics helpers -------------------------------------------------------

def test_resolution_label():
    assert f.resolution_label(0.5, 2.0) == "resolved positive"
    assert f.resolution_label(-2.0, -0.5) == "resolved negative"
    assert f.resolution_label(-1.0, 1.0) == "unresolved"


def test_success_vector_sorted_by_seed():
    df = pd.DataFrame({"environment_seed": [3, 1, 2], "success": [0, 1, 0]})
    vec = f.success_vector(df)
    assert np.array_equal(vec, np.array([1.0, 0.0, 0.0]))  # sorted: seed1->1, seed2->0, seed3->0


def test_success_matrix_shape():
    df1 = pd.DataFrame({"environment_seed": [1, 2], "success": [1, 0]})
    df2 = pd.DataFrame({"environment_seed": [1, 2], "success": [0, 1]})
    mat = f.success_matrix([df1, df2])
    assert mat.shape == (2, 2)


# --- Trajectory fairness helper ----------------------------------------------

def test_trajectories_equal_identical():
    traj = [dict(soldier_pos=np.array([0.0, 0.0, 0.0]), defender_pos=np.array([1.0, 0.0, 0.0]),
                  defender_vel=np.array([0.0, 0.0, 0.0]), enemy_pos=np.array([5.0, 0.0, 0.0]),
                  enemy_vel=np.array([0.0, 0.0, 0.0]), enemy_detected=False)]
    assert f._trajectories_equal(traj, traj)


def test_trajectories_equal_detects_mismatch():
    traj_a = [dict(soldier_pos=np.array([0.0, 0.0, 0.0]), defender_pos=np.array([1.0, 0.0, 0.0]),
                    defender_vel=np.array([0.0, 0.0, 0.0]), enemy_pos=np.array([5.0, 0.0, 0.0]),
                    enemy_vel=np.array([0.0, 0.0, 0.0]), enemy_detected=False)]
    traj_b = [dict(soldier_pos=np.array([0.0, 0.0, 0.0]), defender_pos=np.array([2.0, 0.0, 0.0]),
                    defender_vel=np.array([0.0, 0.0, 0.0]), enemy_pos=np.array([5.0, 0.0, 0.0]),
                    enemy_vel=np.array([0.0, 0.0, 0.0]), enemy_detected=False)]
    assert not f._trajectories_equal(traj_a, traj_b)


def test_trajectories_equal_detects_length_mismatch():
    traj_a = [dict(soldier_pos=np.array([0.0, 0.0, 0.0]), defender_pos=np.array([1.0, 0.0, 0.0]),
                    defender_vel=np.array([0.0, 0.0, 0.0]), enemy_pos=np.array([5.0, 0.0, 0.0]),
                    enemy_vel=np.array([0.0, 0.0, 0.0]), enemy_detected=False)]
    traj_b = traj_a + traj_a
    assert not f._trajectories_equal(traj_a, traj_b)


# --- Episode construction / fairness (real, dev seed 0) ----------------------

def test_all_non_rmpc_instances_share_detection_step_dev_seed():
    rmpc_config, _ = f.load_frozen_rmpc_config()
    instances = f.build_final_instances()
    f._register_instances(instances, rmpc_config)
    detection_steps = set()
    for inst in instances:
        if inst.family == "rmpc":
            continue  # slow; covered by test_rmpc_episode_matches_detection_step below
        row = f.run_final_nominal_episode(inst, seed=0)
        detection_steps.add(row["detection_step"])
    assert len(detection_steps) == 1, f"detection_step differs across instances: {detection_steps}"


def test_rmpc_episode_matches_detection_step():
    rmpc_config, _ = f.load_frozen_rmpc_config()
    instances = f.build_final_instances()
    f._register_instances(instances, rmpc_config)
    greedy = [i for i in instances if i.name == "greedy"][0]
    rmpc = [i for i in instances if i.family == "rmpc"][0]
    greedy_row = f.run_final_nominal_episode(greedy, seed=0)
    rmpc_row = f.run_final_nominal_episode(rmpc, seed=0)
    assert greedy_row["detection_step"] == rmpc_row["detection_step"]
    assert rmpc_row["physical_limit_violations"] == 0


if __name__ == "__main__":
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = failed = 0

    class _FakeTmpPath:
        def __init__(self, base):
            self._base = base

        def __truediv__(self, name):
            return self._base / name

    class _FakeMonkeypatch:
        def __init__(self):
            self._restores = []

        def setattr(self, target, name, value):
            self._restores.append((target, name, getattr(target, name)))
            setattr(target, name, value)

        def undo(self):
            for target, name, old in reversed(self._restores):
                setattr(target, name, old)

    import inspect
    import tempfile

    for test_fn in tests:
        try:
            sig = inspect.signature(test_fn)
            kwargs = {}
            mp = None
            tmp_dir_ctx = None
            if "tmp_path" in sig.parameters:
                tmp_dir_ctx = tempfile.TemporaryDirectory()
                kwargs["tmp_path"] = Path(tmp_dir_ctx.name)
            if "monkeypatch" in sig.parameters:
                mp = _FakeMonkeypatch()
                kwargs["monkeypatch"] = mp
            try:
                test_fn(**kwargs)
                print(f"PASS: {test_fn.__name__}")
                passed += 1
            finally:
                if mp is not None:
                    mp.undo()
                if tmp_dir_ctx is not None:
                    tmp_dir_ctx.cleanup()
        except Exception:
            print(f"FAIL: {test_fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
