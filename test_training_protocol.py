"""Training-protocol correctness tests -- multi-seed PPO campaign guardrails.

Script-style test consistent with the project's other test files
(test_experiment_fairness.py, etc). No external test framework is used;
each test function performs assertions and the runner reports PASS/FAIL
per test with a final summary.

Covers:
    A. Training seeds are (42, 43, 44).
    B. Validation range is 10000..10099 and does not overlap the reserved
       diagnostic or final-test ranges.
    C. Direct and Kalman PPO hyperparameters are exactly identical.
    D. Direct and Kalman SB3 network architectures are identical.
    E. Run output paths differ by track and seed (cannot collide).
    F. Training manifest contains all required fields.
    G. Completed-run protection works (a second call does not retrain).
    H. Direct EnvConfig has use_kalman_tracking=False.
    I. Kalman EnvConfig has use_kalman_tracking=True.
    J. All other behavioral EnvConfig fields match between tracks.
    K. 200k and 400k campaign output roots cannot collide (feature/ppo-400k-replication).
    L. Campaign metadata fields are recorded in the manifest.

Run directly:
    python test_training_protocol.py
"""

from __future__ import annotations

import dataclasses
import shutil
import tempfile
from pathlib import Path

from stable_baselines3 import PPO

from uav_defend.envs import SoldierEnv
from experiments import ppo_hyperparams as H
from experiments.training_protocol import (
    TRAINING_SEEDS,
    VALIDATION_SEED_OFFSET,
    VALIDATION_EPISODES,
    VALIDATION_SEEDS,
    DIAGNOSTIC_SEED_OFFSET,
    DIAGNOSTIC_EPISODES,
    DIAGNOSTIC_SEEDS,
    FINAL_TEST_SEED_OFFSET,
    assert_not_final_test_seed,
    run_output_dir,
    CAMPAIGN_200K_DIRNAME,
    CAMPAIGN_400K_DIRNAME,
    CAMPAIGN_400K_TIMESTEPS,
)
from experiments.training_runner import (
    build_env_config,
    expected_run_signature,
    is_run_complete,
    run_training,
    _sb3_architecture_info,
)


# ---------------------------------------------------------------------------
# A. Training seeds
# ---------------------------------------------------------------------------
def test_training_seeds():
    assert TRAINING_SEEDS == (42, 43, 44), TRAINING_SEEDS


# ---------------------------------------------------------------------------
# B. Validation range and non-overlap with reserved ranges
# ---------------------------------------------------------------------------
def test_validation_range_no_overlap():
    assert VALIDATION_SEED_OFFSET == 10_000
    assert VALIDATION_EPISODES == 100
    assert list(VALIDATION_SEEDS) == list(range(10_000, 10_100))

    assert DIAGNOSTIC_SEED_OFFSET == 12_000
    assert DIAGNOSTIC_EPISODES == 200
    assert list(DIAGNOSTIC_SEEDS) == list(range(12_000, 12_200))

    assert FINAL_TEST_SEED_OFFSET == 20_000

    validation_set = set(VALIDATION_SEEDS)
    diagnostic_set = set(DIAGNOSTIC_SEEDS)
    assert validation_set.isdisjoint(diagnostic_set), "validation/diagnostic seed ranges overlap"
    assert max(VALIDATION_SEEDS) < FINAL_TEST_SEED_OFFSET
    assert max(DIAGNOSTIC_SEEDS) < FINAL_TEST_SEED_OFFSET

    # Guard helper must reject any seed >= the reserved final-test offset.
    assert_not_final_test_seed(FINAL_TEST_SEED_OFFSET - 1)
    raised = False
    try:
        assert_not_final_test_seed(FINAL_TEST_SEED_OFFSET)
    except ValueError:
        raised = True
    assert raised, "assert_not_final_test_seed must reject seed == FINAL_TEST_SEED_OFFSET"


# ---------------------------------------------------------------------------
# C. Direct/Kalman PPO hyperparameters identical
# ---------------------------------------------------------------------------
def test_hyperparameters_identical():
    # There is only ONE hyperparameter module (experiments/ppo_hyperparams.py)
    # consumed identically by run_training() regardless of track -- assert
    # the values match the task's frozen specification exactly.
    assert H.TOTAL_TIMESTEPS == 200_000
    assert H.LEARNING_RATE == 3e-4
    assert H.GAMMA == 0.99
    assert H.BATCH_SIZE == 64
    assert H.N_STEPS == 2048
    assert H.N_EPOCHS == 10
    assert H.CLIP_RANGE == 0.2
    assert H.ENT_COEF == 0.01
    assert H.VF_COEF == 0.5
    assert H.MAX_GRAD_NORM == 0.5
    assert H.GAE_LAMBDA == 0.95
    assert H.EVAL_FREQ == 25_000
    assert H.CHECKPOINT_FREQ == 50_000
    assert H.N_EVAL_EPISODES == 100

    hp_direct = expected_run_signature("direct", 42, H.TOTAL_TIMESTEPS, {
        "total_timesteps": H.TOTAL_TIMESTEPS, "learning_rate": H.LEARNING_RATE,
        "gamma": H.GAMMA, "batch_size": H.BATCH_SIZE, "n_steps": H.N_STEPS,
        "n_epochs": H.N_EPOCHS, "clip_range": H.CLIP_RANGE, "ent_coef": H.ENT_COEF,
        "vf_coef": H.VF_COEF, "max_grad_norm": H.MAX_GRAD_NORM, "gae_lambda": H.GAE_LAMBDA,
    }, "commit")["ppo_hyperparameters"]
    hp_kalman = expected_run_signature("kalman", 42, H.TOTAL_TIMESTEPS, dict(hp_direct), "commit")["ppo_hyperparameters"]
    assert hp_direct == hp_kalman, (hp_direct, hp_kalman)


# ---------------------------------------------------------------------------
# D. Direct/Kalman network architecture identical
# ---------------------------------------------------------------------------
def test_network_architecture_identical():
    arch_by_track = {}
    for track in ("direct", "kalman"):
        env = SoldierEnv(config=build_env_config(track))
        model = PPO(policy="MlpPolicy", env=env, n_steps=64, batch_size=32, verbose=0, seed=42)
        arch_by_track[track] = _sb3_architecture_info(model)

    direct_arch = arch_by_track["direct"]
    kalman_arch = arch_by_track["kalman"]
    assert direct_arch["policy_class"] == kalman_arch["policy_class"]
    assert direct_arch["net_arch"] == kalman_arch["net_arch"]
    assert direct_arch["policy_net_layers"] == kalman_arch["policy_net_layers"]
    assert direct_arch["value_net_layers"] == kalman_arch["value_net_layers"]
    assert direct_arch["activation_fn"] == kalman_arch["activation_fn"]
    assert direct_arch["total_parameters"] == kalman_arch["total_parameters"]
    assert direct_arch["policy_class"] == "ActorCriticPolicy"


# ---------------------------------------------------------------------------
# E. Run output paths differ by track and seed
# ---------------------------------------------------------------------------
def test_output_paths_no_collision():
    root = Path("results")
    paths = set()
    for track in ("direct", "kalman"):
        for seed in TRAINING_SEEDS:
            p = run_output_dir(root, track, seed)
            assert p not in paths, f"duplicate output path: {p}"
            paths.add(p)
    assert run_output_dir(root, "direct", 42) != run_output_dir(root, "kalman", 42)
    assert run_output_dir(root, "direct", 42) != run_output_dir(root, "direct", 43)
    assert "direct" in str(run_output_dir(root, "direct", 42))
    assert "seed_42" in str(run_output_dir(root, "direct", 42))


# ---------------------------------------------------------------------------
# F. Manifest contains all required fields; G. completed-run protection
# ---------------------------------------------------------------------------
def test_manifest_fields_and_completed_run_protection():
    tmp = Path(tempfile.mkdtemp(prefix="ppo_protocol_test_"))
    try:
        output_dir = tmp / "direct_seed42"
        manifest = run_training(
            track="direct",
            seed=42,
            output_dir=output_dir,
            total_timesteps=64,
            validation_seed_offset=10_000,
            validation_episodes=2,
            checkpoint_freq=2048,
            eval_freq=2048,
            overwrite=True,
            verbose=0,
        )

        required_fields = [
            "status", "git_commit", "git_branch", "track", "estimator",
            "training_seed", "training_timesteps", "validation_seed_offset",
            "validation_episodes", "validation_seed_range", "ppo_hyperparameters",
            "checkpoint_freq", "eval_freq", "env_config", "python_version",
            "numpy_version", "torch_version", "stable_baselines3_version",
            "device", "start_timestamp", "end_timestamp", "network_architecture",
            "best_validation_success_rate", "best_checkpoint_timestep",
            "final_validation_success_rate", "validation_history",
            "training_loss_history",
        ]
        for field in required_fields:
            assert field in manifest, f"missing manifest field: {field}"
        assert manifest["status"] == "completed"
        assert manifest["track"] == "direct"
        assert manifest["training_seed"] == 42
        assert (output_dir / "best_model.zip").exists()
        assert (output_dir / "final_model.zip").exists()
        assert (output_dir / "training_manifest.json").exists()

        # G. Completed-run protection: re-calling without overwrite must NOT
        # retrain (same start_timestamp proves the run was skipped, not redone).
        manifest_again = run_training(
            track="direct",
            seed=42,
            output_dir=output_dir,
            total_timesteps=64,
            validation_seed_offset=10_000,
            validation_episodes=2,
            checkpoint_freq=2048,
            eval_freq=2048,
            overwrite=False,
            verbose=0,
        )
        assert manifest_again["start_timestamp"] == manifest["start_timestamp"], (
            "completed-run protection failed: run was retrained"
        )

        expected = expected_run_signature(
            "direct", 42, 64, manifest["ppo_hyperparameters"], manifest["git_commit"]
        )
        assert is_run_complete(output_dir, expected)

        # A DIFFERENT seed must NOT be considered complete against this dir's manifest.
        different_seed_expected = expected_run_signature(
            "direct", 43, 64, manifest["ppo_hyperparameters"], manifest["git_commit"]
        )
        assert not is_run_complete(output_dir, different_seed_expected)
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


# ---------------------------------------------------------------------------
# H, I. Explicit use_kalman_tracking per track
# ---------------------------------------------------------------------------
def test_direct_config_explicit_false():
    config = build_env_config("direct")
    assert config.use_kalman_tracking is False


def test_kalman_config_explicit_true():
    config = build_env_config("kalman")
    assert config.use_kalman_tracking is True


# ---------------------------------------------------------------------------
# J. All other behavioral EnvConfig fields match
# ---------------------------------------------------------------------------
def test_env_configs_identical_except_estimator():
    direct_dict = dataclasses.asdict(build_env_config("direct"))
    kalman_dict = dataclasses.asdict(build_env_config("kalman"))

    assert direct_dict["use_kalman_tracking"] is False
    assert kalman_dict["use_kalman_tracking"] is True

    direct_dict.pop("use_kalman_tracking")
    kalman_dict.pop("use_kalman_tracking")
    assert direct_dict == kalman_dict, (
        set(direct_dict.items()) ^ set(kalman_dict.items())
    )


# ---------------------------------------------------------------------------
# K. 200k/400k campaign output roots cannot collide (feature/ppo-400k-replication)
# ---------------------------------------------------------------------------
def test_campaign_output_roots_no_collision():
    assert CAMPAIGN_400K_TIMESTEPS == 400_000
    assert CAMPAIGN_200K_DIRNAME == "training"
    assert CAMPAIGN_400K_DIRNAME == "training_400k"

    root = Path("results")
    paths_200k = set()
    paths_400k = set()
    for track in ("direct", "kalman"):
        for seed in TRAINING_SEEDS:
            p200 = run_output_dir(root / CAMPAIGN_200K_DIRNAME, track, seed)
            p400 = run_output_dir(root / CAMPAIGN_400K_DIRNAME, track, seed)
            assert p200 != p400, f"200k/400k path collision: {p200}"
            assert p200 not in paths_400k and p400 not in paths_200k, (
                "a 400k run could overwrite its corresponding 200k run (or vice versa)"
            )
            assert p400 not in paths_400k, f"duplicate 400k output path: {p400}"
            paths_200k.add(p200)
            paths_400k.add(p400)

    assert len(paths_200k) == 6, "expected 6 unique 200k paths (2 tracks x 3 seeds)"
    assert len(paths_400k) == 6, "expected 6 unique 400k paths (2 tracks x 3 seeds)"
    assert paths_200k.isdisjoint(paths_400k)


# ---------------------------------------------------------------------------
# L. Campaign metadata fields are recorded in the manifest
# ---------------------------------------------------------------------------
def test_campaign_metadata_fields():
    tmp = Path(tempfile.mkdtemp(prefix="ppo_campaign_meta_test_"))
    try:
        output_dir = tmp / "direct_seed42_400k"
        manifest = run_training(
            track="direct",
            seed=42,
            output_dir=output_dir,
            total_timesteps=64,
            validation_seed_offset=10_000,
            validation_episodes=2,
            checkpoint_freq=2048,
            eval_freq=2048,
            overwrite=True,
            verbose=0,
            campaign="400k_replication",
        )
        assert manifest["campaign"] == "400k_replication"
        assert manifest["training_budget_timesteps"] == 64
        assert manifest["training_mode"] == "fresh_from_scratch"
        assert manifest["previous_200k_campaign_preserved"] is True
        assert manifest["validation_seed_start"] == 10_000
        assert manifest["validation_seed_end"] == 10_001
    finally:
        shutil.rmtree(tmp, ignore_errors=True)


def main() -> int:
    tests = [
        ("A. Training seeds are (42, 43, 44)", test_training_seeds),
        ("B. Validation range and reserved-range non-overlap", test_validation_range_no_overlap),
        ("C. Direct/Kalman PPO hyperparameters identical", test_hyperparameters_identical),
        ("D. Direct/Kalman network architecture identical", test_network_architecture_identical),
        ("E. Run output paths differ by track/seed", test_output_paths_no_collision),
        ("F/G. Manifest fields + completed-run protection", test_manifest_fields_and_completed_run_protection),
        ("H. Direct config use_kalman_tracking=False", test_direct_config_explicit_false),
        ("I. Kalman config use_kalman_tracking=True", test_kalman_config_explicit_true),
        ("J. All other EnvConfig fields match", test_env_configs_identical_except_estimator),
        ("K. 200k/400k campaign output roots cannot collide", test_campaign_output_roots_no_collision),
        ("L. Campaign metadata fields recorded in manifest", test_campaign_metadata_fields),
    ]

    print("=== Training Protocol Tests ===\n")
    n_pass = 0
    n_fail = 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            n_pass += 1
        except AssertionError as e:
            print(f"[FAIL] {name}: {e}")
            n_fail += 1
        except Exception as e:  # noqa: BLE001
            print(f"[ERROR] {name}: {type(e).__name__}: {e}")
            n_fail += 1

    print(f"\n{n_pass} passed, {n_fail} failed out of {len(tests)}")
    return 1 if n_fail else 0


if __name__ == "__main__":
    raise SystemExit(main())
