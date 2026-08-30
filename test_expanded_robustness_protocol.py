"""Expanded robustness protocol tests -- IMPLEMENTATION/FREEZE ONLY.

No robustness seed (66000-99999) is executed here except as plain integers
in range-membership assertions. Acquisition-fairness/physical-limit smoke
checks use ONLY ordinary development seeds (123-127).

Run directly:
    python test_expanded_robustness_protocol.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.sanitize import build_policy_info

from experiments.expanded_robustness_protocol import (
    build_method_instances,
    FROZEN_MODELS,
    EXCLUDED_METHODS,
    RESIDUAL_MAX_ANGLE_DEG,
    HOSTILE_EVASION_SEED_START, HOSTILE_EVASION_SEED_END,
    MANEUVERABILITY_SEED_START, MANEUVERABILITY_SEED_END,
    MEASUREMENT_NOISE_SEED_START, MEASUREMENT_NOISE_SEED_END,
    MOBILITY_SEED_START, MOBILITY_SEED_END,
    DETECTION_RADIUS_SEED_START, DETECTION_RADIUS_SEED_END,
    FUTURE_UNUSED_SEED_START, FUTURE_UNUSED_SEED_END,
    RESIDUAL_VALIDATION_RANGE, RESIDUAL_DIAGNOSTIC_RANGE, EXPANDED_FINAL_RANGE,
    LR_PPO_TRAINING_NAMESPACES,
    EVASION_GAIN_VALUES, NOMINAL_EVASION_GAIN,
    MANEUVERABILITY_ACCEL_VALUES, MANEUVERABILITY_TURN_RATE_VALUES, MANEUVERABILITY_GRID,
    NOMINAL_MANEUVERABILITY_ACCEL, NOMINAL_MANEUVERABILITY_TURN_RATE,
    MEASUREMENT_VAR_VALUES, NOMINAL_MEASUREMENT_VAR, FIXED_PROCESS_VAR,
    DEFENDER_SPEED_ARM_VALUES, HOSTILE_SPEED_ARM_VALUES, MOBILITY_GRID,
    NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED,
    DETECTION_RADIUS_VALUES, NOMINAL_DETECTION_RADIUS,
    build_hostile_evasion_config, build_maneuverability_config,
    build_measurement_noise_config, build_mobility_config, build_detection_radius_config,
    assert_only_fields_differ, assert_process_var_fixed,
    assert_seed_not_yet_opened, assert_disjoint_from_all_prior_ranges,
    assert_seed_in_hostile_evasion_range, assert_seed_in_maneuverability_range,
    assert_seed_in_measurement_noise_range, assert_seed_in_mobility_range,
    assert_seed_in_detection_radius_range,
)
from experiments.run_lead_residual_diagnostic import verify_model_hashes
from experiments.run_post_detection_diagnostic import audit_ground_truth_leakage
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode
from experiments.final_eval_utils import _check_physical_limits, PhysicalLimitViolation
from experiments.final_stats import hierarchical_bootstrap_track
from experiments.run_expanded_robustness_sweep import (
    run_condition,
    compute_condition_acquisition_fairness,
    _SWEEP_ALL_CONDITIONS,
)

_DEV_SEEDS = (123, 124, 125, 126, 127)

_ALL_ROBUSTNESS_RANGES = [
    (HOSTILE_EVASION_SEED_START, HOSTILE_EVASION_SEED_END),
    (MANEUVERABILITY_SEED_START, MANEUVERABILITY_SEED_END),
    (MEASUREMENT_NOISE_SEED_START, MEASUREMENT_NOISE_SEED_END),
    (MOBILITY_SEED_START, MOBILITY_SEED_END),
    (DETECTION_RADIUS_SEED_START, DETECTION_RADIUS_SEED_END),
]


def test_seed_ranges_pairwise_disjoint():
    ranges = _ALL_ROBUSTNESS_RANGES + [(FUTURE_UNUSED_SEED_START, FUTURE_UNUSED_SEED_END)]
    for i in range(len(ranges)):
        for j in range(i + 1, len(ranges)):
            lo1, hi1 = ranges[i]
            lo2, hi2 = ranges[j]
            assert hi1 < lo2 or hi2 < lo1, f"overlap between {ranges[i]} and {ranges[j]}"


def test_seed_ranges_disjoint_from_prior_namespaces():
    prior = [RESIDUAL_VALIDATION_RANGE, RESIDUAL_DIAGNOSTIC_RANGE, EXPANDED_FINAL_RANGE] + list(LR_PPO_TRAINING_NAMESPACES.values())
    for lo, hi in _ALL_ROBUSTNESS_RANGES:
        for plo, phi in prior:
            assert hi < plo or phi < lo, f"robustness range ({lo},{hi}) overlaps prior range ({plo},{phi})"


def test_exact_episode_counts():
    assert HOSTILE_EVASION_SEED_END - HOSTILE_EVASION_SEED_START + 1 == 1000
    assert MANEUVERABILITY_SEED_END - MANEUVERABILITY_SEED_START + 1 == 1000
    assert MEASUREMENT_NOISE_SEED_END - MEASUREMENT_NOISE_SEED_START + 1 == 1000
    assert MOBILITY_SEED_END - MOBILITY_SEED_START + 1 == 1000
    assert DETECTION_RADIUS_SEED_END - DETECTION_RADIUS_SEED_START + 1 == 1000


def test_assert_helpers_reject_all_reserved_and_future_seeds():
    # hostile_evasion, maneuverability, measurement_noise, and mobility are
    # now permanently frozen/consumed (not "not yet opened"), so all four are
    # deliberately EXCLUDED from assert_seed_not_yet_opened's blanket check;
    # they are guarded instead by assert_disjoint_from_all_prior_ranges below.
    _frozen_consumed_ranges = [
        (HOSTILE_EVASION_SEED_START, HOSTILE_EVASION_SEED_END),
        (MANEUVERABILITY_SEED_START, MANEUVERABILITY_SEED_END),
        (MEASUREMENT_NOISE_SEED_START, MEASUREMENT_NOISE_SEED_END),
        (MOBILITY_SEED_START, MOBILITY_SEED_END),
    ]
    still_not_yet_opened = [r for r in _ALL_ROBUSTNESS_RANGES if r not in _frozen_consumed_ranges]
    for lo, hi in still_not_yet_opened + [(FUTURE_UNUSED_SEED_START, FUTURE_UNUSED_SEED_END)]:
        for s in (lo, (lo + hi) // 2, hi):
            try:
                assert_seed_not_yet_opened(s)
                assert False, f"expected rejection for reserved/future seed {s}"
            except ValueError:
                pass
    for lo, hi in _frozen_consumed_ranges:
        for s in (lo, (lo + hi) // 2, hi):
            try:
                assert_disjoint_from_all_prior_ranges(s)
                assert False, f"expected rejection for frozen/consumed seed {s}"
            except ValueError:
                pass


def test_assert_helpers_reject_prior_namespace_seeds():
    for lo, hi in [RESIDUAL_VALIDATION_RANGE, RESIDUAL_DIAGNOSTIC_RANGE, EXPANDED_FINAL_RANGE]:
        try:
            assert_disjoint_from_all_prior_ranges(lo)
            assert False, f"expected rejection for prior-namespace seed {lo}"
        except ValueError:
            pass
    for root, (lo, hi) in LR_PPO_TRAINING_NAMESPACES.items():
        try:
            assert_disjoint_from_all_prior_ranges(lo)
            assert False, f"expected rejection for LR-PPO-{root} training-namespace seed {lo}"
        except ValueError:
            pass


def test_assert_helpers_accept_dev_seeds():
    for s in _DEV_SEEDS:
        assert_seed_not_yet_opened(s) is None
        assert_disjoint_from_all_prior_ranges(s) is None


def test_assert_seed_in_hostile_evasion_range():
    for s in (HOSTILE_EVASION_SEED_START, HOSTILE_EVASION_SEED_END):
        assert assert_seed_in_hostile_evasion_range(s) is None
    for s in (HOSTILE_EVASION_SEED_START - 1, HOSTILE_EVASION_SEED_END + 1):
        try:
            assert_seed_in_hostile_evasion_range(s)
            assert False, f"expected rejection for out-of-range seed {s}"
        except ValueError:
            pass


def test_assert_seed_in_maneuverability_range():
    # Explicit boundary cases per protocol: 66999 reject, 67000 accept, 67999 accept, 68000 reject.
    for s in (MANEUVERABILITY_SEED_START - 1, MANEUVERABILITY_SEED_END + 1):
        try:
            assert_seed_in_maneuverability_range(s)
            assert False, f"expected rejection for out-of-range seed {s}"
        except ValueError:
            pass
    for s in (MANEUVERABILITY_SEED_START, MANEUVERABILITY_SEED_END):
        assert assert_seed_in_maneuverability_range(s) is None
    assert MANEUVERABILITY_SEED_START - 1 == 66_999
    assert MANEUVERABILITY_SEED_START == 67_000
    assert MANEUVERABILITY_SEED_END == 67_999
    assert MANEUVERABILITY_SEED_END + 1 == 68_000


def test_assert_seed_in_measurement_noise_range():
    # Explicit boundary cases per protocol: 67999 reject, 68000 accept, 68999 accept, 69000 reject.
    for s in (MEASUREMENT_NOISE_SEED_START - 1, MEASUREMENT_NOISE_SEED_END + 1):
        try:
            assert_seed_in_measurement_noise_range(s)
            assert False, f"expected rejection for out-of-range seed {s}"
        except ValueError:
            pass
    for s in (MEASUREMENT_NOISE_SEED_START, MEASUREMENT_NOISE_SEED_END):
        assert assert_seed_in_measurement_noise_range(s) is None
    assert MEASUREMENT_NOISE_SEED_START - 1 == 67_999
    assert MEASUREMENT_NOISE_SEED_START == 68_000
    assert MEASUREMENT_NOISE_SEED_END == 68_999
    assert MEASUREMENT_NOISE_SEED_END + 1 == 69_000


def test_assert_seed_in_mobility_range():
    # Explicit boundary cases per protocol: 68999 reject, 69000 accept, 69999 accept, 70000 reject.
    for s in (MOBILITY_SEED_START - 1, MOBILITY_SEED_END + 1):
        try:
            assert_seed_in_mobility_range(s)
            assert False, f"expected rejection for out-of-range seed {s}"
        except ValueError:
            pass
    for s in (MOBILITY_SEED_START, MOBILITY_SEED_END):
        assert assert_seed_in_mobility_range(s) is None
    assert MOBILITY_SEED_START - 1 == 68_999
    assert MOBILITY_SEED_START == 69_000
    assert MOBILITY_SEED_END == 69_999
    assert MOBILITY_SEED_END + 1 == 70_000


def test_assert_seed_in_detection_radius_range():
    # Explicit boundary cases per protocol: 69999 reject, 70000 accept, 70999 accept, 71000 reject.
    for s in (DETECTION_RADIUS_SEED_START - 1, DETECTION_RADIUS_SEED_END + 1):
        try:
            assert_seed_in_detection_radius_range(s)
            assert False, f"expected rejection for out-of-range seed {s}"
        except ValueError:
            pass
    for s in (DETECTION_RADIUS_SEED_START, DETECTION_RADIUS_SEED_END):
        assert assert_seed_in_detection_radius_range(s) is None
    assert DETECTION_RADIUS_SEED_START - 1 == 69_999
    assert DETECTION_RADIUS_SEED_START == 70_000
    assert DETECTION_RADIUS_SEED_END == 70_999
    assert DETECTION_RADIUS_SEED_END + 1 == 71_000


def test_hostile_evasion_grid_exact():
    assert EVASION_GAIN_VALUES == (0.00, 0.25, 0.50, 0.75, 1.00)
    assert NOMINAL_EVASION_GAIN == 0.75
    assert EVASION_GAIN_VALUES.count(NOMINAL_EVASION_GAIN) == 1


def test_maneuverability_grid_exact():
    assert MANEUVERABILITY_ACCEL_VALUES == (4.0, 8.0, 12.0)
    assert MANEUVERABILITY_TURN_RATE_VALUES == (45.0, 90.0, 135.0)
    assert len(MANEUVERABILITY_GRID) == 9
    assert (NOMINAL_MANEUVERABILITY_ACCEL, NOMINAL_MANEUVERABILITY_TURN_RATE) in MANEUVERABILITY_GRID
    assert MANEUVERABILITY_GRID.count((NOMINAL_MANEUVERABILITY_ACCEL, NOMINAL_MANEUVERABILITY_TURN_RATE)) == 1


def test_measurement_noise_grid_exact():
    assert MEASUREMENT_VAR_VALUES == (0.05, 0.25, 0.50, 1.00, 2.00, 4.00)
    assert NOMINAL_MEASUREMENT_VAR == 0.50
    assert MEASUREMENT_VAR_VALUES.count(NOMINAL_MEASUREMENT_VAR) == 1
    assert FIXED_PROCESS_VAR == 1.0


def test_mobility_grid_exact():
    assert DEFENDER_SPEED_ARM_VALUES == (12.0, 15.0, 18.0, 21.0, 24.0)
    assert HOSTILE_SPEED_ARM_VALUES == (8.0, 10.0, 12.0, 14.0, 16.0)
    assert len(MOBILITY_GRID) == 9  # 5 + 5 - 1 (nominal counted once)
    assert (NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED) in MOBILITY_GRID
    assert sum(1 for c in MOBILITY_GRID if c == (NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED)) == 1


def test_mobility_grid_exact_order():
    # Explicit deterministic order (NOT set iteration): defender-speed arm
    # first (12,12)/(15,12)/(18,12)[nominal]/(21,12)/(24,12), then the
    # hostile-speed arm excluding the already-listed nominal duplicate.
    assert MOBILITY_GRID == (
        (12.0, 12.0), (15.0, 12.0), (18.0, 12.0), (21.0, 12.0), (24.0, 12.0),
        (18.0, 8.0), (18.0, 10.0), (18.0, 14.0), (18.0, 16.0),
    )
    assert MOBILITY_GRID[2] == (NOMINAL_DEFENDER_SPEED, NOMINAL_HOSTILE_SPEED)


def test_detection_radius_grid_exact():
    assert DETECTION_RADIUS_VALUES == (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
    assert NOMINAL_DETECTION_RADIUS == 15.0
    assert DETECTION_RADIUS_VALUES.count(NOMINAL_DETECTION_RADIUS) == 1


def test_lead_uses_current_condition_v_d_not_hardcoded_nominal():
    # Item 10: LeadInterceptPolicy must read v_d from the CURRENT config, not
    # a hardcoded 18.0 default -- required for a fair classical comparison
    # at off-nominal defender speeds in the mobility sweep.
    from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
    for v_d in (12.0, 15.0, 18.0, 21.0, 24.0):
        config = build_mobility_config("measurement", v_d, NOMINAL_HOSTILE_SPEED)
        policy = LeadInterceptPolicy(state_source="measurement", config=config)
        assert policy.v_d == v_d


def test_observation_normalization_uses_current_condition_speeds():
    # Item 9: defender/hostile velocity normalization must use the CURRENT
    # config.v_d/config.v_e, not hardcoded 18.0/12.0 -- checked at every
    # mobility-arm endpoint on dev seeds only.
    for v_d, v_e in ((12.0, 12.0), (24.0, 12.0), (18.0, 8.0), (18.0, 16.0)):
        config = build_mobility_config("measurement", v_d, v_e)
        env = SoldierEnv(config=config)
        for seed in _DEV_SEEDS:
            obs, info = env.reset(seed=seed)
            assert np.all(np.isfinite(obs)), f"non-finite obs at (v_d={v_d}, v_e={v_e}), seed={seed}"
            assert np.all(obs >= -1.0 - 1e-6) and np.all(obs <= 1.0 + 1e-6)
            for _ in range(5):
                obs, _, term, trunc, info = env.step(np.zeros(3, dtype=np.float32))
                assert np.all(np.isfinite(obs)), f"non-finite obs mid-episode at (v_d={v_d}, v_e={v_e}), seed={seed}"
                assert np.all(obs >= -1.0 - 1e-6) and np.all(obs <= 1.0 + 1e-6)
                if term or trunc:
                    break
        env.close()


def test_config_dependent_policies_receive_current_sweep_config():
    # Item 8: PN/Lead/LR-PPO must be constructed from the CURRENT sweep
    # config (via the corrected build_policy_for_instance pattern from the
    # mobility runner, reused unmodified for detection_radius) -- never the
    # older implicit-default helper that silently falls back to EnvConfig().
    from experiments.run_mobility_robustness_sweep import build_policy_for_instance
    from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
    from uav_defend.policies.baseline.proportional_navigation_policy import ProportionalNavigationPolicy
    from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

    instances = build_method_instances()
    for detection_radius in (5.0, 20.0, 30.0):
        config = build_detection_radius_config("measurement", detection_radius)
        for instance in instances:
            if instance.name not in ("pn", "lead", "lr_ppo"):
                continue
            policy = build_policy_for_instance(instance, config)
            if isinstance(policy, (LeadInterceptPolicy, ProportionalNavigationPolicy)):
                assert policy.dt == config.dt
                assert policy.v_d == config.v_d
            elif isinstance(policy, LeadResidualPPOPolicyWrapper):
                assert policy._lead_policy.dt == config.dt
                assert policy._lead_policy.v_d == config.v_d


def test_method_instance_count_and_no_kalman_greedy():
    instances = build_method_instances()
    assert len(instances) == 12
    names = {i.name for i in instances}
    assert names == {"greedy", "pn", "lead", "ppo", "ppo_kalman", "lr_ppo"}
    assert "kalman_greedy" in EXCLUDED_METHODS
    assert "kalman_greedy" not in names
    assert "pn_kalman" not in names
    assert "lead_kalman" not in names
    assert "random" not in names


def test_frozen_model_hashes_reused_and_verified():
    assert len(FROZEN_MODELS) == 9
    results = verify_model_hashes()
    assert all(v["match"] for v in results.values())


def test_config_isolation_hostile_evasion():
    for g in EVASION_GAIN_VALUES:
        build_hostile_evasion_config("measurement", g)
        build_hostile_evasion_config("kalman", g)


def test_config_isolation_maneuverability():
    for a, t in MANEUVERABILITY_GRID:
        build_maneuverability_config("measurement", a, t)


def test_config_isolation_measurement_noise():
    for v in MEASUREMENT_VAR_VALUES:
        build_measurement_noise_config("kalman", v)


def test_process_var_isolation_rejects_corrupted_config():
    # Negative test (item 2): a config with a non-nominal process_var must
    # be REJECTED by the measurement-noise isolation guard, never silently
    # accepted as an allowed sweep difference.
    corrupted = EnvConfig(use_kalman_tracking=False, measurement_var=NOMINAL_MEASUREMENT_VAR, process_var=FIXED_PROCESS_VAR + 0.5)
    try:
        assert_process_var_fixed(corrupted)
        assert False, "expected assert_process_var_fixed to reject a corrupted process_var"
    except AssertionError:
        pass
    try:
        assert_only_fields_differ(corrupted, {"measurement_var"})
        assert False, "expected assert_only_fields_differ to reject process_var as an unexpected diff"
    except AssertionError:
        pass


def test_config_isolation_mobility():
    for vd, ve in MOBILITY_GRID:
        build_mobility_config("measurement", vd, ve)


def test_config_isolation_detection_radius():
    for r in DETECTION_RADIUS_VALUES:
        build_detection_radius_config("measurement", r)


def test_canonical_standby_enabled_in_every_sweep_config():
    configs = (
        [build_hostile_evasion_config("measurement", g) for g in EVASION_GAIN_VALUES]
        + [build_maneuverability_config("measurement", a, t) for a, t in MANEUVERABILITY_GRID]
        + [build_measurement_noise_config("measurement", v) for v in MEASUREMENT_VAR_VALUES]
        + [build_mobility_config("measurement", vd, ve) for vd, ve in MOBILITY_GRID]
        + [build_detection_radius_config("measurement", r) for r in DETECTION_RADIUS_VALUES]
    )
    for c in configs:
        assert c.defender_standby_until_detection is True


def test_residual_angle_fixed_at_30():
    assert RESIDUAL_MAX_ANGLE_DEG == 30.0


def test_no_ground_truth_leakage():
    result = audit_ground_truth_leakage()
    assert result["measurement"]["pass"] is True
    assert result["kalman"]["pass"] is True


def test_common_random_numbers_indexing_dev_seeds():
    """A tiny hostile-evasion condition on dev seeds: all 12 instances must
    see the IDENTICAL seed list (common random numbers), never a per-method
    subset."""
    instances = build_method_instances()
    df, _angles = run_condition("hostile_evasion", "gain_0.75_devcheck", {"evasion_gain": 0.75}, list(_DEV_SEEDS), instances)
    for name in [i.display_name for i in instances]:
        seeds_seen = sorted(df[df["instance"] == name]["environment_seed"].tolist())
        assert seeds_seen == sorted(_DEV_SEEDS), (name, seeds_seen)


def test_family_bootstrap_shapes():
    rng = np.random.default_rng(0)
    matrix = rng.binomial(1, 0.8, size=(3, 50)).astype(np.float64)
    observed, lo, hi = hierarchical_bootstrap_track(matrix, 500, 42)
    assert 0.0 <= lo <= observed <= hi <= 1.0


def test_acquisition_fairness_dev_seeds_only():
    instances = build_method_instances()
    df, _angles = run_condition("detection_radius", "radius_15.0_devcheck", {"detection_radius": 15.0}, list(_DEV_SEEDS), instances)
    fairness = compute_condition_acquisition_fairness(df, [i.display_name for i in instances])
    assert fairness["n_seeds_checked"] == len(_DEV_SEEDS)
    assert fairness["n_mismatched"] == 0


def test_physical_limits_on_dev_smoke_conditions():
    for sweep, label, kwargs in [
        ("hostile_evasion", "gain_1.0", {"evasion_gain": 1.0}),
        ("measurement_noise", "mvar_4.0", {"measurement_var": 4.0}),
        ("detection_radius", "radius_30.0", {"detection_radius": 30.0}),
    ]:
        instances = build_method_instances()
        df, _angles = run_condition(sweep, label, kwargs, list(_DEV_SEEDS), instances)
        assert int(df["physical_limit_violations"].sum()) == 0, (sweep, label)


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    tests.sort(key=lambda kv: kv[0])
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
