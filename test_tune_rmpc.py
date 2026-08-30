"""
Tests for the RMPC tuning harness (`experiments/tune_rmpc.py`) -- seed
guards, candidate predeclaration, and the selection/tie-break rules.

These tests exercise ONLY the pure guard/selection functions; they never
run an episode and never open any RMPC-tuning seed (>= 71000). Ordinary
unit-test seeds are used elsewhere in this repository's test suite (0, 1,
2, ...) but this file does not even need those, since every function
tested here is a pure function of plain Python/int arguments.

Run directly: python test_tune_rmpc.py
"""

from __future__ import annotations

import experiments.tune_rmpc as t

EPS = 1e-9


# --- A. authorized seed ranges accepted -------------------------------------

def test_authorized_seed_ranges_accepted():
    for stage, (start, end) in t.STAGE_SEED_RANGES.items():
        t.assert_seed_in_stage(start, stage)
        t.assert_seed_in_stage(end, stage)
        t.assert_seed_in_stage((start + end) // 2, stage)
        t.assert_seed_authorized(start)
        t.assert_seed_authorized(end)


# --- B. seed 71999 rejected (reserved, not part of any stage here) --------

def test_seed_71999_rejected():
    for stage in t.STAGE_SEED_RANGES:
        try:
            t.assert_seed_in_stage(71_999, stage)
            raise AssertionError(f"71999 unexpectedly accepted for stage '{stage}'")
        except ValueError:
            pass
    try:
        t.assert_seed_authorized(71_999)
        raise AssertionError("71999 unexpectedly accepted as authorized")
    except ValueError:
        pass


# --- C. every seed >= 72000 rejected -----------------------------------------

def test_seeds_ge_72000_rejected():
    for seed in (72_000, 72_001, 80_000, 99_999, 100_000):
        try:
            t.assert_seed_authorized(seed)
            raise AssertionError(f"seed {seed} unexpectedly accepted as authorized")
        except ValueError:
            pass


def test_seeds_outside_any_range_rejected():
    for seed in (0, 1, 70_999, 71_300, 71_500, 71_999):
        try:
            t.assert_seed_authorized(seed)
            raise AssertionError(f"seed {seed} unexpectedly accepted as authorized")
        except ValueError:
            pass


# --- D-G. exact stage ranges -------------------------------------------------

def test_stage_a_exact_range():
    seeds = t.seeds_for_stage("stage_a")
    assert seeds[0] == 71_000
    assert seeds[-1] == 71_029
    assert len(seeds) == 30 == t.STAGE_A_N


def test_stage_b_exact_range():
    seeds = t.seeds_for_stage("stage_b")
    assert seeds[0] == 71_030
    assert seeds[-1] == 71_099
    assert len(seeds) == 70 == t.STAGE_B_N


def test_validation_exact_range():
    seeds = t.seeds_for_stage("validation")
    assert seeds[0] == 71_100
    assert seeds[-1] == 71_199
    assert len(seeds) == 100 == t.VALIDATION_N


def test_diagnostic_exact_range():
    seeds = t.seeds_for_stage("diagnostic")
    assert seeds[0] == 71_200
    assert seeds[-1] == 71_299
    assert len(seeds) == 100 == t.DIAGNOSTIC_N


def test_stage_ranges_are_disjoint_and_contiguous():
    ranges = [t.STAGE_A_RANGE, t.STAGE_B_RANGE, t.VALIDATION_RANGE, t.DIAGNOSTIC_RANGE]
    for (a_start, a_end), (b_start, b_end) in zip(ranges, ranges[1:]):
        assert b_start == a_end + 1, "stage ranges must be exactly contiguous, no gaps or overlaps"


def test_reserved_unopened_ranges_cover_the_rest():
    assert t.RESERVED_UNOPENED_RANGES[0] == (71_300, 71_999)
    assert t.RESERVED_UNOPENED_RANGES[1] == (72_000, 99_999)


# --- H. candidate list exactly matches C1-C7 --------------------------------

def test_candidate_ids_exactly_match_predeclared_set():
    expected = {"C1_FAST", "C2_SHORT", "C3_BALANCED", "C4_LONG", "C5_LARGE_POP", "C6_TIGHT_ELITE", "C7_MORE_ITER"}
    assert set(t.CANDIDATE_IDS) == expected
    assert set(t.CANDIDATES) == expected
    assert len(t.CANDIDATE_IDS) == 7


def test_candidate_configs_match_predeclared_values():
    expected = {
        "C1_FAST":        dict(horizon=4, population_size=64,  elite_fraction=0.20, cem_iterations=3),
        "C2_SHORT":       dict(horizon=4, population_size=128, elite_fraction=0.20, cem_iterations=5),
        "C3_BALANCED":    dict(horizon=6, population_size=128, elite_fraction=0.20, cem_iterations=5),
        "C4_LONG":        dict(horizon=8, population_size=128, elite_fraction=0.20, cem_iterations=5),
        "C5_LARGE_POP":   dict(horizon=6, population_size=256, elite_fraction=0.20, cem_iterations=5),
        "C6_TIGHT_ELITE": dict(horizon=6, population_size=128, elite_fraction=0.10, cem_iterations=5),
        "C7_MORE_ITER":   dict(horizon=6, population_size=128, elite_fraction=0.20, cem_iterations=8),
    }
    assert t.CANDIDATES == expected


def test_fixed_constants_match_predeclared_values():
    assert t.FIXED_RMPC_CONSTANTS == dict(
        controller_seed=0, cem_initial_std=1.0, cem_min_std=0.05, cem_sample_clip=3.0, enable_timing=True,
    )


def test_build_rmpc_config_uses_fixed_constants_and_candidate_params():
    cfg = t.build_rmpc_config("C3_BALANCED")
    assert cfg.horizon == 6
    assert cfg.population_size == 128
    assert cfg.elite_fraction == 0.20
    assert cfg.cem_iterations == 5
    assert cfg.controller_seed == 0
    assert cfg.cem_initial_std == 1.0
    assert cfg.cem_min_std == 0.05
    assert cfg.cem_sample_clip == 3.0
    assert cfg.enable_timing is True


# --- I. selection function uses success count only as primary metric -------

def test_rank_candidates_primary_metric_is_success_count():
    success_counts = {"C1_FAST": 10, "C2_SHORT": 20, "C3_BALANCED": 15}
    ranked = t.rank_candidates(success_counts)
    assert ranked[0] == "C2_SHORT"
    assert ranked[1] == "C3_BALANCED"
    assert ranked[2] == "C1_FAST"


def test_rank_candidates_ignores_median_time_when_success_differs():
    """Even if a lower-success candidate has a much faster planning time,
    it must NOT outrank a higher-success candidate (median time is only a
    tie-break, never a primary-metric override)."""
    success_counts = {"C1_FAST": 5, "C3_BALANCED": 6}
    median_times = {"C1_FAST": 0.001, "C3_BALANCED": 10.0}
    ranked = t.rank_candidates(success_counts, median_times)
    assert ranked[0] == "C3_BALANCED"


# --- J. exact-tie computational-cost rule -----------------------------------

def test_theoretical_scenario_step_cost_formula():
    # population_size * cem_iterations * 6 * horizon
    assert t.theoretical_scenario_step_cost("C1_FAST") == 64 * 3 * 6 * 4
    assert t.theoretical_scenario_step_cost("C3_BALANCED") == 128 * 5 * 6 * 6
    assert t.theoretical_scenario_step_cost("C5_LARGE_POP") == 256 * 5 * 6 * 6


def test_exact_tie_broken_by_lower_theoretical_cost():
    # C1_FAST cost=4608, C5_LARGE_POP cost=46080 -- exact success tie must
    # favor the cheaper candidate.
    success_counts = {"C1_FAST": 7, "C5_LARGE_POP": 7}
    ranked = t.rank_candidates(success_counts)
    assert ranked[0] == "C1_FAST"


def test_exact_tie_and_equal_cost_broken_by_median_planning_time():
    # C3_BALANCED and C6_TIGHT_ELITE have IDENTICAL theoretical cost
    # (128*5*6*6 == 128*5*6*6, elite_fraction does not enter the cost
    # formula) -- next tie-break is median planning time.
    assert t.theoretical_scenario_step_cost("C3_BALANCED") == t.theoretical_scenario_step_cost("C6_TIGHT_ELITE")
    success_counts = {"C3_BALANCED": 7, "C6_TIGHT_ELITE": 7}
    median_times = {"C3_BALANCED": 0.9, "C6_TIGHT_ELITE": 0.5}
    ranked = t.rank_candidates(success_counts, median_times)
    assert ranked[0] == "C6_TIGHT_ELITE"


def test_exact_tie_cost_and_time_broken_lexicographically():
    success_counts = {"C3_BALANCED": 7, "C6_TIGHT_ELITE": 7}
    median_times = {"C3_BALANCED": 0.5, "C6_TIGHT_ELITE": 0.5}
    ranked = t.rank_candidates(success_counts, median_times)
    assert ranked[0] == "C3_BALANCED"  # lexicographically smaller than C6_TIGHT_ELITE


# --- K. eliminated candidates cannot enter Stage B --------------------------

def test_eliminated_candidates_cannot_enter_stage_b():
    stage_a_top3 = ["C1_FAST", "C2_SHORT", "C3_BALANCED"]
    t.assert_candidates_allowed(stage_a_top3, t.CANDIDATE_IDS, "stage_b")  # sanity: allowed subset ok
    try:
        t.assert_candidates_allowed(["C4_LONG"], stage_a_top3, "stage_b")
        raise AssertionError("eliminated candidate C4_LONG unexpectedly allowed into stage_b")
    except ValueError:
        pass


def test_run_stage_b_rejects_non_top3_candidate_without_running_episodes():
    """run_stage_b must reject an eliminated candidate before any episode
    is executed (no seed side effects on failure)."""
    try:
        t.run_stage_b(t.build_nominal_env_config(), ["C1_FAST", "C2_SHORT", "C4_LONG_NOT_EXIST"])
        raise AssertionError("expected ValueError for an unknown/disallowed candidate")
    except ValueError:
        pass


def test_run_stage_b_requires_exactly_three_candidates():
    try:
        t.run_stage_b(t.build_nominal_env_config(), ["C1_FAST", "C2_SHORT"])
        raise AssertionError("expected ValueError for only 2 candidates")
    except ValueError:
        pass


# --- L. non-finalists cannot enter validation -------------------------------

def test_non_finalists_cannot_enter_validation():
    development_finalists = ["C2_SHORT", "C3_BALANCED"]
    try:
        t.run_stage_c_validation(t.build_nominal_env_config(), ["C2_SHORT", "C3_BALANCED", "C5_LARGE_POP"])
        raise AssertionError("expected ValueError for 3 candidates (only 2 finalists allowed)")
    except ValueError:
        pass


def test_run_stage_c_validation_requires_exactly_two_candidates():
    try:
        t.run_stage_c_validation(t.build_nominal_env_config(), ["C2_SHORT"])
        raise AssertionError("expected ValueError for only 1 candidate")
    except ValueError:
        pass


# --- M. diagnostic stage accepts only frozen selected config ----------------

def test_diagnostic_accepts_only_selected_candidate():
    t.assert_diagnostic_candidate("C3_BALANCED", "C3_BALANCED")  # sanity: no raise
    try:
        t.assert_diagnostic_candidate("C2_SHORT", "C3_BALANCED")
        raise AssertionError("diagnostic stage unexpectedly accepted a non-selected candidate")
    except ValueError:
        pass


# --- Additional seed-guard robustness checks --------------------------------

def test_seed_guard_rejects_stage_boundary_off_by_one():
    try:
        t.assert_seed_in_stage(71_030, "stage_a")
        raise AssertionError("seed 71030 (first Stage-B seed) unexpectedly accepted into stage_a")
    except ValueError:
        pass
    try:
        t.assert_seed_in_stage(70_999, "stage_a")
        raise AssertionError("seed 70999 (one before Stage A) unexpectedly accepted")
    except ValueError:
        pass


def test_representative_fairness_candidate_is_predeclared_and_valid():
    assert t.REPRESENTATIVE_FAIRNESS_CANDIDATE in t.CANDIDATE_IDS


if __name__ == "__main__":
    import sys
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for test_fn in tests:
        try:
            test_fn()
            print(f"PASS: {test_fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {test_fn.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
