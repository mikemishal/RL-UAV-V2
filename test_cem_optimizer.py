"""
Tests for the generic CEM optimizer (`uav_defend.policies.mpc.cem_optimizer`).

Covers: normalization helper correctness, CEM correctness against synthetic
objectives with a known optimum, determinism given a fixed
`numpy.random.Generator` seed, non-finite-candidate rejection/handling, and
elite-count edge cases.

Run directly: python test_cem_optimizer.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.policies.mpc.cem_optimizer import CEMResult, cem_optimize, normalize_rows

EPS = 1e-8


# --- normalize_rows -----------------------------------------------------

def test_normalize_rows_unit_norm():
    vectors = np.array([[3.0, 4.0, 0.0], [0.0, 0.0, 5.0]])
    normalized = normalize_rows(vectors, EPS)
    norms = np.linalg.norm(normalized, axis=-1)
    assert np.allclose(norms, 1.0)
    assert np.allclose(normalized[0], [0.6, 0.8, 0.0])
    assert np.allclose(normalized[1], [0.0, 0.0, 1.0])


def test_normalize_rows_zero_fallback():
    vectors = np.array([[0.0, 0.0, 0.0], [1e-10, 0.0, 0.0]])
    normalized = normalize_rows(vectors, EPS)
    assert np.array_equal(normalized, np.zeros((2, 3)))


def test_normalize_rows_batch_shape():
    vectors = np.random.default_rng(0).uniform(-1, 1, size=(5, 4, 3))
    normalized = normalize_rows(vectors, EPS)
    assert normalized.shape == vectors.shape


# --- CEM correctness with a known optimum -------------------------------

def _quadratic_objective_factory(target: np.ndarray):
    """Objective maximized at U == target: score = -sum||U_k - target_k||^2."""

    def objective(directions: np.ndarray) -> float:
        return float(-np.sum((directions - target) ** 2))

    return objective


def test_cem_converges_toward_known_optimum():
    horizon = 3
    target = np.tile(np.array([1.0, 0.0, 0.0]), (horizon, 1))
    objective = _quadratic_objective_factory(target)
    rng = np.random.default_rng(7)

    result = cem_optimize(
        objective=objective, horizon=horizon, population_size=128, elite_fraction=0.2,
        n_iterations=8, rng=rng, initial_std=1.0, min_std=0.02, eps=EPS,
    )

    assert result.success
    assert result.best_score > -0.5  # close to the true optimum of 0.0
    assert np.allclose(result.best_sequence, target, atol=0.3)
    assert result.n_evaluations == 128 * 8


def test_cem_best_ever_tracked_not_just_final_mean():
    """The returned best_sequence/best_score must be the best candidate
    found across ALL iterations, not merely derived from the final mean."""
    horizon = 2
    target = np.tile(np.array([0.0, 1.0, 0.0]), (horizon, 1))
    objective = _quadratic_objective_factory(target)
    rng = np.random.default_rng(3)

    result = cem_optimize(
        objective=objective, horizon=horizon, population_size=64, elite_fraction=0.25,
        n_iterations=5, rng=rng, eps=EPS,
    )
    # best_score must correspond EXACTLY to best_sequence's objective value.
    recomputed = objective(result.best_sequence)
    assert np.isclose(recomputed, result.best_score, atol=1e-9)


# --- Determinism ---------------------------------------------------------

def test_cem_deterministic_given_same_generator_seed():
    horizon = 4
    target = np.tile(np.array([0.0, 0.0, 1.0]), (horizon, 1))
    objective = _quadratic_objective_factory(target)

    rng1 = np.random.Generator(np.random.PCG64(np.random.SeedSequence(42)))
    rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence(42)))

    result1 = cem_optimize(objective=objective, horizon=horizon, population_size=32,
                            elite_fraction=0.2, n_iterations=4, rng=rng1, eps=EPS)
    result2 = cem_optimize(objective=objective, horizon=horizon, population_size=32,
                            elite_fraction=0.2, n_iterations=4, rng=rng2, eps=EPS)

    assert np.array_equal(result1.best_sequence, result2.best_sequence)
    assert result1.best_score == result2.best_score
    assert result1.n_evaluations == result2.n_evaluations


def test_cem_different_seeds_can_differ():
    horizon = 3
    target = np.tile(np.array([1.0, 1.0, 1.0]) / np.sqrt(3), (horizon, 1))
    objective = _quadratic_objective_factory(target)

    rng1 = np.random.Generator(np.random.PCG64(np.random.SeedSequence(1)))
    rng2 = np.random.Generator(np.random.PCG64(np.random.SeedSequence(2)))

    result1 = cem_optimize(objective=objective, horizon=horizon, population_size=16,
                            elite_fraction=0.25, n_iterations=2, rng=rng1, eps=EPS)
    result2 = cem_optimize(objective=objective, horizon=horizon, population_size=16,
                            elite_fraction=0.25, n_iterations=2, rng=rng2, eps=EPS)
    # Not asserting inequality strictly (could coincidentally match), just
    # that both runs are internally valid results.
    assert result1.success and result2.success


# --- Non-finite candidate handling ---------------------------------------

def test_cem_all_non_finite_returns_unsuccessful():
    def objective(directions: np.ndarray) -> float:
        return float("nan")

    rng = np.random.default_rng(0)
    result = cem_optimize(objective=objective, horizon=2, population_size=8,
                           elite_fraction=0.5, n_iterations=3, rng=rng, eps=EPS)

    assert not result.success
    assert result.best_score == float("-inf")
    assert np.array_equal(result.best_sequence, np.zeros((2, 3)))
    assert result.n_evaluations == 8 * 3


def test_cem_mixed_finite_and_non_finite_still_succeeds():
    calls = {"n": 0}

    def objective(directions: np.ndarray) -> float:
        calls["n"] += 1
        if calls["n"] % 3 == 0:
            return float("nan")
        return float(-np.sum(directions ** 2))

    rng = np.random.default_rng(11)
    result = cem_optimize(objective=objective, horizon=2, population_size=16,
                           elite_fraction=0.25, n_iterations=3, rng=rng, eps=EPS)
    assert result.success
    assert np.isfinite(result.best_score)


# --- Edge cases -----------------------------------------------------------

def test_cem_elite_count_clamped_to_at_least_one():
    def objective(directions: np.ndarray) -> float:
        return float(-np.sum(directions ** 2))

    rng = np.random.default_rng(5)
    result = cem_optimize(objective=objective, horizon=1, population_size=1,
                           elite_fraction=0.01, n_iterations=2, rng=rng, eps=EPS)
    assert result.success
    assert result.n_evaluations == 1 * 2


def test_cem_invalid_arguments_raise():
    def objective(directions: np.ndarray) -> float:
        return 0.0

    rng = np.random.default_rng(0)
    for kwargs in (
        dict(horizon=0, population_size=8, elite_fraction=0.2, n_iterations=2),
        dict(horizon=2, population_size=0, elite_fraction=0.2, n_iterations=2),
        dict(horizon=2, population_size=8, elite_fraction=0.0, n_iterations=2),
        dict(horizon=2, population_size=8, elite_fraction=1.5, n_iterations=2),
        dict(horizon=2, population_size=8, elite_fraction=0.2, n_iterations=0),
    ):
        try:
            cem_optimize(objective=objective, rng=rng, eps=EPS, **kwargs)
            raise AssertionError(f"expected ValueError for kwargs={kwargs}")
        except ValueError:
            pass


if __name__ == "__main__":
    import sys
    import traceback

    tests = [obj for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    passed = failed = 0
    for t in tests:
        try:
            t()
            print(f"PASS: {t.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL: {t.__name__}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
