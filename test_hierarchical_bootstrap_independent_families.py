"""Unit tests for hierarchical_bootstrap_independent_families (new function,
added for the LR-PPO-vs-standalone-PPO family comparison, which has
unmatched/independent training roots -- see docs task for the
lead-residual-diagnostic-200 evaluation, item 15).

Run directly:
    python test_hierarchical_bootstrap_independent_families.py
"""

from __future__ import annotations

import numpy as np

from experiments.final_stats import hierarchical_bootstrap_independent_families


def test_identical_families_diff_near_zero_ci_includes_zero():
    rng = np.random.default_rng(0)
    a = rng.binomial(1, 0.85, size=(3, 200)).astype(np.float64)
    b = a.copy()
    observed, lo, hi = hierarchical_bootstrap_independent_families(a, b, 5000, 42)
    assert observed == 0.0
    assert lo <= 0.0 <= hi


def test_clearly_higher_family_diff_positive_ci_excludes_zero():
    rng = np.random.default_rng(1)
    a = rng.binomial(1, 0.95, size=(3, 300)).astype(np.float64)
    b = rng.binomial(1, 0.50, size=(3, 300)).astype(np.float64)
    observed, lo, hi = hierarchical_bootstrap_independent_families(a, b, 5000, 42)
    assert observed > 0.3
    assert lo > 0.0


def test_reproducibility_same_seed():
    rng = np.random.default_rng(2)
    a = rng.binomial(1, 0.8, size=(3, 200)).astype(np.float64)
    b = rng.binomial(1, 0.7, size=(3, 200)).astype(np.float64)
    r1 = hierarchical_bootstrap_independent_families(a, b, 2000, 123)
    r2 = hierarchical_bootstrap_independent_families(a, b, 2000, 123)
    assert r1 == r2


def test_different_seeds_give_same_observed_value():
    # The observed (unresampled) difference must not depend on the bootstrap
    # RNG seed at all (only the CI bounds are stochastic); with small
    # discrete sample spaces (3 roots x 200 episodes) percentile bounds can
    # occasionally coincide across different seeds, so only the seed-
    # independent `observed` value is asserted here.
    rng = np.random.default_rng(3)
    a = rng.binomial(1, 0.8, size=(3, 200)).astype(np.float64)
    b = rng.binomial(1, 0.7, size=(3, 200)).astype(np.float64)
    obs1, _, _ = hierarchical_bootstrap_independent_families(a, b, 2000, 1)
    obs2, _, _ = hierarchical_bootstrap_independent_families(a, b, 2000, 2)
    assert obs1 == obs2


def test_asymmetric_root_counts_supported():
    rng = np.random.default_rng(4)
    a = rng.binomial(1, 0.9, size=(3, 150)).astype(np.float64)
    b = rng.binomial(1, 0.6, size=(2, 150)).astype(np.float64)  # different n_roots
    observed, lo, hi = hierarchical_bootstrap_independent_families(a, b, 2000, 7)
    assert observed > 0
    assert lo > 0


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
