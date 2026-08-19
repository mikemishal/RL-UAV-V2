"""
Generic Cross-Entropy Method (CEM) optimizer.

Deliberately decoupled from any RMPC-specific cost/scenario logic: this
module knows nothing about hostile scenarios, defender dynamics, or the
UAV-defense domain. It optimizes an arbitrary scalar objective over a
sequence of unit-direction vectors `U in R^{H x 3}` using a diagonal
per-timestep Gaussian search distribution, matching the design in
`docs/rmpc_baseline_design.md` Section 11.

Determinism: the caller supplies a dedicated `numpy.random.Generator`
instance (never a module-global RNG), so results are bit-for-bit
reproducible given the same generator state and objective function.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Callable

import numpy as np


def normalize_rows(vectors: np.ndarray, eps: float) -> np.ndarray:
    """
    Normalize each row of `vectors` (shape (..., 3)) to a unit vector.

    Mirrors the project's existing eps-guarded zero-vector fallback
    convention (see `compose_lead_residual` / `LeadInterceptPolicy._pursue`):
    rows whose norm is <= eps are mapped to the zero vector rather than
    dividing by (near) zero.
    """
    vectors = np.asarray(vectors, dtype=np.float64)
    norms = np.linalg.norm(vectors, axis=-1, keepdims=True)
    safe_norms = np.where(norms > eps, norms, 1.0)
    normalized = vectors / safe_norms
    mask = (norms > eps)
    return np.where(mask, normalized, 0.0)


@dataclass
class CEMResult:
    """Result of a single `cem_optimize` call."""

    best_sequence: np.ndarray  # shape (horizon, 3), unit directions (or zero)
    best_score: float
    n_evaluations: int
    success: bool  # True iff at least one finite-scored candidate was ever found
    final_mean: np.ndarray  # shape (horizon, 3), raw (pre-normalization) sampling mean
    final_std: np.ndarray  # shape (horizon, 3), raw (pre-normalization) sampling std


def cem_optimize(
    objective: Callable[[np.ndarray], float],
    horizon: int,
    population_size: int,
    elite_fraction: float,
    n_iterations: int,
    rng: np.random.Generator,
    init_mean: np.ndarray | None = None,
    initial_std: float = 1.0,
    min_std: float = 0.05,
    sample_clip: float = 3.0,
    eps: float = 1e-8,
) -> CEMResult:
    """
    Optimize `objective(U) -> float` over `U in R^{horizon x 3}` (each row
    normalized to a unit direction before being passed to `objective`) using
    the Cross-Entropy Method.

    Args:
        objective: Callable taking a (horizon, 3) array of unit directions
            and returning a scalar score to be MAXIMIZED. May return a
            non-finite value (e.g. NaN) for a rejected/invalid candidate;
            such candidates are ignored for elite selection and distribution
            update but do not raise.
        horizon: Number of planning steps `H`.
        population_size: Number of candidates sampled per iteration `N`.
        elite_fraction: Fraction of `population_size` retained as elites
            each iteration (rounded to at least 1).
        n_iterations: Number of CEM iterations.
        rng: Dedicated `numpy.random.Generator` instance used EXCLUSIVELY
            for this call's Gaussian sampling. Never a module-global RNG.
        init_mean: Optional warm-start mean, shape (horizon, 3), in the RAW
            (pre-normalization) sampling space. Defaults to all-zeros.
        initial_std: Initial per-component standard deviation.
        min_std: Floor applied to the per-iteration std update, preventing
            premature distribution collapse.
        sample_clip: Raw sampled components are clipped to
            [-sample_clip, sample_clip] before normalization (numerical
            safeguard against extreme Gaussian draws).
        eps: Epsilon used for direction normalization's zero-vector guard.

    Returns:
        CEMResult with the best candidate found across ALL iterations (not
        merely the final mean), the number of objective evaluations
        performed, and a `success` flag that is False iff every evaluated
        candidate over the entire optimization was non-finite (in which
        case `best_sequence` is the normalized initial mean and
        `best_score` is -inf).
    """
    if horizon <= 0:
        raise ValueError(f"horizon must be > 0, got {horizon}")
    if population_size <= 0:
        raise ValueError(f"population_size must be > 0, got {population_size}")
    if not (0.0 < elite_fraction <= 1.0):
        raise ValueError(f"elite_fraction must be in (0, 1], got {elite_fraction}")
    if n_iterations <= 0:
        raise ValueError(f"n_iterations must be > 0, got {n_iterations}")

    mean = (
        np.zeros((horizon, 3), dtype=np.float64)
        if init_mean is None
        else np.array(init_mean, dtype=np.float64, copy=True)
    )
    std = np.full((horizon, 3), float(initial_std), dtype=np.float64)

    elite_count = max(1, int(round(population_size * elite_fraction)))

    best_sequence = normalize_rows(mean, eps)
    best_score = float("-inf")
    n_evaluations = 0
    any_finite_ever = False

    for _ in range(n_iterations):
        raw_samples = rng.normal(loc=mean, scale=std, size=(population_size, horizon, 3))
        raw_samples = np.clip(raw_samples, -sample_clip, sample_clip)

        candidates = np.stack(
            [normalize_rows(raw_samples[i], eps) for i in range(population_size)], axis=0
        )

        scores = np.full(population_size, float("-inf"), dtype=np.float64)
        for i in range(population_size):
            score = objective(candidates[i])
            n_evaluations += 1
            if np.isfinite(score):
                scores[i] = float(score)

        finite_mask = np.isfinite(scores)
        if not np.any(finite_mask):
            # No usable candidate this iteration -- reject/ignore entirely,
            # keep the previous mean/std, continue to the next iteration.
            continue

        any_finite_ever = True

        # Deterministic tie-breaking: stable sort descending by score, ties
        # resolved by ascending original candidate index.
        order = np.argsort(-scores, kind="stable")
        top_idx = int(order[0])
        if scores[top_idx] > best_score:
            best_score = float(scores[top_idx])
            best_sequence = candidates[top_idx].copy()

        elite_idx = order[:elite_count]
        elite_idx = elite_idx[np.isfinite(scores[elite_idx])]
        if elite_idx.size == 0:
            continue

        elite_raw = raw_samples[elite_idx]
        mean = elite_raw.mean(axis=0)
        std = np.maximum(elite_raw.std(axis=0), min_std)

    return CEMResult(
        best_sequence=best_sequence,
        best_score=best_score,
        n_evaluations=n_evaluations,
        success=any_finite_ever,
        final_mean=mean,
        final_std=std,
    )
