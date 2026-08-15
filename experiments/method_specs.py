"""
Centralized method specification registry for the publication comparison.

This module is the SINGLE SOURCE OF TRUTH mapping each comparison method to:
    - its canonical policy-registry name (uav_defend.policies.registry),
    - which estimator track it uses ("measurement" | "kalman" | "none"),
    - whether it requires a trained model on disk,
    - a filesystem/plot-safe output identifier.

Any evaluation/comparison/reporting script that needs "the list of methods
in the study" should import METHOD_SPECS from here rather than hardcoding
policy names, so that adding/removing a method or renaming a canonical
registry key never requires touching more than this file plus
uav_defend/policies/registry.py.

Usage:
    from experiments.method_specs import METHOD_SPECS, get_method_spec

    for spec in METHOD_SPECS:
        policy = get_policy(spec.policy_name, **kwargs)
        ...

    spec = get_method_spec("pn_kalman")
"""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class MethodSpec:
    """
    Specification of one comparison method.

    Attributes:
        key: Short canonical identifier for this method (matches the
            policy-registry name for all methods in this study).
        display_name: Human-readable name for plots/reports.
        policy_name: Name to pass to uav_defend.policies.registry.get_policy().
        estimator: Which hostile-state estimator this method consumes --
            "measurement" (raw noisy measurement + finite-difference
            velocity, the Direct track), "kalman" (the environment's Kalman
            filter estimate), or "none" (does not use a hostile-state
            estimate at all, e.g. a random baseline).
        requires_model: True if this method needs a trained model file on
            disk (PPO variants); False for scripted/classical baselines.
        output_identifier: Filesystem/plot-safe slug for filenames and plot
            legends (lowercase, underscore-separated).
    """

    key: str
    display_name: str
    policy_name: str
    estimator: str
    requires_model: bool
    output_identifier: str


VALID_ESTIMATORS = ("measurement", "kalman", "none")


METHOD_SPECS: tuple[MethodSpec, ...] = (
    MethodSpec(
        key="greedy",
        display_name="Greedy (Direct)",
        policy_name="greedy",
        estimator="measurement",
        requires_model=False,
        output_identifier="greedy",
    ),
    MethodSpec(
        key="kalman_greedy",
        display_name="Greedy (Kalman)",
        policy_name="kalman_greedy",
        estimator="kalman",
        requires_model=False,
        output_identifier="kalman_greedy",
    ),
    MethodSpec(
        key="pn",
        display_name="Proportional Navigation (Direct)",
        policy_name="pn",
        estimator="measurement",
        requires_model=False,
        output_identifier="pn",
    ),
    MethodSpec(
        key="pn_kalman",
        display_name="Proportional Navigation (Kalman)",
        policy_name="pn_kalman",
        estimator="kalman",
        requires_model=False,
        output_identifier="pn_kalman",
    ),
    MethodSpec(
        key="lead",
        display_name="Lead Intercept (Direct)",
        policy_name="lead",
        estimator="measurement",
        requires_model=False,
        output_identifier="lead",
    ),
    MethodSpec(
        key="lead_kalman",
        display_name="Lead Intercept (Kalman)",
        policy_name="lead_kalman",
        estimator="kalman",
        requires_model=False,
        output_identifier="lead_kalman",
    ),
    MethodSpec(
        key="ppo",
        display_name="PPO (Direct)",
        policy_name="ppo",
        estimator="measurement",
        requires_model=True,
        output_identifier="ppo",
    ),
    MethodSpec(
        key="ppo_kalman",
        display_name="PPO (Kalman)",
        policy_name="ppo_kalman",
        estimator="kalman",
        requires_model=True,
        output_identifier="ppo_kalman",
    ),
    # Random is a diagnostic/sanity-check baseline (uses no hostile-state
    # estimate at all), kept separate from the core 8-method comparison.
    MethodSpec(
        key="random",
        display_name="Random (diagnostic)",
        policy_name="random",
        estimator="none",
        requires_model=False,
        output_identifier="random",
    ),
)


_METHOD_SPECS_BY_KEY: dict[str, MethodSpec] = {spec.key: spec for spec in METHOD_SPECS}


def get_method_spec(key: str) -> MethodSpec:
    """
    Look up a MethodSpec by its canonical key.

    Raises:
        KeyError: If `key` is not a recognized method.
    """
    try:
        return _METHOD_SPECS_BY_KEY[key]
    except KeyError:
        available = ", ".join(_METHOD_SPECS_BY_KEY)
        raise KeyError(f"Unknown method key '{key}'. Available: {available}") from None


def list_method_keys(*, exclude_diagnostic: bool = False) -> list[str]:
    """
    Return all canonical method keys, in the study's canonical order.

    Args:
        exclude_diagnostic: If True, omit "random" (a sanity-check baseline,
            not part of the core 8-method Direct/Kalman comparison).
    """
    if exclude_diagnostic:
        return [spec.key for spec in METHOD_SPECS if spec.key != "random"]
    return [spec.key for spec in METHOD_SPECS]
