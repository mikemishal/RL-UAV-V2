"""
Expanded robustness protocol for Lead-Residual PPO -- SINGLE SOURCE OF TRUTH
for seed allocations, frozen-model reuse, common method definitions,
statistical constants, comparison definitions, and shared parameter grids
across all five expanded robustness sweeps (hostile evasion, defender
maneuverability, measurement noise, mobility, detection radius).

THIS MODULE ONLY FREEZES/DOCUMENTS THE PROTOCOL. No robustness seed
(66000-99999) is executed by importing or using this module.

Frozen source commits (upstream, unrelated files -- NEVER modified here):
    training_source_commit          = 4b0a903bf801c7b439be4671ed822e2f853d72ba
    lr_ppo_training_source_commit   = 3b5bc9a48c0d3d52bcf369aea2d97de4c85d6445
    lr_ppo_diagnostic_source_commit = 80e7aef9777039ae27745dc27f585b312bd9d8ce
    expanded_final_source_commit    = 88100c5fff5ab2ffb4cc4a013dae5439cd44c797

NEW namespace/implementation (this module) -- explicitly does NOT reuse any
historical robustness script's seed ranges, training seeds, model tables,
Kalman-Greedy paper-facing method, or old policy-instance definitions. Only
tested parameter-grid VALUES, config-isolation assertion PATTERNS, and
already-validated statistical helper FUNCTIONS are reused (per this task's
explicit instruction).
"""

from __future__ import annotations

import dataclasses
from pathlib import Path

from uav_defend.config import EnvConfig

from experiments.lead_residual_expanded_final_protocol import (  # noqa: F401 -- single source of truth, re-exported
    FROZEN_MODELS,
    TRAINING_SEEDS,
    LR_TRAINING_ROOTS,
    TRAINING_SOURCE_COMMIT,
    LR_PPO_TRAINING_SOURCE_COMMIT,
    LR_PPO_DIAGNOSTIC_SOURCE_COMMIT,
    SCRIPTED_METHODS,
    PPO_METHODS,
    LR_PPO_METHOD,
    EXCLUDED_METHODS,
    RESIDUAL_MAX_ANGLE_DEG,
)
from experiments.run_lead_residual_diagnostic import MethodInstance, build_instances

PROJECT_ROOT = Path(__file__).parent.parent

EXPANDED_FINAL_SOURCE_COMMIT = "88100c5fff5ab2ffb4cc4a013dae5439cd44c797"

# =============================================================================
# PAPER-FACING METHOD SET -- LOCKED (12 instances, reused verbatim from the
# already-frozen expanded-final evaluation's instance builder; NEVER
# Kalman-Greedy/PN-Kalman/Lead-Kalman/Random/a new prediction-error policy).
# =============================================================================
EXPECTED_N_METHOD_INSTANCES = 12


def build_method_instances() -> tuple[MethodInstance, ...]:
    """Reuses experiments.run_lead_residual_diagnostic.build_instances() --
    the SAME 12-instance definition as the expanded final nominal evaluation
    -- rather than redefining a parallel/divergent instance table."""
    instances = build_instances()
    assert len(instances) == EXPECTED_N_METHOD_INSTANCES
    names = {i.name for i in instances}
    assert names == {"greedy", "pn", "lead", "ppo", "ppo_kalman", "lr_ppo"}
    return instances


# =============================================================================
# NEW EXPANDED ROBUSTNESS SEED NAMESPACE (splits 66000..99999)
# =============================================================================
# Previously-consumed/still-sealed ranges, retained here ONLY for the
# disjointness assertions/tests below -- never reused, never re-opened.
RESIDUAL_VALIDATION_RANGE = (60_000, 60_099)
RESIDUAL_DIAGNOSTIC_RANGE = (60_100, 60_299)
EXPANDED_FINAL_RANGE = (61_000, 65_999)
LR_PPO_TRAINING_NAMESPACES = {  # from experiments/lead_residual_training_protocol.py
    55: (1_000_000, 1_999_999),
    56: (2_000_000, 2_999_999),
    57: (3_000_000, 3_999_999),
}

HOSTILE_EVASION_SEED_START = 66_000
HOSTILE_EVASION_SEED_END = 66_999
HOSTILE_EVASION_EPISODES = HOSTILE_EVASION_SEED_END - HOSTILE_EVASION_SEED_START + 1  # 1000

MANEUVERABILITY_SEED_START = 67_000
MANEUVERABILITY_SEED_END = 67_999
MANEUVERABILITY_EPISODES = MANEUVERABILITY_SEED_END - MANEUVERABILITY_SEED_START + 1  # 1000

MEASUREMENT_NOISE_SEED_START = 68_000
MEASUREMENT_NOISE_SEED_END = 68_999
MEASUREMENT_NOISE_EPISODES = MEASUREMENT_NOISE_SEED_END - MEASUREMENT_NOISE_SEED_START + 1  # 1000

MOBILITY_SEED_START = 69_000
MOBILITY_SEED_END = 69_999
MOBILITY_EPISODES = MOBILITY_SEED_END - MOBILITY_SEED_START + 1  # 1000

DETECTION_RADIUS_SEED_START = 70_000
DETECTION_RADIUS_SEED_END = 70_999
DETECTION_RADIUS_EPISODES = DETECTION_RADIUS_SEED_END - DETECTION_RADIUS_SEED_START + 1  # 1000

FUTURE_UNUSED_SEED_START = 71_000
FUTURE_UNUSED_SEED_END = 99_999

SWEEP_SEED_RANGES: dict[str, tuple[int, int]] = {
    "hostile_evasion": (HOSTILE_EVASION_SEED_START, HOSTILE_EVASION_SEED_END),
    "maneuverability": (MANEUVERABILITY_SEED_START, MANEUVERABILITY_SEED_END),
    "measurement_noise": (MEASUREMENT_NOISE_SEED_START, MEASUREMENT_NOISE_SEED_END),
    "mobility": (MOBILITY_SEED_START, MOBILITY_SEED_END),
    "detection_radius": (DETECTION_RADIUS_SEED_START, DETECTION_RADIUS_SEED_END),
}


def assert_seed_in_hostile_evasion_range(seed: int) -> None:
    """Positive-membership check used for the REAL production hostile-evasion
    run: seed must fall exactly within the range being deliberately opened
    (mirrors experiments.lead_residual_expanded_final_protocol.assert_seed_in_final_range).
    Unlike assert_seed_not_yet_opened (a blanket guard used by dev/smoke
    tasks to keep OFF every reserved range), this function is what actually
    authorizes use of the hostile_evasion range itself."""
    if not (HOSTILE_EVASION_SEED_START <= seed <= HOSTILE_EVASION_SEED_END):
        raise ValueError(f"Seed {seed} is outside the locked hostile-evasion range {HOSTILE_EVASION_SEED_START}..{HOSTILE_EVASION_SEED_END}")


def assert_seed_in_maneuverability_range(seed: int) -> None:
    """Positive-membership check used for the REAL production maneuverability
    run: seed must fall exactly within the range being deliberately opened.
    Unlike assert_seed_not_yet_opened (a blanket guard used by dev/smoke
    tasks to keep OFF every reserved range), this function is what actually
    authorizes use of the maneuverability range itself."""
    if not (MANEUVERABILITY_SEED_START <= seed <= MANEUVERABILITY_SEED_END):
        raise ValueError(f"Seed {seed} is outside the locked maneuverability range {MANEUVERABILITY_SEED_START}..{MANEUVERABILITY_SEED_END}")


def assert_seed_in_measurement_noise_range(seed: int) -> None:
    """Positive-membership check used for the REAL production measurement-noise
    run: seed must fall exactly within the range being deliberately opened.
    Unlike assert_seed_not_yet_opened (a blanket guard used by dev/smoke
    tasks to keep OFF every reserved range), this function is what actually
    authorizes use of the measurement_noise range itself."""
    if not (MEASUREMENT_NOISE_SEED_START <= seed <= MEASUREMENT_NOISE_SEED_END):
        raise ValueError(f"Seed {seed} is outside the locked measurement-noise range {MEASUREMENT_NOISE_SEED_START}..{MEASUREMENT_NOISE_SEED_END}")


def assert_seed_in_mobility_range(seed: int) -> None:
    """Positive-membership check used for the REAL production mobility
    run: seed must fall exactly within the range being deliberately opened.
    Unlike assert_seed_not_yet_opened (a blanket guard used by dev/smoke
    tasks to keep OFF every reserved range), this function is what actually
    authorizes use of the mobility range itself."""
    if not (MOBILITY_SEED_START <= seed <= MOBILITY_SEED_END):
        raise ValueError(f"Seed {seed} is outside the locked mobility range {MOBILITY_SEED_START}..{MOBILITY_SEED_END}")


def assert_seed_in_detection_radius_range(seed: int) -> None:
    """Positive-membership check used for the REAL production detection-radius
    run: seed must fall exactly within the range being deliberately opened.
    Unlike assert_seed_not_yet_opened (a blanket guard used by dev/smoke
    tasks to keep OFF every reserved range), this function is what actually
    authorizes use of the detection_radius range itself."""
    if not (DETECTION_RADIUS_SEED_START <= seed <= DETECTION_RADIUS_SEED_END):
        raise ValueError(f"Seed {seed} is outside the locked detection-radius range {DETECTION_RADIUS_SEED_START}..{DETECTION_RADIUS_SEED_END}")


def assert_seed_not_yet_opened(seed: int) -> None:
    """Raises if `seed` falls in ANY not-yet-opened expanded-robustness
    range -- guards implementation/smoke-test tasks against accidentally
    executing a reserved range. hostile_evasion (66000-66999),
    maneuverability (67000-67999), measurement_noise (68000-68999) and
    mobility (69000-69999) are EXCLUDED from this dict (all four are now
    permanently frozen/consumed, not "not yet opened"); they are instead
    guarded via assert_disjoint_from_all_prior_ranges."""
    _FROZEN_CONSUMED = ("hostile_evasion", "maneuverability", "measurement_noise", "mobility")
    for _name, (lo, hi) in SWEEP_SEED_RANGES.items():
        if _name in _FROZEN_CONSUMED:
            continue
        if lo <= seed <= hi:
            raise ValueError(f"Seed {seed} is in the RESERVED (not-yet-opened) {_name} robustness range.")
    if FUTURE_UNUSED_SEED_START <= seed <= FUTURE_UNUSED_SEED_END:
        raise ValueError(f"Seed {seed} is in the RESERVED (not-yet-opened) future/unused range.")


def assert_disjoint_from_all_prior_ranges(seed: int) -> None:
    """Defense-in-depth: seed must not fall in any PRIOR (already-opened,
    permanently-frozen, or training-namespace) range either."""
    if RESIDUAL_VALIDATION_RANGE[0] <= seed <= RESIDUAL_VALIDATION_RANGE[1]:
        raise ValueError(f"Seed {seed} collides with the residual validation range.")
    if RESIDUAL_DIAGNOSTIC_RANGE[0] <= seed <= RESIDUAL_DIAGNOSTIC_RANGE[1]:
        raise ValueError(f"Seed {seed} collides with the residual diagnostic range.")
    if EXPANDED_FINAL_RANGE[0] <= seed <= EXPANDED_FINAL_RANGE[1]:
        raise ValueError(f"Seed {seed} collides with the expanded final range.")
    if HOSTILE_EVASION_SEED_START <= seed <= HOSTILE_EVASION_SEED_END:
        raise ValueError(f"Seed {seed} collides with the permanently-frozen hostile-evasion range; must not reuse or rerun.")
    if MANEUVERABILITY_SEED_START <= seed <= MANEUVERABILITY_SEED_END:
        raise ValueError(f"Seed {seed} collides with the permanently-frozen maneuverability range; must not reuse or rerun.")
    if MEASUREMENT_NOISE_SEED_START <= seed <= MEASUREMENT_NOISE_SEED_END:
        raise ValueError(f"Seed {seed} collides with the permanently-frozen measurement-noise range; must not reuse or rerun.")
    if MOBILITY_SEED_START <= seed <= MOBILITY_SEED_END:
        raise ValueError(f"Seed {seed} collides with the permanently-frozen mobility range; must not reuse or rerun.")
    for root, (lo, hi) in LR_PPO_TRAINING_NAMESPACES.items():
        if lo <= seed <= hi:
            raise ValueError(f"Seed {seed} collides with LR-PPO-{root}'s training-episode namespace.")


# =============================================================================
# STATISTICS -- SAME validated logic/constants as the expanded final
# evaluation (frozen BEFORE opening any robustness seed).
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260818
CONFIDENCE_LEVEL = 0.95

PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("LR-PPO family - Lead", "lr_ppo", "lead"),
    ("LR-PPO family - PPO family", "lr_ppo", "ppo"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("PPO family - Lead", "ppo", "lead"),
    ("PPO family - Greedy", "ppo", "greedy"),
    ("PPO family - PN", "ppo", "pn"),
    ("PPO-Kalman family - PPO family", "ppo_kalman", "ppo"),
    ("Lead - Greedy", "lead", "greedy"),
    ("Lead - PN", "lead", "pn"),
    ("PN - Greedy", "pn", "greedy"),
)

# =============================================================================
# CANONICAL NOMINAL ENVCONFIG -- the expanded-final's own EnvConfig (Direct
# track), used as the reference point for every sweep's config-isolation
# assertion (item 24: nominal condition must be behaviorally identical to
# the expanded-final canonical EnvConfig except estimator mode).
# =============================================================================
_CANONICAL_NOMINAL_CONFIG_DIRECT = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)
_CANONICAL_NOMINAL_CONFIG_KALMAN = EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True)


def assert_only_fields_differ(swept_config: EnvConfig, allowed_diff_keys: set[str]) -> None:
    """Programmatically verify the ONLY EnvConfig fields differing from the
    canonical nominal (Direct or Kalman, matched by use_kalman_tracking) are
    in `allowed_diff_keys`. Raises AssertionError otherwise.

    NOTE: `use_kalman_tracking` is NEVER itself a swept robustness variable
    -- the reference nominal is always selected to match the swept config's
    own estimator track (Direct vs Kalman), so `use_kalman_tracking` never
    appears in the diff set and must NOT be listed in `allowed_diff_keys`
    (doing so would misleadingly imply it is a sweep dimension).
    """
    assert "use_kalman_tracking" not in allowed_diff_keys, (
        "use_kalman_tracking is never a swept variable -- do not list it in allowed_diff_keys "
        "(the nominal reference is always matched to the swept config's own estimator track)"
    )
    nominal = _CANONICAL_NOMINAL_CONFIG_KALMAN if swept_config.use_kalman_tracking else _CANONICAL_NOMINAL_CONFIG_DIRECT
    swept_dict = dataclasses.asdict(swept_config)
    nominal_dict = dataclasses.asdict(nominal)
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    unexpected = diffs - allowed_diff_keys
    assert not unexpected, f"unexpected EnvConfig fields differ from canonical nominal: {unexpected}"


# =============================================================================
# SWEEP 1: HOSTILE EVASION -- gain grid (reused tested VALUES only)
# =============================================================================
EVASION_GAIN_VALUES: tuple[float, ...] = (0.00, 0.25, 0.50, 0.75, 1.00)
NOMINAL_EVASION_GAIN = 0.75
NOMINAL_EVASION_RADIUS = 20.0
assert NOMINAL_EVASION_GAIN in EVASION_GAIN_VALUES


def build_hostile_evasion_config(estimator: str, evasion_gain: float) -> EnvConfig:
    config = EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        enemy_evasion_enabled=True,  # gain=0.0 means zero reactive-away contribution, NOT a non-maneuvering hostile
        enemy_evasion_radius=NOMINAL_EVASION_RADIUS,
        enemy_evasion_gain=evasion_gain,
    )
    assert_only_fields_differ(config, {"enemy_evasion_gain"})
    return config


# =============================================================================
# SWEEP 2: DEFENDER MANEUVERABILITY -- 3x3 grid
# =============================================================================
MANEUVERABILITY_ACCEL_VALUES: tuple[float, ...] = (4.0, 8.0, 12.0)
MANEUVERABILITY_TURN_RATE_VALUES: tuple[float, ...] = (45.0, 90.0, 135.0)
NOMINAL_MANEUVERABILITY_ACCEL = 8.0
NOMINAL_MANEUVERABILITY_TURN_RATE = 90.0
assert NOMINAL_MANEUVERABILITY_ACCEL in MANEUVERABILITY_ACCEL_VALUES
assert NOMINAL_MANEUVERABILITY_TURN_RATE in MANEUVERABILITY_TURN_RATE_VALUES
MANEUVERABILITY_GRID: tuple[tuple[float, float], ...] = tuple(
    (a, t) for a in MANEUVERABILITY_ACCEL_VALUES for t in MANEUVERABILITY_TURN_RATE_VALUES
)
assert len(MANEUVERABILITY_GRID) == 9


def build_maneuverability_config(estimator: str, defender_max_accel: float, defender_max_turn_rate_deg: float) -> EnvConfig:
    config = EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        defender_max_accel=defender_max_accel,
        defender_max_turn_rate_deg=defender_max_turn_rate_deg,
    )
    assert_only_fields_differ(config, {"defender_max_accel", "defender_max_turn_rate_deg"})
    return config


# =============================================================================
# SWEEP 3: MEASUREMENT NOISE -- variance grid (process_var fixed for PPO-Kalman)
# =============================================================================
MEASUREMENT_VAR_VALUES: tuple[float, ...] = (0.05, 0.25, 0.50, 1.00, 2.00, 4.00)
NOMINAL_MEASUREMENT_VAR = 0.50
FIXED_PROCESS_VAR = 1.0
assert NOMINAL_MEASUREMENT_VAR in MEASUREMENT_VAR_VALUES


# =============================================================================
# SWEEP 3: MEASUREMENT NOISE -- variance grid (process_var FIXED, never a
# swept/allowed-diff field -- see assert_process_var_fixed() below).
# =============================================================================
MEASUREMENT_VAR_VALUES: tuple[float, ...] = (0.05, 0.25, 0.50, 1.00, 2.00, 4.00)
NOMINAL_MEASUREMENT_VAR = 0.50
FIXED_PROCESS_VAR = 1.0
assert NOMINAL_MEASUREMENT_VAR in MEASUREMENT_VAR_VALUES


def assert_process_var_fixed(config: EnvConfig) -> None:
    """Hard guard (item 2 of the hostile-evasion hardening task): measurement_var
    is the ONLY swept estimator parameter for the measurement-noise sweep --
    process_var must never be treated as an allowed sweep difference. Fails
    loudly if `config.process_var`, or either canonical nominal reference's
    process_var, ever drifts from FIXED_PROCESS_VAR (e.g. via an accidental
    future EnvConfig default change)."""
    assert config.process_var == FIXED_PROCESS_VAR, (
        f"process_var must be fixed at {FIXED_PROCESS_VAR}, got {config.process_var}"
    )
    assert _CANONICAL_NOMINAL_CONFIG_DIRECT.process_var == FIXED_PROCESS_VAR, (
        f"Direct nominal reference process_var drifted from {FIXED_PROCESS_VAR}: "
        f"{_CANONICAL_NOMINAL_CONFIG_DIRECT.process_var}"
    )
    assert _CANONICAL_NOMINAL_CONFIG_KALMAN.process_var == FIXED_PROCESS_VAR, (
        f"Kalman nominal reference process_var drifted from {FIXED_PROCESS_VAR}: "
        f"{_CANONICAL_NOMINAL_CONFIG_KALMAN.process_var}"
    )


def build_measurement_noise_config(estimator: str, measurement_var: float) -> EnvConfig:
    config = EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        measurement_var=measurement_var,
        process_var=FIXED_PROCESS_VAR,  # NOT Kalman retuning -- held fixed at every condition
    )
    assert_process_var_fixed(config)
    # process_var is DELIBERATELY excluded from allowed_diff_keys: since it is
    # fixed at the same value as the canonical nominal, it never legitimately
    # differs -- excluding it makes any accidental future drift fail loudly.
    assert_only_fields_differ(config, {"measurement_var"})
    return config


# =============================================================================
# SWEEP 4: MOBILITY -- two 1-D arms sharing the nominal (18, 12) condition
# =============================================================================
DEFENDER_SPEED_ARM_VALUES: tuple[float, ...] = (12.0, 15.0, 18.0, 21.0, 24.0)
HOSTILE_SPEED_ARM_VALUES: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0, 16.0)
NOMINAL_DEFENDER_SPEED = 18.0
NOMINAL_HOSTILE_SPEED = 12.0
assert NOMINAL_DEFENDER_SPEED in DEFENDER_SPEED_ARM_VALUES
assert NOMINAL_HOSTILE_SPEED in HOSTILE_SPEED_ARM_VALUES

# 9 unique (v_d, v_e) conditions, in EXPLICIT deterministic order (never a
# set/set-union -- experiment/output ordering must be reproducible):
# defender-speed arm first (12,12)/(15,12)/(18,12)[nominal]/(21,12)/(24,12),
# then the hostile-speed arm EXCLUDING the already-listed nominal duplicate.
MOBILITY_GRID: tuple[tuple[float, float], ...] = tuple(
    [(v_d, NOMINAL_HOSTILE_SPEED) for v_d in DEFENDER_SPEED_ARM_VALUES]
    + [(NOMINAL_DEFENDER_SPEED, v_e) for v_e in HOSTILE_SPEED_ARM_VALUES if v_e != NOMINAL_HOSTILE_SPEED]
)
assert len(MOBILITY_GRID) == 9
assert MOBILITY_GRID == (
    (12.0, 12.0), (15.0, 12.0), (18.0, 12.0), (21.0, 12.0), (24.0, 12.0),
    (18.0, 8.0), (18.0, 10.0), (18.0, 14.0), (18.0, 16.0),
)


def build_mobility_config(estimator: str, v_d: float, v_e: float) -> EnvConfig:
    config = EnvConfig(use_kalman_tracking=(estimator == "kalman"), v_d=v_d, v_e=v_e)
    assert_only_fields_differ(config, {"v_d", "v_e"})  # maneuverability limits NOT scaled with speed
    return config


# =============================================================================
# SWEEP 5: DETECTION RADIUS
# =============================================================================
DETECTION_RADIUS_VALUES: tuple[float, ...] = (5.0, 10.0, 15.0, 20.0, 25.0, 30.0)
NOMINAL_DETECTION_RADIUS = 15.0
assert NOMINAL_DETECTION_RADIUS in DETECTION_RADIUS_VALUES


def build_detection_radius_config(estimator: str, detection_radius: float) -> EnvConfig:
    config = EnvConfig(use_kalman_tracking=(estimator == "kalman"), detection_radius=detection_radius)
    assert_only_fields_differ(config, {"detection_radius"})
    return config


SWEEP_CONFIG_BUILDERS = {
    "hostile_evasion": build_hostile_evasion_config,
    "maneuverability": build_maneuverability_config,
    "measurement_noise": build_measurement_noise_config,
    "mobility": build_mobility_config,
    "detection_radius": build_detection_radius_config,
}

OUTPUT_ROOT = PROJECT_ROOT / "results" / "expanded_robustness"
SWEEP_OUTPUT_DIRS = {name: OUTPUT_ROOT / name for name in SWEEP_SEED_RANGES}
