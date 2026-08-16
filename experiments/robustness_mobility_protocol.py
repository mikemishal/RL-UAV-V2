"""
Locked mobility robustness sweep protocol -- SINGLE SOURCE OF TRUTH.

The THIRD robustness sweep for the paper. Characterizes how interception
performance changes as the DEFENDER/HOSTILE mobility relationship changes,
via TWO matched one-dimensional sweeps (NOT a 25-cell 2-D Cartesian grid):

    A. DEFENDER-SPEED SWEEP: hold v_e = 12.0 (nominal), vary v_d.
    B. HOSTILE-SPEED SWEEP:  hold v_d = 18.0 (nominal), vary v_e.

The nominal pair (v_d=18, v_e=12) appears in BOTH conceptual sweeps but is
evaluated ONLY ONCE (sweep_axis="nominal") -- 9 unique (v_d, v_e)
configurations total, not 10.

From this task forward (per the measurement-noise sweep), robustness
sweeps use ONLY the six primary paper methods (PN-Kalman and Lead-Kalman
excluded).

Observation-normalization audit (item 19, verified by inspection of
uav_defend/envs/soldier_env.py::_get_observation): defender velocity is
normalized as `defender_vel / self.config.v_d` and hostile velocity as
`.../ self.config.v_e`, BOTH read from the environment's OWN (possibly
swept) config at observation-computation time and clipped to [-1, 1].
Since actual defender/enemy speed is dynamically constrained to <= the
configured v_d/v_e respectively, this normalization is internally
consistent BY CONSTRUCTION at every sweep point -- there is no hardcoded
nominal-speed normalization constant anywhere. test_robustness_mobility.py
item U empirically confirms finiteness/bounds at all tested speeds.

Seed ranges (disjoint from all prior protocol ranges -- see
test_robustness_mobility.py):

    Training seeds                              : 42, 43, 44
    Validation seeds                             : 10000..10099
    Diagnostic seeds                             : 12000..12199
    Final nominal seeds (OPENED, locked)          : 20000..24999
    Acquisition ablation seeds (OPENED, locked)   : 25000..25999
    Detection-radius sweep seeds (OPENED, locked) : 30000..30999
    Measurement-noise sweep seeds (OPENED, locked): 31000..31999
    MOBILITY_SWEEP_SEEDS (OPENED here)            : 32000..32999
    Reserved for later robustness sweeps:
        hostile-evasion robustness                 : 33000..33999
        maneuverability/dynamics robustness          : 34000..34999
"""

from __future__ import annotations

import dataclasses
from dataclasses import dataclass
from pathlib import Path

from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.final_experiment_protocol import PPO_TRAINING_SEEDS, frozen_model_path

PROJECT_ROOT = Path(__file__).parent.parent

# =============================================================================
# NOMINAL MOBILITY
# =============================================================================
NOMINAL_DEFENDER_SPEED = 18.0
NOMINAL_HOSTILE_SPEED = 12.0
NOMINAL_MOBILITY_RATIO = NOMINAL_DEFENDER_SPEED / NOMINAL_HOSTILE_SPEED  # 1.5

# =============================================================================
# ONE-DIMENSIONAL SWEEP GRIDS -- prespecified BEFORE opening seed 32000
# =============================================================================
DEFENDER_SPEED_VALUES: tuple[float, ...] = (12.0, 15.0, 18.0, 21.0, 24.0)
HOSTILE_SPEED_VALUES: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0, 16.0)
assert NOMINAL_DEFENDER_SPEED in DEFENDER_SPEED_VALUES
assert NOMINAL_HOSTILE_SPEED in HOSTILE_SPEED_VALUES

# Fixed maneuverability limits -- NEVER scaled with speed (item 15).
FIXED_DEFENDER_MAX_ACCEL = 8.0
FIXED_DEFENDER_MAX_TURN_RATE_DEG = 90.0
FIXED_DEFENDER_MAX_CLIMB_RATE = 6.0
FIXED_DEFENDER_MAX_DESCENT_RATE = 6.0
FIXED_ENEMY_MAX_ACCEL = 6.0
FIXED_ENEMY_MAX_TURN_RATE_DEG = 75.0
FIXED_ENEMY_MAX_CLIMB_RATE = 5.0
FIXED_ENEMY_MAX_DESCENT_RATE = 5.0

# Fixed sensing -- NEVER changed by this sweep (item 16).
FIXED_DETECTION_RADIUS = 15.0
FIXED_MEASUREMENT_VAR = 0.5
FIXED_PROCESS_VAR = 1.0
FIXED_LEAD_TIME = 0.0

# Fixed evasion -- NEVER changed by this sweep (item 17).
FIXED_ENEMY_EVASION_ENABLED = True
FIXED_ENEMY_EVASION_RADIUS = 20.0
FIXED_ENEMY_EVASION_GAIN = 0.75


@dataclass(frozen=True)
class MobilityCondition:
    """One of the 9 UNIQUE (v_d, v_e) configurations evaluated exactly once."""

    defender_speed: float
    hostile_speed: float
    sweep_axis: str  # "defender_speed" | "hostile_speed" | "nominal"

    @property
    def mobility_ratio(self) -> float:
        return self.defender_speed / self.hostile_speed


def _build_mobility_conditions() -> tuple[MobilityCondition, ...]:
    conditions: list[MobilityCondition] = []
    for v_d in DEFENDER_SPEED_VALUES:
        if v_d == NOMINAL_DEFENDER_SPEED:
            conditions.append(MobilityCondition(v_d, NOMINAL_HOSTILE_SPEED, "nominal"))
        else:
            conditions.append(MobilityCondition(v_d, NOMINAL_HOSTILE_SPEED, "defender_speed"))
    for v_e in HOSTILE_SPEED_VALUES:
        if v_e == NOMINAL_HOSTILE_SPEED:
            continue  # nominal (18,12) already added above -- evaluate ONLY ONCE.
        conditions.append(MobilityCondition(NOMINAL_DEFENDER_SPEED, v_e, "hostile_speed"))
    return tuple(conditions)


MOBILITY_CONDITIONS: tuple[MobilityCondition, ...] = _build_mobility_conditions()
EXPECTED_N_MOBILITY_CONDITIONS = 9
assert len(MOBILITY_CONDITIONS) == EXPECTED_N_MOBILITY_CONDITIONS
assert len({(c.defender_speed, c.hostile_speed) for c in MOBILITY_CONDITIONS}) == EXPECTED_N_MOBILITY_CONDITIONS
assert sum(1 for c in MOBILITY_CONDITIONS if c.sweep_axis == "nominal") == 1
assert sum(1 for c in MOBILITY_CONDITIONS if c.sweep_axis == "defender_speed") == 4
assert sum(1 for c in MOBILITY_CONDITIONS if c.sweep_axis == "hostile_speed") == 4


def defender_speed_conditions() -> tuple[MobilityCondition, ...]:
    """The 5-point defender-speed curve (12,15,18*,21,24), * = shared nominal row."""
    return tuple(c for c in MOBILITY_CONDITIONS if c.sweep_axis in ("defender_speed", "nominal"))


def hostile_speed_conditions() -> tuple[MobilityCondition, ...]:
    """The 5-point hostile-speed curve (8,10,12*,14,16), * = shared nominal row."""
    nominal = next(c for c in MOBILITY_CONDITIONS if c.sweep_axis == "nominal")
    return (nominal,) + tuple(c for c in MOBILITY_CONDITIONS if c.sweep_axis == "hostile_speed")


# =============================================================================
# MOBILITY SWEEP SEEDS -- a previously untouched range
# =============================================================================
MOBILITY_SWEEP_SEED_OFFSET = 32_000
MOBILITY_SWEEP_EPISODES = 1_000
MOBILITY_SWEEP_SEEDS: range = range(
    MOBILITY_SWEEP_SEED_OFFSET, MOBILITY_SWEEP_SEED_OFFSET + MOBILITY_SWEEP_EPISODES
)

# Reserved for LATER robustness sweeps -- do NOT evaluate in this task.
HOSTILE_EVASION_SWEEP_SEED_OFFSET = 33_000
DYNAMICS_SWEEP_SEED_OFFSET = 34_000
RESERVED_FUTURE_ROBUSTNESS_SEEDS: range = range(HOSTILE_EVASION_SWEEP_SEED_OFFSET, 35_000)

# =============================================================================
# STATISTICS PROTOCOL
# =============================================================================
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def assert_mobility_sweep_seed(seed: int) -> None:
    """Raise if `seed` is not in the locked mobility sweep seed range."""
    if seed not in MOBILITY_SWEEP_SEEDS:
        raise ValueError(
            f"Seed {seed} is not in MOBILITY_SWEEP_SEEDS "
            f"({MOBILITY_SWEEP_SEED_OFFSET}..{MOBILITY_SWEEP_SEED_OFFSET + MOBILITY_SWEEP_EPISODES - 1})."
        )


# =============================================================================
# SIX PRIMARY PAPER METHODS ONLY (no PN-Kalman / Lead-Kalman in robustness)
# =============================================================================
PRIMARY_PAPER_METHODS: tuple[str, ...] = ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
SCRIPTED_METHODS: tuple[str, ...] = ("greedy", "kalman_greedy", "pn", "lead")
PPO_METHODS: tuple[str, ...] = ("ppo", "ppo_kalman")


def _estimator_for_method(method: str) -> str:
    return "kalman" if method in ("kalman_greedy", "ppo_kalman") else "measurement"


def _track_for_ppo_method(method: str) -> str:
    return "kalman" if method == "ppo_kalman" else "direct"


@dataclasses.dataclass(frozen=True)
class MobilityPolicyInstance:
    """One concrete, runnable policy instance evaluated across the mobility grid."""

    paper_method: str
    policy_instance: str
    estimator: str
    training_seed: int | None
    model_path: Path | None


def build_mobility_policy_instances() -> tuple[MobilityPolicyInstance, ...]:
    instances: list[MobilityPolicyInstance] = []
    for method in SCRIPTED_METHODS:
        instances.append(MobilityPolicyInstance(
            paper_method=method, policy_instance=method,
            estimator=_estimator_for_method(method), training_seed=None, model_path=None,
        ))
    for method in PPO_METHODS:
        track = _track_for_ppo_method(method)
        for seed in PPO_TRAINING_SEEDS:
            instances.append(MobilityPolicyInstance(
                paper_method=method, policy_instance=f"{method}_seed_{seed}",
                estimator=_estimator_for_method(method), training_seed=seed,
                model_path=frozen_model_path(track, seed),
            ))
    return tuple(instances)


MOBILITY_POLICY_INSTANCES: tuple[MobilityPolicyInstance, ...] = build_mobility_policy_instances()
EXPECTED_N_POLICY_INSTANCES = 10
EXPECTED_RAW_ROW_COUNT = (
    EXPECTED_N_MOBILITY_CONDITIONS * EXPECTED_N_POLICY_INSTANCES * MOBILITY_SWEEP_EPISODES
)  # 90,000


def build_policy(instance: MobilityPolicyInstance):
    """Instantiate the concrete policy for a MobilityPolicyInstance. 'greedy' is
    ALWAYS the corrected GreedyInterceptPolicy from commit 0dbdadd -- never
    reconstruct the old escort-only behavior."""
    if instance.model_path is None:
        return get_policy(instance.paper_method)
    return get_policy(instance.paper_method, model_path=str(instance.model_path), deterministic=True)


def build_mobility_env_config(estimator: str, defender_speed: float, hostile_speed: float) -> EnvConfig:
    """
    Nominal EnvConfig with ONLY v_d, v_e (and, for the Kalman track,
    use_kalman_tracking) changed. All maneuverability, sensing, and evasion
    parameters are explicitly passed at their FIXED nominal values (not
    swept) -- see assert_only_mobility_speed_differs().
    """
    return EnvConfig(
        use_kalman_tracking=(estimator == "kalman"),
        v_d=defender_speed,
        v_e=hostile_speed,
        defender_max_accel=FIXED_DEFENDER_MAX_ACCEL,
        defender_max_turn_rate_deg=FIXED_DEFENDER_MAX_TURN_RATE_DEG,
        defender_max_climb_rate=FIXED_DEFENDER_MAX_CLIMB_RATE,
        defender_max_descent_rate=FIXED_DEFENDER_MAX_DESCENT_RATE,
        enemy_max_accel=FIXED_ENEMY_MAX_ACCEL,
        enemy_max_turn_rate_deg=FIXED_ENEMY_MAX_TURN_RATE_DEG,
        enemy_max_climb_rate=FIXED_ENEMY_MAX_CLIMB_RATE,
        enemy_max_descent_rate=FIXED_ENEMY_MAX_DESCENT_RATE,
        detection_radius=FIXED_DETECTION_RADIUS,
        measurement_var=FIXED_MEASUREMENT_VAR,
        process_var=FIXED_PROCESS_VAR,
        lead_time=FIXED_LEAD_TIME,
        enemy_evasion_enabled=FIXED_ENEMY_EVASION_ENABLED,
        enemy_evasion_radius=FIXED_ENEMY_EVASION_RADIUS,
        enemy_evasion_gain=FIXED_ENEMY_EVASION_GAIN,
    )


def assert_only_mobility_speed_differs(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """
    Programmatically verify the ONLY behavioral EnvConfig fields differing
    between a swept config and the nominal config are v_d/v_e (and
    use_kalman_tracking, expected between tracks). Raises AssertionError
    otherwise -- in particular this catches any accidental maneuverability,
    sensing, evasion, or reward drift.
    """
    swept_dict = dataclasses.asdict(swept_config)
    nominal_dict = dataclasses.asdict(nominal_config)
    allowed_diff_keys = {"v_d", "v_e", "use_kalman_tracking"}
    diffs = {k for k in swept_dict if swept_dict[k] != nominal_dict[k]}
    unexpected = diffs - allowed_diff_keys
    assert not unexpected, f"unexpected EnvConfig fields differ from nominal: {unexpected}"


def assert_defender_arm_config(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """Defender-speed arm: only v_d (+ estimator) may differ; v_e must equal nominal."""
    assert_only_mobility_speed_differs(swept_config, nominal_config)
    assert swept_config.v_e == nominal_config.v_e == NOMINAL_HOSTILE_SPEED


def assert_hostile_arm_config(swept_config: EnvConfig, nominal_config: EnvConfig) -> None:
    """Hostile-speed arm: only v_e (+ estimator) may differ; v_d must equal nominal."""
    assert_only_mobility_speed_differs(swept_config, nominal_config)
    assert swept_config.v_d == nominal_config.v_d == NOMINAL_DEFENDER_SPEED


# =============================================================================
# PRESPECIFIED COMPARISONS (locked BEFORE opening seed 32000)
# =============================================================================
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_lead", "ppo", "lead"),                             # item 21.A
)
ESTIMATOR_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("kalman_greedy_vs_greedy", "kalman_greedy", "greedy"),     # item 22.B
    ("ppo_kalman_vs_ppo", "ppo_kalman", "ppo"),                  # item 22.C
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("ppo_vs_greedy", "ppo", "greedy"),
    ("ppo_vs_pn", "ppo", "pn"),
    ("lead_vs_greedy", "lead", "greedy"),
    ("pn_vs_greedy", "pn", "greedy"),
)
ALL_COMPARISONS: tuple[tuple[str, str, str], ...] = PRIMARY_COMPARISONS + ESTIMATOR_COMPARISONS + SECONDARY_COMPARISONS

# =============================================================================
# OUTPUT LOCATION (gitignored; never committed)
# =============================================================================
ROBUSTNESS_MOBILITY_OUTPUT_ROOT = PROJECT_ROOT / "results" / "robustness" / "mobility"
EXPERIMENT_MANIFEST_FILENAME = "experiment_manifest.json"
MODEL_HASHES_FILENAME = "model_hashes.json"
RAW_EPISODE_RESULTS_FILENAME = "raw_episode_results.csv"
DEFENDER_SPEED_SUMMARY_FILENAME = "defender_speed_summary.csv"
HOSTILE_SPEED_SUMMARY_FILENAME = "hostile_speed_summary.csv"
MOBILITY_RATIO_SUMMARY_FILENAME = "mobility_ratio_summary.csv"
PPO_TRAINING_SEED_SUMMARY_FILENAME = "ppo_training_seed_summary.csv"
PAIRWISE_COMPARISONS_FILENAME = "pairwise_comparisons.csv"
OPERATIONAL_SUMMARY_FILENAME = "operational_summary.csv"
ESTIMATOR_SUMMARY_FILENAME = "estimator_summary.csv"
MOBILITY_SUMMARY_FILENAME = "mobility_summary.json"
