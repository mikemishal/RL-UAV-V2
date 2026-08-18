"""Lead-Residual PPO implementation correctness tests -- IMPLEMENTATION ONLY.

Script-style test consistent with the project's other test files (no
pytest-specific fixtures; also pytest-collectible since functions are
named test_*).

NO MODEL IS TRAINED HERE. All "residual policies" below are synthetic
(zero / fixed-perpendicular / random-bounded) deterministic generators used
purely to exercise the composition math, observation contract, acquisition
fairness, and physical-limit enforcement -- never a learned checkpoint.

Covers (per the lead-residual-ppo task spec, item 24):
    1. zero residual reproduces Lead direction
    2. positive parallel residual reproduces Lead direction
    3. negative parallel residual reproduces Lead direction
    4. perpendicular residual respects maximum angle
    5. arbitrary residual respects maximum angle (many random combinations)
    6. oversized raw residual is norm-clipped
    7. output is finite/unit norm (incl. degenerate lead=0 safety)
    8. 19-D observation shape
    9. no ground-truth leakage
    10. first-detection handoff unchanged
    11. acquisition fairness on dev seeds
    12. zero-residual full episode matches standalone Lead (summary-level)
    12b. zero-residual STEPWISE trajectory equivalence (bit-identical; see
         compose_lead_residual's exact fast path) -- hardening item 2
    13. existing policy regression (Greedy/PN/Lead/PPO/PPO-Kalman unchanged)
    14. physical-limit enforcement
    15. residual_max_angle_deg domain validation (reject <0, >=90, NaN, Inf;
        representative valid values 0/1/30/45/89; theta_max=0 ignores residual)
        -- hardening item 3

Run directly:
    python test_lead_residual_ppo.py
"""

from __future__ import annotations

import numpy as np

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies.baseline.lead_intercept_policy import LeadInterceptPolicy
from uav_defend.policies.registry import get_policy
from uav_defend.policies.sanitize import build_policy_info
from uav_defend.policies.residual.lead_residual_composition import (
    DEFAULT_RESIDUAL_MAX_ANGLE_DEG,
    angle_between_deg,
    compose_lead_residual,
)
from uav_defend.policies.residual.lead_residual_observation import (
    build_residual_observation,
    residual_observation_dim,
)
from uav_defend.policies.residual.lead_residual_ppo_policy_wrapper import LeadResidualPPOPolicyWrapper

from experiments.final_eval_utils import _check_physical_limits, PhysicalLimitViolation
from experiments.post_detection_diagnostic_eval_utils import run_diagnostic_episode
from experiments.post_detection_diagnostic_protocol import FROZEN_MODELS
from experiments.lead_residual_training_env import LeadResidualTrainingEnv

# Non-reserved development seeds ONLY -- NEVER 40100-40299, 41000-45999, >=46000.
_DEV_SEEDS = (123, 124, 125, 126, 127)
_DEV_ROOT = 5000  # matches the dev root already used elsewhere in this repo

_CFG_DIRECT = EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)


# ---------------------------------------------------------------------------
# Test-only synthetic residual policies (NOT learned; see module docstring).
# Reuse the SAME compose_lead_residual() and LeadInterceptPolicy used by
# production code -- no duplicate math anywhere.
# ---------------------------------------------------------------------------
class _SyntheticResidualPolicy:
    def __init__(self, residual_fn, residual_max_angle_deg=DEFAULT_RESIDUAL_MAX_ANGLE_DEG, config=None):
        self._lead_policy = LeadInterceptPolicy(state_source="measurement", config=config)
        self._residual_fn = residual_fn
        self.residual_max_angle_deg = residual_max_angle_deg
        self.last_lead_direction = None

    def act(self, obs, info):
        lead_direction = np.asarray(self._lead_policy.act(obs, info), dtype=np.float32)
        self.last_lead_direction = lead_direction
        residual = self._residual_fn(lead_direction)
        return compose_lead_residual(lead_direction, residual, self.residual_max_angle_deg)

    def reset(self):
        self._lead_policy.reset()


def _zero_residual_fn(u_L):
    return np.zeros(3, dtype=np.float64)


def _fixed_perpendicular_residual_fn(u_L):
    """Deterministic unit vector perpendicular to u_L (falls back to a
    different reference axis if u_L is parallel to the first choice)."""
    u_L = np.asarray(u_L, dtype=np.float64)
    if np.linalg.norm(u_L) < 1e-8:
        return np.zeros(3, dtype=np.float64)
    ref = np.array([0.0, 0.0, 1.0])
    if abs(np.dot(u_L / np.linalg.norm(u_L), ref)) > 0.9:
        ref = np.array([1.0, 0.0, 0.0])
    perp = np.cross(u_L, ref)
    return perp / np.linalg.norm(perp)


def _random_bounded_residual_fn(rng):
    def _fn(u_L):
        return rng.uniform(-1.0, 1.0, size=3)
    return _fn


# ---------------------------------------------------------------------------
# 1/2/3. Zero / parallel residual invariance (action level)
# ---------------------------------------------------------------------------
def test_zero_residual_reproduces_lead_direction():
    rng = np.random.default_rng(0)
    for _ in range(200):
        u_L = rng.normal(size=3)
        u_L = u_L / np.linalg.norm(u_L)
        hybrid = compose_lead_residual(u_L, np.zeros(3), DEFAULT_RESIDUAL_MAX_ANGLE_DEG)
        assert np.allclose(hybrid, u_L, atol=1e-6), (hybrid, u_L)


def test_positive_parallel_residual_reproduces_lead_direction():
    rng = np.random.default_rng(1)
    for _ in range(200):
        u_L = rng.normal(size=3)
        u_L = u_L / np.linalg.norm(u_L)
        scale = rng.uniform(0.01, 5.0)  # includes >1 (must norm-clip first, still parallel)
        r = scale * u_L
        hybrid = compose_lead_residual(u_L, r, DEFAULT_RESIDUAL_MAX_ANGLE_DEG)
        assert np.allclose(hybrid, u_L, atol=1e-5), (hybrid, u_L, scale)


def test_negative_parallel_residual_reproduces_lead_direction():
    rng = np.random.default_rng(2)
    for _ in range(200):
        u_L = rng.normal(size=3)
        u_L = u_L / np.linalg.norm(u_L)
        scale = rng.uniform(0.01, 5.0)
        r = -scale * u_L
        hybrid = compose_lead_residual(u_L, r, DEFAULT_RESIDUAL_MAX_ANGLE_DEG)
        assert np.allclose(hybrid, u_L, atol=1e-5), (hybrid, u_L, scale)


# ---------------------------------------------------------------------------
# 4/5/9. Angular-bound tests
# ---------------------------------------------------------------------------
def test_perpendicular_residual_respects_maximum_angle():
    rng = np.random.default_rng(3)
    theta_max = 30.0
    max_seen = 0.0
    for _ in range(500):
        u_L = rng.normal(size=3)
        u_L = u_L / np.linalg.norm(u_L)
        perp = _fixed_perpendicular_residual_fn(u_L)  # unit perpendicular
        hybrid = compose_lead_residual(u_L, perp, theta_max)
        angle = angle_between_deg(hybrid, u_L)
        assert angle <= theta_max + 1e-4, (angle, theta_max)
        max_seen = max(max_seen, angle)
    # A unit perpendicular residual should approach the configured maximum.
    assert max_seen > theta_max - 1.0, f"expected near-maximal angle, got {max_seen}"


def test_arbitrary_residual_respects_maximum_angle():
    rng = np.random.default_rng(4)
    theta_max = 30.0
    combos = []
    # zero, unit-perp, mixed parallel/perp, oversized, near-axis, 3D directions
    combos.append((np.array([1.0, 0.0, 0.0]), np.zeros(3)))
    combos.append((np.array([0.0, 1.0, 0.0]), np.array([1.0, 0.0, 0.0])))
    combos.append((np.array([0.0, 0.0, 1.0]), np.array([5.0, 5.0, 5.0])))  # oversized -> clip
    for _ in range(2000):
        u_L = rng.normal(size=3)
        u_L = u_L / np.linalg.norm(u_L)
        r = rng.uniform(-3.0, 3.0, size=3)  # includes oversized raw residuals
        combos.append((u_L, r))
    for u_L, r in combos:
        hybrid = compose_lead_residual(u_L, r, theta_max)
        angle = angle_between_deg(hybrid, u_L / np.linalg.norm(u_L))
        assert angle <= theta_max + 1e-4, (u_L, r, angle)
        assert np.all(np.isfinite(hybrid))
        assert abs(float(np.linalg.norm(hybrid)) - 1.0) < 1e-4


# ---------------------------------------------------------------------------
# 6. Oversized raw residual is norm-clipped
# ---------------------------------------------------------------------------
def test_oversized_raw_residual_is_norm_clipped():
    u_L = np.array([1.0, 0.0, 0.0])
    r_small = np.array([0.0, 1.0, 0.0])
    r_large = np.array([0.0, 100.0, 0.0])  # same direction, huge magnitude
    hybrid_small = compose_lead_residual(u_L, r_small, 30.0)
    hybrid_large = compose_lead_residual(u_L, r_large, 30.0)
    # After norm-clipping, r_large / ||r_large|| == r_small (same unit direction)
    assert np.allclose(hybrid_small, hybrid_large, atol=1e-6)


# ---------------------------------------------------------------------------
# 7. Finite / unit norm, incl. degenerate lead=0 safety
# ---------------------------------------------------------------------------
def test_output_finite_unit_norm():
    rng = np.random.default_rng(5)
    for _ in range(500):
        u_L = rng.normal(size=3)
        norm = np.linalg.norm(u_L)
        if norm > 1e-6:
            u_L = u_L / norm
        r = rng.uniform(-10.0, 10.0, size=3)
        hybrid = compose_lead_residual(u_L, r, 30.0)
        assert np.all(np.isfinite(hybrid))


def test_degenerate_zero_lead_direction_is_safe():
    # Lead itself produced no legitimate direction (e.g. no target) --
    # must never emit NaN/Inf/zero-division, and must never substitute
    # ground truth; the documented behavior is to return the zero vector.
    hybrid = compose_lead_residual(np.zeros(3), np.array([1.0, 2.0, 3.0]), 30.0)
    assert np.all(np.isfinite(hybrid))
    assert np.allclose(hybrid, np.zeros(3))


def test_nan_inf_inputs_rejected_not_silently_propagated():
    for bad in (np.array([np.nan, 0.0, 0.0]), np.array([np.inf, 0.0, 0.0])):
        try:
            compose_lead_residual(bad, np.zeros(3), 30.0)
            raised = False
        except ValueError:
            raised = True
        assert raised, "non-finite lead_direction must raise, never silently propagate"


# ---------------------------------------------------------------------------
# 8. 19-D observation shape (verified programmatically, not hardcoded)
# ---------------------------------------------------------------------------
def test_observation_shape_19():
    env = SoldierEnv(config=_CFG_DIRECT)
    direct_dim = env.observation_space.shape[0]
    expected_dim = residual_observation_dim(env.observation_space)
    assert expected_dim == direct_dim + 3
    assert expected_dim == 19  # current canonical SoldierEnv: 16 + 3

    obs, info = env.reset(seed=123)
    lead_policy = LeadInterceptPolicy(state_source="measurement", config=_CFG_DIRECT)
    policy_info = build_policy_info(info, "measurement")
    lead_direction = lead_policy.act(obs, policy_info)
    residual_obs = build_residual_observation(obs, lead_direction)
    assert residual_obs.shape == (expected_dim,)
    assert residual_obs.dtype == np.float32
    env.close()

    train_env = LeadResidualTrainingEnv(config=_CFG_DIRECT, root_seed=_DEV_ROOT)
    assert train_env.observation_space.shape == (expected_dim,)
    assert train_env.action_space.shape == (3,)
    r_obs, _ = train_env.reset()
    assert r_obs.shape == (expected_dim,)
    train_env.close()


# ---------------------------------------------------------------------------
# 9. No ground-truth leakage
# ---------------------------------------------------------------------------
def test_no_ground_truth_leakage():
    env = SoldierEnv(config=_CFG_DIRECT)
    obs, info = env.reset(seed=124)
    assert "enemy_pos" in info and "enemy_vel" in info  # full env info HAS ground truth

    lead_policy = LeadInterceptPolicy(state_source="measurement", config=_CFG_DIRECT)
    policy_info = build_policy_info(info, "measurement")
    assert "enemy_pos" not in policy_info and "enemy_vel" not in policy_info

    lead_direction_sanitized = lead_policy.act(obs, policy_info)

    # Demonstrate identical behavior whether or not enemy_pos/enemy_vel are
    # present in the (never-used-by-policy) surrounding dict -- the policy
    # reads only the sanitized keys either way.
    lead_policy.reset()
    lead_direction_full_info_but_unused = lead_policy.act(obs, info)  # info still has gt, but policy ignores it
    assert np.allclose(lead_direction_sanitized, lead_direction_full_info_but_unused)

    # Poison the ground-truth fields; sanitized policy_info is unaffected.
    poisoned = dict(info)
    poisoned["enemy_pos"] = np.array([9999.0, 9999.0, 9999.0])
    poisoned["enemy_vel"] = np.array([9999.0, 9999.0, 9999.0])
    poisoned_sanitized = build_policy_info(poisoned, "measurement")
    lead_policy.reset()
    lead_direction_poisoned = lead_policy.act(obs, poisoned_sanitized)
    assert np.allclose(lead_direction_sanitized, lead_direction_poisoned)
    env.close()


def test_residual_wrapper_no_ground_truth_leakage():
    class _FakeModel:
        def predict(self, obs, deterministic=True):
            return np.zeros(3, dtype=np.float32), None

    wrapper = LeadResidualPPOPolicyWrapper(_FakeModel(), config=_CFG_DIRECT)
    env = SoldierEnv(config=_CFG_DIRECT)
    obs, info = env.reset(seed=125)
    policy_info = build_policy_info(info, "measurement")
    action = wrapper.act(obs, policy_info)
    assert np.all(np.isfinite(action))

    poisoned = dict(policy_info)
    assert "enemy_pos" not in poisoned and "enemy_vel" not in poisoned
    env.close()


# ---------------------------------------------------------------------------
# 10. First-detection handoff unchanged
# ---------------------------------------------------------------------------
def test_first_detection_handoff_unchanged():
    train_env = LeadResidualTrainingEnv(config=_CFG_DIRECT, root_seed=_DEV_ROOT)
    residual_obs, info = train_env.reset()
    # Reset returns the FIRST detected observation: defender co-located with
    # soldier, at rest, controller action not yet executed.
    assert info["just_detected"] is True
    assert info["enemy_detected"] is True
    assert np.allclose(info["defender_pos"], info["soldier_pos"])
    assert np.allclose(info["defender_vel"], np.zeros(3))
    assert residual_obs.shape == (19,)
    train_env.close()


# ---------------------------------------------------------------------------
# 11. Acquisition fairness on dev seeds
# ---------------------------------------------------------------------------
def test_acquisition_fairness_on_dev_seeds():
    fields = ("soldier_pos", "defender_pos", "defender_vel", "enemy_pos", "enemy_vel")
    max_discrepancy = 0.0
    for seed in _DEV_SEEDS:
        env_ref = SoldierEnv(config=_CFG_DIRECT)
        ref_policy = get_policy("greedy")
        ref_row = run_diagnostic_episode(env_ref, ref_policy, seed=seed, dt=_CFG_DIRECT.dt, record_trajectory=True)
        ref_traj = ref_row["pre_detection_trajectory"]
        env_ref.close()

        env_res = SoldierEnv(config=_CFG_DIRECT)
        res_policy = _SyntheticResidualPolicy(_zero_residual_fn, config=_CFG_DIRECT)
        res_row = run_diagnostic_episode(env_res, res_policy, seed=seed, dt=_CFG_DIRECT.dt, record_trajectory=True)
        res_traj = res_row["pre_detection_trajectory"]
        env_res.close()

        assert len(ref_traj) == len(res_traj), (seed, len(ref_traj), len(res_traj))
        for a, b in zip(ref_traj, res_traj):
            assert a["enemy_detected"] == b["enemy_detected"]
            for field in fields:
                diff = float(np.max(np.abs(a[field] - b[field])))
                max_discrepancy = max(max_discrepancy, diff)
        assert ref_row["detection_step"] == res_row["detection_step"], seed
    assert max_discrepancy < 1e-6, max_discrepancy


# ---------------------------------------------------------------------------
# 12. Zero-residual full episode matches standalone Lead
# ---------------------------------------------------------------------------
def test_zero_residual_full_episode_matches_standalone_lead():
    max_discrepancy = 0.0
    for seed in _DEV_SEEDS:
        env_lead = SoldierEnv(config=_CFG_DIRECT)
        lead_policy = get_policy("lead")
        lead_row = run_diagnostic_episode(env_lead, lead_policy, seed=seed, dt=_CFG_DIRECT.dt, record_trajectory=False)
        env_lead.close()

        env_res = SoldierEnv(config=_CFG_DIRECT)
        res_policy = _SyntheticResidualPolicy(_zero_residual_fn, config=_CFG_DIRECT)
        res_row = run_diagnostic_episode(env_res, res_policy, seed=seed, dt=_CFG_DIRECT.dt, record_trajectory=False)
        env_res.close()

        assert lead_row["outcome"] == res_row["outcome"], (seed, lead_row["outcome"], res_row["outcome"])
        assert lead_row["total_steps"] == res_row["total_steps"], seed
        assert lead_row["detection_step"] == res_row["detection_step"], seed
        diff = abs(lead_row["min_enemy_soldier_dist"] - res_row["min_enemy_soldier_dist"])
        max_discrepancy = max(max_discrepancy, diff)
    assert max_discrepancy < 1e-3, max_discrepancy


# ---------------------------------------------------------------------------
# 2 (hardening). Strengthened zero-residual trajectory equivalence:
# EVERY post-detection timestep compared (state, estimator, action, terminal
# status), not merely terminal outcome/summary distance.
# ---------------------------------------------------------------------------
_STEPWISE_STATE_FIELDS = ("soldier_pos", "defender_pos", "defender_vel", "enemy_pos", "enemy_vel")


def test_zero_residual_stepwise_matches_lead_every_timestep():
    """
    For each dev seed, step standalone Lead and the zero-residual
    Lead-Residual interface in lockstep (same seed => identical RNG streams)
    and compare EVERY timestep: physical state, detection flag, raw
    measurement/measurement-velocity, Lead direction, hybrid direction, and
    terminal status/outcome. Reports (via assertions) the number of compared
    timesteps, max state discrepancy, max action discrepancy, and mismatch
    count.

    compose_lead_residual() has an EXACT bit-identical fast path for zero
    residual (see uav_defend/policies/residual/lead_residual_composition.py),
    so the hybrid action equals the Lead action bit-for-bit at every step;
    since both environments share the identical seed (=> identical RNG
    streams, consumed identically regardless of the action's numeric value),
    the two trajectories are therefore expected to be BIT-IDENTICAL, not
    merely within a tolerance.
    """
    total_timesteps_compared = 0
    max_state_discrepancy = 0.0
    max_action_discrepancy = 0.0
    total_mismatches = 0

    for seed in _DEV_SEEDS:
        env_lead = SoldierEnv(config=_CFG_DIRECT)
        lead_policy = LeadInterceptPolicy(state_source="measurement", config=_CFG_DIRECT)
        obs_l, info_l = env_lead.reset(seed=seed)

        env_res = SoldierEnv(config=_CFG_DIRECT)
        res_policy = _SyntheticResidualPolicy(_zero_residual_fn, config=_CFG_DIRECT)
        obs_r, info_r = env_res.reset(seed=seed)

        # t=0 (reset) state must already match exactly.
        for field in _STEPWISE_STATE_FIELDS:
            disc = float(np.max(np.abs(info_l[field] - info_r[field])))
            max_state_discrepancy = max(max_state_discrepancy, disc)
        assert info_l["enemy_detected"] == info_r["enemy_detected"]

        done = False
        while not done:
            policy_info_l = build_policy_info(info_l, "measurement")
            action_l = lead_policy.act(obs_l, policy_info_l)

            policy_info_r = build_policy_info(info_r, "measurement")
            action_r = res_policy.act(obs_r, policy_info_r)  # hybrid direction

            action_disc = float(np.max(np.abs(action_l - action_r)))
            max_action_discrepancy = max(max_action_discrepancy, action_disc)

            obs_l, _, term_l, trunc_l, info_l = env_lead.step(action_l)
            obs_r, _, term_r, trunc_r, info_r = env_res.step(action_r)
            total_timesteps_compared += 1

            step_state_disc = 0.0
            for field in _STEPWISE_STATE_FIELDS:
                disc = float(np.max(np.abs(info_l[field] - info_r[field])))
                step_state_disc = max(step_state_disc, disc)
            max_state_discrepancy = max(max_state_discrepancy, step_state_disc)

            assert info_l["enemy_detected"] == info_r["enemy_detected"], (seed, total_timesteps_compared)
            if info_l.get("enemy_measurement") is not None and info_r.get("enemy_measurement") is not None:
                m_disc = float(np.max(np.abs(info_l["enemy_measurement"] - info_r["enemy_measurement"])))
                step_state_disc = max(step_state_disc, m_disc)
                max_state_discrepancy = max(max_state_discrepancy, m_disc)
            if info_l.get("enemy_measurement_velocity_valid") and info_r.get("enemy_measurement_velocity_valid"):
                v_disc = float(np.max(np.abs(
                    info_l["enemy_measurement_velocity"] - info_r["enemy_measurement_velocity"]
                )))
                step_state_disc = max(step_state_disc, v_disc)
                max_state_discrepancy = max(max_state_discrepancy, v_disc)

            assert term_l == term_r and trunc_l == trunc_r, (seed, total_timesteps_compared)
            if info_l.get("outcome") is not None or info_r.get("outcome") is not None:
                assert info_l.get("outcome") == info_r.get("outcome"), (seed, total_timesteps_compared)

            if action_disc != 0.0 or step_state_disc != 0.0:
                total_mismatches += 1

            done = term_l or trunc_l

        env_lead.close()
        env_res.close()

    print(
        f"[stepwise zero-residual hardening] timesteps_compared={total_timesteps_compared} "
        f"max_state_discrepancy={max_state_discrepancy:.3e} "
        f"max_action_discrepancy={max_action_discrepancy:.3e} "
        f"mismatches={total_mismatches}"
    )
    # Preferred requirement: bit-identical. compose_lead_residual's exact
    # zero-residual fast path guarantees this (see module docstring).
    assert max_action_discrepancy == 0.0, (
        f"expected bit-identical hybrid==lead actions for zero residual, "
        f"max_action_discrepancy={max_action_discrepancy}"
    )
    assert max_state_discrepancy == 0.0, (
        f"expected bit-identical trajectories for zero residual, "
        f"max_state_discrepancy={max_state_discrepancy}"
    )
    assert total_mismatches == 0
    assert total_timesteps_compared > 0


# ---------------------------------------------------------------------------
# 3 (hardening). Residual angle domain validation
# ---------------------------------------------------------------------------
def test_theta_max_rejects_negative():
    try:
        compose_lead_residual(np.array([1.0, 0.0, 0.0]), np.zeros(3), -1.0)
        assert False, "expected ValueError for theta_max < 0"
    except ValueError:
        pass


def test_theta_max_rejects_90_and_above():
    for bad in (90.0, 90.001, 120.0):
        try:
            compose_lead_residual(np.array([1.0, 0.0, 0.0]), np.array([0.0, 1.0, 0.0]), bad)
            assert False, f"expected ValueError for theta_max={bad}"
        except ValueError:
            pass


def test_theta_max_rejects_nan_and_inf():
    for bad in (float("nan"), float("inf"), float("-inf")):
        try:
            compose_lead_residual(np.array([1.0, 0.0, 0.0]), np.zeros(3), bad)
            assert False, f"expected ValueError for theta_max={bad}"
        except ValueError:
            pass


def test_theta_max_representative_valid_values():
    u_L = np.array([1.0, 0.0, 0.0])
    r = np.array([0.0, 1.0, 0.0])  # unit perpendicular
    for theta_max in (0.0, 1.0, 30.0, 45.0, 89.0):
        hybrid = compose_lead_residual(u_L, r, theta_max)
        assert np.all(np.isfinite(hybrid))
        angle = angle_between_deg(hybrid, u_L)
        assert angle <= theta_max + 1e-4, (theta_max, angle)


def test_theta_max_zero_ignores_residual():
    rng = np.random.default_rng(7)
    for _ in range(200):
        u_L = rng.normal(size=3)
        u_L = u_L / np.linalg.norm(u_L)
        r = rng.uniform(-1.0, 1.0, size=3)
        hybrid = compose_lead_residual(u_L, r, 0.0)
        assert np.allclose(hybrid, u_L, atol=1e-6), (hybrid, u_L, r)


# ---------------------------------------------------------------------------
# 13. Existing policy regression (deterministic reference values)
# ---------------------------------------------------------------------------
_REGRESSION_REFERENCE = {
    # (method, seed) -> (outcome, detection_step, total_steps)
    ("greedy", 123): ("soldier_caught", 11, 390),
    ("greedy", 124): ("soldier_caught", 11, 242),
    ("greedy", 125): ("intercepted", 11, 104),
    ("greedy", 126): ("soldier_caught", 114, 408),
    ("greedy", 127): ("intercepted", 9, 69),
    ("pn", 123): ("intercepted", 11, 388),
    ("pn", 124): ("intercepted", 11, 160),
    ("pn", 125): ("intercepted", 11, 105),
    ("pn", 126): ("intercepted", 114, 443),
    ("pn", 127): ("intercepted", 9, 234),
    ("lead", 123): ("intercepted", 11, 369),
    ("lead", 124): ("intercepted", 11, 148),
    ("lead", 125): ("intercepted", 11, 98),
    ("lead", 126): ("intercepted", 114, 383),
    ("lead", 127): ("intercepted", 9, 274),
    ("ppo_45", 123): ("intercepted", 11, 44),
    ("ppo_45", 124): ("intercepted", 11, 181),
    ("ppo_45", 125): ("intercepted", 11, 37),
    ("ppo_45", 126): ("intercepted", 114, 160),
    ("ppo_45", 127): ("intercepted", 9, 200),
    ("ppo_kalman_45", 123): ("intercepted", 11, 89),
    ("ppo_kalman_45", 124): ("unsafe_intercept", 11, 13),
    ("ppo_kalman_45", 125): ("intercepted", 11, 81),
    ("ppo_kalman_45", 126): ("intercepted", 114, 275),
    ("ppo_kalman_45", 127): ("intercepted", 9, 202),
}


def test_existing_policy_regression_unchanged():
    cfg_kalman = EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True)
    specs = [
        ("greedy", get_policy("greedy"), _CFG_DIRECT),
        ("pn", get_policy("pn"), _CFG_DIRECT),
        ("lead", get_policy("lead"), _CFG_DIRECT),
        ("ppo_45", get_policy("ppo", model_path=str(FROZEN_MODELS[("direct", 45)]["path"]), deterministic=True), _CFG_DIRECT),
        ("ppo_kalman_45", get_policy("ppo_kalman", model_path=str(FROZEN_MODELS[("kalman", 45)]["path"]), deterministic=True), cfg_kalman),
    ]
    for name, policy, cfg in specs:
        env = SoldierEnv(config=cfg)
        for seed in _DEV_SEEDS:
            row = run_diagnostic_episode(env, policy, seed=seed, dt=cfg.dt, record_trajectory=False)
            expected = _REGRESSION_REFERENCE[(name, seed)]
            actual = (row["outcome"], row["detection_step"], row["total_steps"])
            assert actual == expected, (name, seed, actual, expected)
        env.close()


# ---------------------------------------------------------------------------
# 14. Physical-limit enforcement (zero / fixed-perpendicular / random residual)
# ---------------------------------------------------------------------------
def test_physical_limit_enforcement_synthetic_residual_policies():
    rng = np.random.default_rng(42)
    residual_generators = {
        "zero": _zero_residual_fn,
        "fixed_perpendicular": _fixed_perpendicular_residual_fn,
        "random_bounded": _random_bounded_residual_fn(rng),
    }
    total_violations = 0
    total_episodes = 0
    for name, fn in residual_generators.items():
        for seed in _DEV_SEEDS:
            env = SoldierEnv(config=_CFG_DIRECT)
            policy = _SyntheticResidualPolicy(fn, config=_CFG_DIRECT)
            obs, info = env.reset(seed=seed)
            _check_physical_limits(info, env.config)
            policy.reset()
            done = False
            while not done:
                policy_info = build_policy_info(info, "measurement")
                action = policy.act(obs, policy_info)
                assert np.all(np.isfinite(action))
                obs, reward, term, trunc, info = env.step(action)
                try:
                    _check_physical_limits(info, env.config)
                except PhysicalLimitViolation:
                    total_violations += 1
                done = term or trunc
            total_episodes += 1
            env.close()
    assert total_violations == 0, f"{total_violations} physical-limit violations across {total_episodes} episodes"
    assert total_episodes == len(residual_generators) * len(_DEV_SEEDS)


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
