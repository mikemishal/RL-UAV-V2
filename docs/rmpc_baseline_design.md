# Scenario-Based Robust MPC Baseline: Design Note

Status: **IMPLEMENTED / NOT YET TUNED OR PUBLICATION-EVALUATED.** The
controller, CEM optimizer, and scenario model described below have been
implemented and are covered by correctness/unit tests (all passing). NO
hyperparameter tuning, NO Monte-Carlo/publication evaluation, and NO seed
`>= 71000` has been opened. This document's design rationale (Sections
1-19) remains the authoritative source of truth for WHY each choice was
made; this status block records WHAT currently exists in the repository.

### Implementation summary (added post-design, Prompt 2)

**Files added:**
- `uav_defend/dynamics/constrained_point_mass.py` -- pure, dependency-free
  `advance_velocity(current_velocity, desired_velocity, max_speed, max_accel,
  max_turn_rate_rad, max_climb_rate, max_descent_rate, dt, eps) ->
  (next_velocity, diagnostics)`, factored out of
  `SoldierEnv._advance_velocity` (now a thin wrapper delegating to it).
  Regression-tested (`test_constrained_point_mass.py`, 16 tests) against an
  independently re-derived oracle implementation and against the live
  `SoldierEnv` wrapper, proving bit-identical behavior; the full existing
  repository test suite (489 tests before this addition) continues to pass
  unchanged after the refactor.
- `uav_defend/policies/mpc/cem_optimizer.py` -- generic, domain-agnostic
  Cross-Entropy Method optimizer (`cem_optimize`), operating on
  `U in R^{H x 3}` with per-row eps-guarded unit-direction normalization,
  a dedicated caller-supplied `numpy.random.Generator`, deterministic
  stable tie-breaking, best-ever-candidate tracking (not merely the final
  distribution mean), a `min_std` floor, and graceful non-finite-candidate
  rejection (`CEMResult.success=False` if every candidate across the whole
  optimization was non-finite).
- `uav_defend/policies/mpc/hostile_scenarios.py` -- the six-scenario
  hostile-motion model (S1 nominal continuation, S2/S3 bounded left/right
  turn, S4/S5 bounded climb/descent, S6 MAX_AWAY closed-loop), propagated
  through `advance_velocity` using ONLY the hostile's public physical
  limits (`config.v_e`, `enemy_max_accel`, `enemy_max_turn_rate_deg`,
  `enemy_max_climb_rate`, `enemy_max_descent_rate`). S1-S5 are open-loop
  (independent of the candidate defender trajectory); S6 is the sole
  closed-loop scenario, computing `desired_dir = normalize(hostile_pred_k -
  defender_pred_k)` at each predicted step. Never reads
  `enemy_evasion_gain`, `enemy_evasion_radius`, or
  `_compute_enemy_evasion` (verified both by dedicated equal-trajectory
  tests across the full documented range of these config fields and by an
  AST-based static-code audit -- not merely a docstring/comment check).
- `uav_defend/policies/mpc/rmpc_cost.py` -- defender rollout
  (`rollout_defender`), per-scenario terminal-priority scoring
  (`score_trajectory`, mirroring `SoldierEnv.step()`'s exact
  `soldier_caught > unsafe_intercept > intercepted > progress-fallback`
  order and reward magnitudes) and robust `min`-aggregation
  (`evaluate_candidate`). Introduces ZERO new tunable weight constants
  beyond the four already-fixed `EnvConfig` reward fields.
- `uav_defend/policies/mpc/rmpc_policy.py` -- `RMPCConfig` (dataclass) and
  `RMPCPolicy` (the `act(obs, info) -> np.ndarray(3,)` / `reset()`
  controller). Pre-detection short-circuit (zero action, zero CEM
  evaluations); first-measurement fallback (CEM still runs, using the
  scenario model's own pursuit-direction fallback for a missing velocity
  estimate); optimizer-failure fallback to pure pursuit; a dedicated
  `numpy.random.Generator` re-seeded from `controller_seed` in both
  `__init__` and `reset()` (never touching any environment-owned RNG
  stream); diagnostics (`last_guidance_mode`, `last_scenario_scores`,
  `last_robust_score`, `last_best_sequence`, `last_objective_evaluations`)
  and non-action-affecting runtime instrumentation
  (`last_decision_time_s`, `decision_times_s`, `decision_time_stats()`).

**Test files added** (67 new tests total, all passing, both under `pytest`
and each file's direct-execution `__main__` runner):
- `test_constrained_point_mass.py` (16 tests) -- dynamics-refactor
  regression.
- `test_cem_optimizer.py` (11 tests) -- CEM correctness against synthetic
  known-optimum objectives, determinism, non-finite-candidate handling.
- `test_rmpc_scenarios.py` (8 tests) -- per-scenario physical-limit
  satisfaction, S1-S5 open-loop / S6 closed-loop trajectory independence,
  and the evasion-gain/radius independence test.
- `test_rmpc_policy.py` (20 tests) -- information contract (including a
  poisoned-`enemy_pos`/`enemy_vel` identical-action test and an AST-based
  static audit), action validity, determinism, edge cases, pre-detection
  compatibility, no side effects (including a before/after environment-RNG
  bit-generator-state snapshot test), evaluator compatibility (including a
  live `run_diagnostic_episode` smoke test), and runtime-instrumentation
  neutrality.

**Current default TEST configuration** (`RMPCConfig`'s dataclass
defaults) -- an UNTUNED midpoint of the Section 11 candidate grids, NOT a
selected final configuration: `horizon=6`, `population_size=128`,
`elite_fraction=0.2`, `cem_iterations=5`, plus fixed (non-tunable) CEM
numerical safeguards `cem_initial_std=1.0`, `cem_min_std=0.05`,
`cem_sample_clip=3.0`. Section 15's staged hyperparameter-selection plan
(Stage A/B/C, seeds 71000-71399) has NOT been executed; no seed `>= 71000`
has been opened by this implementation work, and no `results/` directory
has been created for RMPC. NO success-rate, interception-performance, or
runtime-benchmark numbers exist yet for this controller.

**Not yet done:** hyperparameter selection (Section 15), integration into
the evaluation-pipeline `MethodInstance`/`build_policy_for_instance`
registries (Section 4.5), the final nominal comparison and targeted
robustness sweeps (Section 14, seeds `>= 72000`), and the manuscript text
(Section 18-19).

### Pre-tuning hardening pass (Prompt 3)

Status remains **IMPLEMENTED / NOT YET TUNED OR PUBLICATION-EVALUATED.**
This pass corrected two correctness/rigor issues found before any
development/validation seed was opened; no seed `>= 71000` was opened by
this pass and no `results/` directory or performance number exists.

1. **CEM is now initialized from the measurement-based Lead direction, not
   an unstructured zero mean.** `uav_defend/policies/mpc/lead_init.py`
   (new) provides a pure `solve_lead_direction(target_pos, target_vel,
   defender_pos, v_d, eps)` function that reuses
   `LeadInterceptPolicy._solve_intercept_time` (an existing, already-tested
   `@staticmethod` with no instance-state dependence) directly, rather than
   re-deriving a second copy of the intercept-time quadratic.
   `LeadInterceptPolicy` itself is UNMODIFIED by this pass (confirmed by an
   empty `git diff` against that file) -- a dedicated consistency-
   regression test (`test_lead_init.py`) proves the new pure helper's
   direction matches `LeadInterceptPolicy(state_source="measurement").act()`
   bit-for-bit across randomized states. `RMPCPolicy.act()` now builds
   `init_mean` by repeating this direction over the full horizon and passes
   it to `cem_optimize(..., init_mean=init_mean)` (previously `None`). If
   the finite-difference hostile velocity is unavailable or no valid
   predictive intercept solution exists, the same helper's pure-pursuit
   fallback toward `enemy_measurement` is used instead. No cross-timestep
   warm-start was added. A new diagnostic, `last_initialization_mode`
   (`"lead"` | `"pursuit"` | `None` when CEM did not run), records which
   case applied.
2. **The nonterminal (no-terminal-event-within-horizon) objective term no
   longer includes an undeclared extra weight.** The previous formula,
   `reward_progress_scale * (dist_de_0 - dist_de_H) - dist_de_H`, is
   corrected to `reward_progress_scale * (dist_de_0 - dist_de_H)` --
   `dist_de_0` is fixed across all candidates at a given decision, so
   maximizing progress is equivalent to minimizing terminal
   defender-hostile distance without an extra, undeclared `-dist_de_H`
   coefficient. Terminal-event rewards (`reward_intercept`,
   `reward_soldier_caught`, `reward_unsafe_intercept`) are unchanged. RMPC
   still introduces ZERO new tunable weight constants beyond the
   already-fixed `EnvConfig.reward_progress_scale`.
   **Publication-safe wording:** "RMPC uses the environment's terminal
   mission rewards and closing-progress scale in its finite-horizon
   objective" -- NOT "RMPC uses exactly the same reward function as PPO"
   (RMPC's objective is mission-aligned but is not an exact replica of the
   RL training reward; `reward_time_penalty`/`reward_proximity_warning`
   remain intentionally omitted).
3. **Corrected theoretical computational accounting** (still NOT a measured
   runtime benchmark) for the current default TEST configuration
   (`N=population_size=128`, `I=cem_iterations=5`, `S`=6 scenarios,
   `H=horizon=6`):
   - Candidate objective evaluations: `N * I = 128 * 5 = 640`, plus 1 final
     selected-candidate reevaluation (for diagnostics).
   - Hostile scenario-step propagations: `N * I * S * H = 128 * 5 * 6 * 6 =
     23,040`, before the final reevaluation's own `S * H = 36` steps.
   - Defender rollout steps: `N * I * H = 128 * 5 * 6 = 3,840`, before the
     final reevaluation's own `H = 6` steps.
   These are theoretical accounting quantities derived directly from the
   implementation's call structure, not measured wall-clock benchmarks.
4. New test files: `test_lead_init.py` (8 tests), `test_rmpc_cost.py`
   (3 tests), plus 5 new tests added to `test_rmpc_policy.py` -- 16 new
   tests total, all passing under `pytest` and each file's direct-execution
   `__main__` runner. Full repository suite: 544 passed (was 528 before
   this pass).

## 1. Scientific motivation

The manuscript currently compares six existing methods: Greedy (simple
classical reference), PN (classical guidance reference), Lead (strong
predictive analytical guidance), PPO (end-to-end learned controller),
PPO-Kalman (estimator ablation), and LR-PPO (the proposed hybrid). None of
these is an optimization-based / receding-horizon optimal-control
comparator. Reviewers of the original submission noted this gap: the paper
lacks a robust-control or model-predictive-control baseline against which a
learned/hybrid method's claimed robustness can be contextualized.

## 2. Reviewer concern being addressed

"The paper compares against classical guidance laws and a pure learning
baseline, but does not compare against an established robust/optimal-control
technique. A model-predictive or robust-optimization baseline would
strengthen the comparison and address whether the learned residual is
providing something beyond what a receding-horizon planner with the same
sensing budget already achieves."

## 3. Role in controller hierarchy

| Method | Role |
|---|---|
| Greedy | simple classical reference |
| PN | classical guidance reference |
| Lead | strong predictive analytical guidance |
| PPO | end-to-end learned controller |
| PPO-Kalman | estimator ablation (not a co-equal method) |
| LR-PPO | proposed hybrid method |
| **RMPC** | **robust optimization-based model-predictive comparator (not a proposed contribution)** |

RMPC is added as a **seventh controller / additional comparator**, not as a competing
proposed method. Its purpose is to show where a receding-horizon robust
planner -- with access to exactly the same sanitized sensing budget as every
other measurement-track controller -- lands relative to Lead, PPO, and
LR-PPO, not to claim it is the best possible controller.

### 3.1 Terminology (mandatory)

Use only:
- "scenario-based robust model predictive controller"
- "robust optimization-based baseline"
- "receding-horizon robust predictive controller"

Never use (unless a specific mathematical result in a later prompt strictly
justifies it, which a sampling-based CEM optimizer over a finite, hand-
specified scenario set does **not**):
- "optimal controller" / "provably optimal"
- "game-theoretically optimal"
- "worst-case optimal over all admissible hostile policies"

CEM with a finite population and a finite, hand-designed scenario set finds
a **locally good, scenario-robust** control sequence -- it is not a proof of
optimality over the true (continuous, stochastic) hostile policy space.

## 4. Repository audit summary

### 4.1 Canonical source of truth (in priority order, per task instruction)
1. Current executable source (`uav_defend/`, `experiments/`).
2. Revised protocol/design documents: `docs/revised_predetection_protocol.md`,
   `docs/lead_residual_ppo_design.md`, `docs/post_detection_training_protocol.md`,
   `docs/lead_residual_training_protocol.md`.
3. Frozen evidence manifests (`results/expanded_robustness/*/protocol_manifest.json`,
   `paper_artifacts/manifests/master_evidence_manifest.json`).

**`docs/experiment_configuration.md` is STALE and must NOT be treated as
authoritative.** It describes a 2-D state space and 2-D action space
(confirmed by direct inspection: lines referencing `s in R^2`, `a in
[-1,1]^2`, and a "2D domain"/"2D grid"), which conflicts with the current
3-D, 16-D-observation / 3-D-action implementation verified throughout this
audit. Any RMPC design decision must be checked against source code first.

### 4.2 Canonical dynamics functions
- `uav_defend/envs/soldier_env.py::SoldierEnv._advance_velocity` -- the
  SINGLE reusable constrained-velocity update shared by both the defender
  and the hostile (a 3-D constrained point-mass model, not six-DOF). Takes
  `current_velocity, desired_velocity, max_speed, max_accel,
  max_turn_rate_rad, max_climb_rate, max_descent_rate` and returns
  `(next_velocity, diagnostics)`. Currently an **instance method** (reads
  `self.config.dt`, `self.config.eps` internally) -- NOT a pure/static
  function today.
- `SoldierEnv._move_defender` / `SoldierEnv._move_enemy` -- call
  `_advance_velocity` with defender-specific or enemy-specific `EnvConfig`
  fields respectively; enemy motion additionally combines pursuit direction
  + AR(1) weave bias + heading noise + reactive evasion (`_compute_enemy_evasion`)
  into a raw desired direction *before* calling `_advance_velocity`.
- Action semantics (confirmed): the environment normalizes a nonzero action
  vector and commands `desired_velocity = config.v_d * normalize(action)`
  for the defender. **RMPC must therefore optimize a DIRECTION, not a
  throttle magnitude** -- any magnitude information in the action is
  discarded by the environment.
- `SoldierEnv.step()` implements the exact pre-detection standby / handoff
  order (see Section 5) and the terminal-condition priority order
  (`soldier_caught` > `unsafe_intercept` > `intercepted` > `timeout`, each a
  pure function of instantaneous 3-D distances -- see Section 10).

### 4.3 Canonical sensing/sanitization functions
- `uav_defend/policies/sanitize.py::build_policy_info(info, estimator_mode)`
  -- the single source of truth for "what is a legitimate policy input".
  `estimator_mode_for_env(env)` infers `"measurement"` or `"kalman"` from
  `env.config.use_kalman_tracking`.
- Legitimate fields (measurement track, which is the track RMPC will use --
  see Section 4.4): `soldier_pos`, `defender_pos`, `defender_vel`,
  `enemy_detected`, `enemy_measurement`, `enemy_measurement_velocity`,
  `enemy_measurement_velocity_valid`.
- Permanently stripped ground-truth keys (defense-in-depth, unconditionally
  removed even if a future edit accidentally allow-lists them):
  `enemy_pos`, `enemy_vel`.

### 4.4 Policy interface
Every policy in this repository implements `policy.act(obs, info) ->
np.ndarray(shape=(3,))` and `policy.reset() -> None`. `obs` is the
environment's 16-D Direct (or Kalman) observation; most existing baseline
policies (Greedy, PN, Lead) read state from `info`, not `obs`, matching the
existing convention RMPC should follow. Compare:
- `GreedyInterceptPolicy.act` -- pursues `info["enemy_measurement"]`
  (falls back to `info["e_hat"]` only if run against a Kalman-track info
  dict), escorts `info["soldier_pos"]` pre-detection.
- `LeadInterceptPolicy.act` -- solves a constant-velocity intercept-time
  quadratic using `target_pos, target_vel` from `_get_target_state(info)`
  (measurement or Kalman track per `state_source`) and `s = self.v_d`
  (read from `config.v_d`, NOT hardcoded -- confirmed during the mobility
  robustness audit).
- `LeadResidualPPOPolicyWrapper` -- composes a Lead direction with a bounded
  PPO residual via `compose_lead_residual` (the single shared math function
  also used by the LR-PPO training environment).

### 4.5 Evaluation integration points
Two evaluation code paths exist in this repository; RMPC must integrate
with the **second** (current/canonical) one, not the first:

1. **Legacy** (`experiments/eval_utils.py`, `experiments/evaluate_for_comparison.py`,
   `experiments/method_specs.py`) -- an earlier, more generic evaluation
   harness predating the expanded-final/expanded-robustness campaign. Still
   present in the repository but **not** the source of any frozen evidence
   used in the current manuscript.
2. **Canonical** (used for every frozen result cited in
   `paper_artifacts/`): `experiments/post_detection_diagnostic_eval_utils.py::run_diagnostic_episode(env, policy, seed, dt, record_trajectory=False)`
   for scripted/PPO/PPO-Kalman policies, and
   `experiments/lead_residual_diagnostic_eval_utils.py::run_lr_ppo_diagnostic_episode`
   for LR-PPO (additionally records per-step residual angles).
   `experiments/run_lead_residual_diagnostic.py::MethodInstance` (fields:
   `name`, `family`, `display_name`, `training_seed`) and `build_instances()`
   define the current frozen 12-instance method table, reused verbatim by
   `experiments/expanded_robustness_protocol.py::build_method_instances()`
   and by every `run_*_robustness_sweep.py` script's
   `build_policy_for_instance(instance, config)` factory function.

**RMPC's future evaluation integration point (Prompt 2+) is a 13th
`MethodInstance(name="rmpc", family="rmpc", display_name="RMPC",
training_seed=None)`** plus a new `elif instance.family == "rmpc":` branch in
each runner's `build_policy_for_instance`. No changes to `run_diagnostic_episode`
itself should be required, since RMPC implements the same `act(obs, info)` /
`reset()` interface as every other policy.

### 4.6 Implementation hazards found during audit
1. **`_advance_velocity` is not currently a pure function.** It is a bound
   method reading `self.config.dt`/`self.config.eps`. RMPC's predictive
   rollout (both for its own defender-dynamics prediction and for each
   hostile scenario) needs the *exact same* update law without constructing
   or mutating a `SoldierEnv` instance. **Recommendation (NOT performed in
   this prompt):** factor a pure, dependency-free function -- e.g.
   `uav_defend/dynamics/constrained_point_mass.py::advance_velocity(current_velocity, desired_velocity, max_speed, max_accel, max_turn_rate_rad, max_climb_rate, max_descent_rate, dt, eps)`
   with the identical signature/semantics already documented in
   `SoldierEnv._advance_velocity`'s docstring, with `SoldierEnv._advance_velocity`
   becoming a thin wrapper that delegates to it. This mirrors the project's
   existing "single shared math function" pattern (`compose_lead_residual`).
   Prompt 2 must perform this refactor (with regression tests proving
   bit-identical output to the current in-place method) before implementing
   RMPC's predictor, rather than duplicating a second, potentially
   inconsistent copy of the dynamics equations.
2. **`enemy_evasion_gain`/`enemy_evasion_radius` are EnvConfig constants,
   but they parameterize the hostile's *decision policy*, not a *vehicle
   limit or safety radius*.** The task's information contract explicitly
   allows "vehicle limits and safety radii" as public knowledge. Evasion
   gain/radius are neither -- they describe how reactively the hostile
   *chooses* to behave, which is analogous to a hidden policy parameter.
   **CORRECTION (post-design-review): RMPC's scenario generator MUST NOT
   use `enemy_evasion_gain` or `enemy_evasion_radius` at all, including
   their documented admissible range or realized value.** These parameters
   define the simulator hostile POLICY, not airframe physics, and are
   excluded entirely from the scenario model (see Section 8.2's revised
   MAX_AWAY scenario, which uses only physical limits and legitimate
   measured state -- never the simulator's evasion formula or parameters).
3. **No existing RMPC-adjacent code exists yet** (`uav_defend/policies/mpc/`,
   CEM optimizer, or dynamics-prediction module all absent) -- this is a
   from-scratch design, not a modification of an existing partial
   implementation.
4. Two different reward-shaping constants already exist in `EnvConfig`
   (`reward_intercept=100`, `reward_soldier_caught=-100`,
   `reward_unsafe_intercept=-150`, `reward_progress_scale=5.0`,
   `reward_time_penalty=-0.05`) that are candidates for RMPC's cost function
   weights (Section 10) -- reusing them avoids introducing brand-new tuned
   constants.

## 5. Pre-detection semantics (inherited, never duplicated)

Per `docs/revised_predetection_protocol.md` and direct inspection of
`SoldierEnv.step()`, with `defender_standby_until_detection=True` (canonical
default, unchanged for RMPC):

1. Defender starts co-located with the soldier, zero velocity.
2. Before detection, the defender is held exactly co-located with the
   soldier's *current* position, zero velocity, every step -- regardless of
   what action RMPC (or any other policy) supplies.
3. The externally supplied action has no effect during standby; it is
   silently discarded by `SoldierEnv.step()`, not by the policy.
4. On the detection-transition step (`just_detected=True`), the supplied
   action is **still discarded** -- the defender remains co-located at the
   moment detection first becomes true.
5. RMPC's first *executed* action occurs on the **next** `step()` call,
   starting from `defender_vel=[0,0,0]` at the detection-time position,
   subject to the unchanged defender dynamics limits.

**RMPC implements NONE of this.** No RMPC-specific search, escort, launch,
or pre-positioning logic will be written. `policy.act()` may be called by
the evaluation harness even during standby (matching how every other policy
is called every step), but its return value is guaranteed to be discarded
by the environment until the first post-detection step -- RMPC's CEM
planner is free to run during standby calls (e.g., returning a degenerate
escort/zero direction, mirroring Lead's own pre-detection behavior of
pursuing `soldier_pos`) without this affecting the canonical protocol in
any way.

## 6. Mathematical formulation (refined after audit)

At each post-detection decision step `t`:

$$
U_t^\* = \arg\max_{U \in \mathcal{U}} \; \min_{j \in S} J_j(U)
\qquad
U = (u_0, u_1, \dots, u_{H-1}), \quad u_k \in \mathbb{R}^3
$$

Only the first element is executed, matching standard receding-horizon MPC:

$$
u_t = u_0^\*
$$

and the problem is re-solved from scratch (with optional warm-start, see
Section 11) at step `t+1`.

**Refinement required by the audit:** each `u_k` is a *direction*, not a
throttle-scaled command -- consistent with the environment's own action
semantics (Section 4.2). CEM therefore samples unconstrained `u_k \in
\mathbb{R}^3` and each is *normalized* (with a zero-vector eps-guarded
fallback, mirroring `compose_lead_residual`'s and `LeadInterceptPolicy._pursue`'s
existing degenerate-safety convention) before being fed to the predictive
dynamics rollout -- exactly how `SoldierEnv` itself treats the action.

## 7. Defender prediction model

Predictive defender state is propagated using the (to-be-factored, Section
4.6 item 1) pure `advance_velocity` helper, called once per predicted step
per candidate `U`, using:

- `current_velocity` = legitimate `info["defender_vel"]` (own true state --
  always legitimate, per `sanitize.py`'s `_COMMON_KEYS`).
- `current_position` = legitimate `info["defender_pos"]`.
- `max_speed = config.v_d`, `max_accel = config.defender_max_accel`,
  `max_turn_rate_rad = radians(config.defender_max_turn_rate_deg)`,
  `max_climb_rate = config.defender_max_climb_rate`,
  `max_descent_rate = config.defender_max_descent_rate`, `dt = config.dt`.

No `SoldierEnv` instance is constructed or mutated during planning -- the
predictor operates entirely on plain NumPy arrays and `EnvConfig` scalar
fields (all legitimately public per Section 8's model-knowledge argument).

## 8. Hostile scenario uncertainty model

### 8.1 Why physical limits are legitimate but hidden state/noise is not

Vehicle physical limits (`v_e`, `enemy_max_accel`, `enemy_max_turn_rate_deg`,
`enemy_max_climb_rate`, `enemy_max_descent_rate`) are **design-time,
publicly documented constants** of the simulated hostile airframe -- exactly
analogous to Lead's own use of `config.v_d` as its assumed interceptor
speed. Knowing that *a* physical vehicle cannot exceed a given
speed/acceleration/turn-rate is categorically different from knowing the
*realized* stochastic trajectory that vehicle will actually fly: the latter
requires knowledge of hidden random-number-generator state (`weave` AR(1)
bias, heading-noise draws) and of a hidden behavioral decision (whether/how
strongly it reactively evades), none of which any deployed sensor could
ever provide. RMPC is therefore designed to bound its uncertainty using only
the *admissible envelope* implied by public physical/config limits, never
the specific realized disturbance.

`enemy_evasion_gain` and `enemy_evasion_radius` describe the simulator
hostile's decision POLICY, not airframe physics, and are therefore EXCLUDED
ENTIRELY from RMPC's scenario generator (Section 8.2's MAX_AWAY scenario
uses only physical limits and legitimate measured state, never these
parameters or the simulator's `_compute_enemy_evasion` formula).

### 8.2 Proposed scenario set (six scenarios, smallest defensible set)

Given current legitimate state `(x_hat, v_hat)` = `(info["enemy_measurement"],
info["enemy_measurement_velocity"]` if `info["enemy_measurement_velocity_valid"]`,
else `None`), each scenario `j` defines a *desired-direction sequence* for
the hostile over the horizon, which is then propagated through the SAME
`advance_velocity` helper using the hostile's own physical limits (never
the defender's):

| # | Scenario | Desired direction rule |
|---|---|---|
| S1 | Nominal continuation | Hold `v_hat`'s direction constant each step (ballistic continuation); if `v_hat` unavailable, hold direction toward the soldier (pursuit-only fallback). |
| S2 | Bounded left turn | Rotate the S1 direction by the hostile's own max turn authority toward a fixed left-lateral offset each step. |
| S3 | Bounded right turn | Mirror of S2, rotated right. |
| S4 | Bounded climb | S1 direction with vertical component driven to `+enemy_max_climb_rate`. |
| S5 | Bounded descent | S1 direction with vertical component driven to `-enemy_max_descent_rate`. |
| S6 | MAX_AWAY (generic worst-direction bounded maneuver) | At each predicted step, command a desired direction AWAY from the *candidate defender's predicted position* (a purely geometric, physical-limits-only construction: e.g. `desired_dir = normalize(hostile_pred_k - defender_pred_k)`), then propagate through the SAME `advance_velocity` helper using ONLY the hostile's public physical limits. This scenario is the only one whose hostile motion depends on the candidate defender rollout being evaluated (closed-loop worst-case direction), all others are open-loop. It MUST NOT use `enemy_evasion_gain`, `enemy_evasion_radius`, or the simulator's `_compute_enemy_evasion` formula/weave/heading-noise state -- it is a generic bounded-maneuver-uncertainty construct, not a reproduction of the simulator's hostile policy. |

This is explicitly a **bounded scenario uncertainty model**, not a
prediction of the true simulator hostile policy: it excludes weave AR(1)
bias, heading noise, and the `enemy_evasion_gain`/`enemy_evasion_radius`
parameters and formula ENTIRELY (not merely their realized values), and the
paper must state this distinction (Section 18).

### 8.3 Feasibility at dt=0.5s

Six scenarios x a horizon of H<=8 steps x a CEM population of up to 256 is
`<=` 256 x 6 x 8 = 12,288 pure-NumPy dynamics-update calls per CEM
iteration, x up to 8 iterations = <=98,304 calls per decision -- expected to
be computationally practical (no environment construction, no I/O, no
neural-network inference) but MUST be empirically confirmed against a
runtime budget in Prompt 2/3 (Section 12), not assumed here.

## 9. Protected-asset (soldier) prediction assumption

RMPC may read the CURRENT `info["soldier_pos"]` (legitimate -- the asset
being defended is directly, continuously observable) but has no legitimate
access to the soldier's future stochastic random-walk realization.

**Recommendation: hold the soldier's position FIXED at its current value
for the entire prediction horizon `H`.** Justification: (a) no future
information is invented; (b) the soldier's speed (`v_s=1.5 m/s`) is slow
relative to the short horizon under consideration (worst case ~
`H*dt*v_s <= 8*0.5*1.5 = 6m` of unmodeled displacement for the largest
candidate `H`), a bounded and stated approximation error, not a silent one;
(c) it keeps the cost function simple (Section 10) since `soldier_pos` only
enters the safety-radius terms (`threat_radius`, `unsafe_intercept_radius`),
for which a fixed conservative reference point is a defensible
simplification. This limitation is stated explicitly in Section 20 and must
appear in the manuscript.

## 10. Robust objective

### 10.1 Per-scenario score `J_j(U)`

Reusing the environment's OWN existing, already-fixed reward magnitudes
(`EnvConfig.reward_intercept=100`, `reward_soldier_caught=-100`,
`reward_unsafe_intercept=-150`, `reward_progress_scale=5.0`) as RMPC's cost
weights -- introducing **zero new tunable weight constants** beyond what the
environment itself already defines (directly satisfying "prefer the fewest
tunable weights necessary").

For each predicted step `k = 1..H` (mirroring the environment's own
per-step terminal-check *order*: `soldier_caught` checked first, then
`unsafe_intercept`, then `intercepted`, exactly as in `SoldierEnv.step()`):

1. `dist_es_k = ||hostile_scenario_k - soldier_pos_fixed||`. If
   `dist_es_k <= threat_radius`: **terminal FAIL** ->
   `J_j(U) = reward_soldier_caught` (`-100`); stop evaluating this
   scenario/candidate pair.
2. `dist_de_k = ||defender_pred_k - hostile_scenario_k||`. If
   `dist_de_k <= intercept_radius`:
   - if `dist_es_k <= unsafe_intercept_radius`: **terminal FAIL (unsafe)**
     -> `J_j(U) = reward_unsafe_intercept` (`-150`); stop.
   - else: **terminal SUCCESS** -> `J_j(U) = reward_intercept` (`100`); stop.
3. If neither triggers by `k=H` (no terminal event within the horizon):
   `J_j(U) = reward_progress_scale * (dist_de_0 - dist_de_H) - dist_de_H`
   -- rewards net closing progress over the horizon and penalizes the
   remaining terminal distance-to-intercept (directly answers "is a terminal
   distance-to-intercept term needed?" -- yes, exactly this term).

### 10.2 Robust aggregation

$$
J_{robust}(U) = \min_{j \in S} J_j(U)
$$

CEM maximizes `J_robust(U)` over the population (Section 11).

### 10.3 Explicit answers to Section 10's required questions
- **Predicted safe interception recognized by:** `dist_de_k <=
  intercept_radius` AND `dist_es_k > unsafe_intercept_radius` at the same
  predicted step (mirrors `SoldierEnv`'s own `intercepted` condition
  exactly).
- **Predicted soldier breach recognized by:** `dist_es_k <= threat_radius`
  (mirrors `soldier_caught`, checked with the SAME priority order as the
  real environment -- soldier-caught dominates a simultaneous intercept
  event, exactly as in `SoldierEnv.step()`).
- **Predicted unsafe intercept recognized by:** `dist_de_k <=
  intercept_radius` AND `dist_es_k <= unsafe_intercept_radius` (and not
  already soldier-caught).
- **No terminal event within horizon:** fallback progress/distance term
  (Section 10.1 item 3) -- never a hard zero or undefined score.
- **Terminal distance-to-intercept term needed:** yes, included (Section
  10.1 item 3).
- **Control-smoothness penalty:** NOT included initially. Justification:
  `advance_velocity` already physically rate-limits acceleration/turn/
  vertical-rate/speed, so the predicted rollout cannot itself be
  non-smooth beyond what the real dynamics permit; an additional penalty
  would be a new tunable weight without a demonstrated need. Defer unless
  Prompt-2/3 development-seed testing reveals pathological chattering, in
  which case any addition must be documented as a correctness fix, not a
  performance-driven tune (mirroring the staged-freeze discipline in
  Section 15).
- **Weights requiring tuning:** none beyond the four already-fixed
  `EnvConfig` reward constants reused verbatim (Section 10.1) -- zero new
  weights.

## 11. CEM optimizer design

- **Optimization variable:** `U in R^{H x 3}` (unconstrained real vectors;
  each row normalized to a unit direction, with an eps-guarded zero-vector
  fallback -- see Section 6).
- **Sampling:** at each CEM iteration, draw `N` candidates from a
  per-timestep diagonal Gaussian `U ~ N(mu, sigma^2 I)` over `R^{H x 3}`
  using the DEDICATED RMPC `Generator` (Section 13); normalize each row
  before rollout.
- **Population size candidates (predeclared, Section 15):** `{64, 128, 256}`.
- **Elite fraction candidates:** `{0.1, 0.2}`.
- **CEM iteration count candidates:** `{3, 5, 8}`.
- **Horizon `H` candidates:** `{4, 6, 8}` steps (2.0s / 3.0s / 4.0s of
  lookahead at `dt=0.5s`).
- **Warm start:** optional; if used, `mu` for step `t+1` is initialized by
  shifting the elite mean from step `t` left by one timestep and appending
  a copy of the last planned direction (standard MPC warm-start). Must be
  disabled by `policy.reset()` (new episode = no carried-over warm-start
  state) and must not break the determinism guarantee (Section 13).
- **Numerical safeguards:** clip raw sampled vectors to a bounded magnitude
  before normalization; guard `||u_k|| < eps` with a zero-direction
  fallback (mirrors `compose_lead_residual`); reject/ignore any candidate
  producing a non-finite `J_robust`; assert at least one finite-scored
  candidate remains in the elite set each iteration.
- **Fallback behavior if optimization fails** (no legitimate hostile
  target yet, i.e. `enemy_measurement is None`; all candidates non-finite;
  or `enemy_detected=False`, i.e. still in standby where the action is
  discarded anyway): fall back to a pure-pursuit direction toward the best
  available legitimate target (`enemy_measurement` if present, else
  `soldier_pos` escort, else the zero vector) -- mirroring
  `LeadInterceptPolicy._pursue`'s existing degenerate-safety convention
  exactly. RMPC must never raise/crash during an evaluation episode.

## 12. Computational fairness / runtime measurement plan (not benchmarked now)

At minimum, later prompts must record: mean, median, P95, and maximum
per-decision wall-clock time; hardware/platform metadata (CPU model, core
count, Python/NumPy version); and the number of objective (rollout)
evaluations per decision (`population_size * CEM_iterations * |S|`).
Manuscript wording must keep "interception performance" and "online
computational cost" as clearly separate reported quantities, and must not
claim real-time feasibility unless measured runtime supports it.

## 13. Deterministic RNG architecture

RMPC owns a single dedicated NumPy `Generator`:

```python
self._rng = np.random.Generator(np.random.PCG64(np.random.SeedSequence(controller_seed)))
```

created in `__init__` (or re-created in `reset()`, TBD in Prompt 2 --
whichever choice is made, it must be documented and tested) and used
EXCLUSIVELY for CEM's Gaussian sampling. It never reads from, seeds, or is
seeded by any `SoldierEnv`-owned RNG stream (the environment's own dynamics
RNG, soldier random-walk RNG, hostile motion/weave RNG, or sensor-noise
RNG are all separate, confirmed-independent streams within `SoldierEnv` by
source inspection). Given the same environment seed, the same
`controller_seed`, and the same `EnvConfig`, repeated evaluation must be
bit-for-bit deterministic -- this is a required, directly testable property
(Section 16.C).

## 14. Future seed protocol -- DESIGN ONLY, NOTHING OPENED

| Range | Purpose | n |
|---|---|---|
| 71000-71099 | RMPC development | 100 |
| 71100-71199 | RMPC validation | 100 |
| 71200-71399 | RMPC diagnostics | 200 |
| 71400-71999 | buffer / unused | 600 |
| 72000-76999 | NEW final nominal comparison (all 7 methods) | 5000 |
| 77000-77999 | NEW targeted robustness (per selected condition) | 1000/condition |
| 78000-99999 | future reserve | -- |

The eventual final comparison (72000-76999) must evaluate Greedy, PN, Lead,
PPO, PPO-Kalman, LR-PPO, **and** RMPC on the SAME new seeds. Existing PPO/
LR-PPO models remain frozen; no retraining. The five planned targeted
robustness conditions (77000-77999, run later, one per condition, not now):
1. strong reactive evasion (`enemy_evasion_gain=1.00`)
2. restrictive defender dynamics (`defender_max_accel=4`,
   `defender_max_turn_rate_deg=45`)
3. high measurement noise (`measurement_var=4.00`)
4. hard mobility (`v_d=18`, `v_e=8`)
5. short detection (`detection_radius=5`)

**No seed in any of these ranges is opened by this prompt.**

## 15. Hyperparameter-selection plan (staged, to avoid tuning on final results)

- **Primary selection metric:** safe-interception success rate (single
  metric, no post-hoc multi-metric blending).
- **Secondary/feasibility constraint:** runtime (a configuration that is
  computationally infeasible is eliminated regardless of success rate, but
  runtime is never used to break ties among feasible configurations beyond
  the primary metric).
- **Stage A (development, 71000-71099):** eliminate clearly unstable or
  computationally infeasible `(H, population, iterations, scenario-set)`
  combinations from the predeclared candidate grid (Section 11). No
  performance-based selection yet -- feasibility screening only.
- **Stage B (validation, 71100-71199):** among the surviving candidates,
  select the SINGLE final configuration using only the primary metric
  (safe-interception success rate).
- **Stage C (diagnostics, 71200-71399):** audit the frozen configuration's
  behavior only (residual-style sanity checks, degenerate-case handling,
  runtime distribution). No further performance-driven tuning permitted
  here unless a genuine correctness defect (not a performance shortfall) is
  discovered, exactly mirroring this project's established freeze
  discipline for every other method in this campaign.
- After Stage B, the configuration is FROZEN before any seed >= 72000 is
  opened.

## 16. Required test plan for Prompt 2 (design only)

**A. Information contract**
- `test_rmpc_never_reads_ground_truth_keys` -- construct/py-inspect the
  policy's `act()` source or monkeypatch `info` to a dict missing
  `enemy_pos`/`enemy_vel`, assert no `KeyError`/no dependence on those keys.
- `test_rmpc_uses_only_sanitized_info` -- run through `build_policy_info`
  exactly like every other measurement-track policy.

**B. Action validity**
- `test_action_shape_is_3` , `test_action_finite`,
  `test_action_within_bounds`, `test_zero_action_fallback_when_no_target`.

**C. Deterministic behavior**
- `test_same_controller_seed_same_state_same_action` -- fixed
  `(obs, info)` + fixed `controller_seed` -> bit-identical action across two
  independent policy instances.

**D. Dynamics prediction**
- `test_pure_predictor_matches_advance_velocity_one_step`,
  `test_pure_predictor_matches_advance_velocity_multi_step` -- against the
  (to-be-factored) pure `advance_velocity` helper, for controlled
  deterministic scenarios (e.g. fixed initial velocity, fixed desired
  direction).

**E. Constraint satisfaction**
- `test_predicted_defender_speed_within_v_d`,
  `test_predicted_defender_accel_within_limit`,
  `test_predicted_defender_turn_within_limit`,
  `test_predicted_defender_climb_descent_within_limit`.

**F. Scenario validity**
- `test_all_six_scenarios_respect_hostile_physical_limits` (speed, accel,
  turn, climb, descent) for a representative set of initial conditions.

**G. Zero/first-measurement edge cases**
- `test_no_velocity_estimate_on_first_detection_step`,
  `test_no_detection_yet_returns_safe_fallback`,
  `test_degenerate_zero_distance_geometry`,
  `test_no_feasible_candidate_falls_back_gracefully`.

**H. Pre-detection behavior**
- `test_environment_ignores_rmpc_action_before_detection` (reuses the
  existing environment-level guarantee -- this test asserts RMPC does not
  bypass it, not that RMPC re-implements it),
  `test_detection_transition_action_not_executed`.

**I. No side effects**
- `test_planning_does_not_mutate_env_state`,
  `test_planning_does_not_consume_env_rng_state` (assert environment RNG
  stream state is bit-identical before/after a `policy.act()` call).

**J. Evaluator compatibility**
- `test_policy_reset_semantics`, `test_matched_seed_evaluation_compatible_with_run_diagnostic_episode`.

**K. Runtime diagnostics**
- `test_timer_instrumentation_does_not_alter_action` -- action returned is
  identical whether or not timing instrumentation is enabled.

## 17. Planned implementation files (Prompt 2, NOT created now)

- `uav_defend/dynamics/constrained_point_mass.py` (pure `advance_velocity`,
  factored out of `SoldierEnv._advance_velocity`, with a regression test
  proving bit-identical behavior).
- `uav_defend/policies/mpc/rmpc_policy.py` (the `RMPCPolicy` class:
  `act(obs, info)`, `reset()`, dedicated RNG per Section 13).
- `uav_defend/policies/mpc/cem_optimizer.py` (generic CEM routine, decoupled
  from RMPC-specific cost logic where practical).
- `uav_defend/policies/mpc/hostile_scenarios.py` (the six-scenario model,
  Section 8.2).
- `uav_defend/policies/mpc/rmpc_cost.py` (Section 10's per-scenario score
  and robust aggregation).
- `test_rmpc_*.py` (Section 16).
- A new `MethodInstance(name="rmpc", family="rmpc", display_name="RMPC",
  training_seed=None)` entry point (Section 4.5), NOT added to
  `build_instances()` until Prompt 2+ actually implements the policy.

## 18. Publication-safe wording

**High-level description (preferred):**

> "We additionally compare against a scenario-based robust model predictive
> controller that replans over a finite horizon using the same noisy target
> state available to the other measurement-based controllers. The
> controller optimizes defender commands against a finite set of bounded
> target-motion scenarios and does not observe the true hostile state or
> future stochastic disturbances."

**Mandatory disclaimer:**

> "The RMPC baseline is a finite-horizon, sampling-based robust optimization
> method and is not claimed to compute the globally optimal pursuit-evasion
> solution."

**Additional required qualifications (from this design):**
- RMPC's scenario set models *bounded physical maneuver uncertainty*, not
  the true simulator hostile policy (excludes weave/heading-noise RNG and
  the realized evasion gain).
- The soldier/protected-asset is held at its current position over the
  short planning horizon; this is a stated simplification, not a claim
  about the true random walk.
- RMPC's computational cost is expected to substantially exceed Lead/PPO/
  LR-PPO; interception performance and online computational cost are always
  reported and discussed separately.

## 19. Explicit limitations (for the manuscript's limitations section)

1. Scenario set is a hand-designed, finite bound on hostile maneuvering --
   not a learned or exhaustive uncertainty model.
2. Soldier future motion is approximated as static over the short horizon.
3. CEM is a sampling-based, not gradient-based or exact, optimizer; no
   convergence guarantee is claimed.
4. `theta_max=30 deg` (LR-PPO) and RMPC's hyperparameters are each frozen
   through a staged, non-performance-final-result-driven process, but
   neither is claimed to be a globally optimal choice.
5. Reported robustness results (once run, in later prompts) will use a NEW
   seed range (72000+), never re-using or comparing against the already-
   frozen 61000-70999 range's exact per-seed outcomes for methods other
   than the newly-added RMPC baseline evaluated under identical seeds.
