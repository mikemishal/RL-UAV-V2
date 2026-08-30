# Lead-Residual PPO: Design Note (Minimal First Implementation)

Status: **implementation + unit/integration testing only — NOT trained, NOT
evaluated on any locked/reserved seed range.**

## Motivation

Existing standalone PPO learns the full interception policy from scratch:

    u_t = pi_theta(o_t)

This couples all of PPO's learning capacity to re-deriving intercept
geometry that a classical analytical guidance law (Lead) already solves
well. Lead-Residual PPO instead keeps the existing analytical
`LeadInterceptPolicy` as the nominal guidance law and asks PPO to learn only
a small, bounded *directional correction* around it:

    r_t = pi_theta(o_t^R)
    u_t = compose(u_L(t), r_t)

This is the **minimal** residual architecture. Deliberately deferred to
later ablations: prediction-error conditioning (e.g. intercept time tau),
residual gating, LSTM/GRU recurrence, Kalman-track variant, curriculum
learning, and any auxiliary reward terms.

## Existing PPO vs. Lead vs. Lead-Residual PPO

- Existing PPO: `u = pi(o)` — full 16-D Direct observation, unconstrained
  unit-direction action.
- Lead: `u_L` — analytical constant-velocity predictive-intercept direction
  (`uav_defend/policies/baseline/lead_intercept_policy.py`, reused verbatim,
  never reimplemented).
- Lead-Residual PPO: `r = pi([o, u_L])`, then `u = compose(u_L, r)`.

## Residual observation (19-D)

    residual_obs = concat(direct_obs, lead_direction)

- `direct_obs`: the environment's existing 16-D Direct observation
  (dimension verified programmatically from `SoldierEnv.observation_space`,
  never hardcoded where avoidable — see
  `uav_defend/policies/residual/lead_residual_observation.py::residual_observation_dim`).
- `lead_direction`: the 3-D unit direction produced by
  `LeadInterceptPolicy(state_source="measurement")` for the SAME state,
  using only legitimate measurement-mode fields (`enemy_measurement`,
  `enemy_measurement_velocity[_valid]`, `defender_pos`, `soldier_pos`,
  `enemy_detected`) — never `info["enemy_pos"]`/`info["enemy_vel"]`.
- Total dimension: 16 + 3 = **19**.

Explicitly excluded from this first implementation: true hostile
position/velocity, future state, terminal information, tracking error,
intercept time tau, prediction error, and any reward component unavailable
to a deployed controller.

## Residual composition

Given the canonical unit Lead direction `u_L` and raw PPO action `r_raw` in
`R^3` (Box(-1, 1, shape=(3,))):

    r = r_raw / ||r_raw||   if ||r_raw|| > 1  else  r_raw          (norm-clip)

    r_perp = r - (r . u_L) u_L                                     (drop the
                                                                      component
                                                                      parallel
                                                                      to Lead)

    theta_max = radians(residual_max_angle_deg)

    u_raw = u_L + tan(theta_max) * r_perp

    u_hybrid = normalize(u_raw)

Implemented once, in exactly one place:
`uav_defend/policies/residual/lead_residual_composition.py::compose_lead_residual`.
Both the training environment
(`experiments/lead_residual_training_env.py`) and the evaluation policy
wrapper (`uav_defend/policies/residual/lead_residual_ppo_policy_wrapper.py`)
call this SAME function — there is no duplicated Lead+residual mathematics
anywhere in the repository.

### Why a directional-only residual (not a magnitude correction)

The canonical environment normalizes any nonzero controller action and
converts it to a desired velocity at the defender's fixed nominal speed
`v_d` (`SoldierEnv._advance_velocity`). Action *magnitude* therefore carries
no meaningful throttle semantics — the environment would discard any such
signal. Projecting out the residual's component parallel to `u_L` forces
the learned signal to be purely angular: "how should I rotate the
analytical Lead command?", not "how hard should I push?"

### Angular bound property

Since `r_perp` is constructed orthogonal to `u_L` (by the projection above)
and `||r_perp|| <= ||r|| <= 1` (projection cannot increase norm), `u_raw`
has a component parallel to `u_L` of magnitude exactly 1 and an orthogonal
component of magnitude at most `tan(theta_max)`. Therefore:

    tan(angle(u_raw, u_L)) <= tan(theta_max)   =>   angle(u_hybrid, u_L) <= theta_max

(normalizing `u_raw` to obtain `u_hybrid` does not change its direction, so
the same bound holds for `u_hybrid`).

### Zero-residual invariance

`r = [0, 0, 0]` => `r_perp = 0` => `u_raw = u_L` => `u_hybrid == u_L` (up to
floating-point tolerance). This is a hard scientific requirement: it lets
Lead-Residual PPO contain standalone Lead as its zero-residual special
case, and is covered by dedicated unit tests (action-level and, separately,
full post-detection episode-level).

### Parallel-residual invariance

If `r` is (positively or negatively) parallel to `u_L`, `r . u_L = ±||r||`
and `r_perp = r - (r . u_L) u_L = 0` exactly (both signs), so
`u_hybrid == u_L` again.

## `residual_max_angle_deg`

Exposed as an explicit configuration value
(`uav_defend.policies.residual.lead_residual_composition.DEFAULT_RESIDUAL_MAX_ANGLE_DEG
= 30.0`), never hidden inside a magic constant.

**This is a PROVISIONAL development default, NOT a publication-locked
hyperparameter.** It may only be changed later using NEW
development/validation seeds; it must NEVER be selected using the
already-opened final-nominal seeds (41000..45999).

## No-ground-truth information contract

- `LeadInterceptPolicy(state_source="measurement")` (the canonical,
  already-existing implementation — never reimplemented) is the sole source
  of the Lead component; it reads only `defender_pos`, `soldier_pos`,
  `enemy_detected`, `enemy_measurement`, `enemy_measurement_velocity[_valid]`.
- The residual observation adds nothing beyond the existing Direct
  observation and this Lead direction.
- Neither the training environment nor the evaluation wrapper ever reads
  `info["enemy_pos"]` or `info["enemy_vel"]`.
- All info passed to policy code goes through
  `uav_defend.policies.sanitize.build_policy_info(info, "measurement")`,
  the repository's single source of truth for legitimate policy inputs.

## Relationship to canonical constrained dynamics

The composed `u_hybrid` is passed unchanged into the canonical
`SoldierEnv.step()`; `_advance_velocity()` (turn-rate/acceleration/speed/
climb/descent limits) is never bypassed or duplicated. The residual
controller only ever supplies a *direction* — physical realizability is
determined entirely by the existing environment dynamics, exactly as for
Greedy/PN/Lead/PPO/PPO-Kalman.

## First-detection handoff

Unchanged from the corrected protocol: before detection the defender
remains exactly co-located with the soldier and at rest, controller actions
are ignored; on the detection transition the defender remains co-located
and at rest and the controller action is *not* executed; control begins on
the *following* step. This timing is delegated entirely to the existing,
already-tested `PostDetectionTrainingEnv` — not reimplemented here. At the
first controlled state, `LeadInterceptPolicy` may legitimately use its
existing `first_measurement_fallback` (target velocity not yet observable)
— this is Lead's own pre-existing behavior, not something invented for this
task.

## Explicitly deferred (later ablations)

This is the **minimal** residual architecture. Intentionally NOT included
in this first implementation: prediction-error conditioning (e.g. an
intercept-time-to-go feature), learned residual gating, LSTM/GRU/recurrent
networks, a Kalman-track residual variant, new/auxiliary reward terms
(residual-norm penalty, Lead-imitation reward, angle penalty, tracking
reward), and curriculum learning. These may be explored as separate,
explicitly-labeled ablations/extensions once this minimal baseline has been
reviewed.
