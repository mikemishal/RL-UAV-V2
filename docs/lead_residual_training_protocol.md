# Lead-Residual PPO (LR-PPO) Training Protocol

This document records the FROZEN protocol for the Lead-Residual PPO training
campaign (see `docs/lead_residual_ppo_design.md` for the architecture design
note). **No full 400k training run has been executed under this protocol.**
This is the implementation-hardening, protocol-freeze, and smoke-test phase
only (see final report in the corresponding task conversation for the exact
smoke-training results).

## Method

- **Method name**: Lead-Residual PPO
- **Abbreviation**: LR-PPO
- **Implementation commit** (architecture/tests, pre-hardening): `3f0d885be7dbca75644fcfe69838111f75a9dc1d`
- **Canonical Lead source**: `uav_defend/policies/baseline/lead_intercept_policy.py::LeadInterceptPolicy(state_source="measurement")` -- reused verbatim, never reimplemented.

## Observation

    o_R = concat(o_direct, u_Lead)

- `o_direct`: existing 16-D Direct PPO observation (unchanged).
- `u_Lead`: canonical Lead controller's 3-D unit direction for the same state.
- **Observation dimension: 19** (verified programmatically, see
  `uav_defend/policies/residual/lead_residual_observation.py::residual_observation_dim`).

## Action

- `r_raw in Box(-1, 1)^3` -- interpreted ONLY through the canonical composition
  function `compose_lead_residual()` (norm-clip, perpendicular projection,
  angular scaling by `tan(theta_max)`, final normalization). No direct
  residual action may bypass this function.
- **Action dimension: 3**

## Residual composition equation

    r = r_raw / ||r_raw||    if ||r_raw|| > 1  else  r_raw
    r_perp = r - (r . u_Lead) u_Lead
    u_hybrid = normalize(u_Lead + tan(theta_max) * r_perp)

Property: `angle(u_hybrid, u_Lead) <= theta_max` (see
`uav_defend/policies/residual/lead_residual_composition.py` for the proof and
`test_lead_residual_ppo.py` for the empirical verification over thousands of
random combinations).

**`residual_max_angle_deg = 30.0`** -- FROZEN for this first minimal
experiment. NOT tuned using old final seeds, new validation seeds, new
diagnostic seeds, or new final seeds. Any future angle search must be
declared a new, separate development experiment.

Angle-domain validation: `compose_lead_residual` requires
`0 <= residual_max_angle_deg < 90`; rejects negative values, values `>= 90`,
NaN, and Inf (the tangent-based construction is undefined/unbounded at 90
degrees). At `theta_max = 0`, `u_hybrid == u_Lead` for every residual action.

## PPO architecture (confirmed programmatically, no training)

- Algorithm: Stable-Baselines3 PPO
- Policy: `MlpPolicy` (no `policy_kwargs` override -- SB3 default architecture)
- Actor (`pi`): `[64, 64]`, Tanh
- Critic (`vf`): `[64, 64]`, Tanh
- Observation dim: 19, Action dim: 3
- **Exact trainable parameter count: 11,143** (vs. standalone PPO's 10,759;
  the difference is exactly 2 networks x 3 extra input features x 64
  first-layer units = 384 extra parameters -- the architecture itself is
  unchanged; see `test_lead_residual_ppo_architecture.py`).

### PPO hyperparameters (identical family/values to standalone PPO)

```json
{
  "learning_rate": 3e-4, "gamma": 0.99, "batch_size": 64, "n_steps": 2048,
  "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
  "max_grad_norm": 0.5, "gae_lambda": 0.95
}
```

## Reward

Identical canonical environment reward used by standalone PPO. NO
residual-specific shaping (no residual-magnitude penalty, Lead-imitation
reward, Lead-deviation penalty, prediction-error reward, angle penalty,
tracking reward, or extra success bonus).

## Training initial condition

Reuses `experiments/lead_residual_training_env.py::LeadResidualTrainingEnv`,
which itself reuses `experiments/post_detection_training_env.py::PostDetectionTrainingEnv`
and the canonical `SoldierEnv(defender_standby_until_detection=True)`. Every
training episode begins at a NATURALLY GENERATED detection state (no
respawning/repositioning/synthetic detected states); the defender remains
co-located with the soldier and stationary at acquisition; the first
PPO-controlled action occurs on the following step.

## Seed protocol (FROZEN; new namespace, completely separate from the old study)

Defined in `experiments/lead_residual_training_protocol.py`.

| Range | Purpose | Status |
|---|---|---|
| 55, 56, 57 | Training roots (LR-PPO-55/56/57) | to be opened by full training |
| 60000-60099 | In-training validation / checkpoint selection (100 seeds) | reserved, unopened |
| 60100-60299 | Post-training diagnostic holdout (200 seeds) | reserved, unopened |
| 61000-65999 | New expanded final nominal test (5000 seeds) | reserved, sealed |
| >=66000 | Future robustness studies | reserved, open-ended |

The OLD baseline-study seed blocks remain permanently reserved/historical and
are **never reused** here: 40000-40099 (old PPO validation), 40100-40299 (old
diagnostic), 41000-45999 (opened final baseline test), >=46000 (was reserved
for the old study).

### Deterministic training-episode-seed mapper (LR-PPO-specific)

`experiments/lead_residual_training_protocol.py::build_lr_ppo_episode_seed_sequence`
reuses the SAME underlying primitive as the existing corrected-PPO campaign
(`numpy.random.SeedSequence(root_seed).spawn(...)`) but is a **new,
LR-PPO-specific function** -- the existing
`experiments/post_detection_training_protocol.py::build_episode_seed_sequence`
(used by the already-frozen 6-model 400k campaign) is **not modified**.
Every generated training episode seed is deterministically folded into
`[0, 60000)` via modulo, which provably guarantees -- by construction -- that
no training episode seed can ever collide with any current or future LR-PPO
evaluation range (all of which begin at 60000). See
`test_lead_residual_training_protocol.py` for verification that the same
root produces an identical sequence, different roots produce different
sequences, and no generated seed ever falls in a reserved range.

## Validation protocol (frozen; NOT executed in this task)

- Cadence: every 25,000 post-detection training steps.
- Full canonical mission (reset through detection to termination), NOT the
  training wrapper.
- Inference: `deterministic=True`.
- **Checkpoint-selection metric: SAFE INTERCEPTION SUCCESS RATE ONLY.**
  Never intercept time, residual magnitude, Lead agreement, or tracking
  metric.
- Tie-break: exact ties -> earliest training step (enforced by only
  overwriting `best_model.zip` on a STRICT `>` improvement -- see
  `experiments/lead_residual_success_rate_eval_callback.py`).
- Validation seeds (60000-60099) are used ONLY for within-training-run
  checkpoint selection -- never to drop a training root, choose the best of
  55/56/57 for publication, tune the residual angle, or change
  reward/observation/architecture. All three frozen training roots remain
  part of later family-level reporting.

## Training length

Requested: 400,000 PPO-controlled post-detection steps per training root.
Actual steps may exceed this exactly (SB3 rollout granularity); both
`requested_timesteps` and `actual_model_num_timesteps` are recorded in
`run_manifest.json`, never truncated to hit exactly 400,000.

## Three independent training runs (prepared, NOT executed)

LR-PPO-55, LR-PPO-56, LR-PPO-57 -- each from scratch: no initialization from
existing PPO checkpoints, no imitation-learning pretraining from Lead, no
warm-start from another residual seed, no shared optimizer state.

## No-ground-truth contract

Identical to every other policy in this codebase: neither the training
environment nor the evaluation wrapper ever reads `info["enemy_pos"]` or
`info["enemy_vel"]`; all info passed to policy code goes through
`uav_defend.policies.sanitize.build_policy_info(info, "measurement")`.

## Future diagnostic / expanded-final protocol (freeze only)

After all three trained models are frozen by validation, the 60100-60299
diagnostic range may be opened ONCE, comparing at minimum Lead, PPO-45/46/47,
LR-PPO-55/56/57 (preferably the complete paper-facing method set). No
residual model changes may occur based solely on favorable/unfavorable
diagnostic outcomes without explicitly declaring a NEW model-development
cycle. The 61000-65999 expanded-final range becomes the new untouched final
nominal test for the EXPANDED paper (Greedy, PN, Lead, PPO, PPO-Kalman,
Lead-Residual PPO, all rerun on the same 5000 matched seeds) if Lead-Residual
PPO survives development and diagnostic review. The old 41000-45999 data
remains valid historical final evidence for the five-controller baseline
study but is NOT the final test for the newly developed residual controller.

## Output namespace

- Models: `models/lead_residual_post_detection/` (new; does not overwrite
  existing PPO/PPO-Kalman models).
- Results: `results/lead_residual_training/` (new; does not overwrite
  `results/training_post_detection_400k/` or `results/final_nominal_post_detection_5000/`).
