# RMPC Targeted Robustness Publication Evaluation Protocol

Status: **PREDECLARED BEFORE ANY SEED >= 77000 IS OPENED.** This document
and `experiments/run_rmpc_targeted_robustness.py` are committed together,
and this commit's SHA is recorded as `RMPC_ROBUSTNESS_PROTOCOL_COMMIT`,
BEFORE seed 77000 is executed. Nothing below may change after any
robustness seed has been observed (Section 17 "source freeze").

**This prompt produces FINAL RAW ROBUSTNESS EVIDENCE only.** No manuscript
files, `paper_artifacts/`, or publication figures are updated by this task.

## 1. Scientific purpose

Evaluate whether the controller hierarchy observed in the final nominal
comparison (`results/rmpc_final_nominal/`) persists under five difficult,
preselected off-nominal conditions. PPO/PPO-Kalman/LR-PPO checkpoints remain
FROZEN (no retraining, no checkpoint selection); RMPC's configuration
remains FROZEN at its tuned nominal selection (no condition-specific
retuning, no controller redesign).

## 2. Ancestry

- RMPC design: `61483f6c448ff2c3b3dfa3bd4c3fa25f18fecf07`
- RMPC implementation: `80ea7bd`
- RMPC hardening: `cb06b04ed39d17ad64b4243c2fe84407bc95e6e4`
- RMPC tuning protocol: `dc6e29ac755e51e5d545a9ddc5bcb5d609eac864`
- RMPC tuning results (frozen selection): `ed4124c`
- Final-nominal base commit: `ed4124c898e095f88db16906712b9e4d48c6b8c6`
- Final-nominal protocol commit: `d017d32`
- Final-nominal results commit (= robustness base commit): `b0f952fbff4a0596239f178222f398c7600f9526`

## 3. Frozen RMPC configuration (verified, never re-derived)

Loaded ONLY from `results/rmpc_tuning/selected_rmpc_config.json` at
runtime, identical to the final nominal campaign:

| Field | Required value |
|---|---|
| `frozen_for_final_evaluation` | `true` |
| `candidate_id` | `"C1_FAST"` |
| `horizon` | `4` |
| `population_size` | `64` |
| `elite_fraction` | `0.20` |
| `cem_iterations` | `3` |
| `controller_seed` | `0` |
| `cem_initial_std` | `1.0` |
| `cem_min_std` | `0.05` |
| `cem_sample_clip` | `3.0` |

Search hyperparameters remain exactly these values under EVERY robustness
condition. RMPC's dedicated CEM RNG re-seeds from `controller_seed=0` at
the start of every episode, independent of the environment seed and of the
active robustness condition.

**RMPC may automatically use changed PUBLIC vehicle-model parameters**
(e.g. R2's `defender_max_accel`/`defender_max_turn_rate_deg`, R4's `v_e`)
because these are ordinary EnvConfig fields passed into `RMPCPolicy(config=...)`
-- this is normal model-based-controller operation using known vehicle
constraints, NOT retuning. RMPC must never receive `enemy_evasion_gain`,
`enemy_evasion_radius`, or any other hidden-state/attacker-parameter value
as a policy input.

## 4. Frozen learned models (verified, never re-derived)

Reused verbatim from `experiments.lead_residual_expanded_final_protocol`:
`FROZEN_MODELS`, `TRAINING_SEEDS=(45,46,47)`, `LR_TRAINING_ROOTS=(55,56,57)`.
Nine SHA-256 hashes re-verified before opening seed 77000 -- any mismatch
is a hard stop.

## 5. Thirteen method instances (identical set to final nominal)

Greedy, PN, Lead, RMPC, PPO-45/46/47, PPO-Kalman-45/46/47, LR-PPO-55/56/57.
Reused verbatim via `experiments.run_rmpc_final_nominal.build_final_instances()`.

## 6. Information contract (unchanged from final nominal)

Measurement/direct track: Greedy, PN, Lead, RMPC, PPO, LR-PPO. Kalman
track: PPO-Kalman only. RMPC continues to receive only sanitized
measurement-track state through `build_policy_info(...)`, plus public
vehicle/task configuration constants. It MUST NOT observe `enemy_pos`,
`enemy_vel`, realized weave state, realized heading noise,
`enemy_evasion_gain`/`enemy_evasion_radius` as policy inputs, future
disturbances, or future sensor measurements. Changing the environment's
evasion gain for R1 does NOT authorize RMPC to observe that value.

## 7. Common pre-detection protocol (identical to final nominal)

Defender starts co-located with the protected asset; held co-located
(zero velocity, actions ignored) before detection; detection-transition
action ignored; first executed controller action occurs on the following
step from rest. No method-specific acquisition/search/pre-positioning
behavior, under every condition.

## 8. Five predeclared robustness conditions

Each condition changes ONLY the fields below relative to the canonical
nominal `EnvConfig` (Direct: `use_kalman_tracking=False`; Kalman:
`use_kalman_tracking=True`; both with `defender_standby_until_detection=True`).
Every condition builds BOTH a Direct and a Kalman variant (the Kalman
variant used only for PPO-Kalman); config-isolation is verified
programmatically via `assert_only_fields_differ`.

| Key | Display name | Changed field(s) | Values |
|---|---|---|---|
| `strong_evasion` | Strong Reactive Evasion | `enemy_evasion_gain` | `1.00` (nominal `0.75`) |
| `restrictive_dynamics` | Restrictive Defender Maneuverability | `defender_max_accel`, `defender_max_turn_rate_deg` | `4.0`, `45.0` (nominal `8.0`, `90.0`) |
| `high_measurement_noise` | High Measurement Noise | `measurement_var` | `4.0` (nominal `0.5`); `process_var` fixed at nominal `1.0`, not retuned |
| `hard_mobility` | Hard Mobility Condition | `v_d=18.0`, `v_e=8.0` -- actual changed-field set derived programmatically (nominal `v_d` is already `18.0`, so the effective changed field is `v_e` only) | `v_e=8.0` (nominal `12.0`) |
| `short_detection` | Short Detection Range | `detection_radius` | `5.0` (nominal `15.0`) |

Paper-facing terminology: R1 is described as "the strongest tested
reactive-evasion condition" (never an optimal/adversarial attacker); R4 is
described as "a hard tested mobility condition" or "defender/hostile
mobility condition (18 m/s, 8 m/s)" (never "slower hostile is inherently
more difficult").

## 9. Robustness seeds

Seeds `77000`-`77999` (1000 unique seeds), reused identically under EVERY
one of the five conditions. The five conditions are therefore **NOT
statistically independent of one another** (same seed IDs intentionally
reused) and must never be described as independent experiments. No random
resampling; no replacement of failed episodes; seed 78000 is never opened.

## 10. Seed guards

`experiments/run_rmpc_targeted_robustness.py` accepts only
`77000 <= seed <= 77999`; no command-line bypass. Proven by
`tests/test_rmpc_targeted_robustness_protocol.py`: 76999 rejected, 77000
accepted, 77999 accepted, 78000 rejected.

## 11. Config-isolation tests

For every condition, the serialized Direct and Kalman `EnvConfig` are
compared field-by-field against the canonical nominal reference matched by
`use_kalman_tracking`. The changed-key set must equal exactly the
predeclared expectation (R4's is derived programmatically, not
hardcoded); any unexpected differing field raises immediately (hard stop).
Estimator-mode differences (`use_kalman_tracking`) are never counted as a
robustness variable.

## 12. Primary endpoint

Safe-interception success: `outcome == "intercepted"`. For every method
instance x condition: successes, n, success rate, Wilson 95% CI.

## 13. Learned-family statistics (identical methodology to final nominal)

For PPO, PPO-Kalman, LR-PPO: three root-specific rates, family mean, root
SD, root-aware/hierarchical 95% interval (`hierarchical_bootstrap_track`).
Never pooled as 3 x 1000 IID episodes. For Greedy, PN, Lead, RMPC: the
single 1000-seed matched instance with a plain Wilson interval. Reused
verbatim via `experiments.run_rmpc_final_nominal.build_family_summary`.

## 14. Predeclared primary/secondary comparisons (per condition)

Primary: `LR-PPO - Lead`, `LR-PPO - RMPC`, `LR-PPO - PPO`, `RMPC - Lead`,
`RMPC - PPO`. Secondary: `PPO - Greedy`, `PPO - PN`, `PPO - Lead`,
`PPO-Kalman - PPO`. Computed independently for each of the five
conditions (25 primary + 20 secondary rows total), using the established
matched-seed/root-aware bootstrap methodology
(`experiments.run_rmpc_final_nominal.build_paired_differences`, reused
verbatim). Reports `difference_pp`, `ci_low_pp`, `ci_high_pp`,
`resolution` in {`resolved positive`, `resolved negative`, `unresolved`}.
This comparison list is never changed after results are seen.

## 15. Nominal reference (descriptive only)

Read only from the already-frozen `results/rmpc_final_nominal/family_summary.csv`
and `instance_summary.csv`. Never recomputed, never combined with
robustness seeds in any statistical test, never paired against off-nominal
results (the seed blocks differ).

## 16. Fairness audit

Detection-step equality across all 13 methods, and full pre-detection
trajectory equality for 7 representative controllers (`greedy`, `pn`,
`lead`, `rmpc`, `ppo_seed45`, `ppo_kalman_seed45`, `lr_ppo_seed55`) on
seeds 77000/77001/77002 -- checked independently WITHIN every condition
(reactive hostile motion is controller-dependent only after the canonical
control handoff; acquisition must remain controller-independent even when
`enemy_evasion_gain`/dynamics/noise/mobility/detection-radius change).
Any mismatch is a hard stop.

## 17. Physical / numerical audit

Across all 65,000 episodes: `physical_limit_violations == 0`, no
unexpected NaN, no Inf, no invalid action, no unknown outcome, no policy
exception. RMPC diagnostics recorded per condition: Lead initialization
count, pursuit initialization count, optimizer fallback count,
planning-call count, numerical optimization failures. A poor RMPC result
under any condition is DATA, not a correctness failure, and RMPC is never
modified in response to a poor result.

## 18. Source freeze

Once seed 77000 is opened, `uav_defend/`, this runner, this protocol
document, statistical code, models, and `selected_rmpc_config.json` must
not be modified. Any correctness defect discovered mid-campaign is a hard
stop -- no patch-and-resume on the same final campaign.

## 19. Parallel execution

Permitted for tractability with a fixed, recorded worker count (avoiding
BLAS oversubscription exactly as in the final nominal campaign). Wall-clock
policy timing collected during the parallel simulation is NOT
publication-grade runtime evidence and is never used for computational-speed
claims.

## 20. Result namespace

`results/rmpc_targeted_robustness/` only, containing `protocol_manifest.json`,
`model_manifest.json`, `selected_rmpc_config_snapshot.json`, five condition
subdirectories (`strong_evasion/`, `restrictive_dynamics/`,
`high_measurement_noise/`, `hard_mobility/`, `short_detection/`, each with
`episodes/`, `instance_summary.csv`, `family_summary.csv`,
`paired_differences.csv`, `condition_audit.json`), and root-level
`robustness_summary.csv`, `primary_differences_summary.csv`,
`fairness_audit.json`, `numerical_audit.json`,
`targeted_robustness_report.md`. Never modifies `results/rmpc_final_nominal/`,
`results/rmpc_tuning/`, `results/expanded_robustness/`, or `paper_artifacts/`.

## 21. Completeness

For each condition: 13 instances x 1000 rows = 13,000 episode rows; across
five conditions, exactly 65,000 rows. For every instance/condition: exactly
seeds 77000-77999, no duplicate, no missing. Terminal categories
(`intercepted`, `soldier_caught`, `unsafe_intercept`, `timeout`) sum to
1000 per instance; no unknown outcome.

## 22. Replication reference values (final nominal, descriptive only)

Greedy=79.12%, PN=75.92%, Lead=86.76%, RMPC=82.04%, PPO family=87.42%,
PPO-Kalman family=83.43%, LR-PPO family=89.43%.

## 23. No-paper-claims disclaimer

This protocol and its execution produce FINAL RAW ROBUSTNESS EVIDENCE only.
No manuscript, abstract, introduction, `paper_artifacts/` entry, or
publication figure is created or edited by this task. Evidence integration
happens only after explicit review.
