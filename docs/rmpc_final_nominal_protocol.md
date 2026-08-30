# RMPC Final Nominal Publication Evaluation Protocol

Status: **PREDECLARED BEFORE ANY SEED >= 72000 IS OPENED.** This document
and `experiments/run_rmpc_final_nominal.py` are committed together, and this
commit's SHA is recorded as `RMPC_FINAL_NOMINAL_PROTOCOL_COMMIT`, BEFORE
seed 72000 is executed. Nothing below may change after any final-nominal
seed has been observed (Section 12 "source freeze").

**This prompt produces FINAL RAW NOMINAL EVIDENCE only.** No manuscript
files, `paper_artifacts/`, or publication figures are updated by this task.

## 1. Scientific purpose

A completely fresh matched-seed nominal comparison among all seven
controller families -- Greedy, PN, Lead, PPO, PPO-Kalman, LR-PPO, RMPC -- on
a brand-new seed block. PPO/PPO-Kalman/LR-PPO checkpoints remain FROZEN (no
retraining); RMPC's configuration remains FROZEN at its tuned selection (no
further tuning).

## 2. Ancestry

- RMPC design: `61483f6c448ff2c3b3dfa3bd4c3fa25f18fecf07`
- RMPC implementation: `80ea7bd`
- RMPC hardening: `cb06b04ed39d17ad64b4243c2fe84407bc95e6e4`
- RMPC tuning protocol: `dc6e29ac755e51e5d545a9ddc5bcb5d609eac864`
- RMPC tuning results (frozen selection): `ed4124c`
- Final-nominal base commit: `ed4124c898e095f88db16906712b9e4d48c6b8c6`

## 3. Frozen RMPC configuration (verified, never re-derived)

Constructed ONLY by loading `results/rmpc_tuning/selected_rmpc_config.json`
at runtime (never a second hardcoded/independently-editable copy). Required
exact field values, verified before opening any seed:

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

**RMPC RNG semantics (frozen, unchanged):** RMPC's dedicated CEM RNG is
re-seeded from `controller_seed=0` at the start of every episode
(`policy.reset()`), independent of the environment seed. Environment
randomness varies by `environment_seed`; RMPC's optimizer search stream
always restarts deterministically from `controller_seed=0` for each
episode. This is the exact frozen evaluated controller definition and is
NOT altered to depend on the environment seed.

## 4. Frozen learned models (verified, never re-derived)

Reused verbatim from `experiments.lead_residual_expanded_final_protocol`
(the SAME module that produced the existing frozen paper evidence):
`FROZEN_MODELS`, `TRAINING_SEEDS=(45,46,47)`, `LR_TRAINING_ROOTS=(55,56,57)`,
`RESIDUAL_MAX_ANGLE_DEG=30.0`, `BOOTSTRAP_REPLICATES=10_000`,
`BOOTSTRAP_SEED=20260818`, `CONFIDENCE_LEVEL=0.95`. No new checkpoint is
selected, no retraining/fine-tuning, no change to PPO hyperparameters,
observation normalization, LR-PPO residual construction, `theta_max`, or
Kalman parameters. All 9 checkpoint SHA-256 hashes are verified against
`FROZEN_MODELS` before any seed is opened (`verify_model_hashes()`); any
mismatch is a STOP condition.

| Family | Training roots | Checkpoint root |
|---|---|---|
| PPO | 45, 46, 47 | `results/training_post_detection_400k/direct/seed_{45,46,47}/best_model.zip` |
| PPO-Kalman | 45, 46, 47 | `results/training_post_detection_400k/kalman/seed_{45,46,47}/best_model.zip` |
| LR-PPO | 55, 56, 57 | `models/lead_residual_post_detection/seed_{55,56,57}/best_model.zip` |

## 5. Thirteen method instances

Reuses `experiments.run_lead_residual_diagnostic.build_instances()`
verbatim for the 12 existing instances, plus exactly one new instance:

| name | display_name | family | training_seed |
|---|---|---|---|
| greedy | Greedy | scripted | None |
| pn | PN | scripted | None |
| lead | Lead | scripted | None |
| **rmpc** | **RMPC** | **rmpc** | **None** |
| ppo | PPO-45/46/47 | ppo | 45,46,47 |
| ppo_kalman | PPOK-45/46/47 | ppo_kalman | 45,46,47 |
| lr_ppo | LR-PPO-55/56/57 | lr_ppo | 55,56,57 |

RMPC is ONE fixed deterministic controller configuration -- never treated
as having "training roots". No Kalman-Greedy, no Lead-Kalman, no additional
controller.

## 6. Estimator/information track

| Instance family | Track |
|---|---|
| Greedy, PN, Lead, RMPC, PPO, LR-PPO | measurement/Direct |
| PPO-Kalman | Kalman |

RMPC receives ONLY `build_policy_info(info, "measurement")`-sanitized
fields, exactly like every other measurement-track policy; no ground truth
is ever passed to RMPC.

## 7. Common pre-detection protocol

Unmodified canonical `SoldierEnv` standby behavior
(`defender_standby_until_detection=True`): defender starts co-located with
the protected asset, remains co-located with zero velocity before
detection, every policy's action is discarded before and during the
detection-transition step, and the first executed action occurs on the
following step from rest. No controller-specific acquisition logic exists
for RMPC or any other method.

## 8. Nominal environment

`EnvConfig(use_kalman_tracking=False, defender_standby_until_detection=True)`
for all measurement-track instances and
`EnvConfig(use_kalman_tracking=True, defender_standby_until_detection=True)`
for PPO-Kalman (identical to `_CFG_DIRECT`/`_CFG_KALMAN` in
`run_lead_residual_diagnostic.py`) -- otherwise all-default canonical
nominal parameters (`v_d=18`, `v_e=12`, `detection_radius=15`,
`measurement_var=0.5`, `threat_radius=2.0`, `intercept_radius=2.5`,
`unsafe_intercept_radius=3.5`, `enemy_evasion_enabled=True`,
`enemy_evasion_gain=0.75`, `enemy_evasion_radius=20.0`, `dt=0.5`,
`max_steps=2000`, defender/hostile accel/turn/climb/descent at their
`EnvConfig` defaults). The full serialized config is recorded in
`protocol_manifest.json`. Environment hostile-policy parameters
(`enemy_evasion_gain`/`enemy_evasion_radius`/weave/heading-noise) remain
part of the environment and are NEVER passed into RMPC's planner.

## 9. Seeds

Exactly `72000-76999` inclusive (5000 unique environment seeds). Every one
of the 13 method instances is evaluated on ALL 5000 seeds -> 65,000 total
episodes. No seed substitution, no skipping, no replacement.
`77000-99999` and `71300-71999` remain unopened by this prompt.

## 10. Canonical episode runner

`experiments.post_detection_diagnostic_eval_utils.run_diagnostic_episode`
for greedy/pn/lead/rmpc/ppo/ppo_kalman, and
`experiments.lead_residual_diagnostic_eval_utils.run_lr_ppo_diagnostic_episode`
for lr_ppo -- BOTH unmodified, exactly as used by the existing frozen
paper-evidence campaigns. The legacy `experiments/eval_utils.py` /
`experiments/evaluate_for_comparison.py` path is not used.

## 11. Primary metric

Safe-interception success: `outcome == "intercepted"`. Success count,
success rate, and Wilson 95% CI reported for every instance.

## 12. Learned-family aggregation

Preserves the exact statistical treatment of the frozen publication
campaign (`experiments/final_stats.py`, reused verbatim, never
reimplemented):

- `hierarchical_bootstrap_track` -- single family's paper-level (equal
  weight, 3-training-seed) success rate + hierarchical 95% CI.
- Per-root success rates, family mean, and root-to-root standard deviation
  also reported.
- The 3 roots x 5000 seeds are NEVER pooled and presented as 15,000 IID
  trials.

Deterministic single-instance methods (Greedy, PN, Lead, RMPC) are reported
as single instances with a plain Wilson CI, never as fake multi-root
families.

## 13. Primary pairwise comparisons (predeclared, fixed order)

| # | Comparison | Statistical method |
|---|---|---|
| A | LR-PPO family - Lead | `hierarchical_paired_bootstrap_vs_scripted` |
| B | LR-PPO family - RMPC | `hierarchical_paired_bootstrap_vs_scripted` (RMPC as the fixed comparator) |
| C | LR-PPO family - PPO family | `hierarchical_bootstrap_independent_families` (independent training roots) |
| D | RMPC - Lead | `paired_bootstrap_ci` (both deterministic, matched seeds) |
| E | RMPC - PPO family | `hierarchical_paired_bootstrap_vs_scripted` (RMPC as the fixed comparator), sign-flipped to report RMPC - PPO |

Secondary comparisons (retained, unchanged methodology):

| # | Comparison | Statistical method |
|---|---|---|
| F | PPO family - Greedy | `hierarchical_paired_bootstrap_vs_scripted` |
| G | PPO family - PN | `hierarchical_paired_bootstrap_vs_scripted` |
| H | PPO family - Lead | `hierarchical_paired_bootstrap_vs_scripted` |
| I | PPO-Kalman family - PPO family | `hierarchical_paired_bootstrap_matched_seeds` (matched training-seed labels) |

No additional headline comparison is invented after results are observed.
All differences reported in percentage points with 95% intervals;
resolution is `"resolved positive"` / `"resolved negative"` / `"unresolved"`
based strictly on whether the 95% interval excludes zero.

## 14. Fairness audit

For every method/seed, `detection_step`/`detection_time_seconds` must be
IDENTICAL across all 13 methods (pre-detection history is
controller-independent under the canonical standby protocol). For the
preregistered subset `{72000, 72001, 72002}`, one representative per
controller family (`greedy`, `pn`, `lead`, `rmpc`, `ppo` seed 45,
`ppo_kalman` seed 45, `lr_ppo` seed 55) is run with `record_trajectory=True`
and compared field-by-field (`soldier_pos`, `defender_pos`, `defender_vel`,
`enemy_pos`, `enemy_vel`, `enemy_detected`) through acquisition. Any
mismatch is a STOP condition. These ground-truth fields are evaluator-only
and are never passed to any policy.

## 15. Physical / numerical audit

`physical_limit_violations == 0` required for all 65,000 episodes. No
unexpected NaN/Inf/invalid action/environment or policy exception (missing
`intercept_time_seconds` for a non-"intercepted" outcome is expected, not a
violation). RMPC-specific diagnostics (optimizer fallback count,
initialization-mode counts, planning call count) are recorded per episode
via the same `_RMPCDiagnosticsWrapper` pattern used in the tuning campaign.
Any systematic optimizer defect is a STOP condition; RMPC is not repaired
after opening final seeds.

## 16. Parallel execution vs runtime benchmarking

The tuning campaign showed severe wall-clock contamination under heavy
parallel execution. This campaign therefore explicitly separates
PERFORMANCE SIMULATION (parallelized, for tractability) from RUNTIME
BENCHMARKING (not attempted here). A fixed, recorded worker count is used;
extreme per-decision scheduling pauses are NOT interpreted as controller
compute time. RMPC decision times collected during this run are saved only
as diagnostic engineering data and are explicitly labeled **"NOT VALID FOR
PUBLICATION RUNTIME COMPARISON"** -- they are never placed in the main
paper table. Final success outcomes are independent of worker scheduling.

## 17. Source freeze after first final seed

Once seed 72000 has been executed, `uav_defend/`,
`experiments/run_rmpc_final_nominal.py`, this document, statistical
analysis code, model files, and `results/rmpc_tuning/selected_rmpc_config.json`
may NOT be modified. Only generated artifacts under
`results/rmpc_final_nominal/` may change. Any correctness defect discovered
after seed 72000 STOPS the campaign; it is not patched and resumed.

## 18. Result namespace

`results/rmpc_final_nominal/` (never overwriting `results/expanded_final/`,
`results/expanded_robustness/`, `results/rmpc_tuning/`, or
`paper_artifacts/`):

```
results/rmpc_final_nominal/
    protocol_manifest.json
    model_manifest.json
    selected_rmpc_config_snapshot.json
    episodes/{greedy,pn,lead,rmpc,
              ppo_seed45,ppo_seed46,ppo_seed47,
              ppo_kalman_seed45,ppo_kalman_seed46,ppo_kalman_seed47,
              lr_ppo_seed55,lr_ppo_seed56,lr_ppo_seed57}.csv
    instance_summary.csv
    family_summary.csv
    paired_differences.csv
    fairness_audit.json
    numerical_audit.json
    final_nominal_report.md
```

## 19. Completeness check

Before any statistical analysis: exactly 5000 rows per instance, exactly
seeds 72000-76999, no duplicate/missing seed, exactly 65,000 total rows,
terminal-outcome counts per instance sum to exactly 5000 across
{intercepted, soldier_caught, unsafe_intercept, timeout} with no unknown
category.

## 20. Replication reference (descriptive only, never used in new statistics)

Previous approximate frozen nominal rates (61000-65999, NOT reused in any
new computation): Greedy=77.62%, PN=74.68%, Lead=85.82%, PPO family=86.83%,
PPO-Kalman family=83.28%, LR-PPO family=88.83%. The new 72000-76999 block is
independent; exact equality is NOT required. Only unexpectedly large
deviations trigger a correctness audit.

## 21. No paper claims

This protocol produces FINAL RAW NOMINAL EVIDENCE only. No manuscript file,
`paper_artifacts/` entry, or publication figure is updated by this task.
