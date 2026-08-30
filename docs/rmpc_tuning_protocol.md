# RMPC Predeclared Development / Validation Tuning Protocol

Status: **PREDECLARED BEFORE ANY SEED >= 71000 IS OPENED.** This document
and `experiments/tune_rmpc.py` are committed together, and this commit's SHA
is recorded as `RMPC_TUNING_PROTOCOL_COMMIT`, BEFORE seed 71000 is executed.
Nothing below may be changed after any development/validation seed has been
observed (see Section 9 "no source changes after first development seed").

**This is a DEVELOPMENT / MODEL-SELECTION campaign. Its outputs are NOT
final publication evidence** and must never be copied into
`paper_artifacts/` or used to update publication figures.

RMPC ancestry for this campaign:
- Design: `docs/rmpc_baseline_design.md` (RMPC_DESIGN_COMMIT `61483f6c448ff2c3b3dfa3bd4c3fa25f18fecf07`)
- Implementation: `RMPC_IMPLEMENTATION_COMMIT 80ea7bd`
- Hardening: `RMPC_HARDENING_COMMIT cb06b04ed39d17ad64b4243c2fe84407bc95e6e4`
- Tuning base commit (this campaign): `cb06b04ed39d17ad64b4243c2fe84407bc95e6e4`

## 1. Purpose

1. Predeclare a small RMPC hyperparameter candidate set (Section 5).
2. Screen on development seeds (Stage A).
3. Confirm on further development seeds (Stage B).
4. Validate finalists on independent seeds (Stage C).
5. Select ONE RMPC configuration and freeze it.
6. Run a non-selection post-selection diagnostic sanity check (Stage D).

This prompt does NOT run the final 5000-seed nominal comparison or the
targeted final robustness campaign (seeds `72000-99999` are never opened
here).

## 2. Fixed (not tuned) RMPC parameters

Frozen for the entire campaign, never varied:

| Field | Value |
|---|---|
| `controller_seed` | `0` |
| `cem_initial_std` | `1.0` |
| `cem_min_std` | `0.05` |
| `cem_sample_clip` | `3.0` |
| `enable_timing` | `True` |

Also frozen (unmodified since the hardening pass, never touched by this
campaign): the six-scenario hostile model, the robust `min` aggregation,
the mission objective (`rmpc_cost.py`), terminal rewards, progress scale,
Lead-based CEM initialization, the first-measurement pursuit fallback, and
the fixed-soldier-position prediction assumption.

Only four RMPC search hyperparameters vary across candidates: `horizon`,
`population_size`, `elite_fraction`, `cem_iterations`.

## 3. Predeclared candidate configurations (C1-C7)

| Candidate | horizon | population_size | elite_fraction | cem_iterations |
|---|---|---|---|---|
| C1_FAST | 4 | 64 | 0.20 | 3 |
| C2_SHORT | 4 | 128 | 0.20 | 5 |
| C3_BALANCED | 6 | 128 | 0.20 | 5 |
| C4_LONG | 8 | 128 | 0.20 | 5 |
| C5_LARGE_POP | 6 | 256 | 0.20 | 5 |
| C6_TIGHT_ELITE | 6 | 128 | 0.10 | 5 |
| C7_MORE_ITER | 6 | 128 | 0.20 | 8 |

No additional candidates may be generated after seeing results. No
exhaustive Cartesian-product grid is performed.

## 4. Nominal environment for tuning

All development/validation evaluation uses the current canonical nominal
`EnvConfig()` (all defaults), explicitly verified and recorded at runtime:
`defender_standby_until_detection = True`, `use_kalman_tracking = False`.
RMPC still never receives the realized hostile-policy parameters
(`enemy_evasion_gain`/`enemy_evasion_radius`/weave/heading-noise state) --
the ENVIRONMENT is allowed to use its canonical nominal hostile behavior;
that is precisely the challenge being evaluated. A full serialized
`EnvConfig` snapshot is stored in `results/rmpc_tuning/protocol_manifest.json`.

## 5. Canonical episode runner

`experiments/post_detection_diagnostic_eval_utils.py::run_diagnostic_episode`
is used unmodified for every episode (RMPC and Lead alike). It already
invokes `build_policy_info(info, estimator_mode)` before `policy.act(...)`;
this behavior is preserved exactly. No simplified RMPC-only environment is
created. The legacy `experiments/eval_utils.py` /
`experiments/evaluate_for_comparison.py` path is NOT used.

Per-episode RMPC-specific diagnostics (optimizer-fallback count,
initialization-mode counts, post-detection planning-time samples) are
captured via a thin PASS-THROUGH observer wrapper
(`_RMPCDiagnosticsWrapper` in `experiments/tune_rmpc.py`) around the
`RMPCPolicy` instance -- it forwards every `act()`/`reset()` call unchanged
and never alters the returned action; it only reads the policy's own
public diagnostic attributes (`last_guidance_mode`,
`last_initialization_mode`, `decision_times_s`) after each call.

## 6. Lead reference

`LeadInterceptPolicy(state_source="measurement", config=config)` is
evaluated once per UNIQUE environment seed opened during this campaign
(contextual reference only -- NOT part of the RMPC hyperparameter-selection
rule, and RMPC-minus-Lead is never used as a criterion).

## 7. Selection metric

The single PRIMARY metric is safe-interception success COUNT
(`outcome == "intercepted"`), exactly as `run_diagnostic_episode` defines
it. Mean reward, minimum distance, tracking error, time-to-intercept,
RMPC-vs-Lead difference, runtime, and subjective trajectory appearance are
never used as the primary selection criterion.

## 8. Predeclared exact-tie break rule

If two candidates have EXACTLY the same success count at a selection stage:

1. Lower theoretical RMPC computation: `population_size * cem_iterations * 6 * horizon`.
2. If still tied: lower measured median POST-DETECTION RMPC planning time.
3. If still tied: lexicographically smaller candidate ID.

No new tie-break is invented after results are observed.

## 9. Seed ranges (authorized stages only)

| Stage | Range | n | Candidates evaluated |
|---|---|---|---|
| A: development screen | 71000-71029 | 30 | ALL SEVEN (C1-C7) |
| B: development confirmation | 71030-71099 | 70 | STAGE_A_TOP3 only |
| C: independent validation | 71100-71199 | 100 | DEVELOPMENT_FINALISTS only |
| D: post-selection diagnostic | 71200-71299 | 100 | SELECTED_RMPC_CONFIG only |

`71300-71999` and `72000-99999` remain UNOPENED reserve for this prompt.
`experiments/tune_rmpc.py` hard-rejects any seed outside these four exact
ranges (`assert_seed_authorized`), and rejects any candidate not in the
allowed set for its stage (`assert_candidates_allowed`).

## 10. Stage procedures

- **Stage A:** all 7 candidates x the same 30 seeds (71000-71029) = 210
  RMPC episodes, + 30 Lead episodes (one per unique seed). Rank by Stage-A
  success count (tie-break Section 8); select the top 3 = `STAGE_A_TOP3`.
  All others are permanently eliminated (never resurrected).
- **Stage B:** `STAGE_A_TOP3` only x 70 NEW seeds (71030-71099) = 210 RMPC
  episodes, + 70 Lead episodes. Combine with each finalist's own Stage-A
  episodes for a 100-seed (71000-71099) development total; rank by this
  combined success count; select the top 2 = `DEVELOPMENT_FINALISTS`.
- **Stage C:** `DEVELOPMENT_FINALISTS` only x 100 independent seeds
  (71100-71199) = 200 RMPC episodes, + 100 Lead episodes. The final
  configuration is chosen using ONLY validation success count on these 100
  seeds (development performance is never combined in). Winner =
  `SELECTED_RMPC_CONFIG`, then FROZEN -- no RMPC scientific/design
  parameter may be modified after observing validation.
- **Stage D:** `SELECTED_RMPC_CONFIG` only, unchanged, x 100 NEW seeds
  (71200-71299) = 100 RMPC episodes, + 100 Lead episodes. DIAGNOSTIC ONLY:
  its outcome must never retune anything.

## 11. Pre-detection fairness audit

For every unique environment seed evaluated with both Lead and RMPC:
`detection_step` (and `detection_time_seconds`) must be IDENTICAL, because
the canonical standby protocol makes pre-detection history
controller-independent. For the first 3 seeds of Stage A, a full
`record_trajectory=True` comparison (`soldier_pos`, `defender_pos`,
`defender_vel`, `enemy_pos`, `enemy_vel`, `enemy_detected` through and
including the detection transition) is performed between Lead and one
predeclared representative RMPC candidate, **`C3_BALANCED`** (chosen here,
before any seed is opened, specifically to avoid post-hoc cherry-picking).
These ground-truth fields are used ONLY by the evaluator for this audit and
are never passed to RMPC. Any discrepancy is a STOP condition.

## 12. Physical / numerical audit

`physical_limit_violations == 0` is required for every episode (any
nonzero count is a STOP condition, not merely a performance note). The
canonical runner's existing numerical-health checks are retained.
Optimizer-fallback count, first-measurement pursuit-initialization count,
and Lead-initialization count are tracked and reported per candidate/stage
(a fallback is not automatically a correctness failure if it is a
legitimate rare degenerate case, but its frequency is always reported).

## 13. Runtime measurement

`RMPCPolicy.decision_times_s` includes cheap pre-detection standby calls.
Per episode: the number of NEW timing entries produced during that episode
is captured, and the first `detection_step` entries (all pre-detection/
transition standby calls) are excluded; only POST-DETECTION planning-call
times are summarized (mean/median/P95/max), per candidate/stage. Timing is
never used for selection except the exact tie-break rule (Section 8).
Theoretical cost (candidate evaluations, hostile scenario-step slots,
defender rollout steps) is also reported per Section 5 of the hardening
pass's accounting. Hardware/platform metadata (OS, CPU identifier, Python
version, NumPy version) is recorded. No real-time-feasibility claim is made.

## 14. Statistics

Success count, success rate, and Wilson 95% CI are reported for every
stage/candidate (and for Lead). `soldier_caught`/`unsafe_intercept`/
`timeout` rates are also reported but never drive selection. These are
DESCRIPTIVE statistics for development/validation only -- no publication
claim is made from this tuning data.

## 15. Failure / correctness stop conditions

Performance itself (RMPC underperforming Lead) is never a stop condition.
Any of the following IS a stop condition: `physical_limit_violations > 0`;
NaN/Inf action or objective; an environment exception; a sanitizer-contract
violation; prohibited information access; a deterministic-reproducibility
failure; a seed-guard violation; systematic optimizer fallback traceable to
a code defect; RMPC mutating `EnvConfig`/environment state during planning;
pre-detection fairness differing across methods for the same seed; or
source code changing unexpectedly after this protocol's commit. If a
correctness defect is discovered after seed 71000 has been opened, the
campaign STOPS immediately -- the code is not "fixed and continued" under
the same seed protocol in this prompt.

## 16. Result artifacts

New isolated namespace: `results/rmpc_tuning/` (never
`results/final_nominal/`, `results/expanded_final/`,
`results/expanded_robustness/`, or `paper_artifacts/`):

```
results/rmpc_tuning/
    protocol_manifest.json
    candidate_configs.json
    stage_a_episodes.csv
    stage_a_summary.csv
    stage_b_episodes.csv
    stage_b_summary.csv
    validation_episodes.csv
    validation_summary.csv
    diagnostic_episodes.csv
    diagnostic_summary.csv
    lead_reference_episodes.csv
    lead_reference_summary.csv
    selected_rmpc_config.json
    tuning_report.md
```

## 17. No source changes after first development seed

Once seed 71000 has been opened, `uav_defend/`, `experiments/tune_rmpc.py`,
this document, and any test file affecting the campaign may NOT be
modified. Only generated artifacts under `results/rmpc_tuning/` may change.
If a source modification appears necessary, the campaign STOPS rather than
continuing under a silently altered protocol.

## 18. Selected-config freeze

`results/rmpc_tuning/selected_rmpc_config.json` becomes the ONLY RMPC
hyperparameter source for the future final campaign. It records
`candidate_id`, all seven `RMPCConfig` fields, validation success
count/n/rate, the selection rule, `source_commit`, a timestamp, and
`"frozen_for_final_evaluation": true`. It is not edited after the Stage D
diagnostic run.
