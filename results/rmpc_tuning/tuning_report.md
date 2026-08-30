# RMPC Predeclared Tuning Campaign Report

**DEVELOPMENT / MODEL-SELECTION DATA -- NOT FINAL PUBLICATION EVIDENCE.**
This report must never be copied into `paper_artifacts/` or used to
update publication figures. See `docs/rmpc_tuning_protocol.md` for the
full predeclared protocol.

## Stage A -- 30-seed development screen (all 7 candidates)

| Candidate | Success/30 | Success % | Wilson 95% CI |
|---|---|---|---|
| C1_FAST | 23/30 | 76.7% | [0.591, 0.882] |
| C2_SHORT | 26/30 | 86.7% | [0.703, 0.947] |
| C3_BALANCED | 13/30 | 43.3% | [0.274, 0.608] |
| C4_LONG | 19/30 | 63.3% | [0.455, 0.781] |
| C5_LARGE_POP | 12/30 | 40.0% | [0.246, 0.577] |
| C6_TIGHT_ELITE | 15/30 | 50.0% | [0.332, 0.668] |
| C7_MORE_ITER | 17/30 | 56.7% | [0.392, 0.726] |

**STAGE_A_TOP3 = ['C2_SHORT', 'C1_FAST', 'C4_LONG']**

## Stage B -- 100-seed development total (top 3)

| Candidate | Combined Success/100 | Success % | Wilson 95% CI |
|---|---|---|---|
| C2_SHORT | 80/100 | 80.0% | [0.711, 0.867] |
| C1_FAST | 81/100 | 81.0% | [0.722, 0.875] |
| C4_LONG | 57/100 | 57.0% | [0.472, 0.663] |

**DEVELOPMENT_FINALISTS = ['C1_FAST', 'C2_SHORT']**

## Validation -- 100 independent seeds (2 finalists)

| Candidate | Success/100 | Success % | Wilson 95% CI |
|---|---|---|---|
| C1_FAST | 75/100 | 75.0% | [0.657, 0.825] |
| C2_SHORT | 72/100 | 72.0% | [0.625, 0.799] |

**SELECTED_RMPC_CONFIG = C1_FAST** (tie-break used: False)

## Stage D -- 100-seed post-selection diagnostic (frozen config, informational only)

Success: 76/100 (76.0%), Wilson 95% CI [0.668, 0.833]

Diagnostic performance did NOT alter the selected configuration.

## Lead reference (contextual only)

- stage_a_71000_71029: 27/30 (90.0%)
- stage_b_71030_71099: 60/70 (85.7%)
- development_combined_71000_71099: 87/100 (87.0%)
- validation_71100_71199: 82/100 (82.0%)
- diagnostic_71200_71299: 88/100 (88.0%)
