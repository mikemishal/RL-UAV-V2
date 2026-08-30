# Resubmission Evidence Policy

Status: **AUTHORITATIVE.** This document defines exactly which result sets
may be used for which paper claims in the conference resubmission. It does
not itself run any experiment, train anything, or open any environment
seed. It is read by `experiments/build_resubmission_evidence.py`, which
extracts every paper-facing number programmatically from the Tier-A
sources below -- no paper number is hardcoded as a source of truth.

## Tier A -- Authoritative final publication evidence

### Nominal

- Source: `results/rmpc_final_nominal/`
- Seeds: `72000-76999` (5000 unique, matched across all 13 method instances)
- These values **supersede** the older `61000-65999` nominal campaign
  (`results/lead_residual_expanded_final_5000/`) for every headline paper
  table and claim.

### Targeted robustness

- Source: `results/rmpc_targeted_robustness/`
- Seeds: `77000-77999` (1000 unique, reused identically across all five
  conditions -- the five conditions are NOT statistically independent of
  one another)
- Five conditions: `strong_evasion`, `restrictive_dynamics`,
  `high_measurement_noise`, `hard_mobility`, `short_detection`
- These are the authoritative paper-facing robustness values whenever the
  paper compares RMPC with the full seven-family controller set.

## Tier B -- Historical / replication / dense-sweep evidence

Existing older evidence, including:

- `results/lead_residual_expanded_final_5000/` (old nominal, seeds
  `61000-65999`)
- `results/expanded_robustness/` (dense one-dimensional sensitivity grids)
- historical `paper_artifacts/` (the pre-RMPC figure/table set)

may be retained ONLY for:

1. independent replication discussion (old vs. new nominal, descriptive
   only -- never pooled),
2. dense one-dimensional sensitivity curves (full evasion-gain grid, full
   maneuverability grid, full noise grid, full mobility grid, full
   detection-radius grid) not repeated at every point in the new targeted
   five-condition campaign,
3. diagnostics not repeated in the new targeted campaign.

Tier-B numerical values must NEVER be silently mixed into a Tier-A headline
table without an explicit, visible campaign label. For example: old
nominal (`61000-65999`) and new nominal (`72000-76999`) must never be
combined into a single reported estimate.

## Tier C -- Model-selection / development data

- Source: `results/rmpc_tuning/`
- This is DEVELOPMENT / MODEL-SELECTION data (Stage A/B/C/D tuning
  campaign that selected the frozen `C1_FAST` configuration).
- It must NOT be used as final RMPC performance evidence -- tuning-stage
  success rates were computed on development/validation seeds explicitly
  reserved from the final evaluation seed ranges and are not comparable to
  Tier-A results.
- The ONLY paper-facing information that may be drawn from this namespace
  is: (a) the frozen RMPC controller definition in
  `results/rmpc_tuning/selected_rmpc_config.json`, and (b) the fact that
  selection used independent development/validation seeds, disjoint from
  the Tier-A nominal (`72000-76999`) and robustness (`77000-77999`) seed
  ranges.

## Enforcement

`experiments/build_resubmission_evidence.py` enforces this hierarchy
programmatically:

- It reads nominal paper values ONLY from `results/rmpc_final_nominal/`.
- It reads robustness paper values ONLY from
  `results/rmpc_targeted_robustness/`.
- It reads the frozen RMPC configuration (never performance data) from
  `results/rmpc_tuning/selected_rmpc_config.json`.
- It reads the old nominal campaign from
  `results/lead_residual_expanded_final_5000/` ONLY for the descriptive
  replication section, and never merges it numerically with the new
  nominal values.
- It performs no environment simulation, no model loading for inference,
  and no training. All work is CSV/JSON parsing and statistics already
  computed by the completed final-nominal and targeted-robustness
  campaigns.

## Seed ledger (evidence consolidation only)

USED FINAL PAPER EVIDENCE:

- `72000-76999` (nominal)
- `77000-77999` (targeted robustness)

RESERVED / UNOPENED (must remain so during evidence consolidation):

- `71300-71999`
- `78000-99999`

No new environment seed is executed during evidence consolidation.
