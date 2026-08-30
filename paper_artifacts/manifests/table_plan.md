# Table Plan (no more than 3 manuscript tables)

## TABLE I — Environment / controller parameters
Fixed nominal values used throughout (v_d=18, v_e=12, defender/enemy
accel/turn/vertical limits, detection_radius=15, measurement_var=0.5,
process_var=1.0, enemy_evasion_gain=0.75/radius=20, θ_max=30°, dt=0.5,
max_steps=2000). Purely descriptive — no CSV needed beyond what already
exists in `uav_defend/config/env_config.py`; to be authored directly in the
manuscript from the frozen config defaults (already audited in this
campaign's config-isolation tests).

## TABLE II — Nominal performance
Source: `paper_artifacts/tables/table_nominal_performance.csv` /
`.tex`. Greedy/PN/Lead/PPO/LR-PPO as headline rows; PPO-Kalman shown as a
clearly labeled estimator-ablation row (not a co-equal method). Includes the
two primary nominal comparisons (LR-PPO−Lead, LR-PPO−PPO) as a footer/second
block within the same table.

## TABLE III — Representative robustness highlights
Source: `paper_artifacts/tables/table_robustness_highlights.csv` / `.tex`.
Six pre-specified rows: Nominal, Strong evasion (gain=1.00), Restrictive
maneuverability (4,45), High measurement noise (var=4.00), Hard mobility
(18,8), Short detection range (Rdet=5m). Each row: Lead / PPO / LR-PPO
success (%) and the two primary differences with 95% CI.

## What belongs in supplemental / internal evidence only (NOT main manuscript)
- Full 9-cell maneuverability grid, full 6-point measurement-noise grid,
  full 9-condition mobility grid, full 6-point detection-radius grid — these
  live in `paper_artifacts/data/figure_*.csv` and appear as Figure 3/4
  panels, not as additional manuscript tables.
- Individual per-training-root success rates (`learned_seed_summary.csv` in
  each sweep directory) — internal evidence only; the manuscript reports
  only equal-weight family means + hierarchical CIs + between-root SD.
- PPO-Kalman vs PPO estimator-ablation detail across all 6 measurement-noise
  variances, individual-root PPOK-vs-PPO descriptive comparisons
  (`estimator_comparisons.csv`) — supplemental/estimator-ablation subsection
  or appendix, referenced from Results D but not tabulated in the main body.
- Kalman tracking-error table (`kalman_tracking_error.csv`) — supports the
  Results D discussion narratively; may appear as a small supplemental table
  if space allows, otherwise summarized in one sentence with the RMSE ratio.
- LR-PPO rescue/harm tables, residual-angle summaries, training-seed
  stability tables, failure-mode tables, post-detection response tables,
  acquisition-fairness/physical-validity/ground-truth-audit reports — all
  internal validation evidence supporting the Methods/Validation subsection
  and the claim-evidence matrix, not manuscript-facing tables.
- Mobility post-hoc audit (`posthoc_audit/`) — clearly labeled POST-HOC /
  EXPLORATORY; belongs in Discussion/Limitations as a qualified mechanistic
  explanation for the slow-hostile inversion, not as a headline result.
