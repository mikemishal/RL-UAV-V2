# Figure Plan (proposal — no figures drawn yet)

Single unifying question: *"Can learning improve a strong classical
predictive interception controller under maneuvering, sensing uncertainty,
and model/dynamics mismatch, rather than replacing classical guidance
altogether?"*

## Recommendation: 3 main figures + 1 optional figure (as proposed)

### FIGURE 1 — Method/system concept (schematic, not a performance graph)
Predictive Lead guidance + bounded (±30°) PPO residual correction, composed
into the hybrid action. Shows: sanitized observation → Lead direction →
residual-PPO correction (clipped to θ_max) → composed unit direction →
environment. No data needed from `paper_artifacts/`; this is a diagram to be
drawn separately by hand/vector tool, not generated from CSVs.

### FIGURE 2 — Nominal performance
Bar/point chart: Greedy, PN, Lead, PPO, LR-PPO with Wilson/hierarchical 95%
CI error bars. **PPO-Kalman visually de-emphasized** (e.g., shown as a small
inset/marker or omitted from the main bars and referenced only in the table),
consistent with its estimator-ablation role.
Data source: `paper_artifacts/tables/table_nominal_performance.csv`.

### FIGURE 3 — Robustness (multi-panel)
One figure, 4 panels:
- (a) hostile evasion: `paper_artifacts/data/figure_evasion.csv` (x=gain)
- (b) maneuverability: `paper_artifacts/data/figure_maneuverability.csv`
  (3x3 heatmap or faceted line, x=accel, series=turn-rate)
- (c) measurement noise: `paper_artifacts/data/figure_measurement_noise.csv`
  (x=sigma, log-ish spacing already reflected in variance grid)
- (d) detection radius: `paper_artifacts/data/figure_detection_radius.csv`
  (x=Rdet)
Each panel: Lead / PPO / LR-PPO lines with CI bands (add PPO-Kalman only to
panel (c), de-emphasized, since it is the estimator-ablation-relevant sweep).

Mobility is **excluded** from Figure 3 (its two-arm/rho structure does not
plot cleanly on a single shared x-axis with the other four 1-D sweeps) and is
instead carried by the compact `table_robustness_highlights` row ("Hard
mobility (v_d=18, v_e=8)") plus the optional Figure 4 below.

### OPTIONAL FIGURE 4 — Mobility operating envelope
`paper_artifacts/data/figure_mobility.csv`: two-panel (defender-speed arm,
hostile-speed arm) or a rho-vs-success scatter with arm-colored markers
(never collapsing the two arms into one curve, per the mobility protocol's
explicit guardrail). Include only if page budget allows; otherwise the
`table_robustness_highlights` "Hard mobility" row plus one sentence in the
narrative suffices.

## Recommended final choice
Adopt the 3-main + 1-optional structure exactly as proposed. If a page-limit
cut is required, drop Figure 4 first (mobility is already represented in
Table III) before cutting any of Figures 1–3.
