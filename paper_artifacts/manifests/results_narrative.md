# Results Narrative Outline

The paper must read as **one argument organized around questions**, not as a
sequential list of five independent sweeps.

## A. Does residual learning improve predictive Lead guidance?
- Nominal comparison (Table II): LR-PPO − Lead = +3.01pp [2.10, 3.91]
  (excludes zero) → Yes, nominally.
- LR-PPO − PPO = +2.01pp [-0.37, 5.04] (includes zero) → does NOT establish
  nominal superiority over standalone PPO (claim C2).
- Frame Lead as already strong (85.82%); the hybrid's role is to *improve*
  it, not replace a weak baseline.

## B. Does the hybrid retain its benefit under maneuvering and dynamics mismatch?
- Maneuverability: advantage over Lead is largest at the most restrictive
  cell (4,45) and shrinks toward zero at permissive cells — advantage is not
  uniform across the grid (claim C4).
- Mobility: LR-PPO's largest absolute advantage over both Lead and PPO
  occurs at the hardest tested condition (v_d=18, v_e=8) (claim C9).
- Use the compact `table_robustness_highlights` rows for both, referencing
  the full grids (Figure 3b, optional Figure 4) for detail.

## C. How does sensing uncertainty affect classical, learned, and hybrid control?
- Measurement noise: LR-PPO's advantage over Lead grows with noise and
  becomes significant at var≥0.50 (claim C5); its advantage over PPO does
  NOT hold universally — point estimate favors PPO at the highest tested
  noise (claim C6).
- Detection radius: LR-PPO's advantage over PPO is concentrated at the
  shortest tested detection range (Rdet=5m), where it also avoids much of
  PPO's elevated unsafe-intercept failure mode (claim C11).
- These two sweeps jointly answer "how does sensing affect control" from
  complementary angles (noise vs. range) — present them together as one
  sub-argument, not as disconnected sweeps.

## D. What does the Kalman ablation tell us?
- PPO-Kalman does not improve on PPO at any tested measurement-noise level in
  this frozen configuration (claim C7).
- The tested Kalman filter's tracking error is 1.9–2.3x the raw/Direct
  track's error at every tested noise level, against this maneuvering
  hostile (claim C8) — correlational evidence consistent with, but not proof
  of, why PPO-Kalman underperforms PPO.
- Frame PPO-Kalman explicitly as a diagnostic answering "does filtering help
  this specific system," not as a competing headline method.

## E. Where does the hybrid help most, and where does it not?
- Synthesize across B/C: the hybrid's advantage over Lead is most reliable
  and grows under adverse/off-nominal conditions (noise, restrictive
  dynamics, hard mobility); its advantage over standalone PPO is most
  reliable specifically in the hardest/shortest-warning conditions (strong
  evasion, restrictive maneuverability, hard mobility, short detection
  range) rather than at nominal.
- Explicitly state where it does NOT help: nominal PPO comparison, highest
  measurement noise, several individual maneuverability/detection-radius
  cells (CIs include zero).
- Mention the mobility post-hoc audit's qualified mechanistic explanation
  for the slow-hostile inversion as a limitation/discussion point, not a
  headline finding (claim C10).

## Explicitly avoid this structure
~~Evasion sweep → Maneuverability sweep → Noise sweep → Mobility sweep →
Detection sweep~~ — this reads as a list of experiments, not an argument.
Sweep-specific numbers should appear as supporting evidence WITHIN sections
A–E above, not as section headers themselves.
