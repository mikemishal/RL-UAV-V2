# Prohibited Claims

Generated from the frozen conference experimental campaign
(source commit `0070946dbb1d1e57f3127ad3347feae94d2818fd`). These statements
must NOT appear in the manuscript, in any form, because the frozen evidence
does not support them.

1. "LR-PPO always beats PPO." — Not supported: nominal (+2.01pp, CI includes
   zero), high measurement noise var=4.00 (point estimate -2.20pp), several
   maneuverability/mobility/detection-radius cells include zero.
2. "LR-PPO always beats Lead." — Not supported: several conditions (e.g.
   maneuverability (24,90)/(24,45), measurement noise low-variance, some
   detection radii) show CIs that include zero.
3. "Kalman filters hurt reinforcement learning." — Not supported as a general
   statement; only shown for this specific frozen `process_var=1.0`
   configuration against this maneuvering hostile (see C7/C8 in the
   claim-evidence matrix).
4. "Lower tracking error causes higher success." — Not supported: the
   tracking-error and success-rate findings are correlational (both directions
   were observed together), never an isolated causal manipulation.
5. "Slower hostile UAVs are inherently harder to intercept." — Not supported
   as a general statement; the mobility post-hoc audit shows this is specific
   to the frozen fixed-acceleration/turn-rate dynamics model and the tested
   speed range, and is explicitly qualified (see C10).
6. "The defender model is high fidelity." — Not supported; it is an explicitly
   simplified constrained point-mass model.
7. "The simulation is six-DOF." — False; it is a 3-D constrained point-mass
   model with acceleration/turn-rate/vertical-rate limits, not six-DOF or
   aerodynamic flight dynamics.
8. "The tested limits are hardware validated." — Not supported; all dynamics
   constants (accel, turn-rate, detection radius, measurement variance, etc.)
   are modeling choices, not validated against any real airframe or sensor.
9. "The hostile is an optimal/intelligent adversary." — Not supported; the
   hostile uses a fixed scripted pursuit+weave+reactive-evasion policy, not
   an adversarially optimized or learning agent.
10. "The 30-degree residual bound is optimal." — Not supported; `theta_max =
    30 deg` was fixed by design before any evaluation and was never tuned or
    swept.
11. "LR-PPO statistically reduces training variance." — Not supported as a
    formal claim; only descriptive between-training-root SD was reported
    (LR-PPO's SD is smaller than PPO/PPO-Kalman's at most but not all tested
    conditions), and no formal variance-reduction test was pre-specified or
    performed.
12. "Detection radius values correspond to a particular real sensor." —
    Not supported; `detection_radius` is a swept behavioral parameter of the
    environment model, not a validated physical sensor specification.

## Additional statements to avoid (identified during evidence consolidation)

13. "Higher mobility ratio (rho = v_d/v_e) guarantees better performance." —
    Directly contradicted by evidence: (18,8) has the highest tested rho
    (2.25) yet is the worst-performing condition for every method.
14. "Larger detection radius always improves interception." — Contradicted:
    success plateaus and slightly declines beyond ~20m for most methods
    (marginal_range_benefit.csv).
15. "Gain=1.00 in the hostile-evasion sweep represents an adversarial or
    optimal hostile policy." — It is simply the upper end of the tested
    reactive-evasion-gain range; the hostile is never adversarially tuned.
16. "PPO-Kalman is a co-equal proposed method." — It is an estimator-ablation
    diagnostic only, per the paper's method-role framework.
17. "The hostile 'knows' the defender's detection state." — False; hostile
    reactive evasion depends only on defender-hostile distance, never on
    whether the defender has detected it.
