# Evidence Summary (Publication-Safe Conclusions)

1. **Nominal:** LR-PPO significantly beat Lead (+2.67 pp,
   resolved positive) and RMPC (+7.39 pp,
   resolved positive).
2. **Nominal:** LR-PPO vs PPO was unresolved
   (+2.01 pp [-0.27, 5.11]).
3. **Robustness:** LR-PPO significantly beat PPO in 4/5 selected off-nominal conditions.
4. **High measurement noise:** PPO significantly beat LR-PPO
   (-5.73 pp,
   resolved negative).
5. **LR-PPO vs Lead:** nominal resolved positive; 2/5 off-nominal resolved positive;
   3/5 unresolved; 0/5 resolved negative (never resolved negative).
6. **RMPC:** condition-dependent; numerical winner under hard mobility; weak under high
   measurement noise.
7. **PPO-Kalman:** no consistent evidence of improvement over PPO
   (-3.99 pp, unresolved).
8. **Reviewer:** the robust-MPC-comparator gap is addressed (see
   `reviewer/reviewer_issue_matrix.md`).

## Recommended contribution framing

**Contribution 1.** A 3-D constrained point-mass counter-UAS interception
framework with common sensing, acquisition, and vehicle-dynamics semantics
across all controllers.

**Contribution 2.** LR-PPO, which retains constant-velocity predictive Lead
guidance as the nominal action and learns a bounded angular residual around
it.

**Contribution 3.** A controlled comparison against classical guidance,
standalone PPO, Kalman-preprocessed PPO, and an independently tuned
scenario-based RMPC.

**Contribution 4.** A matched-seed robustness evaluation showing
controller-dependent strengths across maneuvering, dynamics constraints,
measurement noise, mobility, and detection range.

RMPC is a comparator, not a proposed contribution.

## Recommended central result statement

"LR-PPO retains predictive Lead guidance as a nominal interception policy
and learns a bounded angular residual. On 5,000 fresh nominal Monte Carlo
engagements, LR-PPO improved safe-interception success over Lead by
approximately 2.7 percentage points and over the
independently tuned scenario-based RMPC baseline by approximately
7.4 percentage points. Across five selected
off-nominal conditions, LR-PPO significantly outperformed standalone PPO in
4 conditions, while PPO remained strongest under high measurement
noise."

## Do-not-claim boundaries

- no high-fidelity UAV model
- no six-DOF aerodynamics
- no hardware validation
- no real radar calibration
- no optimal adversarial attacker
- no game-theoretic equilibrium
- no globally optimal RMPC
- no universal superiority over PPO
- no universal superiority over RMPC
- no universal superiority over Lead
- no general claim that Kalman filtering is harmful
- no claim that larger detection radius always improves success
- no claim that lower hostile speed is inherently harder
- no claim PPO significantly beats Lead nominally
- no claim LR-PPO significantly beats PPO nominally
- no causal claim that PPO explicitly observes or corrects Lead prediction error

## Suggested figure/table numbering (manuscript layout recommendation only -- not yet applied)

- Fig. 1: conceptual 3-D engagement scene (existing `images/UAV_intercept_scene.png`)
- Fig. 2: LR-PPO architecture
- Fig. 3: final nominal performance including RMPC (`fig_nominal_performance`)
- Fig. 4: six-condition robustness summary including RMPC (`fig_robustness_summary`)
- Table I: controller definitions / comparison hierarchy
- Table II: nominal success + CI (`nominal_performance`)
- Table III: condensed robustness differences (`robustness_condensed`), IF page budget permits;
  otherwise use the bolded `robustness_success` matrix alone and move differences to an appendix.
