# Robust-Baseline Reviewer Response

We thank the reviewer for pointing out the absence of a robust
optimal-control / MPC comparator in the original submission. We have added
a scenario-based robust Model Predictive Control (RMPC) baseline: a
finite-horizon (H=4) Cross-Entropy-Method (CEM) optimizer (population 64,
elite fraction 0.20, 3 iterations) that plans over a fixed set of bounded
hostile-motion scenarios rather than a single nominal prediction. RMPC was
tuned on independent development/validation seeds, frozen
(`results/rmpc_tuning/selected_rmpc_config.json`,
`frozen_for_final_evaluation=true`), and evaluated -- without further
tuning -- on the same nominal (72000-76999) and off-nominal (77000-77999,
five conditions) seed blocks as every other controller, under an identical
information contract (sanitized measurement-track state only; no ground
truth, no hidden attacker parameters).

RMPC's nominal safe-interception success was 82.04%
(vs. LR-PPO's 89.43%; difference
+7.39 pp, resolved positive). Under the five
off-nominal conditions RMPC's relative standing was strongly condition-dependent: it was the
numerical leader under the hard-mobility condition, significantly outperforming both Lead and PPO
there, while degrading sharply under high measurement noise.

We do not claim RMPC is globally optimal or represents all possible
robust-control formulations; it is one concrete, frozen, scenario-based
comparator evaluated under the same constraints as every learned and
scripted method in this study.
