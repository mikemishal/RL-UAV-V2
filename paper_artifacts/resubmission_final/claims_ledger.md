# Claims Ledger

## C1 -- LR-PPO vs Lead (nominal)

- **Proposed claim**: LR-PPO significantly improved safe-interception success over predictive Lead guidance under nominal conditions.
- **Evidence source**: results/rmpc_final_nominal/paired_differences.csv
- **Point estimate**: +2.67 pp [1.77, 3.57]
- **Statistical status**: resolved positive
- **Allowed wording**: LR-PPO significantly improved safe-interception success over predictive Lead guidance under nominal conditions.
- **Prohibited stronger wording**: LR-PPO always outperforms Lead. | RL corrects all Lead prediction errors.
- **Paper section**: Results (nominal) (priority: high)

## C2 -- LR-PPO vs RMPC (nominal)

- **Proposed claim**: LR-PPO significantly outperformed the scenario-based robust MPC baseline under nominal conditions.
- **Evidence source**: results/rmpc_final_nominal/paired_differences.csv
- **Point estimate**: +7.39 pp [6.31, 8.49]
- **Statistical status**: resolved positive
- **Allowed wording**: LR-PPO significantly outperformed the scenario-based robust MPC baseline under nominal conditions.
- **Prohibited stronger wording**: LR-PPO is superior to robust control. | RL dominates optimal control.
- **Paper section**: Results (nominal) (priority: high)

## C3 -- LR-PPO vs PPO (nominal)

- **Proposed claim**: LR-PPO achieved a numerically higher nominal success rate than PPO, but the root-aware 95% interval for the difference included zero.
- **Evidence source**: results/rmpc_final_nominal/paired_differences.csv
- **Point estimate**: +2.01 pp [-0.27, 5.11]
- **Statistical status**: unresolved
- **Allowed wording**: LR-PPO achieved a numerically higher nominal success rate than PPO, but the root-aware 95% interval for the difference included zero.
- **Prohibited stronger wording**: LR-PPO significantly outperformed PPO nominally.
- **Paper section**: Results (nominal) (priority: high)

## C4 -- LR-PPO vs PPO (robustness)

- **Proposed claim**: LR-PPO significantly outperformed PPO in 4 of five selected off-nominal conditions; high measurement noise is the exception (resolved negative).
- **Evidence source**: results/rmpc_targeted_robustness/primary_differences_summary.csv
- **Point estimate**: -5.73 pp [-10.17, -1.30]
- **Statistical status**: 4/5 resolved positive, 1/5 resolved negative
- **Allowed wording**: LR-PPO significantly outperformed PPO in 4 of five selected off-nominal conditions. High measurement noise is the exception.
- **Prohibited stronger wording**: LR-PPO universally outperforms PPO. | LR-PPO is uniformly more robust to sensing noise.
- **Paper section**: Results (robustness) (priority: high)

## C5 -- LR-PPO vs Lead (robustness)

- **Proposed claim**: LR-PPO significantly improved upon Lead nominally and in 2 of the five selected off-nominal conditions, while the remaining 3 comparisons were statistically unresolved.
- **Evidence source**: results/rmpc_targeted_robustness/primary_differences_summary.csv
- **Point estimate**: +2.67 pp [1.77, 3.57]
- **Statistical status**: nominal resolved positive; 2/5 off-nominal resolved positive; 3/5 unresolved; 0/5 resolved negative
- **Allowed wording**: LR-PPO significantly improved upon Lead nominally and in 2 of the five selected off-nominal conditions, while the remaining comparisons were statistically unresolved. No tested LR-PPO-versus-Lead condition produced a resolved negative difference.
- **Prohibited stronger wording**: LR-PPO significantly beats Lead under every condition.
- **Paper section**: Results (robustness) (priority: high)

## C6 -- RMPC condition-dependent behavior

- **Proposed claim**: The scenario-based RMPC baseline showed strongly condition-dependent behavior: numerical leader under hard mobility, weakest under high measurement noise.
- **Evidence source**: results/rmpc_targeted_robustness/robustness_summary.csv
- **Point estimate**: +19.60 pp [15.70, 23.50]
- **Statistical status**: condition-dependent (see robustness_differences.csv)
- **Allowed wording**: The scenario-based RMPC baseline showed strongly condition-dependent behavior. RMPC was the numerical leader under the hard-mobility condition and significantly outperformed Lead and PPO there. RMPC degraded sharply under high measurement noise.
- **Prohibited stronger wording**: RMPC is weak. | RMPC is inferior to learning. | RMPC is globally optimal. | RMPC represents all robust-control approaches.
- **Paper section**: Results (robustness) / RMPC discussion (priority: high)

## C7 -- Kalman preprocessing

- **Proposed claim**: The tested constant-velocity Kalman preprocessing did not provide a consistent improvement over PPO.
- **Evidence source**: results/rmpc_final_nominal/paired_differences.csv (PPO-Kalman - PPO)
- **Point estimate**: -3.99 pp [-9.81, 1.70]
- **Statistical status**: unresolved
- **Allowed wording**: The tested constant-velocity Kalman preprocessing did not provide a consistent improvement over PPO.
- **Prohibited stronger wording**: Kalman filters are harmful. | Kalman filtering is unsuitable for counter-UAS interception.
- **Paper section**: Results (nominal) / discussion (priority: medium)

## C8 -- Reviewer robust-baseline issue

- **Proposed claim**: The revised evaluation includes a frozen scenario-based robust MPC comparator operating under the same information and engagement constraints.
- **Evidence source**: docs/rmpc_final_nominal_protocol.md, docs/rmpc_targeted_robustness_protocol.md, results/rmpc_tuning/selected_rmpc_config.json
- **Point estimate**: n/a
- **Statistical status**: qualitative / reviewer response
- **Allowed wording**: The revised evaluation includes a frozen scenario-based robust MPC comparator operating under the same information and engagement constraints.
- **Prohibited stronger wording**: We added a provably optimal-control baseline.
- **Paper section**: Response to reviewers (priority: high)

