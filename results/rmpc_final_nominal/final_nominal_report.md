# RMPC Final Nominal Publication Evaluation Report

**FINAL RAW NOMINAL EVIDENCE -- not yet integrated into the manuscript,
paper_artifacts/, or publication figures.**

## Instance results

| Instance | Success/n | Success % | 95% CI |
|---|---|---|---|
| greedy | 3956/5000 | 79.12% | [77.97%, 80.22%] |
| pn | 3796/5000 | 75.92% | [74.72%, 77.08%] |
| lead | 4338/5000 | 86.76% | [85.79%, 87.67%] |
| ppo_seed45 | 4443/5000 | 88.86% | [87.96%, 89.70%] |
| ppo_seed46 | 4465/5000 | 89.30% | [88.41%, 90.13%] |
| ppo_seed47 | 4205/5000 | 84.10% | [83.06%, 85.09%] |
| ppo_kalman_seed45 | 4242/5000 | 84.84% | [83.82%, 85.81%] |
| ppo_kalman_seed46 | 3968/5000 | 79.36% | [78.22%, 80.46%] |
| ppo_kalman_seed47 | 4304/5000 | 86.08% | [85.09%, 87.01%] |
| lr_ppo_seed55 | 4479/5000 | 89.58% | [88.70%, 90.40%] |
| lr_ppo_seed56 | 4469/5000 | 89.38% | [88.50%, 90.20%] |
| lr_ppo_seed57 | 4467/5000 | 89.34% | [88.45%, 90.17%] |
| rmpc | 4102/5000 | 82.04% | [80.95%, 83.08%] |

## Family results

| Family | Success % | Root SD | 95% interval |
|---|---|---|---|
| Greedy | 79.12% | 0.00pp | [77.97%, 80.22%] |
| PN | 75.92% | 0.00pp | [74.72%, 77.08%] |
| Lead | 86.76% | 0.00pp | [85.79%, 87.67%] |
| RMPC | 82.04% | 0.00pp | [80.95%, 83.08%] |
| PPO | 87.42% | 2.88pp | [84.32%, 89.62%] |
| PPO-Kalman | 83.43% | 3.58pp | [79.52%, 86.23%] |
| LR-PPO | 89.43% | 0.13pp | [88.67%, 90.17%] |

## Paired differences

| Comparison | Diff (pp) | 95% CI (pp) | Resolution |
|---|---|---|---|
| LR-PPO - Lead | 2.67 | [1.77, 3.57] | resolved positive |
| LR-PPO - RMPC | 7.39 | [6.31, 8.49] | resolved positive |
| LR-PPO - PPO | 2.01 | [-0.27, 5.11] | unresolved |
| RMPC - Lead | -4.72 | [-5.94, -3.52] | resolved negative |
| RMPC - PPO | -5.38 | [-7.91, -2.22] | resolved negative |
| PPO - Greedy | 8.30 | [5.22, 10.86] | resolved positive |
| PPO - PN | 11.50 | [8.32, 14.09] | resolved positive |
| PPO - Lead | 0.66 | [-2.44, 3.07] | unresolved |
| PPO-Kalman - PPO | -3.99 | [-9.81, 1.70] | unresolved |
