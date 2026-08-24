# RMPC Targeted Robustness Publication Evaluation Report

**FINAL RAW ROBUSTNESS EVIDENCE -- not yet integrated into the manuscript,
paper_artifacts/, or publication figures. The five conditions reuse the SAME
1000 seeds (77000-77999) and are NOT statistically independent of one another.**

## Family success-rate summary (%)

| Condition | Greedy | PN | Lead | RMPC | PPO | PPO-Kalman | LR-PPO |
|---|---|---|---|---|---|---|---|
| Nominal | 79.12 | 75.92 | 86.76 | 82.04 | 87.42 | 83.43 | 89.43 |
| Strong evasion | 78.70 | 76.30 | 87.70 | 80.90 | 84.83 | 82.67 | 89.60 |
| Restrictive dynamics | 61.20 | 58.90 | 65.60 | 67.20 | 56.10 | 59.00 | 67.63 |
| High measurement noise | 70.40 | 64.10 | 73.00 | 52.70 | 83.57 | 81.47 | 77.83 |
| Hard mobility | 52.80 | 32.10 | 49.70 | 69.30 | 40.90 | 49.00 | 58.00 |
| Short detection | 58.10 | 55.60 | 65.20 | 58.90 | 59.53 | 56.57 | 66.00 |

## Primary comparisons by condition

| Condition | Comparison | Diff (pp) | 95% CI (pp) | Resolution |
|---|---|---|---|---|
| Strong evasion | LR-PPO - Lead | 1.90 | [-0.33, 4.20] | unresolved |
| Strong evasion | LR-PPO - RMPC | 8.70 | [6.07, 11.30] | resolved positive |
| Strong evasion | LR-PPO - PPO | 4.77 | [1.47, 8.23] | resolved positive |
| Strong evasion | RMPC - Lead | -6.80 | [-9.60, -4.00] | resolved negative |
| Strong evasion | RMPC - PPO | -3.93 | [-7.67, -0.00] | unresolved |
| Restrictive dynamics | LR-PPO - Lead | 2.03 | [-2.20, 6.03] | unresolved |
| Restrictive dynamics | LR-PPO - RMPC | 0.43 | [-3.63, 4.40] | unresolved |
| Restrictive dynamics | LR-PPO - PPO | 11.53 | [3.10, 18.63] | resolved positive |
| Restrictive dynamics | RMPC - Lead | 1.60 | [-2.00, 5.10] | unresolved |
| Restrictive dynamics | RMPC - PPO | 11.10 | [2.90, 17.90] | resolved positive |
| High measurement noise | LR-PPO - Lead | 4.83 | [0.50, 8.70] | resolved positive |
| High measurement noise | LR-PPO - RMPC | 25.13 | [20.73, 29.37] | resolved positive |
| High measurement noise | LR-PPO - PPO | -5.73 | [-10.17, -1.30] | resolved negative |
| High measurement noise | RMPC - Lead | -20.30 | [-24.00, -16.60] | resolved negative |
| High measurement noise | RMPC - PPO | -30.87 | [-34.93, -26.53] | resolved negative |
| Hard mobility | LR-PPO - Lead | 8.30 | [4.67, 11.87] | resolved positive |
| Hard mobility | LR-PPO - RMPC | -11.30 | [-15.10, -7.60] | resolved negative |
| Hard mobility | LR-PPO - PPO | 17.10 | [12.47, 21.47] | resolved positive |
| Hard mobility | RMPC - Lead | 19.60 | [15.70, 23.50] | resolved positive |
| Hard mobility | RMPC - PPO | 28.40 | [23.97, 32.90] | resolved positive |
| Short detection | LR-PPO - Lead | 0.80 | [-1.13, 2.70] | unresolved |
| Short detection | LR-PPO - RMPC | 7.10 | [4.73, 9.47] | resolved positive |
| Short detection | LR-PPO - PPO | 6.47 | [1.77, 12.37] | resolved positive |
| Short detection | RMPC - Lead | -6.30 | [-8.80, -3.80] | resolved negative |
| Short detection | RMPC - PPO | -0.63 | [-5.60, 5.40] | unresolved |

## Instance results by condition

### Strong Reactive Evasion (`strong_evasion`)

| Instance | Success/n | Success % | 95% CI |
|---|---|---|---|
| greedy | 787/1000 | 78.70% | [76.06%, 81.13%] |
| pn | 763/1000 | 76.30% | [73.57%, 78.83%] |
| lead | 877/1000 | 87.70% | [85.52%, 89.59%] |
| ppo_seed45 | 851/1000 | 85.10% | [82.76%, 87.17%] |
| ppo_seed46 | 874/1000 | 87.40% | [85.20%, 89.31%] |
| ppo_seed47 | 820/1000 | 82.00% | [79.50%, 84.26%] |
| ppo_kalman_seed45 | 846/1000 | 84.60% | [82.23%, 86.70%] |
| ppo_kalman_seed46 | 806/1000 | 80.60% | [78.03%, 82.93%] |
| ppo_kalman_seed47 | 828/1000 | 82.80% | [80.34%, 85.01%] |
| lr_ppo_seed55 | 887/1000 | 88.70% | [86.59%, 90.52%] |
| lr_ppo_seed56 | 894/1000 | 89.40% | [87.34%, 91.16%] |
| lr_ppo_seed57 | 907/1000 | 90.70% | [88.74%, 92.35%] |
| rmpc | 809/1000 | 80.90% | [78.35%, 83.22%] |

### Restrictive Defender Maneuverability (`restrictive_dynamics`)

| Instance | Success/n | Success % | 95% CI |
|---|---|---|---|
| greedy | 612/1000 | 61.20% | [58.14%, 64.17%] |
| pn | 589/1000 | 58.90% | [55.82%, 61.91%] |
| lead | 656/1000 | 65.60% | [62.60%, 68.48%] |
| ppo_seed45 | 510/1000 | 51.00% | [47.90%, 54.09%] |
| ppo_seed46 | 525/1000 | 52.50% | [49.40%, 55.58%] |
| ppo_seed47 | 648/1000 | 64.80% | [61.79%, 67.70%] |
| ppo_kalman_seed45 | 607/1000 | 60.70% | [57.64%, 63.68%] |
| ppo_kalman_seed46 | 591/1000 | 59.10% | [56.02%, 62.11%] |
| ppo_kalman_seed47 | 572/1000 | 57.20% | [54.11%, 60.23%] |
| lr_ppo_seed55 | 699/1000 | 69.90% | [66.99%, 72.66%] |
| lr_ppo_seed56 | 685/1000 | 68.50% | [65.55%, 71.30%] |
| lr_ppo_seed57 | 645/1000 | 64.50% | [61.48%, 67.41%] |
| rmpc | 672/1000 | 67.20% | [64.23%, 70.04%] |

### High Measurement Noise (`high_measurement_noise`)

| Instance | Success/n | Success % | 95% CI |
|---|---|---|---|
| greedy | 704/1000 | 70.40% | [67.50%, 73.15%] |
| pn | 641/1000 | 64.10% | [61.08%, 67.01%] |
| lead | 730/1000 | 73.00% | [70.16%, 75.66%] |
| ppo_seed45 | 849/1000 | 84.90% | [82.55%, 86.99%] |
| ppo_seed46 | 854/1000 | 85.40% | [83.08%, 87.45%] |
| ppo_seed47 | 804/1000 | 80.40% | [77.83%, 82.74%] |
| ppo_kalman_seed45 | 832/1000 | 83.20% | [80.76%, 85.39%] |
| ppo_kalman_seed46 | 788/1000 | 78.80% | [76.16%, 81.22%] |
| ppo_kalman_seed47 | 824/1000 | 82.40% | [79.92%, 84.64%] |
| lr_ppo_seed55 | 802/1000 | 80.20% | [77.62%, 82.55%] |
| lr_ppo_seed56 | 745/1000 | 74.50% | [71.71%, 77.10%] |
| lr_ppo_seed57 | 788/1000 | 78.80% | [76.16%, 81.22%] |
| rmpc | 527/1000 | 52.70% | [49.60%, 55.78%] |

### Hard Mobility Condition (`hard_mobility`)

| Instance | Success/n | Success % | 95% CI |
|---|---|---|---|
| greedy | 528/1000 | 52.80% | [49.70%, 55.88%] |
| pn | 321/1000 | 32.10% | [29.28%, 35.06%] |
| lead | 497/1000 | 49.70% | [46.61%, 52.79%] |
| ppo_seed45 | 441/1000 | 44.10% | [41.05%, 47.19%] |
| ppo_seed46 | 405/1000 | 40.50% | [37.50%, 43.57%] |
| ppo_seed47 | 381/1000 | 38.10% | [35.14%, 41.15%] |
| ppo_kalman_seed45 | 456/1000 | 45.60% | [42.54%, 48.70%] |
| ppo_kalman_seed46 | 652/1000 | 65.20% | [62.19%, 68.09%] |
| ppo_kalman_seed47 | 362/1000 | 36.20% | [33.28%, 39.23%] |
| lr_ppo_seed55 | 584/1000 | 58.40% | [55.32%, 61.42%] |
| lr_ppo_seed56 | 594/1000 | 59.40% | [56.33%, 62.40%] |
| lr_ppo_seed57 | 562/1000 | 56.20% | [53.11%, 59.25%] |
| rmpc | 693/1000 | 69.30% | [66.37%, 72.08%] |

### Short Detection Range (`short_detection`)

| Instance | Success/n | Success % | 95% CI |
|---|---|---|---|
| greedy | 581/1000 | 58.10% | [55.02%, 61.12%] |
| pn | 556/1000 | 55.60% | [52.50%, 58.65%] |
| lead | 652/1000 | 65.20% | [62.19%, 68.09%] |
| ppo_seed45 | 618/1000 | 61.80% | [58.75%, 64.76%] |
| ppo_seed46 | 634/1000 | 63.40% | [60.37%, 66.33%] |
| ppo_seed47 | 534/1000 | 53.40% | [50.30%, 56.47%] |
| ppo_kalman_seed45 | 560/1000 | 56.00% | [52.91%, 59.05%] |
| ppo_kalman_seed46 | 551/1000 | 55.10% | [52.00%, 58.16%] |
| ppo_kalman_seed47 | 586/1000 | 58.60% | [55.52%, 61.61%] |
| lr_ppo_seed55 | 666/1000 | 66.60% | [63.62%, 69.45%] |
| lr_ppo_seed56 | 652/1000 | 65.20% | [62.19%, 68.09%] |
| lr_ppo_seed57 | 662/1000 | 66.20% | [63.21%, 69.06%] |
| rmpc | 589/1000 | 58.90% | [55.82%, 61.91%] |

