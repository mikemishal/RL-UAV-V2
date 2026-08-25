# Statistical Summary -- Source Traceability

Every main estimate below is traced to its exact source file, row identity, seed range, and confidence-interval method. No value here was retyped from a prior Markdown report -- all were re-read from raw CSV/JSON artifacts.

## Nominal families (source: results/rmpc_final_nominal/family_summary.csv, seeds 72000-76999)

- **Greedy**: 79.12% [77.97, 80.22] (n_roots=1, CI method=wilson_interval)
- **PN**: 75.92% [74.72, 77.08] (n_roots=1, CI method=wilson_interval)
- **Lead**: 86.76% [85.79, 87.67] (n_roots=1, CI method=wilson_interval)
- **RMPC**: 82.04% [80.95, 83.08] (n_roots=1, CI method=wilson_interval)
- **PPO**: 87.42% [84.32, 89.62] (n_roots=3, CI method=hierarchical_bootstrap_track)
- **PPO-Kalman**: 83.43% [79.52, 86.23] (n_roots=3, CI method=hierarchical_bootstrap_track)
- **LR-PPO**: 89.43% [88.67, 90.17] (n_roots=3, CI method=hierarchical_bootstrap_track)

## Nominal differences (source: results/rmpc_final_nominal/paired_differences.csv)

- **LR-PPO - Lead**: +2.67 pp [1.77, 3.57] (resolved positive)
- **LR-PPO - RMPC**: +7.39 pp [6.31, 8.49] (resolved positive)
- **LR-PPO - PPO**: +2.01 pp [-0.27, 5.11] (unresolved)
- **RMPC - Lead**: -4.72 pp [-5.94, -3.52] (resolved negative)
- **RMPC - PPO**: -5.38 pp [-7.91, -2.22] (resolved negative)
- **PPO - Greedy**: +8.30 pp [5.22, 10.86] (resolved positive)
- **PPO - PN**: +11.50 pp [8.32, 14.09] (resolved positive)
- **PPO - Lead**: +0.66 pp [-2.44, 3.07] (unresolved)
- **PPO-Kalman - PPO**: -3.99 pp [-9.81, 1.70] (unresolved)

## Robustness matrix (source: results/rmpc_targeted_robustness/robustness_summary.csv, seeds 77000-77999 reused across all five conditions)

- **Nominal**: Greedy=79.12%, PN=75.92%, Lead=86.76%, RMPC=82.04%, PPO=87.42%, PPO-Kalman=83.43%, LR-PPO=89.43%
- **Strong evasion**: Greedy=78.70%, PN=76.30%, Lead=87.70%, RMPC=80.90%, PPO=84.83%, PPO-Kalman=82.67%, LR-PPO=89.60%
- **Restrictive dynamics**: Greedy=61.20%, PN=58.90%, Lead=65.60%, RMPC=67.20%, PPO=56.10%, PPO-Kalman=59.00%, LR-PPO=67.63%
- **High measurement noise**: Greedy=70.40%, PN=64.10%, Lead=73.00%, RMPC=52.70%, PPO=83.57%, PPO-Kalman=81.47%, LR-PPO=77.83%
- **Hard mobility**: Greedy=52.80%, PN=32.10%, Lead=49.70%, RMPC=69.30%, PPO=40.90%, PPO-Kalman=49.00%, LR-PPO=58.00%
- **Short detection**: Greedy=58.10%, PN=55.60%, Lead=65.20%, RMPC=58.90%, PPO=59.53%, PPO-Kalman=56.57%, LR-PPO=66.00%

## Robustness differences (source: results/rmpc_targeted_robustness/primary_differences_summary.csv)

### Strong evasion
- **LR-PPO - Lead**: +1.90 pp [-0.33, 4.20] (unresolved)
- **LR-PPO - RMPC**: +8.70 pp [6.07, 11.30] (resolved positive)
- **LR-PPO - PPO**: +4.77 pp [1.47, 8.23] (resolved positive)
- **RMPC - Lead**: -6.80 pp [-9.60, -4.00] (resolved negative)
- **RMPC - PPO**: -3.93 pp [-7.67, -0.00] (unresolved)

### Restrictive dynamics
- **LR-PPO - Lead**: +2.03 pp [-2.20, 6.03] (unresolved)
- **LR-PPO - RMPC**: +0.43 pp [-3.63, 4.40] (unresolved)
- **LR-PPO - PPO**: +11.53 pp [3.10, 18.63] (resolved positive)
- **RMPC - Lead**: +1.60 pp [-2.00, 5.10] (unresolved)
- **RMPC - PPO**: +11.10 pp [2.90, 17.90] (resolved positive)

### High measurement noise
- **LR-PPO - Lead**: +4.83 pp [0.50, 8.70] (resolved positive)
- **LR-PPO - RMPC**: +25.13 pp [20.73, 29.37] (resolved positive)
- **LR-PPO - PPO**: -5.73 pp [-10.17, -1.30] (resolved negative)
- **RMPC - Lead**: -20.30 pp [-24.00, -16.60] (resolved negative)
- **RMPC - PPO**: -30.87 pp [-34.93, -26.53] (resolved negative)

### Hard mobility
- **LR-PPO - Lead**: +8.30 pp [4.67, 11.87] (resolved positive)
- **LR-PPO - RMPC**: -11.30 pp [-15.10, -7.60] (resolved negative)
- **LR-PPO - PPO**: +17.10 pp [12.47, 21.47] (resolved positive)
- **RMPC - Lead**: +19.60 pp [15.70, 23.50] (resolved positive)
- **RMPC - PPO**: +28.40 pp [23.97, 32.90] (resolved positive)

### Short detection
- **LR-PPO - Lead**: +0.80 pp [-1.13, 2.70] (unresolved)
- **LR-PPO - RMPC**: +7.10 pp [4.73, 9.47] (resolved positive)
- **LR-PPO - PPO**: +6.47 pp [1.77, 12.37] (resolved positive)
- **RMPC - Lead**: -6.30 pp [-8.80, -3.80] (resolved negative)
- **RMPC - PPO**: -0.63 pp [-5.60, 5.40] (unresolved)

## Sign-orientation check

All differences above are computed and reported as `A - B` exactly as labeled (e.g. `LR-PPO - RMPC`, never silently `RMPC - LR-PPO`); orientation is preserved verbatim from `paired_differences.csv` / `primary_differences_summary.csv` with no re-derivation or sign manipulation in this summary.

