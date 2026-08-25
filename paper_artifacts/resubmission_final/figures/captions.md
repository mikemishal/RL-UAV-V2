# Figure captions (stored separately from the rendered figures)

## fig_nominal_performance
Final nominal safe-interception success rate (%) for all seven controller families on 5000 fresh matched environment seeds (72000-76999). Error bars show 95% intervals (Wilson for the four deterministic controllers; root-aware hierarchical bootstrap for the three learned families). LR-PPO (the proposed method) is outlined in black; RMPC is shown with the same visual weight as every other method.

## fig_robustness_summary
Safe-interception success rate (%) for all seven controller families across the nominal condition and five predeclared off-nominal conditions, all evaluated on the same reused 1000-seed block (77000-77999) per condition (nominal uses the independent 5000-seed block 72000-76999). The five off-nominal rows are NOT independent experiments -- they reuse identical environment seeds by design.
