# Reviewer Issue Matrix

| Field | Value |
|---|---|
| Issue | No robust optimal-control / MPC baseline |
| Prior status | Open |
| Revision | Added an independently tuned, frozen scenario-based robust MPC (RMPC) baseline. |
| Evidence | Nominal: 5000 matched seeds (72000-76999). Robustness: five off-nominal conditions x 1000 matched seeds (77000-77999), same information contract, same pre-detection protocol, no privileged true hostile state. |
| Nominal result | LR-PPO - RMPC = +7.39 pp [6.31, 8.49] (resolved positive); RMPC - Lead = -4.72 pp (resolved negative) |
| Status | ADDRESSED |

## Safe response

"We added a scenario-based robust MPC comparator based on finite-horizon CEM
optimization over bounded target-motion scenarios. The controller was tuned
on development/validation seeds, frozen, and subsequently evaluated on
independent nominal and robustness seed blocks."

## Mandatory qualification

RMPC is not globally optimal and is not presented as a complete
differential-game solution. Its relative standing is condition-dependent
(see `paper_artifacts/resubmission_final/tables/robustness_success.csv`).

If an existing reviewer-action document exists elsewhere in the repository,
this record should be cross-referenced from it rather than treated as a
contradictory or duplicate reviewer record.
