# Revised Pre-Detection Engagement Protocol

## Summary

The environment's pre-detection engagement semantics have been revised to match the
intended problem statement. This document describes the OLD and REVISED behavior and
why the change was made. It does **not** alter the manuscript.

## OLD experimental behavior (legacy, `defender_standby_until_detection=False`)

The defender started co-located with the soldier (`d0 = s0`) but was **independently,
continuously controlled** by the external policy/PPO action from the very first step of
the episode -- including before the hostile UAV was ever detected. This meant every
controller (Greedy, PN, Lead, PPO, PPO-Kalman) could exercise learned or scripted
pre-acquisition search/positioning behavior, and PPO in particular could learn to
optimize its position before it had any sensor information about the hostile.

## REVISED canonical behavior (`defender_standby_until_detection=True`, the new default)

1. The protected soldier continues to perform the **existing** stochastic planar random
   walk (`SoldierEnv._move_soldier()`), completely unchanged.
2. The defender starts co-located with the soldier (unchanged).
3. **Before hostile detection**, the defender is held in a fixed standby/escort posture:
   exactly co-located with the soldier's current position, with zero velocity.
4. The externally supplied controller/PPO action has **no effect** during this standby
   phase -- it is silently discarded.
5. When hostile detection first occurs, the defender is still exactly co-located with
   the soldier (this is a direct consequence of item 3).
6. Beginning with the **next** `step()` call after detection was already true at the
   start of that step, the selected interception controller takes over completely.
7. From that point on, the defender follows the existing constrained 3-D point-mass
   dynamics (`_advance_velocity` / `_move_defender`), completely unchanged.

This applies identically to every controller using `SoldierEnv` -- there is no
per-controller special-casing; the standby/handoff logic lives entirely in
`SoldierEnv.step()`.

## Discrete-time handoff (exact step order)

Implemented in `SoldierEnv.step()`:

```
detected_at_step_start = self._enemy_detected      # BEFORE this step's mutations

A. _move_soldier()                                  # existing random walk, unchanged
B. if standby and not detected_at_step_start:
       defender_pos = soldier_pos.copy()             # exact co-location
       defender_vel = [0, 0, 0]
C. _move_enemy()                                     # sees the CURRENT (possibly just-synced) defender_pos
D. _update_detection()                                # may flip enemy_detected False -> True
   just_detected = (not detected_at_step_start) and self._enemy_detected
E. if standby:
       if detected_at_step_start:
           _move_defender(action)                     # controller executes
           controller_action_executed = True
       else:
           # standby / detection-transition step: action discarded
           controller_action_executed = False
   else:  # legacy mode
       _move_defender(action)
       controller_action_executed = True
```

Consequently:

- The **first observation with `detected_flag=1`** is returned with the defender still
  exactly co-located with the soldier (zero velocity) -- the transition step's supplied
  action is never executed, even though detection became true during that same step.
- The controller's **first executed interception action** occurs on the very next
  `step()` call, starting from `defender_vel = [0,0,0]` at the detection-time location,
  subject to the unchanged `defender_max_accel` / `defender_max_turn_rate_deg` /
  `defender_max_climb_rate` / `defender_max_descent_rate` / `v_d` limits.

## What did NOT change

- Soldier random-walk model (`sigma = v_s * dt`, Gaussian, `z = 0`).
- Hostile pursuit + weave + heading-noise + reactive-evasion model (`_move_enemy`,
  `_compute_enemy_evasion`) -- evasion remains independent of defender detection state
  and may still activate before detection (since `enemy_evasion_radius=20m >
  detection_radius=15m` by default); during standby this is simply evaluated relative to
  the co-located defender position.
- Sensing/measurement-noise model, Kalman filter, `use_kalman_tracking` semantics.
- Detection rule: still `||enemy_pos - defender_pos|| <= detection_radius` (true 3-D
  distance) -- unchanged; during standby this is equal to
  `||enemy_pos - soldier_pos||` as a direct consequence of co-location, not a separate
  soldier-centered rule.
- Post-detection defender dynamics (`_advance_velocity`, `_move_defender`), action
  semantics (`desired_velocity = v_d * normalize(action)`), reward coefficients,
  terminal-condition definitions/priority, and observation contract (16-D layout).

## New `EnvConfig` flag

```python
defender_standby_until_detection: bool = True
```

- `True` (default): revised canonical engagement semantics described above.
- `False`: legacy mode, retained for historical reproducibility / diagnostics (e.g. the
  existing acquisition-ablation study, which uses policy-side escort behavior under the
  original environment via `experiments/policy_wrappers.py::EscortUntilDetectionPolicy`).
  `EscortUntilDetectionPolicy` is now explicitly documented as a historical/ablation
  construct, not the canonical mechanism.

## New `info` diagnostics

- `info["defender_standby"]`: True while the current transition used standby behavior.
- `info["controller_action_executed"]`: True iff the supplied action was actually
  applied to the defender's dynamics this transition.
- `info["just_detected"]`: True only on the transition where detection changed
  False -> True this step.

## Implication for existing PPO checkpoints

The existing frozen PPO / PPO-Kalman checkpoints (`results/training_400k/...`) were
trained under the **OLD** policy-controlled pre-detection behavior. Under the revised
canonical environment, their pre-detection state distribution at the moment of handoff
differs from what they were trained on (they never learned a "launch from zero velocity,
co-located with a randomly-walked soldier" initial condition in the same way). **They
should be considered legacy artifacts for the new canonical protocol.** Retraining is
recommended before generating any new publication results under
`defender_standby_until_detection=True`. This document does not perform that retraining.

## Historical results

None of the following were modified, deleted, or overwritten by this change:

- `results/final_nominal/`
- `results/acquisition_ablation/`
- `results/robustness/`
- `results/paper_evidence/`
- `results/paper_figures/`

They remain valid records of the previous (legacy pre-detection) experimental protocol.
A new result namespace will be created for any future evaluation under the revised
protocol.
