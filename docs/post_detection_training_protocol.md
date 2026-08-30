# Post-Detection PPO Training Protocol

This document records the frozen protocol for the REVISED PPO / PPO-Kalman training
campaign, consistent with the corrected canonical engagement
(`defender_standby_until_detection=True`, see `docs/revised_predetection_protocol.md`).
**No full 400k training run has been executed yet** -- this is the implementation,
testing, and small dry-run/smoke-test phase only.

## Protocol manifest template

```json
{
  "protocol_name": "post_detection_v1",
  "source_commit": "<git rev-parse HEAD at training launch time>",
  "defender_standby_until_detection": true,
  "training_roots": [45, 46, 47],
  "old_training_42_43_44_status": "legacy_superseded_for_revised_paper",
  "validation_seeds": "40000-40099",
  "diagnostic_reserved_seeds": "40100-40299",
  "final_nominal_reserved_seeds": "41000-45999",
  "future_robustness_reserved_seeds": ">=46000",
  "ppo_hyperparameters": {
    "learning_rate": 3e-4, "gamma": 0.99, "batch_size": 64, "n_steps": 2048,
    "n_epochs": 10, "clip_range": 0.2, "ent_coef": 0.01, "vf_coef": 0.5,
    "max_grad_norm": 0.5, "gae_lambda": 0.95
  },
  "network_architecture": {"policy": "MlpPolicy", "pi": [64, 64], "vf": [64, 64],
                             "activation": "Tanh", "total_parameters": 10759},
  "validation_frequency_timesteps": 25000,
  "checkpoint_selection_rule": "highest validation success rate; tie -> earliest checkpoint"
}
```

This template is also generated per-run as `training_manifest.json` by
`experiments/train_post_detection_ppo.py`.

## Training-step definition (differs from the legacy campaign)

One PPO "timestep" = one **post-detection, PPO-controlled** environment step, executed
via `PostDetectionTrainingEnv.step()`. Standby fast-forward steps performed inside
`PostDetectionTrainingEnv.reset()` are never counted toward `model.num_timesteps` or the
training budget (verified in `test_post_detection_training_env.py`, test U). Consequently
"400,000 timesteps" under this protocol means 400,000 PPO-controlled post-detection
steps -- a different quantity than the legacy campaign's "400,000 timesteps" (in which
every step from episode start was policy-controlled, since standby did not exist as a
concept).

## Reserved seed ranges

| Range | Purpose | Status |
|---|---|---|
| 45, 46, 47 | Training roots (paired Direct/Kalman) | to be opened by full training |
| 40000-40099 | In-training validation / model selection | reserved, unopened |
| 40100-40299 | Post-training diagnostic holdout | reserved, unopened |
| 41000-45999 | Final paper Monte Carlo evaluation | reserved, untouched |
| >=46000 | Future corrected robustness studies | reserved, untouched |
| 12345 (example) | Development / smoke-test only | non-protocol, freely reusable |

The legacy protocol's seeds (42/43/44 training, 10000-10099 validation, 12000-12199
diagnostic, 20000+ final test) belong to `experiments/training_protocol.py` and are
**not reused** by this module.

## Legacy artifact status

- `old_training_42_43_44_status = "legacy_superseded_for_revised_paper"`: the frozen PPO
  / PPO-Kalman checkpoints trained under seeds 42/43/44 (`results/training_400k/`) were
  trained under the OLD policy-controlled pre-detection behavior and do not reflect the
  revised canonical engagement. They remain untouched historical artifacts.
- **Acquisition-ablation finding is historical/superseded**: the previous finding ("Full
  PPO learns pre-acquisition positioning", `results/acquisition_ablation/`) belongs
  entirely to the OLD protocol, in which PPO could independently control the defender
  before hostile detection. Under the revised canonical engagement, PPO NEVER controls
  the defender before detection at all (standby is enforced by the environment,
  identically for every controller) -- so this finding, and its supporting ablation
  mechanism (`experiments/policy_wrappers.py::EscortUntilDetectionPolicy`), is **not
  valid evidence for the revised paper** and must not be re-cited without this caveat.
  `results/acquisition_ablation/` is retained unmodified as a historical record.
  `results/paper_evidence/` is NOT rebuilt by this task.

## Model-selection protocol (frozen before training begins)

- **Metric**: safe interception success rate only (never mean reward, training loss,
  tracking RMSE, mean detection time, or intercept time).
- **Validation environment**: the FULL canonical `SoldierEnv`
  (`defender_standby_until_detection=True`), evaluated via the shared, unmodified
  `experiments/success_rate_eval_callback.py::SuccessRateEvalCallback` -- **not** the
  `PostDetectionTrainingEnv` wrapper. `test_post_detection_training_env.py` test J
  proves the wrapper's episode outcome matches a full-canonical-mission continuation
  given identical post-detection actions, so this substitution is justified but the
  actual validation always uses the full mission regardless.
- **Frequency**: every 25,000 PPO timesteps (100 episodes, seeds 40000-40099).
- **Tie-break rule**: (1) highest validation success rate; (2) if exactly tied, the
  earliest checkpoint wins. Implemented automatically by
  `SuccessRateEvalCallback`'s strict `>` (never `>=`) best-model update condition.

## Smoke test (development only, never a paper model)

`experiments/train_post_detection_ppo.py --smoke-test` requires a non-protocol
`--training-seed` (rejects 45/46/47) and writes a `SMOKE_TEST_NOT_PAPER_MODEL.txt`
marker file alongside any saved model. Smoke runs deliberately use the DEFAULT
`eval_freq=25000`, so with a small `--total-timesteps` (e.g. 4096-8192) the validation
callback never triggers -- this is intentional: triggering it would consume the reserved
validation seeds (40000-40099), which must remain unopened until full training begins.
