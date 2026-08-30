"""LeadResidualTrainingEnv <-> corrected LR-PPO seed-mapper wiring test
(item 13: smoke-test compatibility). Tiny reset/step integration checks
ONLY -- NOT a training campaign. Uses training root 55 for a single
reset/step (explicitly permitted by the task spec: "A tiny reset/step
integration check is sufficient"; only an ACTUAL PPO training campaign with
55/56/57 is prohibited), plus an ordinary dev root (12345) confirming the
old generic per-episode-seed mechanism is unaffected for non-protocol roots.

Run directly:
    python test_lead_residual_training_env_seed_wiring.py
"""

from __future__ import annotations

import numpy as np

from experiments.lead_residual_training_env import LeadResidualTrainingEnv
from experiments.lead_residual_training_protocol import (
    TRAINING_ROOTS,
    TRAINING_ROOT_NAMESPACE_BASE,
    TRAINING_NAMESPACE_SIZE,
    build_env_config,
)


def test_protocol_root_uses_corrected_namespace():
    for root in TRAINING_ROOTS:
        env = LeadResidualTrainingEnv(config=build_env_config(), root_seed=root)
        obs, info = env.reset()
        seed = env._inner.last_episode_seed
        base = TRAINING_ROOT_NAMESPACE_BASE[root]
        assert base <= seed < base + TRAINING_NAMESPACE_SIZE, (root, seed)
        assert obs.shape == (19,)

        action = np.zeros(3, dtype=np.float32)
        obs2, reward, term, trunc, info2 = env.step(action)
        assert obs2.shape == (19,)
        assert np.isfinite(reward)
        env.close()


def test_dev_root_unaffected_by_correction():
    # Ordinary development/smoke root (NOT 55/56/57): must continue using
    # the untouched old generic mechanism (large, non-namespaced values).
    env = LeadResidualTrainingEnv(config=build_env_config(), root_seed=12345)
    obs, info = env.reset()
    seed = env._inner.last_episode_seed
    for root in TRAINING_ROOTS:
        base = TRAINING_ROOT_NAMESPACE_BASE[root]
        assert not (base <= seed < base + TRAINING_NAMESPACE_SIZE)
    assert obs.shape == (19,)
    env.close()


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    tests.sort(key=lambda kv: kv[0])
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
