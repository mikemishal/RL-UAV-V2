"""LR-PPO architecture confirmation test -- IMPLEMENTATION ONLY (no training).

Constructs an SB3 PPO model against LeadResidualTrainingEnv (seed 99999, a
non-protocol development seed) WITHOUT calling model.learn(), to confirm
programmatically:
    - policy is MlpPolicy, net_arch pi=[64,64] vf=[64,64], activation Tanh
      (SAME architecture family as standalone PPO; no policy_kwargs override)
    - observation_space.shape == (19,), action_space.shape == (3,)
    - exact trainable parameter count (documented in
      experiments/lead_residual_training_protocol.py::EXPECTED_TOTAL_PARAMETERS)
    - hyperparameters match experiments/lead_residual_training_protocol.py::PPO_HYPERPARAMS
      exactly (same family/values as the corrected standalone PPO experiment)

Run directly:
    python test_lead_residual_ppo_architecture.py
"""

from __future__ import annotations

from stable_baselines3 import PPO
from stable_baselines3.common.vec_env import DummyVecEnv

from experiments.lead_residual_training_env import LeadResidualTrainingEnv
from experiments.lead_residual_training_protocol import (
    build_env_config,
    PPO_HYPERPARAMS,
    NET_ARCH,
    ACTIVATION_FN_NAME,
    RESIDUAL_MAX_ANGLE_DEG,
    EXPECTED_TOTAL_PARAMETERS,
)

_DEV_ROOT = 99999  # non-protocol development seed (protocol roots are 55/56/57)


def _build_model() -> PPO:
    def make_env():
        return LeadResidualTrainingEnv(config=build_env_config(), root_seed=_DEV_ROOT, residual_max_angle_deg=RESIDUAL_MAX_ANGLE_DEG)

    env = DummyVecEnv([make_env])
    return PPO(policy="MlpPolicy", env=env, seed=_DEV_ROOT, device="cpu", verbose=0, **PPO_HYPERPARAMS)


def test_observation_and_action_dims():
    model = _build_model()
    assert model.observation_space.shape == (19,)
    assert model.action_space.shape == (3,)
    model.env.close()


def test_network_architecture_matches_standalone_ppo_family():
    model = _build_model()
    policy = model.policy
    pi_layers = [str(m) for m in policy.mlp_extractor.policy_net]
    vf_layers = [str(m) for m in policy.mlp_extractor.value_net]
    assert pi_layers == [
        "Linear(in_features=19, out_features=64, bias=True)", "Tanh()",
        "Linear(in_features=64, out_features=64, bias=True)", "Tanh()",
    ]
    assert vf_layers == [
        "Linear(in_features=19, out_features=64, bias=True)", "Tanh()",
        "Linear(in_features=64, out_features=64, bias=True)", "Tanh()",
    ]
    assert NET_ARCH == dict(pi=[64, 64], vf=[64, 64])
    assert ACTIVATION_FN_NAME == "Tanh"
    model.env.close()


def test_exact_parameter_count():
    model = _build_model()
    n_params = sum(p.numel() for p in model.policy.parameters())
    assert n_params == EXPECTED_TOTAL_PARAMETERS == 11_143
    model.env.close()


def test_hyperparameters_match_frozen_protocol():
    model = _build_model()
    assert model.learning_rate == PPO_HYPERPARAMS["learning_rate"] == 3e-4
    assert model.gamma == PPO_HYPERPARAMS["gamma"] == 0.99
    assert model.batch_size == PPO_HYPERPARAMS["batch_size"] == 64
    assert model.n_steps == PPO_HYPERPARAMS["n_steps"] == 2048
    assert model.n_epochs == PPO_HYPERPARAMS["n_epochs"] == 10
    assert model.clip_range(1.0) == PPO_HYPERPARAMS["clip_range"] == 0.2
    assert model.ent_coef == PPO_HYPERPARAMS["ent_coef"] == 0.01
    assert model.vf_coef == PPO_HYPERPARAMS["vf_coef"] == 0.5
    assert model.max_grad_norm == PPO_HYPERPARAMS["max_grad_norm"] == 0.5
    assert model.gae_lambda == PPO_HYPERPARAMS["gae_lambda"] == 0.95
    model.env.close()


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
