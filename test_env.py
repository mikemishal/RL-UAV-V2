"""Test script for SoldierEnv with RL-ready action space."""
import numpy as np
from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig

env = SoldierEnv()
obs, info = env.reset(seed=42)
print("=== Environment Info ===")
print(f"Observation space: {env.observation_space}")
print(f"  Shape: {env.observation_space.shape}")
print(f"  Low: {env.observation_space.low[0]}, High: {env.observation_space.high[0]}")
print(f"Action space: {env.action_space}")
print(f"\nInitial obs (normalized): {obs}")
print(f"Soldier: {info['soldier_pos']}")
print(f"Defender: {info['defender_pos']}")
print(f"Enemy: {info['enemy_pos']}")
print(f"v_d = {env.config.v_d}, v_e = {env.config.v_e} (equal speeds)")

print("\n=== Running 5 steps with random actions ===")
total_reward = 0
for i in range(5):
    # Sample random action (2D heading)
    action = env.action_space.sample()
    obs, r, term, trunc, info = env.step(action)
    total_reward += r
    dist_es = info['enemy_soldier_dist']
    dist_de = info['defender_enemy_dist']
    print(f"Step {i+1}: action=[{action[0]:.2f}, {action[1]:.2f}], reward={r:.3f}, e→s={dist_es:.1f}, d→e={dist_de:.1f}")
print(f"Total reward (5 steps): {total_reward:.3f}")

print("\n=== Running full episode with random policy ===")
obs, info = env.reset(seed=123)
total_reward = 0
step = 0
while True:
    action = env.action_space.sample()
    obs, reward, term, trunc, info = env.step(action)
    total_reward += reward
    step += 1
    if term or trunc:
        break
print(f"Episode ended at step {step}")
print(f"Outcome: {info['outcome']}")
print(f"Total reward: {total_reward:.2f}")

print("\n=== Multiple random episodes (stats) ===")
outcomes = {"intercepted": 0, "soldier_caught": 0, "timeout": 0, "unsafe_intercept": 0}
for ep in range(20):
    obs, info = env.reset()
    while True:
        action = env.action_space.sample()
        obs, r, term, trunc, info = env.step(action)
        if term or trunc:
            outcomes[info['outcome']] = outcomes.get(info['outcome'], 0) + 1
            break
print(f"Random policy outcomes (20 episodes): {outcomes}")
