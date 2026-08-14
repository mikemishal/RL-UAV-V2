"""Test reward calculation matches specification."""
import numpy as np
from uav_defend import SoldierEnv

env = SoldierEnv()
obs, info = env.reset(seed=42)

print("=== Reward Calculation Verification ===")
print(f"Initial dist_de (defender->enemy): {info['defender_enemy_dist']:.2f}")
print()

# Test a few steps with pursuit action
print("Testing ongoing reward: reward = 5.0 * progress - 0.05")
print("-" * 60)
for i in range(5):
    # Pursuit action: move toward enemy
    direction = info['enemy_pos'] - info['defender_pos']
    norm = np.linalg.norm(direction)
    action = direction / norm if norm > 1e-8 else np.array([0.0, 0.0])
    
    prev_dist = info['defender_enemy_dist']
    obs, reward, term, trunc, info = env.step(action)
    curr_dist = info['defender_enemy_dist']
    progress = prev_dist - curr_dist
    
    expected_reward = 5.0 * progress - 0.05
    match = abs(reward - expected_reward) < 0.001
    print(f"Step {i+1}: progress={progress:.4f}, reward={reward:.4f}, expected={expected_reward:.4f}, match={match}")

print()
print("=== Terminal Rewards ===")

# Test intercept
obs, info = env.reset(seed=0)
step = 0
while True:
    direction = info['enemy_pos'] - info['defender_pos']
    action = direction / np.linalg.norm(direction)
    obs, reward, term, trunc, info = env.step(action)
    step += 1
    if term:
        print(f"Intercept: outcome={info['outcome']}, reward={reward}, expected=100.0")
        break

# Test soldier_caught (random policy)
obs, info = env.reset(seed=123)
step = 0
while True:
    action = env.action_space.sample()  # random
    obs, reward, term, trunc, info = env.step(action)
    step += 1
    if term:
        print(f"Caught: outcome={info['outcome']}, reward={reward}, expected=-100.0")
        break

print()
print("=== Summary ===")
print("Reward structure:")
print("  - Ongoing: reward = 5.0 * (prev_dist_de - curr_dist_de) - 0.05")
print("  - Intercept: +100.0")
print("  - Soldier caught: -100.0")
print("  - Unsafe intercept: -100.0")
print("  - Timeout: -100.0")
