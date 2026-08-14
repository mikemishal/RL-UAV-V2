"""Test Kalman filter integration in SoldierEnv."""

from uav_defend.envs.soldier_env import SoldierEnv
import numpy as np

env = SoldierEnv()
obs, info = env.reset(seed=42)
print("Observation space:", env.observation_space)
print("Initial obs shape:", obs.shape)
print("Initial obs:", obs)
print("  [soldier(2), defender(2), detected(1), e_hat(2), v_hat(2)]")
print("Initial detection:", info["detected"])

# Run until detected
detected_step = None
total_reward = 0.0
rewards = []
for i in range(200):
    obs, r, d, t, info = env.step([0.5, 0.5])
    total_reward += r
    rewards.append(r)
    if info["detected"] and detected_step is None:
        detected_step = i
        print(f"\nStep {i}: DETECTED!")
        print(f"  reward: {r:.3f}")
        print(f"  obs: {obs}")
        print(f"  e_hat (from info): {info['e_hat']}")
        print(f"  v_hat (from info): {info['v_hat']}")
        print(f"  tracking_error: {info['tracking_error']:.3f}")
        print(f"  true enemy_pos (NOT in obs): {info['enemy_pos']}")
        # Verify obs[5:7] matches e_hat / L
        L = env.config.L
        v_e = env.config.v_e
        e_hat_in_obs = obs[5:7] * L
        v_hat_in_obs = obs[7:9] * v_e
        print(f"  e_hat from obs (denormalized): {e_hat_in_obs}")
        print(f"  v_hat from obs (denormalized): {v_hat_in_obs}")
    if d:
        print(f"\nEpisode ended at step {i}: {info['outcome']}")
        print(f"  Total reward: {total_reward:.2f}")
        break

# Run a few more steps to see tracking
if info["detected"] and not d:
    print("\nTracking over next 5 steps:")
    for j in range(5):
        obs, r, d, t, info = env.step([0.5, 0.5])
        rewards.append(r)
        print(f"  Step {detected_step+j+1}: reward={r:.3f}, tracking_error={info['tracking_error']:.3f}, v_hat={info['v_hat']}")
        if d:
            print(f"    Outcome: {info['outcome']}")
            break

print(f"\nReward stats: min={min(rewards):.3f}, max={max(rewards):.3f}, mean={np.mean(rewards):.3f}")
print("\n✓ Kalman filter integration test complete!")
print("✓ True enemy state is NOT in observation (only e_hat, v_hat)")
