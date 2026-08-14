"""Test detection behavior in partial observability environment."""
from uav_defend import SoldierEnv

env = SoldierEnv()
obs, info = env.reset()

print('=== Initial State ===')
print(f'Obs shape: {obs.shape}')
print(f'Obs: {obs}')
print(f'Detection flag (obs[4]): {obs[4]}')
print(f'Enemy detected (info): {info["enemy_detected"]}')
print(f'Enemy pos: {info["enemy_pos"]}')
print(f'Defender-enemy dist: {info["defender_enemy_dist"]:.1f}')
print(f'Detection radius: {env.config.detection_radius}')

# Run until detection
for step in range(200):
    obs, reward, term, trunc, info = env.step(env.action_space.sample())
    if info['enemy_detected']:
        print(f'\n=== Detection at step {step+1} ===')
        print(f'Obs: {obs}')
        print(f'Detection flag (obs[4]): {obs[4]}')
        print(f'Enemy info now visible in obs[5:9]: {obs[5:9]}')
        print(f'Defender-enemy dist: {info["defender_enemy_dist"]:.1f}')
        break
    if term:
        print(f'\nEpisode ended at step {step+1} without detection: {info["outcome"]}')
        break

print('\n=== Done ===')
