"""Quick evaluation of both models with 10 episodes each."""
from uav_defend.envs.soldier_env import SoldierEnv
from uav_defend.config.env_config import EnvConfig
from stable_baselines3 import PPO

def evaluate(model_path, use_kalman, detection_radius, n_episodes=10):
    config = EnvConfig()
    config.detection_radius = detection_radius
    env = SoldierEnv(config, use_kalman_obs=use_kalman)
    model = PPO.load(model_path)
    
    wins = 0
    results = []
    for _ in range(n_episodes):
        obs, _ = env.reset()
        done = False
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, r, done, trunc, info = env.step(action)
            if done or trunc:
                result = info.get('outcome', 'unknown')
                results.append(result)
                if result == 'intercepted':
                    wins += 1
                break
    return wins, results

print("=" * 50)
print("Evaluation Results (10 episodes each)")
print("=" * 50)

# Evaluate No Kalman Model
wins1, results1 = evaluate('ppo_defender_no_kalman.zip', use_kalman=False, detection_radius=30.0)
print(f"No Kalman (r=30): {wins1}/10 wins ({100*wins1/10:.0f}%)")
print(f"  Results: {results1}")

# Evaluate Kalman Model
wins2, results2 = evaluate('ppo_defender.zip', use_kalman=True, detection_radius=15.0)
print(f"Kalman (r=15):    {wins2}/10 wins ({100*wins2/10:.0f}%)")
print(f"  Results: {results2}")

print("=" * 50)
