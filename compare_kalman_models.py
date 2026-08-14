"""Compare PPO models: old (no Kalman) vs new (with Kalman estimation).

Usage:
    python compare_kalman_models.py

This script evaluates and compares:
1. ppo_defender_no_kalman.zip - Model trained without Kalman filter observation
2. ppo_defender.zip - Model trained with Kalman filter observation

Note: Run `python train_ppo.py` first to train the new model with Kalman changes.
"""

import numpy as np
from stable_baselines3 import PPO

from uav_defend import SoldierEnv
from uav_defend.config.env_config import EnvConfig


def evaluate_model(model_path: str, env: SoldierEnv, n_episodes: int = 100, seed: int = 42) -> dict:
    """Evaluate a PPO model on the environment.
    
    Returns:
        dict with win_rate, outcomes, rewards, tracking_errors, etc.
    """
    try:
        model = PPO.load(model_path)
    except Exception as e:
        print(f"  Error loading {model_path}: {e}")
        return None
    
    outcomes = {"intercepted": 0, "soldier_caught": 0, "unsafe_intercept": 0, "timeout": 0}
    total_rewards = []
    episode_lengths = []
    tracking_errors = []
    
    for ep in range(n_episodes):
        obs, info = env.reset(seed=seed + ep)
        done = False
        episode_reward = 0.0
        ep_tracking_errors = []
        
        while not done:
            action, _ = model.predict(obs, deterministic=True)
            obs, reward, terminated, truncated, info = env.step(action)
            episode_reward += reward
            done = terminated or truncated
            
            # Track Kalman tracking error
            if info.get("tracking_error") is not None:
                ep_tracking_errors.append(info["tracking_error"])
        
        outcomes[info["outcome"]] += 1
        total_rewards.append(episode_reward)
        episode_lengths.append(info["step_count"])
        
        if len(ep_tracking_errors) > 0:
            tracking_errors.append(np.mean(ep_tracking_errors))
    
    win_rate = outcomes["intercepted"] / n_episodes * 100
    
    return {
        "win_rate": win_rate,
        "outcomes": outcomes,
        "mean_reward": np.mean(total_rewards),
        "std_reward": np.std(total_rewards),
        "mean_length": np.mean(episode_lengths),
        "mean_tracking_error": np.mean(tracking_errors) if tracking_errors else None,
    }


def main():
    print("=" * 70)
    print("Comparison: PPO with vs without Kalman Filter Observation")
    print("=" * 70)
    
    # Current environment config
    config = EnvConfig()
    print(f"\nEnvironment Configuration:")
    print(f"  detection_radius: {config.detection_radius}")
    print(f"  v_s={config.v_s}, v_e={config.v_e}, v_d={config.v_d}")
    print(f"  intercept_radius: {config.intercept_radius}")
    print(f"  threat_radius: {config.threat_radius}")
    print(f"  unsafe_intercept_radius: {config.unsafe_intercept_radius}")
    
    # Create environment
    env = SoldierEnv(config=config)
    
    n_episodes = 100
    print(f"\nEvaluating each model over {n_episodes} episodes...")
    
    # Evaluate old model (no Kalman)
    print("\n" + "-" * 70)
    print("Model 1: ppo_defender_no_kalman.zip (trained WITHOUT Kalman estimation)")
    print("-" * 70)
    results_old = evaluate_model("ppo_defender_no_kalman.zip", env, n_episodes)
    
    if results_old:
        print(f"  Win rate: {results_old['win_rate']:.1f}%")
        print(f"  Outcomes: {results_old['outcomes']}")
        print(f"  Mean reward: {results_old['mean_reward']:.2f} ± {results_old['std_reward']:.2f}")
        print(f"  Mean episode length: {results_old['mean_length']:.1f}")
        if results_old['mean_tracking_error']:
            print(f"  Mean tracking error: {results_old['mean_tracking_error']:.3f}")
    
    # Evaluate new model (with Kalman)
    print("\n" + "-" * 70)
    print("Model 2: ppo_defender.zip (trained WITH Kalman estimation)")
    print("-" * 70)
    results_new = evaluate_model("ppo_defender.zip", env, n_episodes)
    
    if results_new:
        print(f"  Win rate: {results_new['win_rate']:.1f}%")
        print(f"  Outcomes: {results_new['outcomes']}")
        print(f"  Mean reward: {results_new['mean_reward']:.2f} ± {results_new['std_reward']:.2f}")
        print(f"  Mean episode length: {results_new['mean_length']:.1f}")
        if results_new['mean_tracking_error']:
            print(f"  Mean tracking error: {results_new['mean_tracking_error']:.3f}")
    
    # Summary comparison
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if results_old and results_new:
        print(f"\n{'Metric':<30} {'No Kalman':>15} {'With Kalman':>15} {'Δ':>10}")
        print("-" * 70)
        print(f"{'Win Rate (%)':.<30} {results_old['win_rate']:>15.1f} {results_new['win_rate']:>15.1f} {results_new['win_rate'] - results_old['win_rate']:>+10.1f}")
        print(f"{'Mean Reward':.<30} {results_old['mean_reward']:>15.2f} {results_new['mean_reward']:>15.2f} {results_new['mean_reward'] - results_old['mean_reward']:>+10.2f}")
        print(f"{'Mean Episode Length':.<30} {results_old['mean_length']:>15.1f} {results_new['mean_length']:>15.1f} {results_new['mean_length'] - results_old['mean_length']:>+10.1f}")
        
        if results_new['mean_tracking_error']:
            print(f"\nKalman Tracking Error (new model only): {results_new['mean_tracking_error']:.3f}")
    elif results_old:
        print("\nOnly old model available. Train new model with: python train_ppo.py")
    elif results_new:
        print("\nOnly new model available.")
    else:
        print("\nNo models found. Run training first.")
    
    print("\n" + "=" * 70)
    env.close()


if __name__ == "__main__":
    main()
