"""Train PPO agent on SoldierEnv using Stable-Baselines3."""
import os
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.monitor import Monitor
from stable_baselines3.common.callbacks import BaseCallback

from uav_defend import SoldierEnv


class RewardLoggingCallback(BaseCallback):
    """Callback to print mean reward every n steps."""
    
    def __init__(self, print_freq: int = 1000, verbose: int = 0):
        super().__init__(verbose)
        self.print_freq = print_freq
        self.episode_rewards = []
        self.episode_lengths = []
    
    def _on_step(self) -> bool:
        # Check if episode finished
        if self.locals.get("dones") is not None:
            for idx, done in enumerate(self.locals["dones"]):
                if done:
                    info = self.locals["infos"][idx]
                    if "episode" in info:
                        self.episode_rewards.append(info["episode"]["r"])
                        self.episode_lengths.append(info["episode"]["l"])
        
        # Print every print_freq steps
        if self.n_calls % self.print_freq == 0:
            if len(self.episode_rewards) > 0:
                mean_reward = np.mean(self.episode_rewards[-100:])
                mean_length = np.mean(self.episode_lengths[-100:])
                print(f"Step {self.n_calls}: Mean reward (last 100 eps): {mean_reward:.2f}, "
                      f"Mean length: {mean_length:.1f}, Total episodes: {len(self.episode_rewards)}")
            else:
                print(f"Step {self.n_calls}: No episodes completed yet")
        
        return True


def main():
    # Deterministic seed
    SEED = 42
    np.random.seed(SEED)
    torch.manual_seed(SEED)
    
    # Force CPU
    device = "cpu"
    print(f"Using device: {device}")
    
    # Create environment wrapped with Monitor
    env = SoldierEnv()
    env = Monitor(env)
    env.reset(seed=SEED)
    
    print("=" * 60)
    print("Training PPO on SoldierEnv")
    print("=" * 60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Seed: {SEED}")
    print()
    
    # Create PPO model
    model = PPO(
        "MlpPolicy",
        env,
        verbose=1,
        seed=SEED,
        device=device,
    )
    
    # Create callback for logging
    callback = RewardLoggingCallback(print_freq=1000)
    
    # Train
    print("Starting training...")
    print("=" * 60)
    total_timesteps = 1_000_000
    model.learn(
        total_timesteps=total_timesteps,
        callback=callback,
        progress_bar=True,
    )
    
    # Save model
    model_path = "ppo_defender.zip"
    model.save(model_path)
    print()
    print("=" * 60)
    print(f"Training complete! Model saved to: {model_path}")
    print(f"Total episodes: {len(callback.episode_rewards)}")
    if len(callback.episode_rewards) > 0:
        print(f"Final mean reward (last 100 eps): {np.mean(callback.episode_rewards[-100:]):.2f}")
    print("=" * 60)
    
    env.close()


if __name__ == "__main__":
    main()
