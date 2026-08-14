"""Compare PPO policy vs Random policy and visualize results."""
import numpy as np
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from stable_baselines3 import PPO
from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig


def run_episodes(env, n_episodes=50, policy="random", model=None, max_steps=500):
    """Run episodes with specified policy.
    
    Args:
        env: SoldierEnv
        n_episodes: Number of episodes
        policy: "random" or "ppo"
        model: PPO model (required if policy="ppo")
        max_steps: Max steps per episode
    
    Returns:
        outcomes dict, rewards list
    """
    outcomes = {"intercepted": 0, "soldier_caught": 0, "unsafe_intercept": 0, "timeout": 0}
    rewards = []
    
    for i in range(n_episodes):
        obs, info = env.reset()
        total_reward = 0.0
        
        for _ in range(max_steps):
            if policy == "ppo" and model is not None:
                action, _ = model.predict(obs, deterministic=True)
            else:
                action = env.action_space.sample()
            
            obs, reward, terminated, truncated, info = env.step(action)
            total_reward += reward
            
            if terminated or truncated:
                break
        
        outcome = info.get('outcome', 'unknown')
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        rewards.append(total_reward)
    
    return outcomes, rewards


def main():
    n_episodes = 100
    
    # Create environment
    config = EnvConfig()
    env = SoldierEnv(config=config)
    
    # Load PPO model
    print("Loading PPO model...")
    try:
        model = PPO.load("ppo_defender.zip")
        ppo_available = True
    except Exception as e:
        print(f"Could not load PPO model: {e}")
        ppo_available = False
        model = None
    
    # Evaluate Random Policy
    print(f"\nEvaluating RANDOM policy ({n_episodes} episodes)...")
    random_outcomes, random_rewards = run_episodes(env, n_episodes, policy="random")
    print(f"Random - Outcomes: {random_outcomes}")
    print(f"Random - Win rate: {random_outcomes['intercepted'] / n_episodes * 100:.1f}%")
    print(f"Random - Avg reward: {np.mean(random_rewards):.1f} ± {np.std(random_rewards):.1f}")
    
    # Evaluate PPO Policy
    if ppo_available:
        print(f"\nEvaluating PPO policy ({n_episodes} episodes)...")
        ppo_outcomes, ppo_rewards = run_episodes(env, n_episodes, policy="ppo", model=model)
        print(f"PPO - Outcomes: {ppo_outcomes}")
        print(f"PPO - Win rate: {ppo_outcomes['intercepted'] / n_episodes * 100:.1f}%")
        print(f"PPO - Avg reward: {np.mean(ppo_rewards):.1f} ± {np.std(ppo_rewards):.1f}")
    
    env.close()
    
    # Create comparison visualization
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # Plot 1: Win Rate Comparison
    ax1 = axes[0]
    policies = ['Random']
    win_rates = [random_outcomes['intercepted'] / n_episodes * 100]
    if ppo_available:
        policies.append('PPO')
        win_rates.append(ppo_outcomes['intercepted'] / n_episodes * 100)
    
    colors = ['#ff6b6b', '#4ecdc4']
    bars = ax1.bar(policies, win_rates, color=colors[:len(policies)], edgecolor='black', linewidth=2)
    ax1.set_ylabel('Win Rate (%)', fontsize=12)
    ax1.set_title('Interception Rate', fontsize=14, fontweight='bold')
    ax1.set_ylim(0, 100)
    for bar, rate in zip(bars, win_rates):
        ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2, 
                f'{rate:.1f}%', ha='center', va='bottom', fontsize=12, fontweight='bold')
    
    # Plot 2: Outcome Distribution
    ax2 = axes[1]
    outcome_labels = ['Intercepted', 'Soldier\nCaught', 'Unsafe\nIntercept', 'Timeout']
    x = np.arange(len(outcome_labels))
    width = 0.35
    
    random_counts = [random_outcomes['intercepted'], random_outcomes['soldier_caught'],
                     random_outcomes['unsafe_intercept'], random_outcomes['timeout']]
    ax2.bar(x - width/2, random_counts, width, label='Random', color='#ff6b6b', edgecolor='black')
    
    if ppo_available:
        ppo_counts = [ppo_outcomes['intercepted'], ppo_outcomes['soldier_caught'],
                      ppo_outcomes['unsafe_intercept'], ppo_outcomes['timeout']]
        ax2.bar(x + width/2, ppo_counts, width, label='PPO', color='#4ecdc4', edgecolor='black')
    
    ax2.set_ylabel('Episode Count', fontsize=12)
    ax2.set_title('Outcome Distribution', fontsize=14, fontweight='bold')
    ax2.set_xticks(x)
    ax2.set_xticklabels(outcome_labels)
    ax2.legend()
    ax2.set_ylim(0, n_episodes)
    
    # Plot 3: Reward Distribution
    ax3 = axes[2]
    data = [random_rewards]
    labels = ['Random']
    if ppo_available:
        data.append(ppo_rewards)
        labels.append('PPO')
    
    bp = ax3.boxplot(data, labels=labels, patch_artist=True)
    colors_box = ['#ff6b6b', '#4ecdc4']
    for patch, color in zip(bp['boxes'], colors_box):
        patch.set_facecolor(color)
    ax3.set_ylabel('Episode Reward', fontsize=12)
    ax3.set_title('Reward Distribution', fontsize=14, fontweight='bold')
    
    plt.tight_layout()
    plt.savefig('policy_comparison.png', dpi=150, bbox_inches='tight')
    print("\nSaved comparison plot to: policy_comparison.png")
    
    # Print summary
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"{'Policy':<10} {'Win Rate':<12} {'Avg Reward':<15} {'Soldier Caught':<15}")
    print("-" * 60)
    print(f"{'Random':<10} {random_outcomes['intercepted']/n_episodes*100:>6.1f}%     "
          f"{np.mean(random_rewards):>8.1f}       {random_outcomes['soldier_caught']:>6}")
    if ppo_available:
        print(f"{'PPO':<10} {ppo_outcomes['intercepted']/n_episodes*100:>6.1f}%     "
              f"{np.mean(ppo_rewards):>8.1f}       {ppo_outcomes['soldier_caught']:>6}")
    print("=" * 60)


if __name__ == "__main__":
    main()
