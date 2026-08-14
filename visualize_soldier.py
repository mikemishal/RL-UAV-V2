"""Visualize soldier, defender, and enemy in 2D environment using matplotlib."""

import matplotlib.pyplot as plt
import matplotlib.animation as animation
import numpy as np
from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig


def run_episode(env, max_steps=2000, seed=None, policy="random"):
    """Run an episode and collect trajectories for all entities.
    
    Args:
        env: The SoldierEnv environment.
        max_steps: Maximum steps to run.
        seed: Random seed. If None, uses random seed for different paths each run.
        policy: Policy for defender:
            - "random": Random actions (baseline for RL comparison)
            - "pursuit": Simple pursuit heuristic (chase enemy directly)
    
    Returns:
        soldier_positions: Array of shape (T, 2)
        defender_positions: Array of shape (T, 2)
        enemy_positions: Array of shape (T, 2)
        rewards: Array of shape (T,) with per-step rewards
        total_reward: Total episode reward
        outcome: Episode outcome from environment ("intercepted", "soldier_caught", etc.)
    """
    soldier_positions = []
    defender_positions = []
    enemy_positions = []
    rewards = []
    
    obs, info = env.reset(seed=seed)
    soldier_positions.append(info['soldier_pos'].copy())
    defender_positions.append(info['defender_pos'].copy())
    enemy_positions.append(info['enemy_pos'].copy())
    
    total_reward = 0.0
    outcome = "ongoing"
    for _ in range(max_steps):
        # Select action based on policy
        if policy == "pursuit":
            # Simple pursuit heuristic: move toward enemy
            direction = info['enemy_pos'] - info['defender_pos']
            norm = np.linalg.norm(direction)
            if norm > 1e-8:
                action = direction / norm
            else:
                action = np.array([0.0, 0.0])
        else:
            # Random action (baseline for RL)
            action = env.action_space.sample()
        
        obs, reward, terminated, truncated, info = env.step(action)
        soldier_positions.append(info['soldier_pos'].copy())
        defender_positions.append(info['defender_pos'].copy())
        enemy_positions.append(info['enemy_pos'].copy())
        rewards.append(reward)
        total_reward += reward
        outcome = info.get('outcome', 'ongoing')
        if terminated or truncated:
            break
    
    return (np.array(soldier_positions), np.array(defender_positions), 
            np.array(enemy_positions), np.array(rewards), total_reward, outcome)


def visualize_trajectory(soldier_pos, defender_pos, enemy_pos, L=50.0, 
                         show_defender_trail=False, save_path=None):
    """
    Static plot of all entity trajectories.
    
    Args:
        soldier_pos: Array of shape (T, 2) with soldier positions.
        defender_pos: Array of shape (T, 2) with defender positions.
        enemy_pos: Array of shape (T, 2) with enemy positions.
        L: Domain half-size (domain is [-L, L]²).
        show_defender_trail: If True, show defender trail (overlaps with soldier in escort mode).
        save_path: If provided, save figure to this path.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up domain
    ax.set_xlim(-L * 1.05, L * 1.05)
    ax.set_ylim(-L * 1.05, L * 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Soldier, Defender & Enemy Trajectories')
    ax.grid(True, alpha=0.3)
    
    # Draw domain boundary
    rect = plt.Rectangle((-L, -L), 2*L, 2*L, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Plot soldier trajectory (blue)
    ax.plot(soldier_pos[:, 0], soldier_pos[:, 1], 'b-', alpha=0.5, linewidth=1, label='Soldier Path')
    
    # Plot enemy trajectory (red) - shows weaving pattern
    ax.plot(enemy_pos[:, 0], enemy_pos[:, 1], 'r-', alpha=0.5, linewidth=1, label='Enemy Path')
    
    # Optionally plot defender trajectory (off by default since it overlaps)
    if show_defender_trail:
        ax.plot(defender_pos[:, 0], defender_pos[:, 1], 'g--', alpha=0.5, linewidth=1, label='Defender Path')
    
    # Mark start positions
    ax.scatter(soldier_pos[0, 0], soldier_pos[0, 1], c='blue', s=120, zorder=5, 
               label='Soldier Start', marker='o', edgecolors='black')
    ax.scatter(defender_pos[0, 0], defender_pos[0, 1], c='green', s=100, zorder=6, 
               label='Defender Start', marker='^', edgecolors='black')
    ax.scatter(enemy_pos[0, 0], enemy_pos[0, 1], c='red', s=120, zorder=5, 
               label='Enemy Start', marker='X', edgecolors='black')
    
    # Mark end positions
    ax.scatter(soldier_pos[-1, 0], soldier_pos[-1, 1], c='blue', s=120, zorder=5, 
               marker='s', edgecolors='yellow', linewidths=2)
    ax.scatter(enemy_pos[-1, 0], enemy_pos[-1, 1], c='red', s=120, zorder=5, 
               marker='D', edgecolors='yellow', linewidths=2)
    
    ax.legend(loc='upper right')
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved to {save_path}")
    
    plt.show()


def animate_trajectory(soldier_pos, defender_pos, enemy_pos, L=50.0, interval=50, 
                       show_defender_trail=False, show_enemy_trail=True, save_path=None):
    """
    Animated visualization of all entities movement.
    
    Args:
        soldier_pos: Array of shape (T, 2) with soldier positions.
        defender_pos: Array of shape (T, 2) with defender positions.
        enemy_pos: Array of shape (T, 2) with enemy positions.
        L: Domain half-size.
        interval: Milliseconds between frames.
        show_defender_trail: If True, show defender trail (off by default).
        show_enemy_trail: If True, show enemy trail (on by default to show weaving).
        save_path: If provided, save animation to this path (e.g., 'simulation.gif').
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Set up domain
    ax.set_xlim(-L * 1.05, L * 1.05)
    ax.set_ylim(-L * 1.05, L * 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title('Soldier, Defender & Enemy Pursuit')
    ax.grid(True, alpha=0.3)
    
    # Draw domain boundary
    rect = plt.Rectangle((-L, -L), 2*L, 2*L, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Initialize plot elements - trails
    soldier_trail, = ax.plot([], [], 'b-', alpha=0.4, linewidth=1, label='Soldier Path')
    defender_trail, = ax.plot([], [], 'g--', alpha=0.3, linewidth=1, label='Defender Path')
    enemy_trail, = ax.plot([], [], 'r-', alpha=0.4, linewidth=1, label='Enemy Path')
    
    # Current position markers: soldier (circle), defender (triangle), enemy (X)
    soldier_marker, = ax.plot([], [], 'bo', markersize=14, label='Soldier', markeredgecolor='black')
    defender_marker, = ax.plot([], [], 'g^', markersize=12, label='Defender', markeredgecolor='black')
    enemy_marker, = ax.plot([], [], 'rX', markersize=14, label='Enemy', markeredgecolor='darkred')
    
    # Start markers
    ax.plot(soldier_pos[0, 0], soldier_pos[0, 1], 'ko', markersize=6, alpha=0.5)
    ax.plot(enemy_pos[0, 0], enemy_pos[0, 1], 'ko', markersize=6, alpha=0.5)
    
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
    
    # Distance text
    dist_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, verticalalignment='top',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    ax.legend(loc='upper right')
    
    def init():
        soldier_trail.set_data([], [])
        defender_trail.set_data([], [])
        enemy_trail.set_data([], [])
        soldier_marker.set_data([], [])
        defender_marker.set_data([], [])
        enemy_marker.set_data([], [])
        step_text.set_text('')
        dist_text.set_text('')
        return (soldier_trail, defender_trail, enemy_trail, 
                soldier_marker, defender_marker, enemy_marker, step_text, dist_text)
    
    def animate(frame):
        # Draw soldier trail
        soldier_trail.set_data(soldier_pos[:frame+1, 0], soldier_pos[:frame+1, 1])
        
        # Optionally draw defender trail
        if show_defender_trail:
            defender_trail.set_data(defender_pos[:frame+1, 0], defender_pos[:frame+1, 1])
        
        # Draw enemy trail (shows weaving pattern)
        if show_enemy_trail:
            enemy_trail.set_data(enemy_pos[:frame+1, 0], enemy_pos[:frame+1, 1])
        
        # Current positions
        soldier_marker.set_data([soldier_pos[frame, 0]], [soldier_pos[frame, 1]])
        defender_marker.set_data([defender_pos[frame, 0]], [defender_pos[frame, 1]])
        enemy_marker.set_data([enemy_pos[frame, 0]], [enemy_pos[frame, 1]])
        
        # Calculate distance between enemy and soldier
        dist = np.linalg.norm(soldier_pos[frame] - enemy_pos[frame])
        
        step_text.set_text(f'Step: {frame}\nSoldier: ({soldier_pos[frame, 0]:.1f}, {soldier_pos[frame, 1]:.1f})')
        dist_text.set_text(f'Enemy→Soldier: {dist:.1f}')
        
        return (soldier_trail, defender_trail, enemy_trail, 
                soldier_marker, defender_marker, enemy_marker, step_text, dist_text)
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(soldier_pos), interval=interval, blit=True
    )
    
    if save_path:
        anim.save(save_path, writer='pillow', fps=1000//interval)
        print(f"Animation saved to {save_path}")
    
    plt.show()
    
    return anim


def run_multiple_episodes(env, n_episodes=5, max_steps=500, policy="random"):
    """Run multiple episodes and collect all trajectories with rewards.
    
    Args:
        env: The SoldierEnv environment.
        n_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        policy: Policy for defender - "random" (default) or "pursuit".
    
    Returns:
        episodes: List of tuples (soldier_pos, defender_pos, enemy_pos, outcome, total_reward)
    """
    episodes = []
    outcomes = {"intercepted": 0, "soldier_caught": 0, "unsafe_intercept": 0, "timeout": 0}
    total_rewards = []
    
    for i in range(n_episodes):
        soldier_pos, defender_pos, enemy_pos, rewards, total_reward, outcome = run_episode(
            env, max_steps=max_steps, seed=None, policy=policy
        )
        
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_rewards.append(total_reward)
        episodes.append((soldier_pos, defender_pos, enemy_pos, outcome, total_reward))
        print(f"Episode {i+1}/{n_episodes}: {outcome} (steps={len(soldier_pos)}, reward={total_reward:.1f})")
    
    print(f"\nOutcome summary: {outcomes}")
    print(f"Average reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    return episodes


def animate_multiple_episodes(episodes, L=50.0, interval=50, pause_between=30,
                              show_defender_trail=True, show_enemy_trail=True, 
                              save_path=None, title="RL Environment Demo"):
    """
    Animated visualization of multiple episodes played sequentially.
    
    Args:
        episodes: List of (soldier_pos, defender_pos, enemy_pos, outcome, total_reward) tuples.
        L: Domain half-size.
        interval: Milliseconds between frames.
        pause_between: Frames to pause between episodes.
        show_defender_trail: If True, show defender trail.
        show_enemy_trail: If True, show enemy trail.
        save_path: If provided, save animation to this path.
        title: Title for the animation.
    """
    fig, ax = plt.subplots(figsize=(10, 10))
    
    # Flatten all episode frames with pauses
    all_frames = []
    for ep_idx, episode_data in enumerate(episodes):
        # Handle both old (4-tuple) and new (5-tuple) formats
        if len(episode_data) == 5:
            soldier_pos, defender_pos, enemy_pos, outcome, total_reward = episode_data
        else:
            soldier_pos, defender_pos, enemy_pos, outcome = episode_data
            total_reward = 0.0
        
        for frame_idx in range(len(soldier_pos)):
            all_frames.append((ep_idx, frame_idx, soldier_pos, defender_pos, enemy_pos, outcome, total_reward))
        # Add pause frames at end of episode
        for _ in range(pause_between):
            all_frames.append((ep_idx, len(soldier_pos)-1, soldier_pos, defender_pos, enemy_pos, outcome, total_reward))
    
    # Set up domain
    ax.set_xlim(-L * 1.05, L * 1.05)
    ax.set_ylim(-L * 1.05, L * 1.05)
    ax.set_aspect('equal')
    ax.set_xlabel('X')
    ax.set_ylabel('Y')
    ax.set_title(title)
    ax.grid(True, alpha=0.3)
    
    # Draw domain boundary
    rect = plt.Rectangle((-L, -L), 2*L, 2*L, fill=False, edgecolor='black', linewidth=2)
    ax.add_patch(rect)
    
    # Color scheme for outcomes
    outcome_colors = {
        "intercepted": "limegreen",
        "soldier_caught": "red", 
        "unsafe_intercept": "orange",
        "timeout": "gray",
        "unknown": "purple"
    }
    
    # Initialize plot elements
    soldier_trail, = ax.plot([], [], 'b-', alpha=0.4, linewidth=1.5)
    defender_trail, = ax.plot([], [], 'g--', alpha=0.5, linewidth=1.5)
    enemy_trail, = ax.plot([], [], 'r-', alpha=0.4, linewidth=1.5)
    
    soldier_marker, = ax.plot([], [], 'bo', markersize=14, markeredgecolor='black')
    defender_marker, = ax.plot([], [], 'g^', markersize=12, markeredgecolor='black')
    enemy_marker, = ax.plot([], [], 'rX', markersize=14, markeredgecolor='darkred')
    
    # Text displays
    episode_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                           fontsize=12, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.8))
    
    step_text = ax.text(0.02, 0.88, '', transform=ax.transAxes, verticalalignment='top',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.7))
    
    dist_text = ax.text(0.02, 0.78, '', transform=ax.transAxes, verticalalignment='top',
                        fontsize=9, bbox=dict(boxstyle='round', facecolor='lightcoral', alpha=0.7))
    
    outcome_text = ax.text(0.98, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                           horizontalalignment='right', fontsize=11, fontweight='bold',
                           bbox=dict(boxstyle='round', facecolor='white', alpha=0.8))
    
    # Legend (static)
    ax.plot([], [], 'bo', markersize=10, label='Soldier')
    ax.plot([], [], 'g^', markersize=10, label='Defender')
    ax.plot([], [], 'rX', markersize=10, label='Enemy')
    ax.legend(loc='lower right')
    
    # Add reward text display
    reward_text = ax.text(0.02, 0.68, '', transform=ax.transAxes, verticalalignment='top',
                          fontsize=10, bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.7))
    
    def init():
        soldier_trail.set_data([], [])
        defender_trail.set_data([], [])
        enemy_trail.set_data([], [])
        soldier_marker.set_data([], [])
        defender_marker.set_data([], [])
        enemy_marker.set_data([], [])
        episode_text.set_text('')
        step_text.set_text('')
        dist_text.set_text('')
        outcome_text.set_text('')
        reward_text.set_text('')
        return (soldier_trail, defender_trail, enemy_trail,
                soldier_marker, defender_marker, enemy_marker,
                episode_text, step_text, dist_text, outcome_text, reward_text)
    
    def animate(i):
        ep_idx, frame, soldier_pos, defender_pos, enemy_pos, outcome, total_reward = all_frames[i]
        
        # Update trails
        soldier_trail.set_data(soldier_pos[:frame+1, 0], soldier_pos[:frame+1, 1])
        if show_defender_trail:
            defender_trail.set_data(defender_pos[:frame+1, 0], defender_pos[:frame+1, 1])
        if show_enemy_trail:
            enemy_trail.set_data(enemy_pos[:frame+1, 0], enemy_pos[:frame+1, 1])
        
        # Update markers
        soldier_marker.set_data([soldier_pos[frame, 0]], [soldier_pos[frame, 1]])
        defender_marker.set_data([defender_pos[frame, 0]], [defender_pos[frame, 1]])
        enemy_marker.set_data([enemy_pos[frame, 0]], [enemy_pos[frame, 1]])
        
        # Calculate distances
        enemy_soldier_dist = np.linalg.norm(soldier_pos[frame] - enemy_pos[frame])
        defender_enemy_dist = np.linalg.norm(defender_pos[frame] - enemy_pos[frame])
        
        # Update text
        episode_text.set_text(f'Episode {ep_idx + 1}/{len(episodes)}')
        step_text.set_text(f'Step: {frame}/{len(soldier_pos)-1}')
        dist_text.set_text(f'Enemy→Soldier: {enemy_soldier_dist:.1f}\nDefender→Enemy: {defender_enemy_dist:.1f}')
        reward_text.set_text(f'Total Reward: {total_reward:.1f}')
        
        # Show outcome with color
        outcome_color = outcome_colors.get(outcome, "gray")
        outcome_text.set_text(outcome.upper().replace('_', ' '))
        outcome_text.set_bbox(dict(boxstyle='round', facecolor=outcome_color, alpha=0.7))
        
        return (soldier_trail, defender_trail, enemy_trail,
                soldier_marker, defender_marker, enemy_marker,
                episode_text, step_text, dist_text, outcome_text, reward_text)
    
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(all_frames), interval=interval, blit=True
    )
    
    if save_path:
        print(f"Saving animation with {len(all_frames)} frames...")
        if save_path.endswith('.mp4'):
            anim.save(save_path, writer='ffmpeg', fps=1000//interval)
        else:
            anim.save(save_path, writer='pillow', fps=1000//interval)
        print(f"Animation saved to {save_path}")
    
    plt.show()
    
    return anim


if __name__ == "__main__":
    # Create environment with default config (equal speeds: v_d = v_e = 3.0)
    config = EnvConfig()
    env = SoldierEnv(config=config)
    
    print("=" * 60)
    print("UAV Defend Environment - RL Ready")
    print("=" * 60)
    print(f"Observation space: {env.observation_space}")
    print(f"Action space: {env.action_space}")
    print(f"Defender speed: {config.v_d}, Enemy speed: {config.v_e}")
    print(f"Reward shaping: step={config.reward_time_penalty}, "
          f"progress_scale={config.reward_progress_scale}, "
          f"intercept={config.reward_intercept}, caught={config.reward_soldier_caught}")
    
    # Run random policy episodes (baseline - what RL needs to beat)
    print("\n" + "=" * 60)
    print("RANDOM POLICY (Baseline - RL needs to beat this)")
    print("=" * 60)
    random_episodes = run_multiple_episodes(env, n_episodes=10, max_steps=300, policy="random")
    
    # Run pursuit policy episodes (heuristic upper bound)
    print("\n" + "=" * 60)
    print("PURSUIT POLICY (Heuristic - simple strategy)")
    print("=" * 60)
    pursuit_episodes = run_multiple_episodes(env, n_episodes=10, max_steps=300, policy="pursuit")
    
    # Generate comparison video
    print("\n" + "=" * 60)
    print("Generating comparison video...")
    print("=" * 60)
    
    # Combine some episodes from each policy for video
    # Take 3 random + 3 pursuit episodes
    combined_episodes = random_episodes[:3] + pursuit_episodes[:3]
    
    animate_multiple_episodes(
        combined_episodes, 
        L=config.L, 
        interval=40, 
        pause_between=30,
        show_defender_trail=True, 
        show_enemy_trail=True,
        save_path='rl_environment_demo.mp4',
        title='UAV Defend: Random vs Pursuit Policy (v_d = v_e = 3.0)'
    )
    
    print("\nVideo saved: rl_environment_demo.mp4")
    print("First 3 episodes: RANDOM policy | Last 3 episodes: PURSUIT policy")
