"""Evaluate trained PPO model and generate video of episodes."""
import numpy as np
import cv2
from stable_baselines3 import PPO
from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def run_episode_with_model(env, model, max_steps=500, deterministic=True):
    """Run a single episode using the trained model.
    
    Args:
        env: The SoldierEnv environment.
        model: Trained PPO model.
        max_steps: Maximum steps per episode.
        deterministic: If True, use deterministic actions.
    
    Returns:
        soldier_pos, defender_pos, enemy_pos, outcome, total_reward
    """
    soldier_positions = []
    defender_positions = []
    enemy_positions = []
    
    obs, info = env.reset()
    soldier_positions.append(info['soldier_pos'].copy())
    defender_positions.append(info['defender_pos'].copy())
    enemy_positions.append(info['enemy_pos'].copy())
    
    total_reward = 0.0
    for _ in range(max_steps):
        # Get action from trained model
        action, _ = model.predict(obs, deterministic=deterministic)
        
        obs, reward, terminated, truncated, info = env.step(action)
        soldier_positions.append(info['soldier_pos'].copy())
        defender_positions.append(info['defender_pos'].copy())
        enemy_positions.append(info['enemy_pos'].copy())
        total_reward += reward
        
        if terminated or truncated:
            break
    
    outcome = info.get('outcome', 'unknown')
    
    return (np.array(soldier_positions), np.array(defender_positions),
            np.array(enemy_positions), outcome, total_reward)


def evaluate_model(model_path, n_episodes=10, max_steps=500, deterministic=True):
    """Evaluate trained model and collect episodes.
    
    Args:
        model_path: Path to saved model.
        n_episodes: Number of episodes to run.
        max_steps: Maximum steps per episode.
        deterministic: If True, use deterministic actions.
    
    Returns:
        episodes: List of episode data tuples.
    """
    # Load model
    print(f"Loading model from: {model_path}")
    model = PPO.load(model_path)
    
    # Create environment
    config = EnvConfig()
    env = SoldierEnv(config=config)
    
    print(f"\nRunning {n_episodes} episodes with trained PPO agent...")
    print("=" * 60)
    
    episodes = []
    outcomes = {"intercepted": 0, "soldier_caught": 0, "unsafe_intercept": 0, "timeout": 0}
    total_rewards = []
    
    for i in range(n_episodes):
        soldier_pos, defender_pos, enemy_pos, outcome, total_reward = run_episode_with_model(
            env, model, max_steps=max_steps, deterministic=deterministic
        )
        
        outcomes[outcome] = outcomes.get(outcome, 0) + 1
        total_rewards.append(total_reward)
        episodes.append((soldier_pos, defender_pos, enemy_pos, outcome, total_reward))
        
        print(f"Episode {i+1}/{n_episodes}: {outcome} "
              f"(steps={len(soldier_pos)}, reward={total_reward:.1f})")
    
    print("=" * 60)
    print(f"\nOutcome summary: {outcomes}")
    print(f"Win rate: {outcomes['intercepted'] / n_episodes * 100:.1f}%")
    print(f"Average reward: {np.mean(total_rewards):.1f} ± {np.std(total_rewards):.1f}")
    
    env.close()
    return episodes, config


def create_video_opencv(episodes, config, video_path, fps=20):
    """Create MP4 video using OpenCV.
    
    Args:
        episodes: List of (soldier_pos, defender_pos, enemy_pos, outcome, reward) tuples.
        config: EnvConfig with L parameter.
        video_path: Output video path.
        fps: Frames per second.
    """
    L = config.L
    fig_size = (800, 800)
    
    # Setup video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, fig_size)
    
    # Color map for outcomes
    outcome_colors = {
        'intercepted': 'green',
        'soldier_caught': 'red',
        'unsafe_intercept': 'orange',
        'timeout': 'gray'
    }
    
    total_frames = sum(len(ep[0]) for ep in episodes) + len(episodes) * 20  # + pause frames
    frame_count = 0
    
    for ep_idx, (soldier_pos, defender_pos, enemy_pos, outcome, reward) in enumerate(episodes):
        n_steps = len(soldier_pos)
        
        for step in range(n_steps):
            # Create matplotlib figure
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            ax.set_xlim(-L - 5, L + 5)
            ax.set_ylim(-L - 5, L + 5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            
            # Draw domain boundary
            rect = plt.Rectangle((-L, -L), 2*L, 2*L, fill=False, 
                                 edgecolor='blue', linewidth=2, linestyle='--')
            ax.add_patch(rect)
            
            # Draw trails
            if step > 0:
                ax.plot(soldier_pos[:step+1, 0], soldier_pos[:step+1, 1], 
                       'g-', alpha=0.3, linewidth=1)
                ax.plot(defender_pos[:step+1, 0], defender_pos[:step+1, 1], 
                       'b-', alpha=0.5, linewidth=1.5)
                ax.plot(enemy_pos[:step+1, 0], enemy_pos[:step+1, 1], 
                       'r-', alpha=0.5, linewidth=1.5)
            
            # Draw agents
            ax.plot(soldier_pos[step, 0], soldier_pos[step, 1], 'go', 
                   markersize=12, label='Soldier')
            ax.plot(defender_pos[step, 0], defender_pos[step, 1], 'b^', 
                   markersize=12, label='Defender')
            ax.plot(enemy_pos[step, 0], enemy_pos[step, 1], 'rv', 
                   markersize=12, label='Enemy')
            
            # Draw detection radius around defender
            detection_circle = plt.Circle(
                (defender_pos[step, 0], defender_pos[step, 1]),
                config.detection_radius,
                fill=False, edgecolor='blue', linestyle='--', linewidth=1.5, alpha=0.6,
                label='Detection Radius'
            )
            ax.add_patch(detection_circle)
            
            # Title with episode info
            color = outcome_colors.get(outcome, 'black')
            title = f"Episode {ep_idx + 1}/{len(episodes)} | Step {step}/{n_steps-1}"
            if step == n_steps - 1:
                title += f" | {outcome.upper()} (R={reward:.1f})"
            ax.set_title(title, fontsize=12, color=color if step == n_steps - 1 else 'black')
            ax.legend(loc='upper right')
            
            # Convert to image
            canvas = FigureCanvasAgg(fig)
            canvas.draw()
            img = np.frombuffer(canvas.buffer_rgba(), dtype=np.uint8)
            img = img.reshape(fig.canvas.get_width_height()[::-1] + (4,))
            img = cv2.cvtColor(img, cv2.COLOR_RGBA2BGR)
            
            out.write(img)
            plt.close(fig)
            
            frame_count += 1
            if frame_count % 50 == 0:
                print(f"  Frame {frame_count}/{total_frames}")
        
        # Add pause frames at end of episode
        for _ in range(20):
            out.write(img)
            frame_count += 1
    
    out.release()
    print(f"Video saved: {video_path}")


def main():
    model_path = "ppo_defender.zip"
    n_episodes = 5
    video_path = "ppo_evaluation.mp4"
    
    print("=" * 60)
    print("PPO Defender Evaluation")
    print("=" * 60)
    
    # Run evaluation
    episodes, config = evaluate_model(
        model_path=model_path,
        n_episodes=n_episodes,
        max_steps=500,
        deterministic=True
    )
    
    # Generate video using OpenCV
    print(f"\nGenerating video: {video_path}")
    print("=" * 60)
    
    create_video_opencv(episodes, config, video_path, fps=20)
    
    print("\nDone!")


if __name__ == "__main__":
    main()
