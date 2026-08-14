"""Compare PPO models in animation: No Kalman vs With Kalman estimation.

Creates a side-by-side or sequential animation showing:
1. First few episodes using the old model (no Kalman)
2. Then episodes using the new Kalman model

Also visualizes Kalman tracking estimate vs true enemy position.
"""
import numpy as np
import cv2
from stable_baselines3 import PPO
from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg


def run_episode_with_tracking(env, model, max_steps=500, deterministic=True):
    """Run a single episode, collecting Kalman tracking data.
    
    Returns:
        dict with positions, tracking info, outcome, reward
    """
    soldier_positions = []
    defender_positions = []
    enemy_positions = []
    e_hat_positions = []  # Kalman estimates
    tracking_errors = []
    detected_flags = []
    
    obs, info = env.reset()
    soldier_positions.append(info['soldier_pos'].copy())
    defender_positions.append(info['defender_pos'].copy())
    enemy_positions.append(info['enemy_pos'].copy())
    e_hat_positions.append(info.get('e_hat'))
    tracking_errors.append(info.get('tracking_error'))
    detected_flags.append(info.get('detected', False))
    
    total_reward = 0.0
    for _ in range(max_steps):
        action, _ = model.predict(obs, deterministic=deterministic)
        obs, reward, terminated, truncated, info = env.step(action)
        
        soldier_positions.append(info['soldier_pos'].copy())
        defender_positions.append(info['defender_pos'].copy())
        enemy_positions.append(info['enemy_pos'].copy())
        e_hat_positions.append(info.get('e_hat'))
        tracking_errors.append(info.get('tracking_error'))
        detected_flags.append(info.get('detected', False))
        total_reward += reward
        
        if terminated or truncated:
            break
    
    return {
        'soldier': np.array(soldier_positions),
        'defender': np.array(defender_positions),
        'enemy': np.array(enemy_positions),
        'e_hat': e_hat_positions,
        'tracking_error': tracking_errors,
        'detected': detected_flags,
        'outcome': info.get('outcome', 'unknown'),
        'reward': total_reward,
    }


def create_comparison_video(episodes_no_kalman, episodes_kalman, config, 
                            video_path, fps=20):
    """Create comparison video showing both models.
    
    Args:
        episodes_no_kalman: Episodes from old model (list of dicts)
        episodes_kalman: Episodes from Kalman model (list of dicts)
        config: EnvConfig
        video_path: Output path
        fps: Frames per second
    """
    L = config.L
    fig_size = (800, 800)
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, fig_size)
    
    outcome_colors = {
        'intercepted': 'green',
        'soldier_caught': 'red',
        'unsafe_intercept': 'orange',
        'timeout': 'gray'
    }
    
    all_episodes = []
    # Add no-Kalman episodes with label
    for ep in episodes_no_kalman:
        ep['model'] = 'No Kalman (r=30)'
        all_episodes.append(ep)
    # Add Kalman episodes with label
    for ep in episodes_kalman:
        ep['model'] = 'With Kalman (r=15)'
        all_episodes.append(ep)
    
    total_frames = sum(len(ep['soldier']) for ep in all_episodes) + len(all_episodes) * 30
    frame_count = 0
    
    for ep_idx, ep in enumerate(all_episodes):
        soldier_pos = ep['soldier']
        defender_pos = ep['defender']
        enemy_pos = ep['enemy']
        e_hat_list = ep['e_hat']
        detected_list = ep['detected']
        outcome = ep['outcome']
        reward = ep['reward']
        model_name = ep['model']
        ep_detection_radius = ep.get('detection_radius', config.detection_radius)
        
        n_steps = len(soldier_pos)
        is_kalman = 'Kalman' in model_name
        
        for step in range(n_steps):
            fig, ax = plt.subplots(figsize=(8, 8), dpi=100)
            ax.set_xlim(-L - 5, L + 5)
            ax.set_ylim(-L - 5, L + 5)
            ax.set_aspect('equal')
            ax.grid(True, alpha=0.3)
            ax.axhline(y=0, color='k', linewidth=0.5)
            ax.axvline(x=0, color='k', linewidth=0.5)
            
            # Domain boundary
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
                
                # Draw Kalman estimate trail (if available)
                if is_kalman:
                    e_hat_trail = [e for e in e_hat_list[:step+1] if e is not None]
                    if len(e_hat_trail) > 1:
                        e_hat_arr = np.array(e_hat_trail)
                        ax.plot(e_hat_arr[:, 0], e_hat_arr[:, 1], 
                               'm--', alpha=0.6, linewidth=1.5, label='_nolegend_')
            
            # Draw agents
            ax.plot(soldier_pos[step, 0], soldier_pos[step, 1], 'go', 
                   markersize=12, label='Soldier')
            ax.plot(defender_pos[step, 0], defender_pos[step, 1], 'b^', 
                   markersize=12, label='Defender')
            ax.plot(enemy_pos[step, 0], enemy_pos[step, 1], 'rv', 
                   markersize=12, label='Enemy (True)')
            
            # Draw Kalman estimate marker
            if is_kalman and e_hat_list[step] is not None:
                e_hat = e_hat_list[step]
                ax.plot(e_hat[0], e_hat[1], 'm*', markersize=14, 
                       label='Enemy (Estimated)')
                # Draw line from estimate to true position (tracking error)
                ax.plot([e_hat[0], enemy_pos[step, 0]], 
                       [e_hat[1], enemy_pos[step, 1]], 
                       'm:', alpha=0.5, linewidth=1)
            
            # Detection radius (use per-episode value)
            detection_circle = plt.Circle(
                (defender_pos[step, 0], defender_pos[step, 1]),
                ep_detection_radius,
                fill=False, edgecolor='blue', linestyle='--', linewidth=1.5, alpha=0.6
            )
            ax.add_patch(detection_circle)
            
            # Title
            color = outcome_colors.get(outcome, 'black')
            detected_str = "DETECTED" if detected_list[step] else "searching..."
            title = f"[{model_name}] Episode {ep_idx + 1}/{len(all_episodes)}"
            title += f"\nStep {step}/{n_steps-1} | {detected_str}"
            
            if step == n_steps - 1:
                title += f"\n{outcome.upper()} (R={reward:.1f})"
                ax.set_title(title, fontsize=11, color=color, fontweight='bold')
            else:
                ax.set_title(title, fontsize=11)
            
            ax.legend(loc='upper right', fontsize=9)
            
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
        
        # Pause frames at end
        for _ in range(30):
            out.write(img)
            frame_count += 1
    
    out.release()
    print(f"Video saved: {video_path}")


def main():
    print("=" * 70)
    print("Animation: No Kalman (radius=30) vs With Kalman (radius=15)")
    print("=" * 70)
    
    n_episodes_each = 3
    
    # Load both models
    print("\nLoading models...")
    try:
        model_no_kalman = PPO.load("ppo_defender_no_kalman.zip")
        print("  ✓ Loaded ppo_defender_no_kalman.zip (trained with radius=30)")
    except Exception as e:
        print(f"  ✗ Could not load ppo_defender_no_kalman.zip: {e}")
        model_no_kalman = None
    
    try:
        model_kalman = PPO.load("ppo_defender.zip")
        print("  ✓ Loaded ppo_defender.zip (trained with radius=15)")
    except Exception as e:
        print(f"  ✗ Could not load ppo_defender.zip: {e}")
        model_kalman = None
    
    if model_no_kalman is None and model_kalman is None:
        print("\nNo models available. Please train models first.")
        return
    
    episodes_no_kalman = []
    episodes_kalman = []
    
    # Run episodes with no-Kalman model using detection_radius=30 and legacy obs
    if model_no_kalman:
        print(f"\nRunning {n_episodes_each} episodes with NO KALMAN model (radius=30, legacy obs)...")
        config_30 = EnvConfig()
        config_30.detection_radius = 30.0  # Original training config
        env_30 = SoldierEnv(config=config_30, use_kalman_obs=False)  # Legacy observation format
        for i in range(n_episodes_each):
            env_30.reset(seed=100 + i)
            ep = run_episode_with_tracking(env_30, model_no_kalman)
            ep['detection_radius'] = 30.0
            episodes_no_kalman.append(ep)
            print(f"  Episode {i+1}: {ep['outcome']} (steps={len(ep['soldier'])}, R={ep['reward']:.1f})")
        env_30.close()
    
    # Run episodes with Kalman model using detection_radius=15 and Kalman obs
    if model_kalman:
        print(f"\nRunning {n_episodes_each} episodes with KALMAN model (radius=15, Kalman obs)...")
        config_15 = EnvConfig()
        config_15.detection_radius = 15.0  # Current training config
        env_15 = SoldierEnv(config=config_15, use_kalman_obs=True)  # Kalman observation format
        for i in range(n_episodes_each):
            env_15.reset(seed=100 + i)  # Same seeds for comparison
            ep = run_episode_with_tracking(env_15, model_kalman)
            ep['detection_radius'] = 15.0
            episodes_kalman.append(ep)
            print(f"  Episode {i+1}: {ep['outcome']} (steps={len(ep['soldier'])}, R={ep['reward']:.1f})")
        env_15.close()
    
    # Use config for video (use the larger radius for display)
    config_display = EnvConfig()
    config_display.detection_radius = 30.0
    
    # Generate video
    video_path = "kalman_comparison.mp4"
    print(f"\nGenerating comparison video: {video_path}")
    print("-" * 70)
    
    create_comparison_video(
        episodes_no_kalman, 
        episodes_kalman, 
        config_display, 
        video_path, 
        fps=20
    )
    
    # Summary
    print("\n" + "=" * 70)
    print("SUMMARY")
    print("=" * 70)
    
    if episodes_no_kalman:
        wins = sum(1 for ep in episodes_no_kalman if ep['outcome'] == 'intercepted')
        print(f"No Kalman Model (r=30): {wins}/{len(episodes_no_kalman)} wins")
    
    if episodes_kalman:
        wins = sum(1 for ep in episodes_kalman if ep['outcome'] == 'intercepted')
        print(f"With Kalman Model (r=15): {wins}/{len(episodes_kalman)} wins")
    
    print(f"\nVideo saved: {video_path}")


if __name__ == "__main__":
    main()
