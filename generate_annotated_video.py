"""Generate annotated evaluation video comparing PPO vs Random policy."""
import numpy as np
import cv2
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from matplotlib.backends.backend_agg import FigureCanvasAgg
from matplotlib.patches import Circle
from stable_baselines3 import PPO
from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig


def run_episode_detailed(env, model=None, policy="ppo", max_steps=500):
    """Run episode collecting detailed state info for annotations.
    
    Returns:
        dict with positions, distances, detection states, actions, etc.
    """
    data = {
        'soldier_pos': [],
        'defender_pos': [],
        'enemy_pos': [],
        'defender_enemy_dist': [],
        'enemy_soldier_dist': [],
        'defender_soldier_dist': [],
        'detected': [],
        'actions': [],
        'rewards': [],
        'outcome': 'ongoing'
    }
    
    obs, info = env.reset()
    data['soldier_pos'].append(info['soldier_pos'].copy())
    data['defender_pos'].append(info['defender_pos'].copy())
    data['enemy_pos'].append(info['enemy_pos'].copy())
    data['detected'].append(info.get('enemy_detected', False))
    
    # Initial distances
    data['defender_enemy_dist'].append(np.linalg.norm(info['defender_pos'] - info['enemy_pos']))
    data['enemy_soldier_dist'].append(np.linalg.norm(info['enemy_pos'] - info['soldier_pos']))
    data['defender_soldier_dist'].append(np.linalg.norm(info['defender_pos'] - info['soldier_pos']))
    
    total_reward = 0.0
    for step in range(max_steps):
        if policy == "ppo" and model is not None:
            action, _ = model.predict(obs, deterministic=True)
        else:
            action = env.action_space.sample()
        
        data['actions'].append(action.copy())
        
        obs, reward, terminated, truncated, info = env.step(action)
        data['rewards'].append(reward)
        total_reward += reward
        
        data['soldier_pos'].append(info['soldier_pos'].copy())
        data['defender_pos'].append(info['defender_pos'].copy())
        data['enemy_pos'].append(info['enemy_pos'].copy())
        data['detected'].append(info.get('enemy_detected', False))
        
        data['defender_enemy_dist'].append(np.linalg.norm(info['defender_pos'] - info['enemy_pos']))
        data['enemy_soldier_dist'].append(np.linalg.norm(info['enemy_pos'] - info['soldier_pos']))
        data['defender_soldier_dist'].append(np.linalg.norm(info['defender_pos'] - info['soldier_pos']))
        
        if terminated or truncated:
            data['outcome'] = info.get('outcome', 'unknown')
            break
    
    # Convert to arrays
    for key in ['soldier_pos', 'defender_pos', 'enemy_pos', 'actions']:
        data[key] = np.array(data[key])
    data['total_reward'] = total_reward
    
    return data


def create_annotated_video(episodes_data, config, video_path, fps=15):
    """Create annotated MP4 video with comments and info overlays.
    
    Args:
        episodes_data: List of episode data dicts (from run_episode_detailed)
        config: EnvConfig
        video_path: Output path
        fps: Frames per second
    """
    L = config.L
    fig_size = (1200, 800)  # Wider for annotations panel
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_path, fourcc, fps, fig_size)
    
    outcome_colors = {
        'intercepted': '#2ecc71',
        'soldier_caught': '#e74c3c',
        'unsafe_intercept': '#f39c12',
        'timeout': '#95a5a6',
        'ongoing': '#3498db'
    }
    
    total_frames = sum(len(ep['soldier_pos']) for ep in episodes_data) + len(episodes_data) * 30
    frame_count = 0
    
    for ep_idx, data in enumerate(episodes_data):
        n_steps = len(data['soldier_pos'])
        policy_name = data.get('policy', 'Unknown')
        
        for step in range(n_steps):
            # Create figure with two panels
            fig = plt.figure(figsize=(12, 8), dpi=100)
            
            # Main visualization panel (left)
            ax_main = fig.add_axes([0.05, 0.1, 0.55, 0.8])
            ax_main.set_xlim(-L - 5, L + 5)
            ax_main.set_ylim(-L - 5, L + 5)
            ax_main.set_aspect('equal')
            ax_main.grid(True, alpha=0.3)
            
            # Domain boundary
            rect = plt.Rectangle((-L, -L), 2*L, 2*L, fill=False,
                                edgecolor='#3498db', linewidth=2, linestyle='--')
            ax_main.add_patch(rect)
            
            # Current positions
            s_pos = data['soldier_pos'][step]
            d_pos = data['defender_pos'][step]
            e_pos = data['enemy_pos'][step]
            detected = data['detected'][step]
            
            # Draw detection radius around defender
            detection_circle = Circle(d_pos, config.detection_radius,
                                      fill=False, edgecolor='cyan', 
                                      linestyle=':', linewidth=2, alpha=0.7)
            ax_main.add_patch(detection_circle)
            
            # Draw intercept radius around defender (if detected)
            if detected:
                intercept_circle = Circle(d_pos, config.intercept_radius,
                                         fill=True, facecolor='blue', alpha=0.2)
                ax_main.add_patch(intercept_circle)
            
            # Draw threat radius around soldier
            threat_circle = Circle(s_pos, config.threat_radius,
                                   fill=True, facecolor='red', alpha=0.15)
            ax_main.add_patch(threat_circle)
            
            # Draw trails
            if step > 0:
                ax_main.plot(data['soldier_pos'][:step+1, 0], data['soldier_pos'][:step+1, 1],
                           'g-', alpha=0.4, linewidth=1.5, label='_nolegend_')
                ax_main.plot(data['defender_pos'][:step+1, 0], data['defender_pos'][:step+1, 1],
                           'b-', alpha=0.5, linewidth=2, label='_nolegend_')
                ax_main.plot(data['enemy_pos'][:step+1, 0], data['enemy_pos'][:step+1, 1],
                           'r-', alpha=0.5, linewidth=2, label='_nolegend_')
            
            # Draw agents
            ax_main.plot(s_pos[0], s_pos[1], 'go', markersize=15, label='Soldier', zorder=10)
            ax_main.plot(d_pos[0], d_pos[1], 'b^', markersize=15, label='Defender', zorder=10)
            
            # Enemy: show differently if detected vs not
            if detected:
                ax_main.plot(e_pos[0], e_pos[1], 'rv', markersize=15, label='Enemy (DETECTED)', zorder=10)
            else:
                ax_main.plot(e_pos[0], e_pos[1], 'r*', markersize=12, alpha=0.5, 
                           label='Enemy (hidden)', zorder=10)
            
            # Draw action vector (if detected and we have action)
            if detected and step < len(data['actions']):
                action = data['actions'][step]
                action_norm = np.linalg.norm(action)
                if action_norm > 0.1:
                    ax_main.arrow(d_pos[0], d_pos[1], 
                                 action[0] * 5, action[1] * 5,
                                 head_width=1.5, head_length=0.8,
                                 fc='blue', ec='darkblue', alpha=0.7, zorder=5)
            
            ax_main.legend(loc='upper left', fontsize=9)
            ax_main.set_xlabel('X', fontsize=10)
            ax_main.set_ylabel('Y', fontsize=10)
            
            # Info panel (right side)
            ax_info = fig.add_axes([0.65, 0.1, 0.32, 0.8])
            ax_info.axis('off')
            
            # Title
            outcome_color = outcome_colors.get(data['outcome'], '#333')
            if step == n_steps - 1:
                title_color = outcome_color
            else:
                title_color = '#333'
            
            # Episode header
            ax_info.text(0.5, 0.98, f"Episode {ep_idx + 1}", fontsize=16, fontweight='bold',
                        ha='center', va='top', transform=ax_info.transAxes)
            ax_info.text(0.5, 0.93, f"Policy: {policy_name}", fontsize=12,
                        ha='center', va='top', transform=ax_info.transAxes,
                        color='#2980b9' if policy_name == 'PPO' else '#e67e22')
            
            # Step counter
            ax_info.text(0.5, 0.86, f"Step: {step}/{n_steps-1}", fontsize=11,
                        ha='center', va='top', transform=ax_info.transAxes)
            
            # Detection status box
            det_color = '#2ecc71' if detected else '#e74c3c'
            det_text = "✓ ENEMY DETECTED" if detected else "✗ Searching..."
            ax_info.text(0.5, 0.78, det_text, fontsize=14, fontweight='bold',
                        ha='center', va='top', transform=ax_info.transAxes,
                        color=det_color,
                        bbox=dict(boxstyle='round', facecolor='white', edgecolor=det_color, linewidth=2))
            
            # Distances section
            ax_info.text(0.5, 0.68, "─── DISTANCES ───", fontsize=10,
                        ha='center', va='top', transform=ax_info.transAxes, color='#7f8c8d')
            
            d_e_dist = data['defender_enemy_dist'][step]
            e_s_dist = data['enemy_soldier_dist'][step]
            d_s_dist = data['defender_soldier_dist'][step]
            
            # Color code distances
            de_color = '#2ecc71' if d_e_dist < config.intercept_radius * 2 else '#333'
            es_color = '#e74c3c' if e_s_dist < config.threat_radius * 2 else '#333'
            
            ax_info.text(0.1, 0.62, f"Defender → Enemy:", fontsize=10,
                        ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.9, 0.62, f"{d_e_dist:.1f}", fontsize=11, fontweight='bold',
                        ha='right', va='top', transform=ax_info.transAxes, color=de_color)
            
            ax_info.text(0.1, 0.56, f"Enemy → Soldier:", fontsize=10,
                        ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.9, 0.56, f"{e_s_dist:.1f}", fontsize=11, fontweight='bold',
                        ha='right', va='top', transform=ax_info.transAxes, color=es_color)
            
            ax_info.text(0.1, 0.50, f"Defender → Soldier:", fontsize=10,
                        ha='left', va='top', transform=ax_info.transAxes)
            ax_info.text(0.9, 0.50, f"{d_s_dist:.1f}", fontsize=11,
                        ha='right', va='top', transform=ax_info.transAxes)
            
            # Thresholds reference
            ax_info.text(0.5, 0.42, "─── THRESHOLDS ───", fontsize=10,
                        ha='center', va='top', transform=ax_info.transAxes, color='#7f8c8d')
            
            ax_info.text(0.1, 0.36, f"Detection radius:", fontsize=9,
                        ha='left', va='top', transform=ax_info.transAxes, color='#7f8c8d')
            ax_info.text(0.9, 0.36, f"{config.detection_radius:.1f}", fontsize=9,
                        ha='right', va='top', transform=ax_info.transAxes, color='cyan')
            
            ax_info.text(0.1, 0.31, f"Intercept radius:", fontsize=9,
                        ha='left', va='top', transform=ax_info.transAxes, color='#7f8c8d')
            ax_info.text(0.9, 0.31, f"{config.intercept_radius:.1f}", fontsize=9,
                        ha='right', va='top', transform=ax_info.transAxes, color='blue')
            
            ax_info.text(0.1, 0.26, f"Threat radius:", fontsize=9,
                        ha='left', va='top', transform=ax_info.transAxes, color='#7f8c8d')
            ax_info.text(0.9, 0.26, f"{config.threat_radius:.1f}", fontsize=9,
                        ha='right', va='top', transform=ax_info.transAxes, color='red')
            
            # Cumulative reward
            cum_reward = sum(data['rewards'][:step]) if step > 0 else 0
            ax_info.text(0.5, 0.18, "─── REWARD ───", fontsize=10,
                        ha='center', va='top', transform=ax_info.transAxes, color='#7f8c8d')
            ax_info.text(0.5, 0.12, f"{cum_reward:.1f}", fontsize=16, fontweight='bold',
                        ha='center', va='top', transform=ax_info.transAxes)
            
            # Final outcome (show on last frame)
            if step == n_steps - 1:
                outcome_text = data['outcome'].upper()
                ax_info.text(0.5, 0.04, outcome_text, fontsize=18, fontweight='bold',
                            ha='center', va='top', transform=ax_info.transAxes,
                            color='white',
                            bbox=dict(boxstyle='round', facecolor=outcome_color, 
                                     edgecolor='black', linewidth=2, pad=0.5))
            
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
    config = EnvConfig()
    env = SoldierEnv(config=config)
    
    # Load PPO model
    print("Loading PPO model...")
    try:
        model = PPO.load("ppo_defender.zip")
    except Exception as e:
        print(f"Error loading model: {e}")
        return
    
    n_episodes = 5
    
    # Collect episodes with both policies
    episodes_data = []
    
    print(f"\nRunning {n_episodes} PPO episodes...")
    for i in range(n_episodes):
        data = run_episode_detailed(env, model=model, policy="ppo")
        data['policy'] = 'PPO'
        episodes_data.append(data)
        print(f"  PPO Episode {i+1}: {data['outcome']} (reward={data['total_reward']:.1f})")
    
    print(f"\nRunning {n_episodes} Random episodes...")
    for i in range(n_episodes):
        data = run_episode_detailed(env, model=None, policy="random")
        data['policy'] = 'Random'
        episodes_data.append(data)
        print(f"  Random Episode {i+1}: {data['outcome']} (reward={data['total_reward']:.1f})")
    
    env.close()
    
    # Generate video
    video_path = "evaluation_annotated.mp4"
    print(f"\nGenerating annotated video: {video_path}")
    print("=" * 60)
    create_annotated_video(episodes_data, config, video_path, fps=15)
    
    # Summary
    ppo_wins = sum(1 for d in episodes_data if d['policy'] == 'PPO' and d['outcome'] == 'intercepted')
    random_wins = sum(1 for d in episodes_data if d['policy'] == 'Random' and d['outcome'] == 'intercepted')
    
    print("\n" + "=" * 60)
    print("SUMMARY")
    print("=" * 60)
    print(f"PPO Win Rate: {ppo_wins}/{n_episodes} ({ppo_wins/n_episodes*100:.0f}%)")
    print(f"Random Win Rate: {random_wins}/{n_episodes} ({random_wins/n_episodes*100:.0f}%)")
    print(f"\nVideo saved to: {video_path}")


if __name__ == "__main__":
    main()
