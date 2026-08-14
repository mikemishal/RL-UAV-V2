"""Generate video file showing soldier, defender, and enemy movement."""

import numpy as np
import matplotlib.pyplot as plt
import matplotlib.animation as animation
from uav_defend import SoldierEnv
from uav_defend.config import EnvConfig
from visualize_soldier import run_episode


def create_video(soldier_pos, defender_pos, enemy_pos, L=50.0, fps=30,
                 show_defender_trail=False, show_enemy_trail=True, 
                 output_file="pursuit_video.mp4"):
    """
    Create a video file of the simulation for easy playback control.
    
    Args:
        soldier_pos: Array of shape (T, 2) with soldier positions.
        defender_pos: Array of shape (T, 2) with defender positions.
        enemy_pos: Array of shape (T, 2) with enemy positions.
        L: Domain half-size.
        fps: Frames per second for video.
        show_defender_trail: If True, show defender trail.
        show_enemy_trail: If True, show enemy trail.
        output_file: Output video filename (.mp4).
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
    
    # Current position markers
    soldier_marker, = ax.plot([], [], 'bo', markersize=14, label='Soldier', markeredgecolor='black')
    defender_marker, = ax.plot([], [], 'g^', markersize=12, label='Defender', markeredgecolor='black')
    enemy_marker, = ax.plot([], [], 'rX', markersize=14, label='Enemy', markeredgecolor='darkred')
    
    # Start markers
    ax.plot(soldier_pos[0, 0], soldier_pos[0, 1], 'ko', markersize=6, alpha=0.5)
    ax.plot(enemy_pos[0, 0], enemy_pos[0, 1], 'ko', markersize=6, alpha=0.5)
    
    # Text displays
    step_text = ax.text(0.02, 0.98, '', transform=ax.transAxes, verticalalignment='top',
                        fontsize=10, bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.7))
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
        # Draw trails
        soldier_trail.set_data(soldier_pos[:frame+1, 0], soldier_pos[:frame+1, 1])
        if show_defender_trail:
            defender_trail.set_data(defender_pos[:frame+1, 0], defender_pos[:frame+1, 1])
        if show_enemy_trail:
            enemy_trail.set_data(enemy_pos[:frame+1, 0], enemy_pos[:frame+1, 1])
        
        # Current positions
        soldier_marker.set_data([soldier_pos[frame, 0]], [soldier_pos[frame, 1]])
        defender_marker.set_data([defender_pos[frame, 0]], [defender_pos[frame, 1]])
        enemy_marker.set_data([enemy_pos[frame, 0]], [enemy_pos[frame, 1]])
        
        # Distance and step info
        dist = np.linalg.norm(soldier_pos[frame] - enemy_pos[frame])
        step_text.set_text(f'Step: {frame}/{len(soldier_pos)-1}\nSoldier: ({soldier_pos[frame, 0]:.1f}, {soldier_pos[frame, 1]:.1f})')
        dist_text.set_text(f'Enemy→Soldier: {dist:.1f}')
        
        return (soldier_trail, defender_trail, enemy_trail, 
                soldier_marker, defender_marker, enemy_marker, step_text, dist_text)
    
    # Create animation
    anim = animation.FuncAnimation(
        fig, animate, init_func=init,
        frames=len(soldier_pos), interval=1000//fps, blit=True
    )
    
    # Save as MP4 video (requires ffmpeg)
    print(f"Saving video to {output_file}...")
    try:
        writer = animation.FFMpegWriter(fps=fps, metadata=dict(artist='UAV_Defend'), bitrate=1800)
        anim.save(output_file, writer=writer)
        print(f"Video saved successfully: {output_file}")
    except Exception as e:
        print(f"FFmpeg not available, trying alternative writer...")
        # Fallback to pillow for GIF if ffmpeg not available
        fallback_file = output_file.replace('.mp4', '.gif')
        anim.save(fallback_file, writer='pillow', fps=fps)
        print(f"Saved as GIF instead: {fallback_file}")
    
    plt.close(fig)
    return anim


def main():
    # Create environment
    config = EnvConfig()  # Uses defaults: L=50, v_s=1.5, v_e=12.0, v_d=18.0
    env = SoldierEnv(config=config)
    
    # Run episode until termination (soldier caught or timeout)
    max_steps = 500
    print(f"Running episode (max {max_steps} steps)...")
    soldier_pos, defender_pos, enemy_pos, weave_biases = run_episode(env, max_steps=max_steps)
    print(f"Collected {len(soldier_pos)} positions")
    print(f"Soldier - Start: {soldier_pos[0]}, End: {soldier_pos[-1]}")
    print(f"Enemy - Start: {enemy_pos[0]}, End: {enemy_pos[-1]}")
    
    # Calculate final distance
    final_dist = np.linalg.norm(soldier_pos[-1] - enemy_pos[-1])
    print(f"Final enemy→soldier distance: {final_dist:.2f}")
    
    # Generate video file
    output_file = "pursuit_video.mp4"
    print(f"\nGenerating video: {output_file}")
    print("This may take a moment...")
    
    create_video(
        soldier_pos, 
        defender_pos,
        enemy_pos,
        L=config.L, 
        fps=30,  # 30 frames per second
        show_defender_trail=False,
        show_enemy_trail=True,
        output_file=output_file
    )
    
    print("\nDone!")


if __name__ == "__main__":
    main()
