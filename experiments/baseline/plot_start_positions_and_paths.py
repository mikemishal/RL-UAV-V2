"""Generate two static PNG snapshots from one episode rollout.

The script saves:
- A start-frame image at step 0
- A second image after a chosen number of steps

Both images show start positions, detection radius, and path lines up to that frame.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import os
import sys

# Add project root to path
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", ".."))

import matplotlib.pyplot as plt
import numpy as np

from uav_defend.config import EnvConfig
from uav_defend.envs import SoldierEnv
from uav_defend.policies import GreedyInterceptPolicy

IEEE_FIGURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}


def _plot_snapshot(
    *,
    config: EnvConfig,
    seed: int,
    frame_index: int,
    output_path: Path,
    soldier_positions: np.ndarray,
    defender_positions: np.ndarray,
    enemy_positions: np.ndarray,
    start_soldier: np.ndarray,
    start_defender: np.ndarray,
    start_enemy: np.ndarray,
    detection_step: int | None,
    outcome: str,
) -> Path:
    output_path.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(IEEE_FIGURE_RC):
        fig, ax = plt.subplots(figsize=(3.5, 3.5), constrained_layout=True)

        # Domain boundary
        L = config.L
        ax.set_xlim(-L, L)
        ax.set_ylim(-L, L)

        # Detection radius circle centered on current defender position
        det_circle = plt.Circle(
            (defender_positions[frame_index, 0], defender_positions[frame_index, 1]),
            config.detection_radius,
            color="tab:green",
            alpha=0.12,
            label=f"Detection radius ({config.detection_radius:.0f})",
        )
        ax.add_patch(det_circle)
        det_edge = plt.Circle(
            (defender_positions[frame_index, 0], defender_positions[frame_index, 1]),
            config.detection_radius,
            fill=False,
            edgecolor="tab:green",
            linewidth=1.0,
            linestyle="--",
        )
        ax.add_patch(det_edge)

        path_end = frame_index + 1

        # Trail paths up to the current frame (no labels; icons carry the legend)
        ax.plot(
            enemy_positions[:path_end, 0],
            enemy_positions[:path_end, 1],
            "-",
            color="tab:red",
            lw=1.5,
            alpha=0.5,
        )
        ax.plot(
            defender_positions[:path_end, 0],
            defender_positions[:path_end, 1],
            "-",
            color="tab:blue",
            lw=1.5,
            alpha=0.5,
        )
        ax.plot(
            soldier_positions[:path_end, 0],
            soldier_positions[:path_end, 1],
            "--",
            color="tab:gray",
            lw=1.2,
            alpha=0.5,
        )

        # Current-position icons (the item itself, no separate "start" markers)
        ax.scatter(
            soldier_positions[frame_index, 0],
            soldier_positions[frame_index, 1],
            s=120,
            marker="o",
            color="tab:gray",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Soldier",
        )
        ax.scatter(
            defender_positions[frame_index, 0],
            defender_positions[frame_index, 1],
            s=130,
            marker="^",
            color="tab:blue",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Defender Drone",
        )
        ax.scatter(
            enemy_positions[frame_index, 0],
            enemy_positions[frame_index, 1],
            s=130,
            marker="s",
            color="tab:red",
            edgecolors="black",
            linewidths=0.8,
            zorder=5,
            label="Enemy Drone",
        )

        ax.set_xlabel("x", fontsize=8)
        ax.set_ylabel("y", fontsize=8)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis='both', labelsize=7)
        ax.legend(loc="upper left", fontsize=7, frameon=False, borderaxespad=0.3)

        fig.savefig(output_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
        pdf_path = output_path.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    return output_path


def generate_plots(
    seed: int,
    output_start_path: Path,
    output_after_path: Path,
    after_steps: int,
    max_steps: int = 800,
    detection_radius: float | None = None,
    soldier_start: tuple[float, float] | None = None,
    defender_start: tuple[float, float] | None = None,
    enemy_start: tuple[float, float] | None = None,
) -> tuple[Path, Path]:
    config = EnvConfig(use_kalman_tracking=False)
    
    # Override detection radius if provided
    if detection_radius is not None:
        config.detection_radius = detection_radius
    
    env = SoldierEnv(config=config)
    policy = GreedyInterceptPolicy()

    obs, info = env.reset(seed=seed)
    
    # Optionally override initial positions
    if soldier_start is not None:
        info["soldier_pos"] = np.array(soldier_start, dtype=np.float32)
        obs[:2] = soldier_start
    if defender_start is not None:
        info["defender_pos"] = np.array(defender_start, dtype=np.float32)
    if enemy_start is not None:
        info["enemy_pos"] = np.array(enemy_start, dtype=np.float32)
        obs[2:4] = enemy_start
    
    policy.reset()

    soldier_positions = [info["soldier_pos"].copy()]
    defender_positions = [info["defender_pos"].copy()]
    enemy_positions = [info["enemy_pos"].copy()]

    start_soldier = info["soldier_pos"].copy()
    start_defender = info["defender_pos"].copy()
    start_enemy = info["enemy_pos"].copy()

    detection_step = None
    done = False
    step_count = 0

    while not done and step_count < max_steps:
        action = policy.act(obs, info)
        obs, _, terminated, truncated, info = env.step(action)
        done = terminated or truncated
        step_count += 1

        soldier_positions.append(info["soldier_pos"].copy())
        defender_positions.append(info["defender_pos"].copy())
        enemy_positions.append(info["enemy_pos"].copy())

        if detection_step is None and info.get("enemy_detected", False):
            detection_step = step_count

    env.close()

    soldier_positions = np.asarray(soldier_positions)
    defender_positions = np.asarray(defender_positions)
    enemy_positions = np.asarray(enemy_positions)

    available_last_step = len(enemy_positions) - 1
    after_steps_clamped = max(0, min(after_steps, available_last_step))

    saved_start = _plot_snapshot(
        config=config,
        seed=seed,
        frame_index=0,
        output_path=output_start_path,
        soldier_positions=soldier_positions,
        defender_positions=defender_positions,
        enemy_positions=enemy_positions,
        start_soldier=start_soldier,
        start_defender=start_defender,
        start_enemy=start_enemy,
        detection_step=detection_step,
        outcome=info.get("outcome", "N/A"),
    )

    saved_after = _plot_snapshot(
        config=config,
        seed=seed,
        frame_index=after_steps_clamped,
        output_path=output_after_path,
        soldier_positions=soldier_positions,
        defender_positions=defender_positions,
        enemy_positions=enemy_positions,
        start_soldier=start_soldier,
        start_defender=start_defender,
        start_enemy=start_enemy,
        detection_step=detection_step,
        outcome=info.get("outcome", "N/A"),
    )

    return saved_start, saved_after


def main() -> None:
    parser = argparse.ArgumentParser(description="Create two PNG snapshots: start and after a few steps")
    parser.add_argument("--seed", type=int, default=7, help="Random seed for the episode")
    parser.add_argument(
        "--output-start",
        type=str,
        default="media/images/start_positions_step0.png",
        help="Output path for step-0 snapshot",
    )
    parser.add_argument(
        "--output-after",
        type=str,
        default="media/images/start_positions_after_steps.png",
        help="Output path for after-steps snapshot",
    )
    parser.add_argument("--after-steps", type=int, default=8, help="Step index for second snapshot")
    parser.add_argument("--max-steps", type=int, default=800, help="Maximum rollout steps")
    parser.add_argument("--detection-radius", type=float, default=None, help="Detection radius (optional)")
    parser.add_argument("--soldier-start-x", type=float, default=None, help="Soldier start x position")
    parser.add_argument("--soldier-start-y", type=float, default=None, help="Soldier start y position")
    parser.add_argument("--enemy-start-x", type=float, default=None, help="Enemy start x position")
    parser.add_argument("--enemy-start-y", type=float, default=None, help="Enemy start y position")
    parser.add_argument("--defender-start-x", type=float, default=None, help="Defender start x position")
    parser.add_argument("--defender-start-y", type=float, default=None, help="Defender start y position")
    args = parser.parse_args()

    soldier_start = None
    if args.soldier_start_x is not None and args.soldier_start_y is not None:
        soldier_start = (args.soldier_start_x, args.soldier_start_y)
    
    defender_start = None
    if args.defender_start_x is not None and args.defender_start_y is not None:
        defender_start = (args.defender_start_x, args.defender_start_y)
    
    enemy_start = None
    if args.enemy_start_x is not None and args.enemy_start_y is not None:
        enemy_start = (args.enemy_start_x, args.enemy_start_y)

    output_start_path, output_after_path = generate_plots(
        seed=args.seed,
        output_start_path=Path(args.output_start),
        output_after_path=Path(args.output_after),
        after_steps=args.after_steps,
        max_steps=args.max_steps,
        detection_radius=args.detection_radius,
        soldier_start=soldier_start,
        defender_start=defender_start,
        enemy_start=enemy_start,
    )
    print(f"Saved: {output_start_path}")
    print(f"Saved: {output_after_path}")


if __name__ == "__main__":
    main()
