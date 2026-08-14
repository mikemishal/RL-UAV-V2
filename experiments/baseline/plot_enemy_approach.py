"""Generate a conceptual diagram showing the enemy drone approaching the detection radius.

The image shows:
- Soldier and Defender Drone colocated at the origin
- Detection radius circle centered on the defender
- Enemy Drone positioned at the edge of the detection radius
- A noisy trail from the enemy's far starting position to the radius edge
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
import numpy as np

IEEE_FIGURE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

OUTPUT_PATH = PROJECT_ROOT / "media" / "images" / "enemy_approach_radius_edge_v3.png"

# Global visual scale for a compact paper-ready version while preserving layout.
VISUAL_SCALE = 0.85

# ── scene parameters ─────────────────────────────────────────────────────────
FIELD_L = 50.0          # half-width of the domain
DETECTION_RADIUS = 30.0

SOLDIER_POS  = np.array([-10.0, -20.0])
DEFENDER_POS = np.array([-10.0, -20.0])   # colocated

ENEMY_FAR    = np.array([35.0, 45.0]) # where the enemy was
# Enemy arrives at the detection-radius edge, along the same direction
_dir = ENEMY_FAR - DEFENDER_POS
_dir = _dir / np.linalg.norm(_dir)
ENEMY_EDGE   = DEFENDER_POS + _dir * DETECTION_RADIUS

# Defender moves partway toward the enemy edge
DEFENDER_NEW = DEFENDER_POS + _dir * (DETECTION_RADIUS * 0.55)

NOISE_STD    = 1.8      # lateral noise on the approach trail
N_TRAIL_PTS  = 60       # resolution of the noisy line
SEED         = 42


def _noisy_trail(start: np.ndarray, end: np.ndarray,
                 n: int, noise_std: float, rng: np.random.Generator) -> np.ndarray:
    """Straight-line path from start→end with perpendicular Gaussian noise."""
    t = np.linspace(0, 1, n)
    base = np.outer(1 - t, start) + np.outer(t, end)   # (n, 2) straight line

    # perpendicular unit vector (rotate direction 90°)
    direction = end - start
    direction = direction / np.linalg.norm(direction)
    perp = np.array([-direction[1], direction[0]])

    lateral_noise = rng.normal(0, noise_std, n)
    base += np.outer(lateral_noise, perp)
    return base


def main() -> None:
    rng = np.random.default_rng(SEED)
    trail = _noisy_trail(ENEMY_FAR, ENEMY_EDGE, N_TRAIL_PTS, NOISE_STD, rng)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)

    with plt.rc_context(IEEE_FIGURE_RC):
        fig, ax = plt.subplots(
            figsize=(2.6 * VISUAL_SCALE, 2.6 * VISUAL_SCALE),
            constrained_layout=True,
        )

        ax.set_xlim(-FIELD_L, FIELD_L)
        ax.set_ylim(-FIELD_L, FIELD_L)
        ax.set_aspect("equal", adjustable="box")
        ax.grid(True, alpha=0.25)
        ax.tick_params(axis="both", labelsize=7 * VISUAL_SCALE)

        # Detection radius
        det_fill = plt.Circle(DEFENDER_POS, DETECTION_RADIUS,
                              color="lightgray", alpha=0.12,
                              label="Detection radius")
        det_edge = plt.Circle(DEFENDER_POS, DETECTION_RADIUS,
                              fill=False, edgecolor="lightgray",
                              linewidth=1.0 * VISUAL_SCALE, linestyle="--")
        ax.add_patch(det_fill)
        ax.add_patch(det_edge)

        # Noisy enemy approach trail
        ax.plot(trail[:, 0], trail[:, 1],
            "-", color="tab:red", lw=1.5 * VISUAL_SCALE, alpha=0.55,
                label="Enemy Drone path")

        # Soldier icon (stays at original position)
        ax.scatter(*SOLDIER_POS, s=120 * VISUAL_SCALE, marker="o",
               color="tab:gray", edgecolors="black", linewidths=0.8 * VISUAL_SCALE,
                   zorder=5, label="Soldier")

        # Defender Drone path: dark green line from start to new position
        ax.plot([DEFENDER_POS[0], DEFENDER_NEW[0]],
                [DEFENDER_POS[1], DEFENDER_NEW[1]],
            "-", color="darkgreen", lw=1.8 * VISUAL_SCALE, alpha=0.85,
                label="Defender Drone path")
        ax.annotate("", xy=DEFENDER_NEW, xytext=DEFENDER_POS,
                    arrowprops=dict(arrowstyle="->", color="darkgreen",
                       lw=1.8 * VISUAL_SCALE))

        # Defender Drone icon at new (moved) position
        ax.scatter(*DEFENDER_NEW, s=130 * VISUAL_SCALE, marker="^",
               color="tab:blue", edgecolors="black", linewidths=0.8 * VISUAL_SCALE,
                   zorder=6, label="Defender Drone")

        # Enemy Drone icon at radius edge
        ax.scatter(*ENEMY_EDGE, s=130 * VISUAL_SCALE, marker="s",
               color="tab:red", edgecolors="black", linewidths=0.8 * VISUAL_SCALE,
                   zorder=5, label="Enemy Drone")

        ax.legend(
            loc="upper left",
            fontsize=7 * VISUAL_SCALE,
            frameon=False,
            borderaxespad=0.3 * VISUAL_SCALE,
            handlelength=2.0 * VISUAL_SCALE,
            markerscale=VISUAL_SCALE,
        )

        fig.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight", pad_inches=0.02)
        pdf_path = OUTPUT_PATH.with_suffix(".pdf")
        fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
        plt.close(fig)

    print(f"Saved: {OUTPUT_PATH}")
    print(f"Saved: {pdf_path}")


if __name__ == "__main__":
    main()
