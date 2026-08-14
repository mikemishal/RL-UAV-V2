"""
Shared Experiment Configuration.

This module centralizes all experiment parameters to ensure baseline,
kalman-baseline, RL, and RL-with-Kalman evaluations use identical settings.
Any changes here automatically apply to all synchronized experiment pipelines.

=============================================================================
SYNCHRONIZED EXPERIMENT TRACKS
=============================================================================

    1. Baseline (greedy pursuit)
    2. Kalman Baseline (greedy pursuit on Kalman estimates)
    3. Direct RL (PPO with true observations)
    4. RL-with-Kalman (PPO with Kalman-filtered observations)

All tracks share:
    - Parameter sweep ranges
    - Episode counts per evaluation
    - Evaluation seed sequences
    - CSV column schemas
    - Plot formatting

Usage:
    from experiments.experiment_config import (
        CONFIG, SWEEP_CONFIG, EVAL_CONFIG, 
        KALMAN_CONFIG, METHOD_STYLES
    )
    
    # Get parameter ranges
    speeds = CONFIG.DEFENDER_SPEEDS
    
    # Use sweep-specific defaults
    n_episodes = SWEEP_CONFIG["defender_speed"]["n_episodes"]
    
    # Get Kalman parameters
    process_var = KALMAN_CONFIG["process_var"]
    
    # Get method styling
    color = METHOD_STYLES["rl_kalman"]["color"]

Modification Guide:
    - To change sweep ranges: Update the *_SPEEDS, *_RADII class attributes
    - To change noise sweep: Update NOISE_SWEEP_VALUES (variance values)
    - To change episode counts: Update SWEEP_CONFIG or EVAL_CONFIG
    - To change seeds: Update SEED_OFFSET
    - To change Kalman params: Update KALMAN_CONFIG
    - All changes propagate to ALL experiment pipelines automatically
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any


@dataclass(frozen=True)
class ExperimentConfig:
    """
    Centralized experiment configuration.
    
    All parameter ranges, episode counts, and evaluation settings are
    defined here to prevent drift between the four synchronized experiment
    tracks: baseline, kalman baseline, direct RL, and RL-Kalman.
    """
    
    # =========================================================================
    # PARAMETER SWEEP RANGES
    # =========================================================================
    
    # Defender speed sweep (1D)
    DEFENDER_SPEEDS: tuple[float, ...] = (10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
    
    # Enemy speed sweep (1D)
    ENEMY_SPEEDS: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0, 16.0)
    
    # Detection radius sweep (1D)
    DETECTION_RADII: tuple[float, ...] = (5.0, 8.0, 10.0, 12.0, 15.0)

    # Measurement-noise sweep values (variance, not standard deviation).
    # These are the canonical shared values for all four experiment tracks.
    NOISE_SWEEP_VALUES: tuple[float, ...] = (0.0, 0.1, 0.25, 0.5, 1.0, 2.0)
    # Backward-compatible alias used by existing scripts/config keys.
    MEASUREMENT_VARS: tuple[float, ...] = NOISE_SWEEP_VALUES
    
    # 2D speed grid (defender × enemy)
    GRID_DEFENDER_SPEEDS: tuple[float, ...] = (10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0)
    GRID_ENEMY_SPEEDS: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0)
    
    # =========================================================================
    # EPISODE COUNTS
    # =========================================================================
    
    # Main evaluation (Monte Carlo) for final runs
    NUM_EPISODES: int = 100000
    # Backward-compatible alias used by existing scripts/config keys.
    EVAL_EPISODES: int = NUM_EPISODES
    
    # 1D parameter sweeps (per parameter value)
    SWEEP_EPISODES: int = 200
    
    # 2D grid sweep (per grid point) - lower due to N×M scaling
    GRID_EPISODES: int = 100
    
    # Trajectory capture (for qualitative analysis)
    TRAJECTORY_EPISODES: int = 50
    
    # =========================================================================
    # REPRODUCIBILITY
    # =========================================================================
    
    SEED_OFFSET: int = 0
    
    # =========================================================================
    # PLOT FORMATTING
    # =========================================================================
    
    PLOT_DPI: int = 150
    PLOT_FIGSIZE: tuple[float, float] = (10, 6)
    GRID_FIGSIZE: tuple[float, float] = (12, 5)
    
    # Colors for four-track comparison
    BASELINE_COLOR: str = "#2E86AB"  # Blue
    KALMAN_BASELINE_COLOR: str = "#F39C12"  # Orange
    RL_COLOR: str = "#9B59B6"        # Purple
    RL_KALMAN_COLOR: str = "#E74C3C" # Red
    
    # =========================================================================
    # DEFAULT ENVIRONMENT PARAMETERS
    # =========================================================================
    
    DEFAULT_ENEMY_SPEED: float = 12.0
    DEFAULT_DEFENDER_SPEED: float = 18.0
    DEFAULT_DETECTION_RADIUS: float = 10.0
    
    # =========================================================================
    # CSV SCHEMA
    # =========================================================================
    
    # Core columns that must appear in all sweep results
    SWEEP_COLUMNS: tuple[str, ...] = (
        "n_episodes",
        "n_success",
        "success_rate",
        "success_se",
        "mean_episode_length",
        "std_episode_length",
        "mean_detection_time",
        "mean_intercept_time",
    )
    
    # Additional columns for comparison DataFrames
    COMPARISON_COLUMNS: tuple[str, ...] = (
        "seed",
        "outcome",
        "success",
        "episode_length",
        "total_reward",
        "detected",
        "detection_time",
        "intercept_time",
    )


# Singleton instance for easy import
CONFIG = ExperimentConfig()


# =========================================================================
# SWEEP-SPECIFIC CONFIGURATIONS
# =========================================================================

SWEEP_CONFIG: dict[str, dict[str, Any]] = {
    "defender_speed": {
        "parameter_name": "defender_speed",
        "parameter_values": CONFIG.DEFENDER_SPEEDS,
        "n_episodes": CONFIG.SWEEP_EPISODES,
        "csv_filename": "sweep_defender_speed.csv",
        "plot_filename": "sweep_defender_speed.png",
        "xlabel": "Defender Speed (v_d)",
        "title_suffix": "Defender Speed",
    },
    "enemy_speed": {
        "parameter_name": "enemy_speed",
        "parameter_values": CONFIG.ENEMY_SPEEDS,
        "n_episodes": CONFIG.SWEEP_EPISODES,
        "csv_filename": "sweep_enemy_speed.csv",
        "plot_filename": "sweep_enemy_speed.png",
        "xlabel": "Enemy Speed (v_e)",
        "title_suffix": "Enemy Speed",
    },
    "detection_radius": {
        "parameter_name": "detection_radius",
        "parameter_values": CONFIG.DETECTION_RADII,
        "n_episodes": CONFIG.SWEEP_EPISODES,
        "csv_filename": "sweep_detection_radius.csv",
        "plot_filename": "sweep_detection_radius.png",
        "xlabel": "Detection Radius (r_det)",
        "title_suffix": "Detection Radius",
    },
    "measurement_var": {
        "parameter_name": "measurement_var",
        "parameter_values": CONFIG.NOISE_SWEEP_VALUES,
        "n_episodes": CONFIG.SWEEP_EPISODES,
        "csv_filename": "sweep_measurement_var.csv",
        "plot_filename": "sweep_measurement_var.png",
        "xlabel": "Measurement Noise Variance (sigma^2)",
        "title_suffix": "Measurement Noise Variance",
    },
    "speed_grid": {
        "defender_speeds": CONFIG.GRID_DEFENDER_SPEEDS,
        "enemy_speeds": CONFIG.GRID_ENEMY_SPEEDS,
        "n_episodes": CONFIG.GRID_EPISODES,
        "csv_filename": "sweep_speed_grid.csv",
        "plot_filename": "sweep_speed_grid.png",
    },
}


EVAL_CONFIG: dict[str, Any] = {
    "n_episodes": CONFIG.EVAL_EPISODES,
    "seed_offset": CONFIG.SEED_OFFSET,
    "trajectory_episodes": CONFIG.TRAJECTORY_EPISODES,
    "baseline_csv": "baseline_results.csv",
    "kalman_baseline_csv": "kalman_baseline_results.csv",
    "rl_csv": "rl_results.csv",
    "rl_kalman_csv": "rl_kalman_results.csv",
}


# =========================================================================
# KALMAN FILTER CONFIGURATION
# =========================================================================

KALMAN_CONFIG: dict[str, Any] = {
    # Process noise (model uncertainty)
    "process_var": 1.0,
    
    # Measurement noise variance (sigma^2), kept consistent with EnvConfig.measurement_var
    "measurement_var": 0.5,
    
    # Prediction lead time (for anticipatory pursuit)
    "lead_time": 0.0,
}


# =========================================================================
# METHOD-SPECIFIC STYLING
# =========================================================================

METHOD_STYLES: dict[str, dict[str, Any]] = {
    "baseline": {
        "name": "Greedy Baseline",
        "short_name": "Baseline",
        "color": CONFIG.BASELINE_COLOR,
        "marker": "o",
        "linestyle": "-",
        "output_dir": "results/baseline",
    },
    "kalman_baseline": {
        "name": "Kalman Baseline",
        "short_name": "Kalman-Baseline",
        "color": CONFIG.KALMAN_BASELINE_COLOR,
        "marker": "D",
        "linestyle": "-",
        "output_dir": "results/baseline",
    },
    "rl": {
        "name": "PPO (Direct RL)",
        "short_name": "RL",
        "color": CONFIG.RL_COLOR,
        "marker": "s",
        "linestyle": "--",
        "output_dir": "results/rl",
    },
    "rl_kalman": {
        "name": "PPO (RL + Kalman)",
        "short_name": "RL-Kalman",
        "color": CONFIG.RL_KALMAN_COLOR,
        "marker": "^",
        "linestyle": "-.",
        "output_dir": "results/rl_kalman",
    },
}


# =========================================================================
# OUTPUT PATHS
# =========================================================================

OUTPUT_DIRS: dict[str, str] = {
    "baseline": "results/baseline",
    "kalman_baseline": "results/baseline",
    "rl": "results/rl",
    "rl_kalman": "results/rl_kalman",
    "comparison": "results/comparison",
}

METHOD_KEYS: tuple[str, ...] = (
    "baseline",
    "kalman_baseline",
    "rl",
    "rl_kalman",
)


# =========================================================================
# HELPER FUNCTIONS
# =========================================================================

def get_sweep_defaults(sweep_name: str) -> dict[str, Any]:
    """
    Get default configuration for a specific sweep type.
    
    Args:
        sweep_name: One of "defender_speed", "enemy_speed", "detection_radius", "speed_grid"
    
    Returns:
        Dictionary with sweep configuration.
    
    Example:
        >>> defaults = get_sweep_defaults("defender_speed")
        >>> speeds = defaults["parameter_values"]  # (10.0, 12.0, ...)
        >>> n_eps = defaults["n_episodes"]  # 200
    """
    if sweep_name not in SWEEP_CONFIG:
        raise ValueError(f"Unknown sweep: {sweep_name}. "
                        f"Available: {list(SWEEP_CONFIG.keys())}")
    return SWEEP_CONFIG[sweep_name].copy()


def get_kalman_defaults() -> dict[str, Any]:
    """
    Get default Kalman filter configuration.
    
    Returns:
        Dictionary with process_var, measurement_var, lead_time.
    
    Example:
        >>> kalman = get_kalman_defaults()
        >>> process_var = kalman["process_var"]  # 1.0
    """
    return KALMAN_CONFIG.copy()


def get_method_style(method: str) -> dict[str, Any]:
    """
    Get styling configuration for a method.
    
    Args:
        method: One of "baseline", "kalman_baseline", "rl", "rl_kalman"
    
    Returns:
        Dictionary with name, color, marker, linestyle, output_dir.
    """
    if method not in METHOD_STYLES:
        raise ValueError(f"Unknown method: {method}. "
                        f"Available: {list(METHOD_STYLES.keys())}")
    return METHOD_STYLES[method].copy()


def format_speeds_for_cli(speeds: tuple[float, ...] | list[float]) -> str:
    """Convert speed tuple to comma-separated string for CLI defaults."""
    return ",".join(str(s) for s in speeds)


def parse_speeds_from_cli(speeds_str: str) -> list[float]:
    """Parse comma-separated speed string from CLI."""
    return [float(s.strip()) for s in speeds_str.split(",")]


def get_output_dir(method: str) -> str:
    """Get output directory path for a method."""
    if method not in OUTPUT_DIRS:
        raise ValueError(f"Unknown method: {method}. "
                        f"Available: {list(OUTPUT_DIRS.keys())}")
    return OUTPUT_DIRS[method]


def get_eval_csv_filename(method: str) -> str:
    """Get the standard evaluation CSV filename for a method."""
    mapping = {
        "baseline": EVAL_CONFIG["baseline_csv"],
        "kalman_baseline": EVAL_CONFIG["kalman_baseline_csv"],
        "rl": EVAL_CONFIG["rl_csv"],
        "rl_kalman": EVAL_CONFIG["rl_kalman_csv"],
    }
    if method not in mapping:
        raise ValueError(f"Unknown method: {method}. Available: {list(mapping.keys())}")
    return mapping[method]


def get_sweep_csv_filename(sweep_name: str, method: str) -> str:
    """
    Get the standard sweep CSV filename for a method.

    Kalman baseline keeps the same sweep names as the other tracks with an
    added `_kalman_baseline` suffix to avoid collisions inside results/baseline.
    """
    if sweep_name not in SWEEP_CONFIG:
        raise ValueError(f"Unknown sweep: {sweep_name}. Available: {list(SWEEP_CONFIG.keys())}")

    base_name = SWEEP_CONFIG[sweep_name]["csv_filename"]
    if method == "kalman_baseline":
        stem, ext = base_name.rsplit(".", 1)
        return f"{stem}_kalman_baseline.{ext}"
    if method not in METHOD_KEYS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_KEYS)}")
    return base_name


def get_sweep_plot_filename(sweep_name: str, method: str, variant: str | None = None) -> str:
    """
    Get the standard sweep plot filename for a method.

    Args:
        sweep_name: Sweep key from SWEEP_CONFIG.
        method: Method key.
        variant: Optional suffix such as "heatmap" or "contour".
    """
    if sweep_name not in SWEEP_CONFIG:
        raise ValueError(f"Unknown sweep: {sweep_name}. Available: {list(SWEEP_CONFIG.keys())}")

    base_name = SWEEP_CONFIG[sweep_name]["plot_filename"]
    stem, ext = base_name.rsplit(".", 1)

    if method == "kalman_baseline":
        stem = f"{stem}_kalman_baseline"
    elif method not in METHOD_KEYS:
        raise ValueError(f"Unknown method: {method}. Available: {list(METHOD_KEYS)}")

    if variant:
        stem = f"{stem}_{variant}"
    return f"{stem}.{ext}"


# =========================================================================
# VERSION INFO
# =========================================================================

__version__ = "2.1.0"
__doc_version__ = """
Experiment Configuration v2.1.0 (Four-Method Synchronization)

Methods Supported:
  - Greedy Baseline (direct pursuit)
    - Kalman Baseline (greedy pursuit on Kalman estimates)
  - PPO Direct RL (raw observations)
  - PPO RL-Kalman (Kalman-filtered observations)

Parameter Ranges:
  - Defender speeds (1D): {defender_speeds}
  - Enemy speeds (1D):    {enemy_speeds}
  - Detection radii:      {detection_radii}
  - Grid defender:        {grid_defender}
  - Grid enemy:           {grid_enemy}

Episode Counts:
  - Main evaluation:      {eval_episodes}
  - 1D sweeps:            {sweep_episodes} per value
  - 2D grid:              {grid_episodes} per point

Kalman Filter Config:
  - Process variance:     {process_var}
  - Measurement variance: {measurement_var}
  - Lead time (s):        {lead_time}

Seed offset: {seed_offset}
""".format(
    defender_speeds=CONFIG.DEFENDER_SPEEDS,
    enemy_speeds=CONFIG.ENEMY_SPEEDS,
    detection_radii=CONFIG.DETECTION_RADII,
    grid_defender=CONFIG.GRID_DEFENDER_SPEEDS,
    grid_enemy=CONFIG.GRID_ENEMY_SPEEDS,
    eval_episodes=CONFIG.EVAL_EPISODES,
    sweep_episodes=CONFIG.SWEEP_EPISODES,
    grid_episodes=CONFIG.GRID_EPISODES,
    process_var=KALMAN_CONFIG["process_var"],
    measurement_var=KALMAN_CONFIG["measurement_var"],
    lead_time=KALMAN_CONFIG["lead_time"],
    seed_offset=CONFIG.SEED_OFFSET,
)


def print_config_summary():
    """Print current experiment configuration."""
    print(__doc_version__)


if __name__ == "__main__":
    print_config_summary()
