"""Export defender-speed plot data to a clean CSV for advisor review.

This script only reads existing CSVs and does not run experiments.
"""

from __future__ import annotations

from pathlib import Path
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd

from experiments.experiment_config import (
    METHOD_STYLES,
    get_output_dir,
    get_sweep_csv_filename,
)

METHOD_ORDER = ("baseline", "kalman_baseline", "rl", "rl_kalman")

# Possible column name variants to handle
DEFENDER_SPEED_COLS = ["defender_speed", "parameter_value", "value"]
SUCCESS_RATE_COLS = ["success_rate"]
SUCCESS_SE_COLS = ["success_se", "standard_error"]
SUCCESS_CI_LOW_COLS = ["success_ci_low"]
SUCCESS_CI_HIGH_COLS = ["success_ci_high"]
N_EPISODES_COLS = ["n_episodes"]
N_SUCCESS_COLS = ["n_success"]


def _get_column(df: pd.DataFrame, candidates: list[str], default=None):
    """Find first matching column name from candidates, or return default."""
    for col in candidates:
        if col in df.columns:
            return col
    return default


def _load_defender_speed_csv(method: str) -> pd.DataFrame:
    """Load defender-speed sweep CSV for a method."""
    method_dir = PROJECT_ROOT / get_output_dir(method)
    csv_path = method_dir / get_sweep_csv_filename("defender_speed", method)
    
    if not csv_path.exists():
        # Try alternative paths
        alt_paths = [
            PROJECT_ROOT / "results" / method / get_sweep_csv_filename("defender_speed", method),
            PROJECT_ROOT / "results" / "baseline" / get_sweep_csv_filename("defender_speed", method),
        ]
        for alt in alt_paths:
            if alt.exists():
                csv_path = alt
                break
    
    if not csv_path.exists():
        raise FileNotFoundError(f"Could not find defender_speed CSV for {method}. Tried: {csv_path}")
    
    print(f"Loading defender_speed for {method}: {csv_path}")
    return pd.read_csv(csv_path)


def _extract_columns(df: pd.DataFrame, method_key: str, method_name: str) -> pd.DataFrame:
    """Extract and standardize relevant columns from raw CSV."""
    result_data = {}
    
    # Add metadata columns
    result_data["method_key"] = [method_key] * len(df)
    result_data["method_name"] = [method_name] * len(df)
    
    # Defender speed column
    defender_speed_col = _get_column(df, DEFENDER_SPEED_COLS)
    if defender_speed_col:
        result_data["defender_speed"] = df[defender_speed_col].tolist()
    
    # Success rate column
    success_rate_col = _get_column(df, SUCCESS_RATE_COLS)
    if success_rate_col:
        result_data["success_rate"] = df[success_rate_col].tolist()
        # success_percent is just success_rate * 100
        result_data["success_percent"] = (df[success_rate_col] * 100).tolist()
    
    # Optional columns: SE and CI
    success_se_col = _get_column(df, SUCCESS_SE_COLS)
    if success_se_col:
        result_data["success_se"] = df[success_se_col].tolist()
    
    success_ci_low_col = _get_column(df, SUCCESS_CI_LOW_COLS)
    if success_ci_low_col:
        result_data["success_ci_low"] = df[success_ci_low_col].tolist()
    
    success_ci_high_col = _get_column(df, SUCCESS_CI_HIGH_COLS)
    if success_ci_high_col:
        result_data["success_ci_high"] = df[success_ci_high_col].tolist()
    
    n_episodes_col = _get_column(df, N_EPISODES_COLS)
    if n_episodes_col:
        result_data["n_episodes"] = df[n_episodes_col].tolist()
    
    n_success_col = _get_column(df, N_SUCCESS_COLS)
    if n_success_col:
        result_data["n_success"] = df[n_success_col].tolist()
    
    return pd.DataFrame(result_data)


def main():
    """Load and export defender-speed plot data."""
    frames = []
    
    for method_key in METHOD_ORDER:
        df = _load_defender_speed_csv(method_key)
        method_name = METHOD_STYLES[method_key]["name"]
        extracted = _extract_columns(df, method_key, method_name)
        frames.append(extracted)
    
    combined = pd.concat(frames, ignore_index=True)
    
    # Ensure output directory exists
    output_dir = PROJECT_ROOT / "results" / "comparison"
    output_dir.mkdir(parents=True, exist_ok=True)
    
    output_path = output_dir / "defender_speed_plot_data_for_advisor.csv"
    combined.to_csv(output_path, index=False)
    
    print(f"\n✓ Saved to: {output_path}")
    print(f"\nPreview ({len(combined)} rows):")
    print(combined.to_string())


if __name__ == "__main__":
    main()
