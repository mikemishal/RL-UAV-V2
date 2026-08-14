"""Create compact IEEE-ready paper figures from existing comparison sweep CSV outputs.

This script only reads existing CSV files and produces publication figures.
It does not run experiments or change evaluation logic.
"""

from __future__ import annotations

import math
from pathlib import Path
from typing import Dict
import sys

PROJECT_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(PROJECT_ROOT))

import matplotlib.pyplot as plt
from matplotlib.ticker import PercentFormatter
import numpy as np
import pandas as pd

from experiments.experiment_config import (
    METHOD_STYLES,
    SWEEP_CONFIG,
    get_output_dir,
    get_sweep_csv_filename,
)

FIGURES_DIR = PROJECT_ROOT / "figures"

METHOD_ORDER = ("baseline", "kalman_baseline", "rl", "rl_kalman")

SWEEP_PANEL_CONFIG = [
    ("defender_speed", "(a)", ["defender_speed", "parameter_value"]),
    ("enemy_speed", "(b)", ["enemy_speed", "parameter_value"]),
    ("detection_radius", "(c)", ["detection_radius", "parameter_value"]),
]

NOISE_X_CANDIDATES = ["measurement_var", "noise_value", "parameter_value"]

IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}

LABEL_FONTSIZE = 12
TICK_FONTSIZE = 10
LEGEND_FONTSIZE = 10
PANEL_LABEL_FONTSIZE = 12


def _search_roots_for_method(method: str) -> list[Path]:
    method_dir = PROJECT_ROOT / get_output_dir(method)
    return [
        method_dir,
        PROJECT_ROOT / "results" / "baseline",
        PROJECT_ROOT / "results" / "rl",
        PROJECT_ROOT / "results" / "rl_kalman",
        PROJECT_ROOT / "results" / "comparison",
    ]


def _resolve_csv_path(method: str, filename_candidates: list[str]) -> Path:
    roots = _search_roots_for_method(method)

    for root in roots:
        for name in filename_candidates:
            direct = root / name
            if direct.exists():
                return direct

            recursive_matches = sorted(root.glob(f"**/{name}"))
            if recursive_matches:
                return recursive_matches[0]

    roots_text = "\n  - ".join(str(p) for p in roots)
    names_text = ", ".join(filename_candidates)
    raise FileNotFoundError(
        f"Missing required CSV for method '{method}'. Tried filenames [{names_text}] in:\n"
        f"  - {roots_text}"
    )


def _required_column(df: pd.DataFrame, candidates: list[str], context: str) -> str:
    for col in candidates:
        if col in df.columns:
            return col
    raise ValueError(
        f"Required column for {context} not found. Tried {candidates}. "
        f"Available columns: {list(df.columns)}"
    )


def _error_bars(df: pd.DataFrame, y_col: str) -> np.ndarray | None:
    if "success_se" in df.columns:
        return df["success_se"].to_numpy()

    low_col = next((c for c in ["success_ci_low", "ci_low", "ci_lower"] if c in df.columns), None)
    high_col = next((c for c in ["success_ci_high", "ci_high", "ci_upper"] if c in df.columns), None)
    if low_col and high_col:
        y = df[y_col].to_numpy()
        y_low = np.clip(y - df[low_col].to_numpy(), a_min=0.0, a_max=None)
        y_high = np.clip(df[high_col].to_numpy() - y, a_min=0.0, a_max=None)
        return np.vstack((y_low, y_high))

    return None


def _compute_y_limits(values: list[float]) -> tuple[float, float]:
    if not values:
        return (0.2, 0.9)

    min_success_rate = min(values)
    max_success_rate = max(values)

    y_min = max(0.0, math.floor((min_success_rate - 0.05) * 10.0) / 10.0)
    y_max = min(1.0, math.ceil((max_success_rate + 0.05) * 10.0) / 10.0)

    if y_max <= y_min:
        return (0.2, 0.9)
    return (y_min, y_max)


def _save_figure(fig: plt.Figure, png_path: Path, pdf_path: Path) -> None:
    png_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    print(f"Saved PDF: {pdf_path}")
    print(f"Saved PNG: {png_path}")


def _load_sweep_df(sweep_name: str, method: str) -> pd.DataFrame:
    filename = get_sweep_csv_filename(sweep_name, method)
    path = _resolve_csv_path(method, [filename])
    df = pd.read_csv(path)
    print(f"Loaded {sweep_name} for {method}: {path}")
    return df


def _noise_filenames_for_method(method: str) -> list[str]:
    primary = get_sweep_csv_filename("measurement_var", method)
    fallback = {
        "baseline": ["sweep_noise_baseline.csv"],
        "kalman_baseline": ["sweep_noise_kalman_baseline.csv"],
        "rl": ["sweep_noise_rl.csv"],
        "rl_kalman": ["sweep_noise_rl_kalman.csv"],
    }
    names = [primary]
    names.extend(fallback.get(method, []))
    return names


def _load_noise_df(method: str) -> pd.DataFrame:
    path = _resolve_csv_path(method, _noise_filenames_for_method(method))
    df = pd.read_csv(path)
    print(f"Loaded measurement noise sweep for {method}: {path}")
    return df


def plot_compact_sweeps_figure() -> None:
    with plt.rc_context(IEEE_RC):
        fig, axes = plt.subplots(1, 3, figsize=(7.1, 2.1), sharey=True, constrained_layout=True)

        handles = {}
        all_success = []

        for ax, (sweep_name, panel_label, x_candidates) in zip(axes, SWEEP_PANEL_CONFIG):
            x_label = SWEEP_CONFIG[sweep_name]["xlabel"]
            if sweep_name == "defender_speed":
                x_label = "Defender Speed ($v_d$)"
            elif sweep_name == "enemy_speed":
                x_label = "Enemy Speed ($v_e$)"
            elif sweep_name == "detection_radius":
                x_label = r"Detection Radius ($R_{\mathrm{det}}$)"

            for method in METHOD_ORDER:
                style = METHOD_STYLES[method]
                df = _load_sweep_df(sweep_name, method)

                x_col = _required_column(df, x_candidates, f"x-axis ({sweep_name}, {method})")
                y_col = _required_column(df, ["success_rate"], f"success rate ({sweep_name}, {method})")
                df_plot = df.sort_values(x_col)

                x = df_plot[x_col].to_numpy()
                y = df_plot[y_col].to_numpy()
                all_success.extend(y.tolist())

                yerr = _error_bars(df_plot, y_col)
                if yerr is not None:
                    line = ax.errorbar(
                        x,
                        y,
                        yerr=yerr,
                        fmt=f"{style['marker']}{style['linestyle']}",
                        color=style["color"],
                        ecolor=style["color"],
                        linewidth=1.2,
                        markersize=3.8,
                        capsize=2.5,
                        capthick=0.8,
                        label=style["name"],
                    )
                    handles.setdefault(method, line[0])
                else:
                    line = ax.plot(
                        x,
                        y,
                        f"{style['marker']}{style['linestyle']}",
                        color=style["color"],
                        linewidth=1.2,
                        markersize=3.8,
                        label=style["name"],
                    )[0]
                    handles.setdefault(method, line)

            ax.set_xlabel(x_label, fontsize=LABEL_FONTSIZE)
            ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
            ax.grid(True, alpha=0.25, linewidth=0.5)
            ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)
            ax.text(
                0.02,
                0.98,
                panel_label,
                transform=ax.transAxes,
                ha="left",
                va="top",
                fontsize=PANEL_LABEL_FONTSIZE,
            )

        axes[0].set_ylabel("Success Rate (%)", fontsize=LABEL_FONTSIZE)
        y_min, y_max = _compute_y_limits(all_success)
        for ax in axes:
            ax.set_ylim(y_min, y_max)

        ordered_handles = [handles[m] for m in METHOD_ORDER]
        ordered_labels = [METHOD_STYLES[m]["name"] for m in METHOD_ORDER]
        fig.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.18),
            ncol=4,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.7,
            handletextpad=0.4,
        )

        _save_figure(
            fig,
            FIGURES_DIR / "compare_all_sweeps_compact.png",
            FIGURES_DIR / "compare_all_sweeps_compact.pdf",
        )
        plt.close(fig)


def plot_compact_noise_figure() -> None:
    with plt.rc_context(IEEE_RC):
        fig, ax = plt.subplots(1, 1, figsize=(7.1, 2.2), constrained_layout=True)

        handles = {}
        all_success = []

        for method in METHOD_ORDER:
            style = METHOD_STYLES[method]
            df = _load_noise_df(method)

            x_col = _required_column(df, NOISE_X_CANDIDATES, f"x-axis (measurement noise, {method})")
            y_col = _required_column(df, ["success_rate"], f"success rate (measurement noise, {method})")
            df_plot = df.sort_values(x_col)

            x = df_plot[x_col].to_numpy()
            y = df_plot[y_col].to_numpy()
            all_success.extend(y.tolist())

            yerr = _error_bars(df_plot, y_col)
            if yerr is not None:
                line = ax.errorbar(
                    x,
                    y,
                    yerr=yerr,
                    fmt=f"{style['marker']}{style['linestyle']}",
                    color=style["color"],
                    ecolor=style["color"],
                    linewidth=1.2,
                    markersize=3.8,
                    capsize=2.5,
                    capthick=0.8,
                    label=style["name"],
                )
                handles.setdefault(method, line[0])
            else:
                line = ax.plot(
                    x,
                    y,
                    f"{style['marker']}{style['linestyle']}",
                    color=style["color"],
                    linewidth=1.2,
                    markersize=3.8,
                    label=style["name"],
                )[0]
                handles.setdefault(method, line)

        ax.set_xlabel("Measurement Noise Variance ($\\sigma^2$)", fontsize=LABEL_FONTSIZE)
        ax.set_ylabel("Success Rate (%)", fontsize=LABEL_FONTSIZE)
        ax.yaxis.set_major_formatter(PercentFormatter(xmax=1.0, decimals=0))
        y_min, y_max = _compute_y_limits(all_success)
        ax.set_ylim(y_min, y_max)
        ax.grid(True, alpha=0.25, linewidth=0.5)
        ax.tick_params(axis="both", labelsize=TICK_FONTSIZE)

        ordered_handles = [handles[m] for m in METHOD_ORDER]
        ordered_labels = [METHOD_STYLES[m]["name"] for m in METHOD_ORDER]
        ax.legend(
            ordered_handles,
            ordered_labels,
            loc="upper center",
            bbox_to_anchor=(0.5, 1.22),
            ncol=4,
            fontsize=LEGEND_FONTSIZE,
            frameon=False,
            columnspacing=1.0,
            handlelength=1.7,
            handletextpad=0.4,
        )

        _save_figure(
            fig,
            FIGURES_DIR / "sweep_noise_all_methods_compact.png",
            FIGURES_DIR / "sweep_noise_all_methods_compact.pdf",
        )
        plt.close(fig)


def main() -> None:
    plot_compact_sweeps_figure()
    plot_compact_noise_figure()


if __name__ == "__main__":
    main()
