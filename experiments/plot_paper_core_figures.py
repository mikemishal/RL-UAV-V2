"""
Publication figure generator -- CORE figures only (nominal six-method
performance + acquisition ablation).

Reads ONLY frozen paper-facing evidence under results/paper_evidence/ and
writes vector PDF + 300 DPI PNG figures, plotted-data CSVs, captions, and a
generation manifest under results/paper_figures/.

This script performs NO simulation of any kind:
    - NEVER instantiates SoldierEnv
    - NEVER imports stable_baselines3 / loads a PPO model
    - NEVER runs policy inference or an environment step
    - NEVER consumes an evaluation seed

Only pandas, numpy, matplotlib, pathlib, hashlib, json, subprocess are used.

Usage:
    python experiments/plot_paper_core_figures.py
"""

from __future__ import annotations

import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
EVIDENCE_ROOT = PROJECT_ROOT / "results" / "paper_evidence"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "paper_figures"

SIX_METHODS = ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
METHOD_DISPLAY_NAME = {
    "greedy": "Greedy", "kalman_greedy": "Kalman-Greedy", "pn": "PN",
    "lead": "Lead", "ppo": "PPO", "ppo_kalman": "PPO-Kalman",
}

# =============================================================================
# CENTRAL METHOD STYLE MAP -- reused by every current and future figure.
#
# Colorblind-accessible Okabe-Ito palette. Semantic grouping:
#   - Direct (unfiltered) methods use a SOLID line style.
#   - Kalman-filtered variants use a DASHED line style and a color drawn from
#     the SAME hue family as their direct counterpart (Kalman-Greedy shares
#     Greedy's blue family; PPO-Kalman shares PPO's warm family) -- so
#     filtering is encoded redundantly via linestyle AND color family, not
#     color alone. Marker shapes are unique per method for grayscale/print
#     distinguishability. PPO is NOT given a visually dominant color/marker
#     (it uses the same visual weight as every other method).
# =============================================================================
METHOD_STYLE = {
    "greedy":        {"color": "#0072B2", "marker": "o", "linestyle": "-",  "label": "Greedy"},
    "kalman_greedy": {"color": "#56B4E9", "marker": "D", "linestyle": "--", "label": "Kalman-Greedy"},
    "pn":            {"color": "#009E73", "marker": "s", "linestyle": "-",  "label": "PN"},
    "lead":          {"color": "#D55E00", "marker": "^", "linestyle": "-",  "label": "Lead"},
    "ppo":           {"color": "#E69F00", "marker": "*", "linestyle": "-",  "label": "PPO"},
    "ppo_kalman":    {"color": "#CC79A7", "marker": "v", "linestyle": "--", "label": "PPO-Kalman"},
}

# =============================================================================
# GLOBAL PUBLICATION STYLE (IEEE two-column scale; no proprietary font)
# =============================================================================
plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 8,
    "axes.labelsize": 8.5,
    "axes.titlesize": 8.5,
    "xtick.labelsize": 7.5,
    "ytick.labelsize": 7.5,
    "legend.fontsize": 7.5,
    "figure.dpi": 300,
    "savefig.dpi": 300,
    "axes.grid": False,
    "axes.spines.top": False,
    "axes.spines.right": False,
})

ONE_COLUMN_WIDTH_IN = 3.45   # IEEE one-column width
TWO_COLUMN_WIDTH_IN = 7.16   # IEEE two-column width
PNG_DPI = 300


def _get_git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _get_git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _sha256_of_file(path: Path) -> str:
    h = hashlib.sha256()
    h.update(path.read_bytes())
    return h.hexdigest()


def _assert_six_methods_only(df: pd.DataFrame, method_col: str) -> None:
    values = set(str(v) for v in df[method_col].dropna().unique())
    forbidden = {"pn_kalman", "lead_kalman", "PN-Kalman", "Lead-Kalman"}
    assert not (values & forbidden), f"forbidden method(s) found in {method_col}: {values & forbidden}"


def _save_figure(fig: plt.Figure, stem: str) -> tuple[Path, Path]:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pdf_path = OUTPUT_ROOT / f"{stem}.pdf"
    png_path = OUTPUT_ROOT / f"{stem}.png"
    fig.savefig(pdf_path, format="pdf")
    fig.savefig(png_path, format="png", dpi=PNG_DPI)
    return pdf_path, png_path


# =============================================================================
# FIGURE A -- NOMINAL SIX-METHOD PERFORMANCE (forest / dot-whisker plot)
# =============================================================================
_EXPECTED_NOMINAL = {
    "greedy": 0.7576, "kalman_greedy": 0.6912, "pn": 0.7304,
    "lead": 0.8310, "ppo": 0.8539, "ppo_kalman": 0.8157,
}

# Grouped presentation order (NOT sorted by outcome) -- chosen over an
# outcome-sorted order to avoid visually editorializing the comparison; see
# module docstring / final report for rationale. Top-to-bottom as listed.
NOMINAL_ORDER = ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")


def load_and_validate_nominal_table() -> pd.DataFrame:
    df = pd.read_csv(EVIDENCE_ROOT / "final_nominal_paper_table.csv")
    _assert_six_methods_only(df, "method_key")
    assert set(df["method_key"]) == set(SIX_METHODS), "final_nominal_paper_table.csv must contain exactly the six canonical methods"

    discrepancies = []
    for method_key, expected in _EXPECTED_NOMINAL.items():
        actual = float(df[df["method_key"] == method_key]["success_rate"].iloc[0])
        if abs(actual - expected) > 0.005:
            discrepancies.append(f"{method_key}: expected~{expected}, actual={actual}")
    if discrepancies:
        raise RuntimeError(f"Nominal values differ materially from expected values -- STOPPING: {discrepancies}")
    return df


def plot_nominal_performance(df: pd.DataFrame) -> dict:
    ordered = df.set_index("method_key").loc[list(NOMINAL_ORDER)].reset_index()
    y_positions = np.arange(len(ordered))[::-1]  # first method plotted at the TOP

    fig, ax = plt.subplots(figsize=(ONE_COLUMN_WIDTH_IN, 2.55), constrained_layout=True)

    for y, (_, row) in zip(y_positions, ordered.iterrows()):
        style = METHOD_STYLE[row["method_key"]]
        rate_pct = row["success_rate"] * 100
        lo_pct = row["success_ci_low"] * 100
        hi_pct = row["success_ci_high"] * 100
        ax.plot([lo_pct, hi_pct], [y, y], color=style["color"], linestyle=style["linestyle"], linewidth=1.3, zorder=2)
        ax.plot(rate_pct, y, marker=style["marker"], color=style["color"], markersize=6,
                 markeredgecolor="black", markeredgewidth=0.4, zorder=3)
        ax.annotate(f"{rate_pct:.1f}%", xy=(hi_pct, y), xytext=(3, 0), textcoords="offset points",
                    va="center", ha="left", fontsize=7.5)

    ax.set_yticks(y_positions)
    ax.set_yticklabels([METHOD_STYLE[m]["label"] for m in ordered["method_key"]])
    ax.set_xlabel("Safe Interception Success Rate (%)")
    # X-axis range: data spans ~69-85%; a modest truncation (60-92%) keeps CI
    # whiskers and right-side percentage annotations legible without
    # implying a zero baseline is meaningful for a CI-comparison forest plot.
    ax.set_xlim(60, 92)
    ax.set_xticks([60, 65, 70, 75, 80, 85, 90])
    ax.set_ylim(-0.7, len(ordered) - 0.3)
    ax.tick_params(axis="y", length=0)

    pdf_path, png_path = _save_figure(fig, "fig_nominal_performance")
    plt.close(fig)

    plotted = ordered[["method_key", "success_rate", "success_ci_low", "success_ci_high"]].copy()
    plotted["method"] = plotted["method_key"].map(METHOD_DISPLAY_NAME)
    plotted["success_rate_pct"] = plotted["success_rate"] * 100
    plotted["success_ci_low_pct"] = plotted["success_ci_low"] * 100
    plotted["success_ci_high_pct"] = plotted["success_ci_high"] * 100
    plotted = plotted[["method", "method_key", "success_rate_pct", "success_ci_low_pct", "success_ci_high_pct"]]
    plotted.to_csv(OUTPUT_ROOT / "fig_nominal_performance_data.csv", index=False)

    caption = (
        "Safe-interception success under the locked nominal configuration. PPO significantly outperforms Greedy "
        "and PN and is statistically competitive with predictive Lead interception. Error bars denote 95% "
        "confidence intervals; PPO-based intervals additionally account for variation across the three "
        "independently trained policies."
    )
    (OUTPUT_ROOT / "fig_nominal_performance_caption.txt").write_text(caption)

    return {"pdf": pdf_path, "png": png_path, "caption": caption, "dimensions_in": (ONE_COLUMN_WIDTH_IN, 2.55)}


# =============================================================================
# FIGURE B -- ACQUISITION ABLATION (two-panel: success ; detection time)
# =============================================================================
ACQUISITION_ORDER = ("lead", "escorted_ppo", "full_ppo")
ACQUISITION_DISPLAY_NAME = {"lead": "Lead", "escorted_ppo": "Escorted PPO", "full_ppo": "Full PPO"}
# Escorted/Full PPO are ablation CONDITIONS of the single PPO policy family --
# reuse PPO's style (color/marker) for both, distinguished by linestyle
# (Full PPO solid, Escorted PPO dashed) and Lead keeps its own style.
ACQUISITION_STYLE = {
    "lead": {"color": METHOD_STYLE["lead"]["color"], "marker": METHOD_STYLE["lead"]["marker"], "linestyle": "-"},
    "escorted_ppo": {"color": METHOD_STYLE["ppo"]["color"], "marker": METHOD_STYLE["ppo"]["marker"], "linestyle": "--"},
    "full_ppo": {"color": METHOD_STYLE["ppo"]["color"], "marker": METHOD_STYLE["ppo"]["marker"], "linestyle": "-"},
}

_EXPECTED_ACQUISITION_SUCCESS = {"lead": 0.848, "escorted_ppo": 0.8243, "full_ppo": 0.8607}
_EXPECTED_ACQUISITION_DETECTION_S = {"lead": 26.93, "escorted_ppo": 26.93, "full_ppo": 10.97}
_EXPECTED_FULL_MINUS_ESCORTED_PP = 3.63
_EXPECTED_ESCORTED_MINUS_LEAD_PP = -2.37


def load_and_validate_acquisition_summary() -> tuple[pd.DataFrame, pd.DataFrame]:
    df = pd.read_csv(EVIDENCE_ROOT / "acquisition_paper_summary.csv")
    conditions = df[df["row_type"] == "condition_success"].copy()
    comparisons = df[df["row_type"].isin(["comparison", "detection_time_comparison"])].copy()

    _assert_six_methods_only(conditions, "condition")  # condition col reuses method-like values here
    assert set(conditions["condition"]) == set(ACQUISITION_ORDER), \
        f"acquisition conditions must be exactly {ACQUISITION_ORDER}, got {set(conditions['condition'])}"

    discrepancies = []
    for cond, expected in _EXPECTED_ACQUISITION_SUCCESS.items():
        actual = float(conditions[conditions["condition"] == cond]["success_rate"].iloc[0])
        if abs(actual - expected) > 0.005:
            discrepancies.append(f"{cond} success: expected~{expected}, actual={actual}")
    for cond, expected in _EXPECTED_ACQUISITION_DETECTION_S.items():
        actual = float(conditions[conditions["condition"] == cond]["mean_detection_time_seconds"].iloc[0])
        if abs(actual - expected) > 0.5:
            discrepancies.append(f"{cond} detection time: expected~{expected}, actual={actual}")

    full_minus_escorted = comparisons[comparisons["comparison"] == "Full PPO - Escorted PPO"].iloc[0]
    if abs(full_minus_escorted["observed_diff_pp"] - _EXPECTED_FULL_MINUS_ESCORTED_PP) > 0.1:
        discrepancies.append(f"Full-Escorted diff: expected~{_EXPECTED_FULL_MINUS_ESCORTED_PP}, actual={full_minus_escorted['observed_diff_pp']}")
    if full_minus_escorted["ci_low_pp"] <= 0 <= full_minus_escorted["ci_high_pp"]:
        discrepancies.append("Full PPO - Escorted PPO CI unexpectedly includes 0")

    escorted_minus_lead = comparisons[comparisons["comparison"] == "Escorted PPO - Lead"].iloc[0]
    if abs(escorted_minus_lead["observed_diff_pp"] - _EXPECTED_ESCORTED_MINUS_LEAD_PP) > 0.1:
        discrepancies.append(f"Escorted-Lead diff: expected~{_EXPECTED_ESCORTED_MINUS_LEAD_PP}, actual={escorted_minus_lead['observed_diff_pp']}")
    if not (escorted_minus_lead["ci_low_pp"] <= 0 <= escorted_minus_lead["ci_high_pp"]):
        discrepancies.append("Escorted PPO - Lead CI unexpectedly excludes 0")

    if discrepancies:
        raise RuntimeError(f"Acquisition ablation values differ materially from expected values -- STOPPING: {discrepancies}")

    return conditions, comparisons


def plot_acquisition_ablation(conditions: pd.DataFrame, comparisons: pd.DataFrame) -> dict:
    ordered = conditions.set_index("condition").loc[list(ACQUISITION_ORDER)].reset_index()
    y_positions = np.arange(len(ordered))[::-1]

    fig, (ax_a, ax_b) = plt.subplots(1, 2, figsize=(TWO_COLUMN_WIDTH_IN, 2.3), constrained_layout=True)

    # --- Panel (a): success rate, point + 95% CI ---
    for y, (_, row) in zip(y_positions, ordered.iterrows()):
        style = ACQUISITION_STYLE[row["condition"]]
        rate_pct = row["success_rate"] * 100
        lo_pct = row["success_ci_low"] * 100
        hi_pct = row["success_ci_high"] * 100
        ax_a.plot([lo_pct, hi_pct], [y, y], color=style["color"], linestyle=style["linestyle"], linewidth=1.3, zorder=2)
        ax_a.plot(rate_pct, y, marker=style["marker"], color=style["color"], markersize=6,
                   markeredgecolor="black", markeredgewidth=0.4, zorder=3)
        ax_a.annotate(f"{rate_pct:.1f}%", xy=(hi_pct, y), xytext=(3, 0), textcoords="offset points",
                       va="center", ha="left", fontsize=7)

    ax_a.set_yticks(y_positions)
    ax_a.set_yticklabels([ACQUISITION_DISPLAY_NAME[c] for c in ordered["condition"]])
    ax_a.set_xlabel("Safe Interception Success Rate (%)")
    ax_a.set_xlim(72, 94)
    ax_a.set_ylim(-0.7, len(ordered) - 0.3)
    ax_a.tick_params(axis="y", length=0)
    ax_a.text(0.02, 0.98, "(a) Interception success", transform=ax_a.transAxes, va="top", ha="left", fontsize=7.5)

    # Bracket annotation: Full PPO (top, y=2) vs Escorted PPO (middle, y=1).
    full_row = ordered[ordered["condition"] == "full_ppo"].iloc[0]
    escorted_row = ordered[ordered["condition"] == "escorted_ppo"].iloc[0]
    y_full = y_positions[ordered["condition"].tolist().index("full_ppo")]
    y_escorted = y_positions[ordered["condition"].tolist().index("escorted_ppo")]
    bracket_x = max(full_row["success_ci_high"], escorted_row["success_ci_high"]) * 100 + 5.5
    ax_a.plot([bracket_x, bracket_x], [y_escorted, y_full], color="black", linewidth=0.8, zorder=1)
    ax_a.plot([bracket_x - 0.6, bracket_x], [y_escorted, y_escorted], color="black", linewidth=0.8, zorder=1)
    ax_a.plot([bracket_x - 0.6, bracket_x], [y_full, y_full], color="black", linewidth=0.8, zorder=1)
    full_minus_escorted = comparisons[comparisons["comparison"] == "Full PPO - Escorted PPO"].iloc[0]
    ax_a.text(bracket_x + 0.8, (y_full + y_escorted) / 2,
               f"$\\Delta$ = +{full_minus_escorted['observed_diff_pp']:.2f} pp\n95% CI [{full_minus_escorted['ci_low_pp']:.2f}, {full_minus_escorted['ci_high_pp']:.2f}]",
               va="center", ha="left", fontsize=6.5)
    ax_a.set_xlim(72, 100)

    # --- Panel (b): mean detection time, means only (no frozen CI available) ---
    for y, (_, row) in zip(y_positions, ordered.iterrows()):
        style = ACQUISITION_STYLE[row["condition"]]
        det_time = row["mean_detection_time_seconds"]
        ax_b.plot(det_time, y, marker=style["marker"], color=style["color"], markersize=6,
                   markeredgecolor="black", markeredgewidth=0.4, zorder=3)
        ax_b.plot([0, det_time], [y, y], color=style["color"], linestyle=style["linestyle"], linewidth=1.0, zorder=2, alpha=0.5)
        ax_b.annotate(f"{det_time:.1f} s", xy=(det_time, y), xytext=(4, 0), textcoords="offset points",
                       va="center", ha="left", fontsize=7)

    ax_b.set_yticks(y_positions)
    ax_b.set_yticklabels([ACQUISITION_DISPLAY_NAME[c] for c in ordered["condition"]])
    ax_b.set_xlabel("Mean Detection Time (s)")
    ax_b.set_xlim(0, 34)
    ax_b.set_ylim(-0.7, len(ordered) - 0.3)
    ax_b.tick_params(axis="y", length=0)
    ax_b.text(0.02, 0.98, "(b) Detection time", transform=ax_b.transAxes, va="top", ha="left", fontsize=7.5)

    pdf_path, png_path = _save_figure(fig, "fig_acquisition_ablation")
    plt.close(fig)

    plotted_conditions = ordered[["condition", "success_rate", "success_ci_low", "success_ci_high", "mean_detection_time_seconds"]].copy()
    plotted_conditions["condition_display"] = plotted_conditions["condition"].map(ACQUISITION_DISPLAY_NAME)
    plotted_conditions["success_rate_pct"] = plotted_conditions["success_rate"] * 100
    plotted_conditions["success_ci_low_pct"] = plotted_conditions["success_ci_low"] * 100
    plotted_conditions["success_ci_high_pct"] = plotted_conditions["success_ci_high"] * 100
    plotted_comparisons = comparisons[comparisons["comparison"].isin(
        ["Full PPO - Escorted PPO", "Escorted PPO - Lead"]
    )][["comparison", "observed_diff_pp", "ci_low_pp", "ci_high_pp"]].copy()

    plotted_conditions[["condition_display", "success_rate_pct", "success_ci_low_pct", "success_ci_high_pct",
                        "mean_detection_time_seconds"]].to_csv(OUTPUT_ROOT / "fig_acquisition_ablation_data.csv", index=False)
    # Append the annotated comparison rows to the same reproducibility CSV (clearly labeled row_type).
    with open(OUTPUT_ROOT / "fig_acquisition_ablation_data.csv", "a") as f:
        f.write("\n# comparisons annotated on panel (a):\n")
        plotted_comparisons.to_csv(f, index=False)

    caption = (
        "Acquisition-behavior ablation. Forcing PPO to use the same scripted pre-detection escort behavior as "
        "Lead reduces its safe-interception rate by 3.63 percentage points (95% CI [1.30, 5.93]). Full PPO also "
        "acquires the hostile substantially earlier. Once pre-detection behavior is equalized, Escorted PPO is "
        "statistically indistinguishable from Lead, indicating that learned pre-acquisition positioning accounts "
        "for a meaningful portion of PPO's overall performance. Panel (b) shows descriptive mean detection time; "
        "no frozen uncertainty interval is available for this quantity."
    )
    (OUTPUT_ROOT / "fig_acquisition_ablation_caption.txt").write_text(caption)

    return {"pdf": pdf_path, "png": png_path, "caption": caption, "dimensions_in": (TWO_COLUMN_WIDTH_IN, 2.3)}


# =============================================================================
# MANIFEST
# =============================================================================
def build_manifest(fig_a_info: dict, fig_b_info: dict) -> dict:
    input_files = ["final_nominal_paper_table.csv", "acquisition_paper_summary.csv"]
    input_hashes = {f: _sha256_of_file(EVIDENCE_ROOT / f) for f in input_files}

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "matplotlib_version": matplotlib.__version__,
        "python_version": sys.version,
        "input_evidence_files": input_files,
        "input_evidence_sha256": input_hashes,
        "generated_figures": {
            "fig_nominal_performance": {
                "pdf": str(fig_a_info["pdf"].relative_to(PROJECT_ROOT)),
                "png": str(fig_a_info["png"].relative_to(PROJECT_ROOT)),
                "width_in": fig_a_info["dimensions_in"][0], "height_in": fig_a_info["dimensions_in"][1],
                "dpi": PNG_DPI,
            },
            "fig_acquisition_ablation": {
                "pdf": str(fig_b_info["pdf"].relative_to(PROJECT_ROOT)),
                "png": str(fig_b_info["png"].relative_to(PROJECT_ROOT)),
                "width_in": fig_b_info["dimensions_in"][0], "height_in": fig_b_info["dimensions_in"][1],
                "dpi": PNG_DPI,
            },
        },
        "method_style_map": METHOD_STYLE,
        "simulations_run": 0,
        "episodes_run": 0,
        "new_evaluation_seeds_used": 0,
    }
    (OUTPUT_ROOT / "core_figures_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    return manifest


def main() -> int:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    nominal_df = load_and_validate_nominal_table()
    fig_a_info = plot_nominal_performance(nominal_df)
    print(f"Figure A written: {fig_a_info['pdf']}, {fig_a_info['png']}")

    conditions, comparisons = load_and_validate_acquisition_summary()
    fig_b_info = plot_acquisition_ablation(conditions, comparisons)
    print(f"Figure B written: {fig_b_info['pdf']}, {fig_b_info['png']}")

    manifest = build_manifest(fig_a_info, fig_b_info)
    print(json.dumps(manifest, indent=2, default=str))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
