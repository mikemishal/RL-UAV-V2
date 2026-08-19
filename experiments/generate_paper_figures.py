"""
Publication figure generator -- DATA ANALYSIS / FIGURE PRODUCTION ONLY.
Reads exclusively from the frozen paper_artifacts/data/ and
paper_artifacts/tables/ CSVs (commit 129beee5f79892c9922fe1179586a745704d3cc4).
Does NOT construct SoldierEnv, does NOT import stable_baselines3, does NOT
run any episode or load any model. Deterministically regenerates every
figure from the frozen CSVs.

Usage:
    python experiments/generate_paper_figures.py
"""

from __future__ import annotations

import hashlib
import json
import math
from datetime import datetime, timezone
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
from matplotlib.gridspec import GridSpecFromSubplotSpec
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
DATA_ROOT = PROJECT_ROOT / "paper_artifacts" / "data"
TABLES_ROOT = PROJECT_ROOT / "paper_artifacts" / "tables"
FIG_ROOT = PROJECT_ROOT / "paper_artifacts" / "figures"
EVIDENCE_SOURCE_COMMIT = "129beee5f79892c9922fe1179586a745704d3cc4"

# =============================================================================
# CENTRALIZED STYLE -- reused across every figure. Colorblind-safe (Okabe-Ito)
# palette; every method also distinguishable by marker AND line style so the
# figures remain interpretable in grayscale print.
# =============================================================================
STYLE = {
    "Greedy":     dict(color="#999999", marker="s", linestyle=":",  label="Greedy"),
    "PN":         dict(color="#56B4E9", marker="^", linestyle="-.", label="PN"),
    "Lead":       dict(color="#0072B2", marker="o", linestyle="-",  label="Lead"),
    "PPO":        dict(color="#E69F00", marker="D", linestyle="--", label="PPO"),
    "LR-PPO":     dict(color="#009E73", marker="*", linestyle="-",  label="LR-PPO"),
    "PPO-Kalman": dict(color="#777777", marker="x", linestyle=":",  label="PPO-Kalman (ablation)"),
}


def setup_style():
    plt.rcParams.update({
        "font.family": "serif",
        "font.serif": ["DejaVu Serif"],
        "font.size": 8,
        "axes.titlesize": 9,
        "axes.labelsize": 8,
        "xtick.labelsize": 7.5,
        "ytick.labelsize": 7.5,
        "legend.fontsize": 8,
        "figure.facecolor": "white",
        "axes.facecolor": "white",
        "savefig.facecolor": "white",
        "axes.linewidth": 0.8,
        "lines.linewidth": 1.3,
        "lines.markersize": 5,
        "axes.grid": True,
        "grid.linewidth": 0.4,
        "grid.alpha": 0.35,
        "text.usetex": False,
        "svg.fonttype": "none",
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    })


VALIDATION_LOG: list[dict] = []
GENERATED_FILES: list[Path] = []


def save_all_formats(fig, name: str) -> dict:
    paths = {}
    for ext in ("pdf", "svg", "png"):
        p = FIG_ROOT / f"{name}.{ext}"
        fig.savefig(p, format=ext, dpi=300 if ext == "png" else None, bbox_inches="tight")
        paths[ext] = p
        GENERATED_FILES.append(p)
    plt.close(fig)
    return paths


def _errbar(ax, x, y, lo, hi, style, zorder=3, alpha=1.0, ms_scale=1.0):
    yerr = np.array([[y - lo], [hi - y]])
    ax.errorbar(x, y, yerr=yerr, color=style["color"], marker=style["marker"],
                linestyle=style["linestyle"], markersize=plt.rcParams["lines.markersize"] * ms_scale,
                label=style["label"], capsize=2.5, elinewidth=1.0, alpha=alpha, zorder=zorder)


# =============================================================================
# FIGURE 1 -- proposed method schematic (vector, no data)
# =============================================================================
def make_figure1():
    fig, ax = plt.subplots(figsize=(3.4, 5.8))
    ax.axis("off")
    ax.set_xlim(0, 10)
    ax.set_ylim(0, 16)

    def box(cx, cy, w, h, text, fc="#f2f2f2", ec="#333333", fontsize=7.3):
        b = mpatches.FancyBboxPatch((cx - w / 2, cy - h / 2), w, h,
                                     boxstyle="round,pad=0.08,rounding_size=0.12",
                                     facecolor=fc, edgecolor=ec, linewidth=0.9, zorder=2)
        ax.add_patch(b)
        ax.text(cx, cy, text, ha="center", va="center", fontsize=fontsize, zorder=3, linespacing=1.5)
        return (cx, cy, w, h)

    def arrow(p_from, p_to, **kw):
        a = mpatches.FancyArrowPatch(p_from, p_to, arrowstyle="-|>", mutation_scale=8,
                                      color="#333333", linewidth=0.9, zorder=1, **kw)
        ax.add_patch(a)

    box(5, 15.2, 6.4, 0.9, r"Sanitized observation $s_t$")
    box(2.6, 13.1, 4.2, 1.3, "Predictive Lead guidance\n" + r"$u_L = \mathrm{Lead}(s_t)$")
    box(7.4, 13.1, 4.2, 1.3, "PPO residual policy\n" + r"$\Delta u_t = \pi_\theta(s_t)$")
    box(7.4, 10.7, 4.2, 1.3, "Bounded angular residual\n" + r"$\theta \leq \theta_{max}=30^\circ$")
    box(5, 8.0, 6.6, 1.5, "Residual composition\nand normalization")
    box(5, 5.5, 5.2, 1.0, r"Commanded direction $u_t$")
    box(5, 3.0, 8.6, 1.7,
        "Constrained UAV point-mass dynamics\n(bounded acceleration, horizontal turn rate,\nclimb/descent rate, maximum speed)")

    arrow((5, 14.75), (2.6, 13.75))
    arrow((5, 14.75), (7.4, 13.75))
    arrow((2.6, 12.45), (4.15, 8.75))
    arrow((7.4, 12.45), (7.4, 11.35))
    arrow((7.4, 10.05), (5.85, 8.75))
    arrow((5, 7.25), (5, 6.0))
    arrow((5, 5.0), (5, 3.85))

    ax.text(5, 1.0,
            r"$u_t = \mathrm{normalize}\left(u_L + \tan(\theta_{max})\,\Delta u_\perp\right),\ \ "
            r"\Delta u_\perp = \Delta u_t - (\Delta u_t \cdot u_L)\,u_L$",
            ha="center", va="center", fontsize=6.6, style="italic", zorder=3)
    ax.text(5, 0.55, "Lead uses the same sanitized target-state observation as the residual policy.\n"
                     "Output is a desired velocity DIRECTION only; the environment enforces all physical limits.",
            ha="center", va="center", fontsize=6.2, color="#444444", linespacing=1.5)

    fig.subplots_adjust(left=0.02, right=0.98, top=0.99, bottom=0.02)
    return save_all_formats(fig, "fig1_lrppo_architecture")


# =============================================================================
# FIGURE 2 -- nominal performance (forest / point-range plot)
# =============================================================================
def make_figure2():
    df = pd.read_csv(TABLES_ROOT / "table_nominal_performance.csv")
    methods_df = df[df["row_type"] == "method"].set_index("label")
    order = ["Greedy", "PN", "Lead", "PPO", "LR-PPO"]

    expected = {"Greedy": 77.62, "PN": 74.68, "Lead": 85.82, "PPO": 86.83, "LR-PPO": 88.83}
    checked = {}
    for m in order:
        val = float(methods_df.loc[m, "success_rate_pct"])
        checked[m] = val
        assert abs(val - expected[m]) < 0.01, f"FIG2 mismatch for {m}: {val} vs {expected[m]}"

    fig, ax = plt.subplots(figsize=(3.4, 2.6))
    y_positions = list(range(len(order)))[::-1]
    emphasized = {"Lead", "PPO", "LR-PPO"}
    for y, m in zip(y_positions, order):
        r = methods_df.loc[m]
        style = STYLE[m]
        alpha = 1.0 if m in emphasized else 0.72
        ms = 7 if m == "LR-PPO" else 6
        ax.errorbar(r["success_rate_pct"], y, xerr=[[r["success_rate_pct"] - r["ci_low_pct"]], [r["ci_high_pct"] - r["success_rate_pct"]]],
                    color=style["color"], marker=style["marker"], markersize=ms, alpha=alpha,
                    capsize=3, elinewidth=1.2, markeredgecolor="black" if m == "LR-PPO" else style["color"],
                    markeredgewidth=0.6 if m == "LR-PPO" else 0.0)
        ax.text(r["success_rate_pct"] + 1.0, y, f"{r['success_rate_pct']:.2f}", va="center", ha="left", fontsize=7.2)

    ax.set_yticks(y_positions)
    ax.set_yticklabels(order)
    ax.set_xlim(65, 95)
    ax.set_xlabel("Safe-interception success rate (%)")
    ax.set_title("Nominal performance", fontsize=9)
    ax.grid(axis="x", linewidth=0.4, alpha=0.35)
    ax.grid(axis="y", visible=False)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)

    fig.tight_layout()
    paths = save_all_formats(fig, "fig2_nominal_performance")
    VALIDATION_LOG.append({
        "figure": "fig2_nominal_performance", "source_csv": "table_nominal_performance.csv",
        "n_plotted_points": len(order), "min_success_pct": min(checked.values()), "max_success_pct": max(checked.values()),
        "key_expected_values_checked": expected, "key_values_from_csv": checked, "match": True,
    })
    return paths


# =============================================================================
# FIGURE 3 -- robustness multi-panel (2x2: evasion, maneuverability, noise, detection radius)
# =============================================================================
def _panel_line(ax, df, xcol, methods, x_is_categorical=False):
    x = df[xcol].to_numpy(dtype=float)
    for m in methods:
        key = {"lead": "Lead", "ppo": "PPO", "lr_ppo": "LR-PPO", "ppo_kalman": "PPO-Kalman"}[m]
        style = STYLE[key]
        succ = df[f"{m}_success_pct"].to_numpy(dtype=float)
        lo = df[f"{m}_ci_low_pct"].to_numpy(dtype=float)
        hi = df[f"{m}_ci_high_pct"].to_numpy(dtype=float)
        alpha = 0.65 if m == "ppo_kalman" else 1.0
        ax.errorbar(x, succ, yerr=[succ - lo, hi - succ], color=style["color"], marker=style["marker"],
                    linestyle=style["linestyle"], label=style["label"], capsize=2.2, elinewidth=0.9, alpha=alpha,
                    markersize=plt.rcParams["lines.markersize"] * (0.85 if m == "ppo_kalman" else 1.0))


def make_figure3a(ax):
    df = pd.read_csv(DATA_ROOT / "figure_evasion.csv").sort_values("reactive_evasion_gain")
    _panel_line(ax, df, "reactive_evasion_gain", ["lead", "ppo", "lr_ppo"])
    ax.axvline(0.75, color="#555555", linestyle=":", linewidth=0.8, zorder=0)
    ax.text(0.75, ax.get_ylim()[1] if ax.get_ylim()[1] else 96, "Nominal", rotation=90, fontsize=6.5,
            ha="right", va="top", color="#555555")
    ax.set_xlabel("Reactive evasion gain")
    ax.set_ylabel("Safe-interception success (%)")
    ax.set_title("(a) Reactive evasion", fontsize=8.5)
    ax.set_ylim(55, 95)

    row1 = df[df["reactive_evasion_gain"] == 1.00].iloc[0]
    expected = {"lead": 86.8, "ppo": 84.33, "lr_ppo": 90.2}
    got = {"lead": float(row1["lead_success_pct"]), "ppo": float(row1["ppo_success_pct"]), "lr_ppo": float(row1["lr_ppo_success_pct"])}
    for k in expected:
        assert abs(got[k] - expected[k]) < 0.02, f"FIG3A gain=1 mismatch {k}: {got[k]} vs {expected[k]}"
    VALIDATION_LOG.append({
        "figure": "fig3_robustness/panel_a_evasion", "source_csv": "figure_evasion.csv",
        "n_plotted_points": len(df) * 3, "min_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "lr_ppo_success_pct"]].min().min()),
        "max_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "lr_ppo_success_pct"]].max().max()),
        "key_expected_values_checked": expected, "key_values_from_csv": got, "match": True,
    })


def make_figure3b(fig, gs_cell):
    df = pd.read_csv(DATA_ROOT / "figure_maneuverability.csv")
    accels = sorted(df["defender_max_accel"].unique())
    turns = sorted(df["defender_max_turn_rate_deg"].unique())

    def matrix(colname):
        m = np.full((len(accels), len(turns)), np.nan)
        for i, a in enumerate(accels):
            for j, t in enumerate(turns):
                row = df[(df["defender_max_accel"] == a) & (df["defender_max_turn_rate_deg"] == t)]
                if len(row):
                    m[i, j] = float(row.iloc[0][colname])
        return m

    def sig_matrix(lo_col, hi_col):
        m = np.full((len(accels), len(turns)), False)
        for i, a in enumerate(accels):
            for j, t in enumerate(turns):
                row = df[(df["defender_max_accel"] == a) & (df["defender_max_turn_rate_deg"] == t)]
                if len(row):
                    lo, hi = float(row.iloc[0][lo_col]), float(row.iloc[0][hi_col])
                    m[i, j] = (lo > 0) or (hi < 0)
        return m

    lr_lead = matrix("lr_minus_lead_pp")
    lr_ppo = matrix("lr_minus_ppo_pp")
    sig_lead = sig_matrix("lr_minus_lead_ci_low_pp", "lr_minus_lead_ci_high_pp")
    sig_ppo = sig_matrix("lr_minus_ppo_ci_low_pp", "lr_minus_ppo_ci_high_pp")

    vmax = max(np.nanmax(np.abs(lr_lead)), np.nanmax(np.abs(lr_ppo)))
    vmin = -vmax

    inner = GridSpecFromSubplotSpec(1, 3, subplot_spec=gs_cell, width_ratios=[1, 1, 0.07], wspace=0.35)
    ax_left = fig.add_subplot(inner[0, 0])
    ax_right = fig.add_subplot(inner[0, 1])
    cax = fig.add_subplot(inner[0, 2])

    nominal_idx = (accels.index(8.0), turns.index(90.0))

    for ax, mat, sig, title in ((ax_left, lr_lead, sig_lead, "LR-PPO $-$ Lead"), (ax_right, lr_ppo, sig_ppo, "LR-PPO $-$ PPO")):
        im = ax.imshow(mat, cmap="PRGn", vmin=vmin, vmax=vmax, aspect="auto")
        for i in range(len(accels)):
            for j in range(len(turns)):
                ax.text(j, i, f"{mat[i, j]:+.1f}", ha="center", va="center", fontsize=6.3,
                        color="black" if abs(mat[i, j]) < 0.6 * vmax else "white")
                if sig[i, j]:
                    ax.add_patch(mpatches.Rectangle((j - 0.5, i - 0.5), 1, 1, fill=False, edgecolor="black", linewidth=1.1))
        ax.add_patch(mpatches.Rectangle((nominal_idx[1] - 0.5, nominal_idx[0] - 0.5), 1, 1, fill=False, edgecolor="#0072B2", linewidth=1.6))
        ax.set_xticks(range(len(turns)))
        ax.set_xticklabels([f"{t:g}" for t in turns])
        ax.set_yticks(range(len(accels)))
        ax.set_yticklabels([f"{a:g}" for a in accels])
        ax.set_xlabel("Turn rate (deg/s)", fontsize=7)
        ax.set_title(title, fontsize=7.5)
    ax_left.set_ylabel("Max accel (m/s$^2$)", fontsize=7)
    fig.colorbar(im, cax=cax).set_label("pp", fontsize=6.5)
    cax.tick_params(labelsize=6)
    ax_left.text(-0.9, -0.9, "(b) Maneuverability sensitivity", fontsize=8.5, transform=ax_left.transData, ha="left", va="bottom")

    r45 = df[(df["defender_max_accel"] == 4.0) & (df["defender_max_turn_rate_deg"] == 45.0)].iloc[0]
    assert abs(float(r45["lr_minus_lead_pp"]) - 4.73) < 0.01
    assert abs(float(r45["lr_minus_ppo_pp"]) - 14.40) < 0.01
    VALIDATION_LOG.append({
        "figure": "fig3_robustness/panel_b_maneuverability", "source_csv": "figure_maneuverability.csv",
        "n_plotted_points": 9 * 2, "min_success_pct": None, "max_success_pct": None,
        "key_expected_values_checked": {"lr_minus_lead_pp_4_45": 4.73, "lr_minus_ppo_pp_4_45": 14.40},
        "key_values_from_csv": {"lr_minus_lead_pp_4_45": float(r45["lr_minus_lead_pp"]), "lr_minus_ppo_pp_4_45": float(r45["lr_minus_ppo_pp"])},
        "match": True,
    })


def make_figure3c(ax):
    df = pd.read_csv(DATA_ROOT / "figure_measurement_noise.csv").sort_values("sigma")
    _panel_line(ax, df, "sigma", ["lead", "ppo", "ppo_kalman", "lr_ppo"])
    nominal_sigma = math.sqrt(0.5)
    ax.axvline(nominal_sigma, color="#555555", linestyle=":", linewidth=0.8, zorder=0)
    ax.text(nominal_sigma, 96, "Nominal", rotation=90, fontsize=6.5, ha="right", va="top", color="#555555")
    ax.set_xlabel(r"Measurement-noise standard deviation, $\sigma$")
    ax.set_ylabel("Safe-interception success (%)")
    ax.set_title("(c) Measurement noise", fontsize=8.5)
    ax.set_ylim(55, 95)
    first = df.iloc[0]
    ax.annotate("PPO-Kalman (ablation)", xy=(float(first["sigma"]), float(first["ppo_kalman_success_pct"])),
                xytext=(0.75, 60), fontsize=6.3, color="#777777", ha="left",
                arrowprops=dict(arrowstyle="-", color="#777777", linewidth=0.6))

    row = df[df["measurement_var"] == 4.00].iloc[0]
    expected = {"lead": 71.8, "ppo": 82.1, "lr_ppo": 79.9}
    got = {"lead": float(row["lead_success_pct"]), "ppo": float(row["ppo_success_pct"]), "lr_ppo": float(row["lr_ppo_success_pct"])}
    for k in expected:
        assert abs(got[k] - expected[k]) < 0.02, f"FIG3C var=4 mismatch {k}: {got[k]} vs {expected[k]}"
    VALIDATION_LOG.append({
        "figure": "fig3_robustness/panel_c_measurement_noise", "source_csv": "figure_measurement_noise.csv",
        "n_plotted_points": len(df) * 4, "min_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "ppo_kalman_success_pct", "lr_ppo_success_pct"]].min().min()),
        "max_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "ppo_kalman_success_pct", "lr_ppo_success_pct"]].max().max()),
        "key_expected_values_checked": expected, "key_values_from_csv": got, "match": True,
    })


def make_figure3d(ax):
    df = pd.read_csv(DATA_ROOT / "figure_detection_radius.csv").sort_values("detection_radius")
    _panel_line(ax, df, "detection_radius", ["lead", "ppo", "lr_ppo"])
    ax.axvline(15, color="#555555", linestyle=":", linewidth=0.8, zorder=0)
    ax.text(15, 96, "Nominal $R_{det}$", rotation=90, fontsize=6.2, ha="right", va="top", color="#555555")
    ax.axvline(20, color="#7a3b8a", linestyle=":", linewidth=0.8, zorder=0)
    ax.text(20, 96, "Evasion radius", rotation=90, fontsize=6.2, ha="left", va="top", color="#7a3b8a")
    ax.set_xlabel("Detection radius (m)")
    ax.set_ylabel("Safe-interception success (%)")
    ax.set_title("(d) Detection radius", fontsize=8.5)
    ax.set_ylim(55, 95)

    row = df[df["detection_radius"] == 5.0].iloc[0]
    expected = {"lead": 68.7, "ppo": 62.27, "lr_ppo": 69.63}
    got = {"lead": float(row["lead_success_pct"]), "ppo": float(row["ppo_success_pct"]), "lr_ppo": float(row["lr_ppo_success_pct"])}
    for k in expected:
        assert abs(got[k] - expected[k]) < 0.02, f"FIG3D R=5 mismatch {k}: {got[k]} vs {expected[k]}"
    VALIDATION_LOG.append({
        "figure": "fig3_robustness/panel_d_detection_radius", "source_csv": "figure_detection_radius.csv",
        "n_plotted_points": len(df) * 3, "min_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "lr_ppo_success_pct"]].min().min()),
        "max_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "lr_ppo_success_pct"]].max().max()),
        "key_expected_values_checked": expected, "key_values_from_csv": got, "match": True,
    })


def make_figure3():
    fig = plt.figure(figsize=(7.0, 5.6))
    gs = fig.add_gridspec(2, 2, hspace=0.55, wspace=0.35)

    ax_a = fig.add_subplot(gs[0, 0])
    make_figure3a(ax_a)

    make_figure3b(fig, gs[0, 1])

    ax_c = fig.add_subplot(gs[1, 0])
    make_figure3c(ax_c)

    ax_d = fig.add_subplot(gs[1, 1])
    make_figure3d(ax_d)

    handles, labels = ax_a.get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.28, -0.02), frameon=False, fontsize=7.5)

    for ax in (ax_a, ax_c, ax_d):
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)

    return save_all_formats(fig, "fig3_robustness")


# =============================================================================
# FIGURE 4 (optional) -- mobility, two panels (defender-speed arm, hostile-speed arm)
# =============================================================================
def make_figure4():
    df = pd.read_csv(DATA_ROOT / "figure_mobility.csv")
    fig, axes = plt.subplots(1, 2, figsize=(6.8, 2.9))

    defender_arm = df[df["arm"].isin(["defender_speed", "nominal"])].sort_values("v_d")
    _panel_line(axes[0], defender_arm.rename(columns={"v_d": "x"}), "x", ["lead", "ppo", "lr_ppo"])
    axes[0].axvline(18.0, color="#555555", linestyle=":", linewidth=0.8, zorder=0)
    axes[0].set_xlabel(r"Defender speed $v_d$ (m/s), $v_e=12$ fixed")
    axes[0].set_ylabel("Safe-interception success (%)")
    axes[0].set_title("(a) Defender-speed arm", fontsize=8.5)

    hostile_arm = df[df["arm"].isin(["hostile_speed", "nominal"])].sort_values("v_e")
    _panel_line(axes[1], hostile_arm.rename(columns={"v_e": "x"}), "x", ["lead", "ppo", "lr_ppo"])
    axes[1].axvline(12.0, color="#555555", linestyle=":", linewidth=0.8, zorder=0)
    axes[1].set_xlabel(r"Hostile speed $v_e$ (m/s), $v_d=18$ fixed")
    axes[1].set_title("(b) Hostile-speed arm", fontsize=8.5)

    for ax in axes:
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
    handles, labels = axes[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="lower center", ncol=3, bbox_to_anchor=(0.5, -0.08), frameon=False, fontsize=7.5)

    row = df[(df["v_d"] == 18.0) & (df["v_e"] == 8.0)].iloc[0]
    expected = {"lead": 48.8, "ppo": 39.47, "lr_ppo": 57.17}
    got = {"lead": float(row["lead_success_pct"]), "ppo": float(row["ppo_success_pct"]), "lr_ppo": float(row["lr_ppo_success_pct"])}
    for k in expected:
        assert abs(got[k] - expected[k]) < 0.02, f"FIG4 (18,8) mismatch {k}: {got[k]} vs {expected[k]}"
    VALIDATION_LOG.append({
        "figure": "fig4_mobility_optional", "source_csv": "figure_mobility.csv",
        "n_plotted_points": len(df) * 3, "min_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "lr_ppo_success_pct"]].min().min()),
        "max_success_pct": float(df[["lead_success_pct", "ppo_success_pct", "lr_ppo_success_pct"]].max().max()),
        "key_expected_values_checked": expected, "key_values_from_csv": got, "match": True,
    })

    fig.tight_layout(rect=[0, 0.08, 1, 1])
    return save_all_formats(fig, "fig4_mobility_optional")


# =============================================================================
# CAPTIONS
# =============================================================================
def write_captions():
    text = f"""# Figure Captions

## Figure 1 -- Proposed method (Lead-Residual PPO) architecture
The Lead-Residual PPO (LR-PPO) controller composes a bounded angular residual
correction, learned by PPO, around a predictive Lead-guidance direction. Both
the Lead policy and the residual policy consume the same sanitized
target-state observation $s_t$. The residual policy's output $\\Delta u_t$ is
clipped to a bounded angular offset $\\theta_{{max}} = 30^\\circ$ from the Lead
direction before composition and normalization; the result $u_t$ is a
commanded *direction*, not a physical turn command -- the environment
separately enforces bounded acceleration, horizontal turn rate,
climb/descent rate, and maximum speed.

## Figure 2 -- Nominal performance
Safe-interception success rate (%) at the nominal training condition
(5000 evaluation seeds). Classical-method (Greedy, PN, Lead) intervals are
Wilson 95% intervals ($n=5000$). Learned-family (PPO, LR-PPO) intervals are
hierarchical 95% intervals computed across 3 independently trained roots
(3 training roots, 5000 evaluation episodes each); PPO-Kalman is omitted from
this headline comparison and reported separately as an estimator-ablation
diagnostic (Table II). The LR-PPO nominal gain over Lead is +3.01 pp
(95% CI [2.10, 3.91]), excluding zero. The apparent LR-PPO vs. PPO gain
(+2.01 pp, 95% CI [-0.37, 5.04]) includes zero and is not a nominal
superiority claim.

## Figure 3 -- Robustness across four independent sweeps
(a) Reactive hostile evasion: safe-interception success (%) vs. reactive
evasion gain (5 tested values); the vertical dashed line marks the nominal
training gain (0.75). LR-PPO remains near 90% at the highest tested gain
(1.00) while standalone PPO declines; error bars are 95% intervals (Wilson
for Lead, hierarchical for PPO/LR-PPO).
(b) Maneuverability sensitivity: LR-PPO's advantage (percentage points) over
Lead (left) and over PPO (right) across the full 3$\\times$3 grid of defender
maximum acceleration (4/8/12 m/s$^2$) and maximum horizontal turn rate
(45/90/135 deg/s); this panel plots the *hybrid advantage*, not absolute
success. The nominal cell (8 m/s$^2$, 90 deg/s) is outlined in blue; cells
with a bold black border have a 95% difference interval that excludes zero.
(c) Measurement noise: safe-interception success (%) vs. measurement-noise
standard deviation $\\sigma=\\sqrt{{\\mathrm{{variance}}}}$ (6 tested values); the
vertical dashed line marks the nominal $\\sigma=\\sqrt{{0.5}}\\approx0.707$.
PPO-Kalman (estimator ablation) is shown de-emphasized in gray; it does not
outperform PPO at any tested noise level in this frozen configuration.
(d) Detection radius: safe-interception success (%) vs. defender detection
radius (6 tested values); dashed reference lines mark the nominal detection
radius (15 m) and the (independent, fixed) hostile reactive-evasion radius
(20 m) -- the hostile has no knowledge of the defender's detection state.
All panels: Lead, PPO, and LR-PPO error bars are 95% intervals (Wilson for
Lead/scripted methods, hierarchical across 3 training roots for PPO/LR-PPO).

## Figure 4 (optional) -- Mobility operating envelope
Two matched one-dimensional speed arms sharing one nominal condition
($v_d=18$, $v_e=12$ m/s): (a) defender speed varied with hostile speed fixed
at 12 m/s; (b) hostile speed varied with defender speed fixed at 18 m/s.
Because acceleration and turn-rate limits remain fixed while speed changes,
these curves reflect speed-maneuverability coupling in the tested
constrained point-mass model, not speed alone -- they should not be
interpreted as a statement about speed in isolation. The slow-hostile
condition ($v_d=18$, $v_e=8$) is the hardest of the 9 tested mobility
conditions for every method shown.
"""
    (FIG_ROOT / "figure_captions.md").write_text(text)
    GENERATED_FILES.append(FIG_ROOT / "figure_captions.md")


# =============================================================================
# VALIDATION / MANIFEST
# =============================================================================
def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def validate_images():
    from PIL import Image
    report = []
    for f in FIG_ROOT.glob("*"):
        if f.suffix not in (".png", ".pdf", ".svg"):
            continue
        size = f.stat().st_size
        entry = {"file": f.name, "size_bytes": size, "nonzero": size > 0}
        if f.suffix == ".png":
            with Image.open(f) as im:
                entry["png_dimensions"] = im.size
        report.append(entry)
        assert size > 0, f"{f} has zero size"
    return report


def build_manifest(image_report):
    figures = {}
    for stem in ("fig1_lrppo_architecture", "fig2_nominal_performance", "fig3_robustness", "fig4_mobility_optional"):
        entry = {"files": {}, "source_datasets": [], "generation_script": "experiments/generate_paper_figures.py",
                 "evidence_source_commit": EVIDENCE_SOURCE_COMMIT, "new_simulation": False, "new_training": False}
        for ext in ("pdf", "svg", "png"):
            p = FIG_ROOT / f"{stem}.{ext}"
            if p.exists():
                entry["files"][ext] = {"path": str(p.relative_to(PROJECT_ROOT)), "sha256": sha256_of_file(p), "size_bytes": p.stat().st_size}
        png_entry = next((r for r in image_report if r["file"] == f"{stem}.png"), None)
        if png_entry:
            entry["png_dimensions"] = png_entry["png_dimensions"]
        figures[stem] = entry

    figures["fig1_lrppo_architecture"]["source_datasets"] = []
    figures["fig2_nominal_performance"]["source_datasets"] = ["paper_artifacts/tables/table_nominal_performance.csv"]
    figures["fig3_robustness"]["source_datasets"] = [
        "paper_artifacts/data/figure_evasion.csv", "paper_artifacts/data/figure_maneuverability.csv",
        "paper_artifacts/data/figure_measurement_noise.csv", "paper_artifacts/data/figure_detection_radius.csv",
    ]
    figures["fig4_mobility_optional"]["source_datasets"] = ["paper_artifacts/data/figure_mobility.csv"]

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "evidence_source_commit": EVIDENCE_SOURCE_COMMIT,
        "new_simulation": False, "new_training": False,
        "figures": figures,
        "validation_records": VALIDATION_LOG,
    }
    (FIG_ROOT / "figure_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    GENERATED_FILES.append(FIG_ROOT / "figure_manifest.json")


def main() -> int:
    FIG_ROOT.mkdir(parents=True, exist_ok=True)
    setup_style()

    make_figure1()
    make_figure2()
    make_figure3()
    make_figure4()
    write_captions()

    image_report = validate_images()
    build_manifest(image_report)

    print("Validation log:")
    for v in VALIDATION_LOG:
        print(" -", v["figure"], "match:", v["match"])
    print("\nImage validation:")
    for r in image_report:
        print(" -", r["file"], r["size_bytes"], "bytes", r.get("png_dimensions", ""))
    print(f"\n{len(GENERATED_FILES)} files written to {FIG_ROOT}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
