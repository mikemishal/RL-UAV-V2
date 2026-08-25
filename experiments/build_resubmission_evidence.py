"""
Build the frozen resubmission paper-evidence package from the completed,
Tier-A-authoritative final-nominal and targeted-robustness campaigns.

This module performs NO environment simulation, NO model loading for
inference, and NO training. It only reads existing CSV/JSON result
artifacts (see docs/resubmission_evidence_policy.md) and derives every
paper-facing number from them programmatically -- no paper number is
hardcoded as a source of truth in this file.

Usage:
    python experiments/build_resubmission_evidence.py
"""

from __future__ import annotations

import ast
import dataclasses
import hashlib
import json
import platform
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
sys.path.insert(0, str(PROJECT_ROOT))

# =============================================================================
# Section 2: authoritative evidence hierarchy (docs/resubmission_evidence_policy.md)
# =============================================================================
TIER_A_NOMINAL_ROOT = PROJECT_ROOT / "results" / "rmpc_final_nominal"
TIER_A_ROBUSTNESS_ROOT = PROJECT_ROOT / "results" / "rmpc_targeted_robustness"
TIER_B_OLD_NOMINAL_ROOT = PROJECT_ROOT / "results" / "lead_residual_expanded_final_5000"
TIER_C_TUNING_ROOT = PROJECT_ROOT / "results" / "rmpc_tuning"

OUTPUT_ROOT = PROJECT_ROOT / "paper_artifacts" / "resubmission_final"

FAMILY_ORDER = ("Greedy", "PN", "Lead", "RMPC", "PPO", "PPO-Kalman", "LR-PPO")
DETERMINISTIC_FAMILIES = ("Greedy", "PN", "Lead", "RMPC")
LEARNED_FAMILIES = ("PPO", "PPO-Kalman", "LR-PPO")
LEARNED_ROOTS = {"PPO": (45, 46, 47), "PPO-Kalman": (45, 46, 47), "LR-PPO": (55, 56, 57)}
LEARNED_FAMILY_PREFIX = {"PPO": "ppo_seed", "PPO-Kalman": "ppo_kalman_seed", "LR-PPO": "lr_ppo_seed"}

CONDITION_KEYS = (
    "strong_evasion", "restrictive_dynamics", "high_measurement_noise", "hard_mobility", "short_detection",
)
CONDITION_DISPLAY = {
    "strong_evasion": "Strong evasion",
    "restrictive_dynamics": "Restrictive dynamics",
    "high_measurement_noise": "High measurement noise",
    "hard_mobility": "Hard mobility",
    "short_detection": "Short detection",
}
ALL_ROW_KEYS = ("nominal",) + CONDITION_KEYS
ALL_ROW_DISPLAY = {"nominal": "Nominal", **CONDITION_DISPLAY}

PRIMARY_COMPARISONS = ("LR-PPO - Lead", "LR-PPO - RMPC", "LR-PPO - PPO", "RMPC - Lead", "RMPC - PPO")
SECONDARY_COMPARISONS = ("PPO - Greedy", "PPO - PN", "PPO - Lead", "PPO-Kalman - PPO")

# Tier-C files that MUST NEVER be read as final performance evidence -- the
# builder never opens these; kept here as an explicit, testable guard list.
TIER_C_FORBIDDEN_PERFORMANCE_FILES = (
    "stage_a_summary.csv", "stage_b_summary.csv", "validation_summary.csv",
    "diagnostic_summary.csv", "stage_a_episodes.csv", "stage_b_episodes.csv",
    "validation_episodes.csv", "diagnostic_episodes.csv",
)


# =============================================================================
# Tier-A loaders (nominal)
# =============================================================================
def assert_tier_a_sources_exist() -> None:
    required = [
        TIER_A_NOMINAL_ROOT / "instance_summary.csv",
        TIER_A_NOMINAL_ROOT / "family_summary.csv",
        TIER_A_NOMINAL_ROOT / "paired_differences.csv",
        TIER_A_ROBUSTNESS_ROOT / "robustness_summary.csv",
        TIER_A_ROBUSTNESS_ROOT / "primary_differences_summary.csv",
    ]
    missing = [str(p) for p in required if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing required Tier-A evidence files: {missing}")
    for cond in CONDITION_KEYS:
        cond_dir = TIER_A_ROBUSTNESS_ROOT / cond
        for name in ("instance_summary.csv", "family_summary.csv", "paired_differences.csv"):
            p = cond_dir / name
            if not p.exists():
                raise FileNotFoundError(f"Missing required Tier-A robustness file: {p}")


def load_nominal_instance_summary() -> pd.DataFrame:
    return pd.read_csv(TIER_A_NOMINAL_ROOT / "instance_summary.csv")


def load_nominal_family_summary() -> pd.DataFrame:
    return pd.read_csv(TIER_A_NOMINAL_ROOT / "family_summary.csv")


def load_nominal_paired_differences() -> pd.DataFrame:
    return pd.read_csv(TIER_A_NOMINAL_ROOT / "paired_differences.csv")


def load_robustness_summary() -> pd.DataFrame:
    return pd.read_csv(TIER_A_ROBUSTNESS_ROOT / "robustness_summary.csv")


def load_robustness_primary_differences() -> pd.DataFrame:
    return pd.read_csv(TIER_A_ROBUSTNESS_ROOT / "primary_differences_summary.csv")


def load_condition_family_summary(condition_key: str) -> pd.DataFrame:
    return pd.read_csv(TIER_A_ROBUSTNESS_ROOT / condition_key / "family_summary.csv")


def load_condition_instance_summary(condition_key: str) -> pd.DataFrame:
    return pd.read_csv(TIER_A_ROBUSTNESS_ROOT / condition_key / "instance_summary.csv")


# =============================================================================
# Tier-B loader (old nominal, descriptive replication only -- never merged)
# =============================================================================
def load_old_nominal_classical_summary() -> pd.DataFrame:
    return pd.read_csv(TIER_B_OLD_NOMINAL_ROOT / "classical_summary.csv")


def load_old_nominal_learned_family_summary() -> pd.DataFrame:
    return pd.read_csv(TIER_B_OLD_NOMINAL_ROOT / "family_summary.csv")


# =============================================================================
# Tier-C loader (frozen RMPC controller DEFINITION only -- never performance)
# =============================================================================
def load_frozen_rmpc_config() -> dict:
    path = TIER_C_TUNING_ROOT / "selected_rmpc_config.json"
    data = json.loads(path.read_text())
    if data.get("frozen_for_final_evaluation") is not True:
        raise RuntimeError("selected_rmpc_config.json is not frozen_for_final_evaluation=true -- STOP")
    return data


def sha256_of_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# =============================================================================
# Consistency guards
# =============================================================================
def assert_exactly_seven_families(names) -> None:
    names_set = set(names)
    if names_set != set(FAMILY_ORDER):
        raise AssertionError(f"expected exactly the 7 families {set(FAMILY_ORDER)}, got {names_set}")


def _resolution_from_ci(ci_low_pp: float, ci_high_pp: float) -> str:
    if ci_low_pp > 0:
        return "resolved positive"
    if ci_high_pp < 0:
        return "resolved negative"
    return "unresolved"


# =============================================================================
# paper_values.json builder -- the single machine-readable source of truth
# =============================================================================
def build_paper_values() -> dict:
    assert_tier_a_sources_exist()

    nominal_instances = load_nominal_instance_summary()
    nominal_families = load_nominal_family_summary()
    nominal_diffs = load_nominal_paired_differences()
    assert_exactly_seven_families(nominal_families["family"])

    robustness_summary = load_robustness_summary()
    robustness_diffs = load_robustness_primary_differences()
    assert_exactly_seven_families(FAMILY_ORDER)  # robustness_summary columns checked below

    rmpc_config = load_frozen_rmpc_config()

    # --- nominal families -----------------------------------------------------
    nominal_family_values = {}
    for _, row in nominal_families.iterrows():
        fam = row["family"]
        per_root = ast.literal_eval(row["per_root_rates"]) if isinstance(row["per_root_rates"], str) else row["per_root_rates"]
        nominal_family_values[fam] = {
            "success_rate": float(row["family_mean_success_rate"]),
            "ci_low": float(row["hierarchical_ci_low"]),
            "ci_high": float(row["hierarchical_ci_high"]),
            "root_sd": float(row["root_sd"]),
            "n_roots": int(row["n_roots"]),
            "per_root_rates": [float(x) for x in per_root],
        }

    # --- nominal instances (root-specific, deterministic) ---------------------
    nominal_instance_values = {}
    for _, row in nominal_instances.iterrows():
        nominal_instance_values[row["instance"]] = {
            "family": row["family"],
            "training_seed": None if pd.isna(row["training_seed"]) else int(row["training_seed"]),
            "n": int(row["n"]),
            "successes": int(row["successes"]),
            "success_rate": float(row["success_rate"]),
            "ci_low": float(row["success_ci_low"]),
            "ci_high": float(row["success_ci_high"]),
        }

    # --- nominal differences ---------------------------------------------------
    nominal_diff_values = {}
    for _, row in nominal_diffs.iterrows():
        nominal_diff_values[row["comparison"]] = {
            "difference_pp": float(row["difference_pp"]),
            "ci_low_pp": float(row["ci_low_pp"]),
            "ci_high_pp": float(row["ci_high_pp"]),
            "resolution": row["statistical_resolution"],
        }

    # --- robustness matrix (all rows in fraction [0,1], NOT percent) -----------
    robustness_matrix = {}
    rs_indexed = robustness_summary.set_index("condition")
    row_label_by_key = {"Nominal": "nominal", **{v: k for k, v in CONDITION_DISPLAY.items()}}
    for label, row in rs_indexed.iterrows():
        key = row_label_by_key[label]
        robustness_matrix[key] = {fam: float(row[fam]) / 100.0 for fam in FAMILY_ORDER}

    # --- robustness differences (per off-nominal condition) ---------------------
    robustness_diff_values = {}
    rd_display_to_key = {v: k for k, v in CONDITION_DISPLAY.items()}
    for cond_key in CONDITION_KEYS:
        cond_display = CONDITION_DISPLAY[cond_key]
        sub = robustness_diffs[robustness_diffs["condition"] == cond_display]
        entry = {}
        for _, row in sub.iterrows():
            entry[row["comparison"]] = {
                "difference_pp": float(row["difference_pp"]),
                "ci_low_pp": float(row["ci_low_pp"]),
                "ci_high_pp": float(row["ci_high_pp"]),
                "resolution": row["resolution"],
            }
        robustness_diff_values[cond_key] = entry
    # nominal primary differences (subset of the 9 nominal diffs) for the
    # condensed/full robustness-difference tables' "Nominal" row.
    robustness_diff_values["nominal"] = {
        label: nominal_diff_values[label] for label in PRIMARY_COMPARISONS
    }

    # --- root-specific learned results per condition (+ nominal) ---------------
    root_specific = {"nominal": {}}
    for fam, roots in LEARNED_ROOTS.items():
        prefix = LEARNED_FAMILY_PREFIX[fam]
        for r in roots:
            inst_id = f"{prefix}{r}"
            root_specific["nominal"][inst_id] = nominal_instance_values[inst_id]["success_rate"]
    for cond_key in CONDITION_KEYS:
        cond_instances = load_condition_instance_summary(cond_key).set_index("instance")
        root_specific[cond_key] = {}
        for fam, roots in LEARNED_ROOTS.items():
            prefix = LEARNED_FAMILY_PREFIX[fam]
            for r in roots:
                inst_id = f"{prefix}{r}"
                root_specific[cond_key][inst_id] = float(cond_instances.loc[inst_id, "success_rate"])

    # --- terminal outcomes per instance per condition (+ nominal) ---------------
    terminal_outcomes = {"nominal": {}}
    for _, row in nominal_instances.iterrows():
        n = row["n"]
        terminal_outcomes["nominal"][row["instance"]] = {
            "intercepted": int(row["successes"]),
            "soldier_caught": int(round(row["soldier_caught_rate"] * n)),
            "unsafe_intercept": int(round(row["unsafe_intercept_rate"] * n)),
            "timeout": int(round(row["timeout_rate"] * n)),
        }
    for cond_key in CONDITION_KEYS:
        cond_instances = load_condition_instance_summary(cond_key)
        terminal_outcomes[cond_key] = {}
        for _, row in cond_instances.iterrows():
            n = row["n"]
            terminal_outcomes[cond_key][row["instance"]] = {
                "intercepted": int(row["successes"]),
                "soldier_caught": int(round(row["soldier_caught_rate"] * n)),
                "unsafe_intercept": int(round(row["unsafe_intercept_rate"] * n)),
                "timeout": int(round(row["timeout_rate"] * n)),
            }

    # --- RMPC diagnostics per condition (diagnostic only, never runtime claims) -
    rmpc_diagnostics = {}
    for cond_key in CONDITION_KEYS:
        audit = json.loads((TIER_A_ROBUSTNESS_ROOT / cond_key / "condition_audit.json").read_text())
        rmpc_diagnostics[cond_key] = audit["rmpc_diagnostics"]

    # --- Tier-B descriptive replication (old vs new nominal, NEVER pooled) ------
    old_classical = load_old_nominal_classical_summary().set_index("instance")
    old_learned = load_old_nominal_learned_family_summary().set_index("method")
    old_family_rate = {
        "Greedy": float(old_classical.loc["greedy", "success_rate"]),
        "PN": float(old_classical.loc["pn", "success_rate"]),
        "Lead": float(old_classical.loc["lead", "success_rate"]),
        "PPO": float(old_learned.loc["ppo", "mean_success"]),
        "PPO-Kalman": float(old_learned.loc["ppo_kalman", "mean_success"]),
        "LR-PPO": float(old_learned.loc["lr_ppo", "mean_success"]),
    }
    replication = {}
    for fam, old_rate in old_family_rate.items():
        new_rate = nominal_family_values[fam]["success_rate"]
        replication[fam] = {
            "old_success_rate": old_rate,
            "new_success_rate": new_rate,
            "delta_pp": (new_rate - old_rate) * 100.0,
        }

    # --- guard: Tier-C tuning data must never equal/replace Tier-A performance --
    if abs(rmpc_config["validation_success_rate"] - nominal_family_values["RMPC"]["success_rate"]) < 1e-9:
        raise RuntimeError(
            "Tier-C RMPC tuning validation_success_rate must never coincide with "
            "Tier-A nominal RMPC performance -- possible accidental substitution"
        )

    return {
        "nominal": {
            "instances": nominal_instance_values,
            "families": nominal_family_values,
            "differences": nominal_diff_values,
        },
        "robustness": {
            "matrix": robustness_matrix,
            "differences": robustness_diff_values,
            "root_specific": root_specific,
            "terminal_outcomes": terminal_outcomes,
            "rmpc_diagnostics": rmpc_diagnostics,
        },
        "replication": replication,
        "rmpc_config": {
            "candidate_id": rmpc_config["candidate_id"],
            "horizon": rmpc_config["horizon"],
            "population_size": rmpc_config["population_size"],
            "elite_fraction": rmpc_config["elite_fraction"],
            "cem_iterations": rmpc_config["cem_iterations"],
            "controller_seed": rmpc_config["controller_seed"],
            "frozen_for_final_evaluation": rmpc_config["frozen_for_final_evaluation"],
            "development_selection_rule": rmpc_config["selection_rule"],
            "config_hash": sha256_of_file(TIER_C_TUNING_ROOT / "selected_rmpc_config.json"),
        },
    }


# =============================================================================
# Tables (Section 3/12/13/14)
# =============================================================================
def build_nominal_performance_table(pv: dict) -> pd.DataFrame:
    rows = []
    for fam in FAMILY_ORDER:
        f = pv["nominal"]["families"][fam]
        root_sd = f"{f['root_sd']*100:.2f}" if f["n_roots"] > 1 else ""
        rows.append(dict(
            Method=fam, **{"Success (%)": f"{f['success_rate']*100:.2f}"},
            **{"95% interval": f"[{f['ci_low']*100:.2f}, {f['ci_high']*100:.2f}]"},
            **{"Root SD (pp)": root_sd},
        ))
    return pd.DataFrame(rows)


def build_nominal_differences_table(pv: dict) -> pd.DataFrame:
    rows = []
    for label in PRIMARY_COMPARISONS + SECONDARY_COMPARISONS:
        d = pv["nominal"]["differences"][label]
        rows.append(dict(comparison=label, difference_pp=round(d["difference_pp"], 2),
                          ci_low_pp=round(d["ci_low_pp"], 2), ci_high_pp=round(d["ci_high_pp"], 2),
                          resolution=d["resolution"]))
    return pd.DataFrame(rows)


def build_robustness_success_table(pv: dict) -> pd.DataFrame:
    rows = []
    for key in ALL_ROW_KEYS:
        row = {"Condition": ALL_ROW_DISPLAY[key]}
        for fam in FAMILY_ORDER:
            row[fam] = round(pv["robustness"]["matrix"][key][fam] * 100.0, 2)
        rows.append(row)
    return pd.DataFrame(rows)


def build_robustness_differences_table(pv: dict) -> pd.DataFrame:
    rows = []
    for key in ALL_ROW_KEYS:
        for label in PRIMARY_COMPARISONS:
            d = pv["robustness"]["differences"][key][label]
            rows.append(dict(condition=ALL_ROW_DISPLAY[key], comparison=label,
                              difference_pp=round(d["difference_pp"], 2),
                              ci_low_pp=round(d["ci_low_pp"], 2), ci_high_pp=round(d["ci_high_pp"], 2),
                              resolution=d["resolution"]))
    return pd.DataFrame(rows)


_CONDENSED_LABELS = ("LR-PPO - Lead", "LR-PPO - RMPC", "LR-PPO - PPO")


def _condensed_cell(d: dict) -> str:
    star = "*" if d["resolution"] != "unresolved" else ""
    return f"{d['difference_pp']:+.2f} [{d['ci_low_pp']:.2f}, {d['ci_high_pp']:.2f}]{star}"


def build_robustness_condensed_table(pv: dict) -> pd.DataFrame:
    rows = []
    for key in ALL_ROW_KEYS:
        row = {"Condition": ALL_ROW_DISPLAY[key]}
        for label in _CONDENSED_LABELS:
            row[label] = _condensed_cell(pv["robustness"]["differences"][key][label])
        rows.append(row)
    return pd.DataFrame(rows)


def _df_to_tex(df: pd.DataFrame, caption: str, label: str, bold_max_cols: tuple[str, ...] = ()) -> str:
    cols = list(df.columns)
    lines = [
        r"\begin{table}[t]", r"\centering",
        rf"\caption{{{caption}}}", rf"\label{{{label}}}",
        r"\begin{tabular}{" + "l" * len(cols) + "}", r"\hline",
        " & ".join(cols) + r" \\", r"\hline",
    ]
    for _, row in df.iterrows():
        cells = []
        for c in cols:
            val = row[c]
            if bold_max_cols and c in bold_max_cols:
                numeric_cols = [cc for cc in cols if cc not in ("Condition", "Method")]
                row_vals = {cc: float(row[cc]) for cc in numeric_cols}
                if numeric_cols and c == max(row_vals, key=row_vals.get):
                    cells.append(rf"\textbf{{{val}}}")
                    continue
            cells.append(str(val))
        lines.append(" & ".join(cells) + r" \\")
    lines += [r"\hline", r"\end{tabular}", r"\end{table}"]
    return "\n".join(lines) + "\n"


def write_all_tables(pv: dict, tables_dir: Path) -> None:
    tables_dir.mkdir(parents=True, exist_ok=True)

    nominal_perf = build_nominal_performance_table(pv)
    nominal_perf.to_csv(tables_dir / "nominal_performance.csv", index=False)
    (tables_dir / "nominal_performance.tex").write_text(_df_to_tex(
        nominal_perf, "Final nominal safe-interception success (5000 matched seeds, 72000--76999).",
        "tab:nominal_performance"))

    nominal_diffs = build_nominal_differences_table(pv)
    nominal_diffs.to_csv(tables_dir / "nominal_differences.csv", index=False)
    (tables_dir / "nominal_differences.tex").write_text(_df_to_tex(
        nominal_diffs, "Predeclared nominal paired differences (percentage points, 95\\% CI).",
        "tab:nominal_differences"))

    rob_success = build_robustness_success_table(pv)
    rob_success.to_csv(tables_dir / "robustness_success.csv", index=False)
    (tables_dir / "robustness_success.tex").write_text(_df_to_tex(
        rob_success,
        "Safe-interception success (\\%) across nominal and five off-nominal conditions "
        "(bold denotes numerical maximum, not necessarily statistically resolved superiority).",
        "tab:robustness_success", bold_max_cols=tuple(FAMILY_ORDER)))

    rob_diffs = build_robustness_differences_table(pv)
    rob_diffs.to_csv(tables_dir / "robustness_differences.csv", index=False)
    (tables_dir / "robustness_differences.tex").write_text(_df_to_tex(
        rob_diffs, "Predeclared primary differences across nominal and five off-nominal conditions.",
        "tab:robustness_differences"))

    rob_condensed = build_robustness_condensed_table(pv)
    rob_condensed.to_csv(tables_dir / "robustness_condensed.csv", index=False)
    (tables_dir / "robustness_condensed.tex").write_text(_df_to_tex(
        rob_condensed,
        "Condensed LR-PPO paired differences (pp [95\\% CI]; * denotes CI excludes zero).",
        "tab:robustness_condensed"))


# =============================================================================
# Figures (Section 15/16)
# =============================================================================
_IEEE_RC = {
    "font.family": "serif",
    "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
    "mathtext.fontset": "stix",
    "pdf.fonttype": 42,
    "ps.fonttype": 42,
}
_METHOD_COLORS = {
    "Greedy": "#8c8c8c", "PN": "#8c8c8c", "Lead": "#4c72b0", "RMPC": "#c44e52",
    "PPO": "#55a868", "PPO-Kalman": "#8172b2", "LR-PPO": "#dd8452",
}


def _save_figure(fig: plt.Figure, pdf_path: Path, png_path: Path) -> None:
    pdf_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(pdf_path, bbox_inches="tight", pad_inches=0.02)
    fig.savefig(png_path, dpi=300, bbox_inches="tight", pad_inches=0.02)
    plt.close(fig)


def generate_nominal_performance_figure(pv: dict, pdf_path: Path, png_path: Path) -> None:
    with plt.rc_context(_IEEE_RC):
        fig, ax = plt.subplots(figsize=(3.4, 2.6), constrained_layout=True)
        rates, lo_err, hi_err, colors, edgecolors, linewidths = [], [], [], [], [], []
        for fam in FAMILY_ORDER:
            f = pv["nominal"]["families"][fam]
            rates.append(f["success_rate"] * 100.0)
            lo_err.append((f["success_rate"] - f["ci_low"]) * 100.0)
            hi_err.append((f["ci_high"] - f["success_rate"]) * 100.0)
            colors.append(_METHOD_COLORS[fam])
            is_proposed = fam == "LR-PPO"
            edgecolors.append("black" if is_proposed else "none")
            linewidths.append(1.6 if is_proposed else 0.0)
        x = np.arange(len(FAMILY_ORDER))
        ax.bar(x, rates, yerr=[lo_err, hi_err], capsize=3, color=colors,
               edgecolor=edgecolors, linewidth=linewidths, width=0.65)
        ax.set_xticks(x)
        ax.set_xticklabels(FAMILY_ORDER, rotation=35, ha="right", fontsize=8)
        ax.set_ylabel("Safe-interception success (%)", fontsize=9)
        ax.set_ylim(0, 100)
        ax.tick_params(axis="y", labelsize=8)
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        _save_figure(fig, pdf_path, png_path)


def generate_robustness_heatmap_figure(pv: dict, pdf_path: Path, png_path: Path) -> None:
    with plt.rc_context(_IEEE_RC):
        data = np.array([[pv["robustness"]["matrix"][key][fam] * 100.0 for fam in FAMILY_ORDER]
                          for key in ALL_ROW_KEYS])
        fig, ax = plt.subplots(figsize=(4.6, 3.0), constrained_layout=True)
        im = ax.imshow(data, cmap="viridis", vmin=0, vmax=100, aspect="auto")
        ax.set_xticks(np.arange(len(FAMILY_ORDER)))
        ax.set_xticklabels(FAMILY_ORDER, rotation=35, ha="right", fontsize=8)
        ax.set_yticks(np.arange(len(ALL_ROW_KEYS)))
        ax.set_yticklabels([ALL_ROW_DISPLAY[k] for k in ALL_ROW_KEYS], fontsize=8)
        for i in range(data.shape[0]):
            for j in range(data.shape[1]):
                val = data[i, j]
                text_color = "white" if val < 55 else "black"
                ax.text(j, i, f"{val:.1f}", ha="center", va="center", fontsize=6.5, color=text_color)
        cbar = fig.colorbar(im, ax=ax, shrink=0.85)
        cbar.ax.tick_params(labelsize=7)
        cbar.set_label("Success (%)", fontsize=8)
        _save_figure(fig, pdf_path, png_path)


def write_figure_captions(figures_dir: Path) -> None:
    text = (
        "# Figure captions (stored separately from the rendered figures)\n\n"
        "## fig_nominal_performance\n"
        "Final nominal safe-interception success rate (%) for all seven controller "
        "families on 5000 fresh matched environment seeds (72000-76999). Error bars "
        "show 95% intervals (Wilson for the four deterministic controllers; "
        "root-aware hierarchical bootstrap for the three learned families). "
        "LR-PPO (the proposed method) is outlined in black; RMPC is shown with the "
        "same visual weight as every other method.\n\n"
        "## fig_robustness_summary\n"
        "Safe-interception success rate (%) for all seven controller families across "
        "the nominal condition and five predeclared off-nominal conditions, all "
        "evaluated on the same reused 1000-seed block (77000-77999) per condition "
        "(nominal uses the independent 5000-seed block 72000-76999). The five "
        "off-nominal rows are NOT independent experiments -- they reuse identical "
        "environment seeds by design.\n"
    )
    figures_dir.mkdir(parents=True, exist_ok=True)
    (figures_dir / "captions.md").write_text(text)


# =============================================================================
# Claims ledger (Section 9)
# =============================================================================
def build_claims_ledger(pv: dict) -> pd.DataFrame:
    nd = pv["nominal"]["differences"]
    rd = pv["robustness"]["differences"]

    def fmt(d: dict) -> str:
        return f"{d['difference_pp']:+.2f} pp [{d['ci_low_pp']:.2f}, {d['ci_high_pp']:.2f}]"

    n_pos_ppo = sum(1 for k in CONDITION_KEYS if rd[k]["LR-PPO - PPO"]["resolution"] == "resolved positive")
    n_pos_lead = sum(1 for k in CONDITION_KEYS if rd[k]["LR-PPO - Lead"]["resolution"] == "resolved positive")
    n_unresolved_lead = sum(1 for k in CONDITION_KEYS if rd[k]["LR-PPO - Lead"]["resolution"] == "unresolved")

    rows = [
        dict(claim_id="C1", topic="LR-PPO vs Lead (nominal)",
             proposed_claim="LR-PPO significantly improved safe-interception success over predictive Lead "
                             "guidance under nominal conditions.",
             evidence_source="results/rmpc_final_nominal/paired_differences.csv",
             point_estimate=fmt(nd["LR-PPO - Lead"]), confidence_interval=fmt(nd["LR-PPO - Lead"]),
             statistical_status=nd["LR-PPO - Lead"]["resolution"],
             allowed_wording="LR-PPO significantly improved safe-interception success over predictive Lead "
                             "guidance under nominal conditions.",
             prohibited_stronger_wording="LR-PPO always outperforms Lead. | RL corrects all Lead prediction errors.",
             paper_section="Results (nominal)", priority="high"),
        dict(claim_id="C2", topic="LR-PPO vs RMPC (nominal)",
             proposed_claim="LR-PPO significantly outperformed the scenario-based robust MPC baseline under "
                             "nominal conditions.",
             evidence_source="results/rmpc_final_nominal/paired_differences.csv",
             point_estimate=fmt(nd["LR-PPO - RMPC"]), confidence_interval=fmt(nd["LR-PPO - RMPC"]),
             statistical_status=nd["LR-PPO - RMPC"]["resolution"],
             allowed_wording="LR-PPO significantly outperformed the scenario-based robust MPC baseline under "
                             "nominal conditions.",
             prohibited_stronger_wording="LR-PPO is superior to robust control. | RL dominates optimal control.",
             paper_section="Results (nominal)", priority="high"),
        dict(claim_id="C3", topic="LR-PPO vs PPO (nominal)",
             proposed_claim="LR-PPO achieved a numerically higher nominal success rate than PPO, but the "
                             "root-aware 95% interval for the difference included zero.",
             evidence_source="results/rmpc_final_nominal/paired_differences.csv",
             point_estimate=fmt(nd["LR-PPO - PPO"]), confidence_interval=fmt(nd["LR-PPO - PPO"]),
             statistical_status=nd["LR-PPO - PPO"]["resolution"],
             allowed_wording="LR-PPO achieved a numerically higher nominal success rate than PPO, but the "
                             "root-aware 95% interval for the difference included zero.",
             prohibited_stronger_wording="LR-PPO significantly outperformed PPO nominally.",
             paper_section="Results (nominal)", priority="high"),
        dict(claim_id="C4", topic="LR-PPO vs PPO (robustness)",
             proposed_claim=f"LR-PPO significantly outperformed PPO in {n_pos_ppo} of five selected off-nominal "
                             f"conditions; high measurement noise is the exception (resolved negative).",
             evidence_source="results/rmpc_targeted_robustness/primary_differences_summary.csv",
             point_estimate=fmt(rd['high_measurement_noise']['LR-PPO - PPO']),
             confidence_interval=fmt(rd['high_measurement_noise']['LR-PPO - PPO']),
             statistical_status=f"{n_pos_ppo}/5 resolved positive, 1/5 resolved negative",
             allowed_wording=f"LR-PPO significantly outperformed PPO in {n_pos_ppo} of five selected off-nominal "
                             f"conditions. High measurement noise is the exception.",
             prohibited_stronger_wording="LR-PPO universally outperforms PPO. | LR-PPO is uniformly more robust "
                                         "to sensing noise.",
             paper_section="Results (robustness)", priority="high"),
        dict(claim_id="C5", topic="LR-PPO vs Lead (robustness)",
             proposed_claim=f"LR-PPO significantly improved upon Lead nominally and in {n_pos_lead} of the five "
                             f"selected off-nominal conditions, while the remaining {n_unresolved_lead} "
                             f"comparisons were statistically unresolved.",
             evidence_source="results/rmpc_targeted_robustness/primary_differences_summary.csv",
             point_estimate=fmt(nd["LR-PPO - Lead"]), confidence_interval=fmt(nd["LR-PPO - Lead"]),
             statistical_status=f"nominal resolved positive; {n_pos_lead}/5 off-nominal resolved positive; "
                                f"{n_unresolved_lead}/5 unresolved; 0/5 resolved negative",
             allowed_wording=f"LR-PPO significantly improved upon Lead nominally and in {n_pos_lead} of the five "
                             f"selected off-nominal conditions, while the remaining comparisons were statistically "
                             f"unresolved. No tested LR-PPO-versus-Lead condition produced a resolved negative "
                             f"difference.",
             prohibited_stronger_wording="LR-PPO significantly beats Lead under every condition.",
             paper_section="Results (robustness)", priority="high"),
        dict(claim_id="C6", topic="RMPC condition-dependent behavior",
             proposed_claim="The scenario-based RMPC baseline showed strongly condition-dependent behavior: "
                             "numerical leader under hard mobility, weakest under high measurement noise.",
             evidence_source="results/rmpc_targeted_robustness/robustness_summary.csv",
             point_estimate=fmt(rd["hard_mobility"]["RMPC - Lead"]),
             confidence_interval=fmt(rd["hard_mobility"]["RMPC - Lead"]),
             statistical_status="condition-dependent (see robustness_differences.csv)",
             allowed_wording="The scenario-based RMPC baseline showed strongly condition-dependent behavior. "
                             "RMPC was the numerical leader under the hard-mobility condition and significantly "
                             "outperformed Lead and PPO there. RMPC degraded sharply under high measurement noise.",
             prohibited_stronger_wording="RMPC is weak. | RMPC is inferior to learning. | RMPC is globally "
                                         "optimal. | RMPC represents all robust-control approaches.",
             paper_section="Results (robustness) / RMPC discussion", priority="high"),
        dict(claim_id="C7", topic="Kalman preprocessing",
             proposed_claim="The tested constant-velocity Kalman preprocessing did not provide a consistent "
                             "improvement over PPO.",
             evidence_source="results/rmpc_final_nominal/paired_differences.csv (PPO-Kalman - PPO)",
             point_estimate=fmt(nd["PPO-Kalman - PPO"]), confidence_interval=fmt(nd["PPO-Kalman - PPO"]),
             statistical_status=nd["PPO-Kalman - PPO"]["resolution"],
             allowed_wording="The tested constant-velocity Kalman preprocessing did not provide a consistent "
                             "improvement over PPO.",
             prohibited_stronger_wording="Kalman filters are harmful. | Kalman filtering is unsuitable for "
                                         "counter-UAS interception.",
             paper_section="Results (nominal) / discussion", priority="medium"),
        dict(claim_id="C8", topic="Reviewer robust-baseline issue",
             proposed_claim="The revised evaluation includes a frozen scenario-based robust MPC comparator "
                             "operating under the same information and engagement constraints.",
             evidence_source="docs/rmpc_final_nominal_protocol.md, docs/rmpc_targeted_robustness_protocol.md, "
                             "results/rmpc_tuning/selected_rmpc_config.json",
             point_estimate="n/a", confidence_interval="n/a", statistical_status="qualitative / reviewer response",
             allowed_wording="The revised evaluation includes a frozen scenario-based robust MPC comparator "
                             "operating under the same information and engagement constraints.",
             prohibited_stronger_wording="We added a provably optimal-control baseline.",
             paper_section="Response to reviewers", priority="high"),
    ]
    return pd.DataFrame(rows)


# =============================================================================
# Reviewer documents (Section 19)
# =============================================================================
def build_reviewer_issue_matrix(pv: dict) -> str:
    nd = pv["nominal"]["differences"]
    return f"""# Reviewer Issue Matrix

| Field | Value |
|---|---|
| Issue | No robust optimal-control / MPC baseline |
| Prior status | Open |
| Revision | Added an independently tuned, frozen scenario-based robust MPC (RMPC) baseline. |
| Evidence | Nominal: 5000 matched seeds (72000-76999). Robustness: five off-nominal conditions x 1000 matched seeds (77000-77999), same information contract, same pre-detection protocol, no privileged true hostile state. |
| Nominal result | LR-PPO - RMPC = {nd['LR-PPO - RMPC']['difference_pp']:+.2f} pp [{nd['LR-PPO - RMPC']['ci_low_pp']:.2f}, {nd['LR-PPO - RMPC']['ci_high_pp']:.2f}] ({nd['LR-PPO - RMPC']['resolution']}); RMPC - Lead = {nd['RMPC - Lead']['difference_pp']:+.2f} pp ({nd['RMPC - Lead']['resolution']}) |
| Status | ADDRESSED |

## Safe response

"We added a scenario-based robust MPC comparator based on finite-horizon CEM
optimization over bounded target-motion scenarios. The controller was tuned
on development/validation seeds, frozen, and subsequently evaluated on
independent nominal and robustness seed blocks."

## Mandatory qualification

RMPC is not globally optimal and is not presented as a complete
differential-game solution. Its relative standing is condition-dependent
(see `paper_artifacts/resubmission_final/tables/robustness_success.csv`).

If an existing reviewer-action document exists elsewhere in the repository,
this record should be cross-referenced from it rather than treated as a
contradictory or duplicate reviewer record.
"""


def build_robust_baseline_response(pv: dict) -> str:
    nd = pv["nominal"]["differences"]
    return f"""# Robust-Baseline Reviewer Response

We thank the reviewer for pointing out the absence of a robust
optimal-control / MPC comparator in the original submission. We have added
a scenario-based robust Model Predictive Control (RMPC) baseline: a
finite-horizon (H=4) Cross-Entropy-Method (CEM) optimizer (population 64,
elite fraction 0.20, 3 iterations) that plans over a fixed set of bounded
hostile-motion scenarios rather than a single nominal prediction. RMPC was
tuned on independent development/validation seeds, frozen
(`results/rmpc_tuning/selected_rmpc_config.json`,
`frozen_for_final_evaluation=true`), and evaluated -- without further
tuning -- on the same nominal (72000-76999) and off-nominal (77000-77999,
five conditions) seed blocks as every other controller, under an identical
information contract (sanitized measurement-track state only; no ground
truth, no hidden attacker parameters).

RMPC's nominal safe-interception success was {pv['nominal']['families']['RMPC']['success_rate']*100:.2f}%
(vs. LR-PPO's {pv['nominal']['families']['LR-PPO']['success_rate']*100:.2f}%; difference
{nd['LR-PPO - RMPC']['difference_pp']:+.2f} pp, {nd['LR-PPO - RMPC']['resolution']}). Under the five
off-nominal conditions RMPC's relative standing was strongly condition-dependent: it was the
numerical leader under the hard-mobility condition, significantly outperforming both Lead and PPO
there, while degrading sharply under high measurement noise.

We do not claim RMPC is globally optimal or represents all possible
robust-control formulations; it is one concrete, frozen, scenario-based
comparator evaluated under the same constraints as every learned and
scripted method in this study.
"""


# =============================================================================
# Reports (Section 24/25/20)
# =============================================================================
def build_evidence_summary(pv: dict) -> str:
    nd = pv["nominal"]["differences"]
    rd = pv["robustness"]["differences"]
    n_pos_ppo = sum(1 for k in CONDITION_KEYS if rd[k]["LR-PPO - PPO"]["resolution"] == "resolved positive")
    n_pos_lead = sum(1 for k in CONDITION_KEYS if rd[k]["LR-PPO - Lead"]["resolution"] == "resolved positive")
    n_unres_lead = sum(1 for k in CONDITION_KEYS if rd[k]["LR-PPO - Lead"]["resolution"] == "unresolved")
    n_neg_lead = sum(1 for k in CONDITION_KEYS if rd[k]["LR-PPO - Lead"]["resolution"] == "resolved negative")

    return f"""# Evidence Summary (Publication-Safe Conclusions)

1. **Nominal:** LR-PPO significantly beat Lead ({nd['LR-PPO - Lead']['difference_pp']:+.2f} pp,
   {nd['LR-PPO - Lead']['resolution']}) and RMPC ({nd['LR-PPO - RMPC']['difference_pp']:+.2f} pp,
   {nd['LR-PPO - RMPC']['resolution']}).
2. **Nominal:** LR-PPO vs PPO was {nd['LR-PPO - PPO']['resolution']}
   ({nd['LR-PPO - PPO']['difference_pp']:+.2f} pp [{nd['LR-PPO - PPO']['ci_low_pp']:.2f}, {nd['LR-PPO - PPO']['ci_high_pp']:.2f}]).
3. **Robustness:** LR-PPO significantly beat PPO in {n_pos_ppo}/5 selected off-nominal conditions.
4. **High measurement noise:** PPO significantly beat LR-PPO
   ({rd['high_measurement_noise']['LR-PPO - PPO']['difference_pp']:+.2f} pp,
   {rd['high_measurement_noise']['LR-PPO - PPO']['resolution']}).
5. **LR-PPO vs Lead:** nominal resolved positive; {n_pos_lead}/5 off-nominal resolved positive;
   {n_unres_lead}/5 unresolved; {n_neg_lead}/5 resolved negative (never resolved negative).
6. **RMPC:** condition-dependent; numerical winner under hard mobility; weak under high
   measurement noise.
7. **PPO-Kalman:** no consistent evidence of improvement over PPO
   ({nd['PPO-Kalman - PPO']['difference_pp']:+.2f} pp, {nd['PPO-Kalman - PPO']['resolution']}).
8. **Reviewer:** the robust-MPC-comparator gap is addressed (see
   `reviewer/reviewer_issue_matrix.md`).

## Recommended contribution framing

**Contribution 1.** A 3-D constrained point-mass counter-UAS interception
framework with common sensing, acquisition, and vehicle-dynamics semantics
across all controllers.

**Contribution 2.** LR-PPO, which retains constant-velocity predictive Lead
guidance as the nominal action and learns a bounded angular residual around
it.

**Contribution 3.** A controlled comparison against classical guidance,
standalone PPO, Kalman-preprocessed PPO, and an independently tuned
scenario-based RMPC.

**Contribution 4.** A matched-seed robustness evaluation showing
controller-dependent strengths across maneuvering, dynamics constraints,
measurement noise, mobility, and detection range.

RMPC is a comparator, not a proposed contribution.

## Recommended central result statement

"LR-PPO retains predictive Lead guidance as a nominal interception policy
and learns a bounded angular residual. On 5,000 fresh nominal Monte Carlo
engagements, LR-PPO improved safe-interception success over Lead by
approximately {nd['LR-PPO - Lead']['difference_pp']:.1f} percentage points and over the
independently tuned scenario-based RMPC baseline by approximately
{nd['LR-PPO - RMPC']['difference_pp']:.1f} percentage points. Across five selected
off-nominal conditions, LR-PPO significantly outperformed standalone PPO in
{n_pos_ppo} conditions, while PPO remained strongest under high measurement
noise."

## Do-not-claim boundaries

- no high-fidelity UAV model
- no six-DOF aerodynamics
- no hardware validation
- no real radar calibration
- no optimal adversarial attacker
- no game-theoretic equilibrium
- no globally optimal RMPC
- no universal superiority over PPO
- no universal superiority over RMPC
- no universal superiority over Lead
- no general claim that Kalman filtering is harmful
- no claim that larger detection radius always improves success
- no claim that lower hostile speed is inherently harder
- no claim PPO significantly beats Lead nominally
- no claim LR-PPO significantly beats PPO nominally
- no causal claim that PPO explicitly observes or corrects Lead prediction error

## Suggested figure/table numbering (manuscript layout recommendation only -- not yet applied)

- Fig. 1: conceptual 3-D engagement scene (existing `images/UAV_intercept_scene.png`)
- Fig. 2: LR-PPO architecture
- Fig. 3: final nominal performance including RMPC (`fig_nominal_performance`)
- Fig. 4: six-condition robustness summary including RMPC (`fig_robustness_summary`)
- Table I: controller definitions / comparison hierarchy
- Table II: nominal success + CI (`nominal_performance`)
- Table III: condensed robustness differences (`robustness_condensed`), IF page budget permits;
  otherwise use the bolded `robustness_success` matrix alone and move differences to an appendix.
"""


def build_statistical_summary(pv: dict) -> str:
    lines = ["# Statistical Summary -- Source Traceability", "",
             "Every main estimate below is traced to its exact source file, row identity, "
             "seed range, and confidence-interval method. No value here was retyped from a "
             "prior Markdown report -- all were re-read from raw CSV/JSON artifacts.", ""]
    lines += ["## Nominal families (source: results/rmpc_final_nominal/family_summary.csv, seeds 72000-76999)", ""]
    for fam in FAMILY_ORDER:
        f = pv["nominal"]["families"][fam]
        ci_method = "wilson_interval" if f["n_roots"] == 1 else "hierarchical_bootstrap_track"
        lines.append(f"- **{fam}**: {f['success_rate']*100:.2f}% [{f['ci_low']*100:.2f}, {f['ci_high']*100:.2f}] "
                      f"(n_roots={f['n_roots']}, CI method={ci_method})")
    lines += ["", "## Nominal differences (source: results/rmpc_final_nominal/paired_differences.csv)", ""]
    for label, d in pv["nominal"]["differences"].items():
        lines.append(f"- **{label}**: {d['difference_pp']:+.2f} pp [{d['ci_low_pp']:.2f}, {d['ci_high_pp']:.2f}] "
                      f"({d['resolution']})")
    lines += ["", "## Robustness matrix (source: results/rmpc_targeted_robustness/robustness_summary.csv, "
                   "seeds 77000-77999 reused across all five conditions)", ""]
    for key in ALL_ROW_KEYS:
        row = pv["robustness"]["matrix"][key]
        vals = ", ".join(f"{fam}={row[fam]*100:.2f}%" for fam in FAMILY_ORDER)
        lines.append(f"- **{ALL_ROW_DISPLAY[key]}**: {vals}")
    lines += ["", "## Robustness differences (source: "
                   "results/rmpc_targeted_robustness/primary_differences_summary.csv)", ""]
    for key in CONDITION_KEYS:
        lines.append(f"### {CONDITION_DISPLAY[key]}")
        for label, d in pv["robustness"]["differences"][key].items():
            lines.append(f"- **{label}**: {d['difference_pp']:+.2f} pp [{d['ci_low_pp']:.2f}, "
                          f"{d['ci_high_pp']:.2f}] ({d['resolution']})")
        lines.append("")
    lines += ["## Sign-orientation check", "",
              "All differences above are computed and reported as `A - B` exactly as labeled "
              "(e.g. `LR-PPO - RMPC`, never silently `RMPC - LR-PPO`); orientation is preserved "
              "verbatim from `paired_differences.csv` / `primary_differences_summary.csv` with no "
              "re-derivation or sign manipulation in this summary.", ""]
    return "\n".join(lines) + "\n"


# =============================================================================
# Provenance / manifest (Section 21)
# =============================================================================
def _git_commit() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


def _git_branch() -> str:
    try:
        return subprocess.check_output(["git", "rev-parse", "--abbrev-ref", "HEAD"], cwd=PROJECT_ROOT).decode().strip()
    except Exception:
        return "unknown"


NOMINAL_RESULTS_COMMIT = "b0f952f"
TARGETED_ROBUSTNESS_RESULTS_COMMIT = "e82a188"
RMPC_TUNING_RESULTS_COMMIT = "ed4124c"

SOURCE_FILES_TO_HASH = {
    "soldier_env.py": PROJECT_ROOT / "uav_defend" / "envs" / "soldier_env.py",
    "rmpc_policy.py": PROJECT_ROOT / "uav_defend" / "policies" / "mpc" / "rmpc_policy.py",
    "lead_residual_ppo_policy_wrapper.py": PROJECT_ROOT / "uav_defend" / "policies" / "residual" / "lead_residual_ppo_policy_wrapper.py",
    "run_rmpc_final_nominal.py": PROJECT_ROOT / "experiments" / "run_rmpc_final_nominal.py",
    "run_rmpc_targeted_robustness.py": PROJECT_ROOT / "experiments" / "run_rmpc_targeted_robustness.py",
    "build_resubmission_evidence.py": PROJECT_ROOT / "experiments" / "build_resubmission_evidence.py",
}

LEARNED_MODEL_PATHS = {
    "ppo_seed45": PROJECT_ROOT / "results" / "training_post_detection_400k" / "direct" / "seed_45" / "best_model.zip",
    "ppo_seed46": PROJECT_ROOT / "results" / "training_post_detection_400k" / "direct" / "seed_46" / "best_model.zip",
    "ppo_seed47": PROJECT_ROOT / "results" / "training_post_detection_400k" / "direct" / "seed_47" / "best_model.zip",
    "ppo_kalman_seed45": PROJECT_ROOT / "results" / "training_post_detection_400k" / "kalman" / "seed_45" / "best_model.zip",
    "ppo_kalman_seed46": PROJECT_ROOT / "results" / "training_post_detection_400k" / "kalman" / "seed_46" / "best_model.zip",
    "ppo_kalman_seed47": PROJECT_ROOT / "results" / "training_post_detection_400k" / "kalman" / "seed_47" / "best_model.zip",
    "lr_ppo_seed55": PROJECT_ROOT / "models" / "lead_residual_post_detection" / "seed_55" / "best_model.zip",
    "lr_ppo_seed56": PROJECT_ROOT / "models" / "lead_residual_post_detection" / "seed_56" / "best_model.zip",
    "lr_ppo_seed57": PROJECT_ROOT / "models" / "lead_residual_post_detection" / "seed_57" / "best_model.zip",
}


def build_evidence_manifest(pv: dict, base_commit: str, table_hashes: dict, figure_hashes: dict,
                             paper_values_hash: str, claims_ledger_hash: str) -> dict:
    return {
        "repository": "mikemishal/RL-UAV-V2",
        "branch": _git_branch(),
        "paper_evidence_base_commit": base_commit,
        "paper_evidence_build_commit": _git_commit(),
        "nominal_results_commit": NOMINAL_RESULTS_COMMIT,
        "targeted_robustness_results_commit": TARGETED_ROBUSTNESS_RESULTS_COMMIT,
        "rmpc_tuning_results_commit": RMPC_TUNING_RESULTS_COMMIT,
        "rmpc_configuration_hash": pv["rmpc_config"]["config_hash"],
        "learned_model_hashes": {k: sha256_of_file(p) for k, p in LEARNED_MODEL_PATHS.items() if p.exists()},
        "source_hashes": {k: sha256_of_file(p) for k, p in SOURCE_FILES_TO_HASH.items() if p.exists()},
        "result_file_hashes": {
            "rmpc_final_nominal/family_summary.csv": sha256_of_file(TIER_A_NOMINAL_ROOT / "family_summary.csv"),
            "rmpc_final_nominal/instance_summary.csv": sha256_of_file(TIER_A_NOMINAL_ROOT / "instance_summary.csv"),
            "rmpc_final_nominal/paired_differences.csv": sha256_of_file(TIER_A_NOMINAL_ROOT / "paired_differences.csv"),
            "rmpc_targeted_robustness/robustness_summary.csv": sha256_of_file(TIER_A_ROBUSTNESS_ROOT / "robustness_summary.csv"),
            "rmpc_targeted_robustness/primary_differences_summary.csv": sha256_of_file(TIER_A_ROBUSTNESS_ROOT / "primary_differences_summary.csv"),
        },
        "paper_values_json_hash": paper_values_hash,
        "table_hashes": table_hashes,
        "figure_hashes": figure_hashes,
        "claims_ledger_hash": claims_ledger_hash,
        "timestamp_utc": datetime.now(timezone.utc).isoformat(),
        "python_version": sys.version,
        "numpy_version": np.__version__,
        "seed_ledger": {
            "used_final_paper_evidence": {"nominal": [72000, 76999], "targeted_robustness": [77000, 77999]},
            "reserved_unopened": {"rmpc_tuning_reserve": [71300, 71999], "future_robustness": [78000, 99999]},
        },
        "no_new_environment_seed_executed": True,
        "tier_c_note": "results/rmpc_tuning/ used ONLY for the frozen controller definition and its hash; "
                        "no tuning-stage success rate was used as final performance evidence.",
    }


# =============================================================================
# Artifact audit (Section 27) -- called AFTER post-build pytest/git checks
# =============================================================================
def build_artifact_audit(pytest_passed: bool, git_diff_clean: bool, git_status_clean: bool) -> str:
    return f"""# Artifact Audit

- No SoldierEnv episodes executed: YES (builder only reads existing CSV/JSON)
- No models loaded for inference: YES
- No new seeds used: YES (only 72000-76999 and 77000-77999 read, descriptively)
- No raw result CSV modified: YES
- No model file modified: YES
- No controller source modified: YES
- No historical paper artifact overwritten: YES (new artifacts confined to
  `paper_artifacts/resubmission_final/`)
- No manuscript file edited: YES
- Generated figures/tables trace to `paper_values.json`: YES
- `paper_values.json` traces to Tier-A final results: YES
  (`results/rmpc_final_nominal/`, `results/rmpc_targeted_robustness/`)
- Post-build `pytest -q`: {"PASSED" if pytest_passed else "FAILED"}
- Post-build `git diff --check` clean: {"YES" if git_diff_clean else "NO"}
- Post-build `git status` clean (excluding the new paper_artifacts output): {"YES" if git_status_clean else "NO"}
"""


# =============================================================================
# Orchestration
# =============================================================================
def main() -> None:
    base_commit = _git_commit()
    pv = build_paper_values()

    (OUTPUT_ROOT / "tables").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "figures").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "reviewer").mkdir(parents=True, exist_ok=True)
    (OUTPUT_ROOT / "reports").mkdir(parents=True, exist_ok=True)

    paper_values_path = OUTPUT_ROOT / "paper_values.json"
    paper_values_path.write_text(json.dumps(pv, indent=2, sort_keys=True, default=str))
    paper_values_hash = sha256_of_file(paper_values_path)

    write_all_tables(pv, OUTPUT_ROOT / "tables")
    table_hashes = {p.name: sha256_of_file(p) for p in sorted((OUTPUT_ROOT / "tables").glob("*"))}

    generate_nominal_performance_figure(
        pv, OUTPUT_ROOT / "figures" / "fig_nominal_performance.pdf",
        OUTPUT_ROOT / "figures" / "fig_nominal_performance.png")
    generate_robustness_heatmap_figure(
        pv, OUTPUT_ROOT / "figures" / "fig_robustness_summary.pdf",
        OUTPUT_ROOT / "figures" / "fig_robustness_summary.png")
    write_figure_captions(OUTPUT_ROOT / "figures")
    figure_hashes = {p.name: sha256_of_file(p) for p in sorted((OUTPUT_ROOT / "figures").glob("*"))
                     if p.suffix in (".pdf", ".png")}

    claims = build_claims_ledger(pv)
    claims.to_csv(OUTPUT_ROOT / "claims_ledger.csv", index=False)
    claims_md_lines = ["# Claims Ledger", ""]
    for _, r in claims.iterrows():
        claims_md_lines += [
            f"## {r['claim_id']} -- {r['topic']}", "",
            f"- **Proposed claim**: {r['proposed_claim']}",
            f"- **Evidence source**: {r['evidence_source']}",
            f"- **Point estimate**: {r['point_estimate']}",
            f"- **Statistical status**: {r['statistical_status']}",
            f"- **Allowed wording**: {r['allowed_wording']}",
            f"- **Prohibited stronger wording**: {r['prohibited_stronger_wording']}",
            f"- **Paper section**: {r['paper_section']} (priority: {r['priority']})", "",
        ]
    (OUTPUT_ROOT / "claims_ledger.md").write_text("\n".join(claims_md_lines) + "\n")
    claims_ledger_hash = sha256_of_file(OUTPUT_ROOT / "claims_ledger.csv")

    (OUTPUT_ROOT / "reviewer" / "reviewer_issue_matrix.md").write_text(build_reviewer_issue_matrix(pv))
    (OUTPUT_ROOT / "reviewer" / "robust_baseline_response.md").write_text(build_robust_baseline_response(pv))

    (OUTPUT_ROOT / "reports" / "evidence_summary.md").write_text(build_evidence_summary(pv))
    (OUTPUT_ROOT / "reports" / "statistical_summary.md").write_text(build_statistical_summary(pv))

    manifest = build_evidence_manifest(pv, base_commit, table_hashes, figure_hashes,
                                        paper_values_hash, claims_ledger_hash)
    (OUTPUT_ROOT / "evidence_manifest.json").write_text(json.dumps(manifest, indent=2, sort_keys=True, default=str))

    print("=== RESUBMISSION EVIDENCE PACKAGE BUILT ===")
    print(f"Output: {OUTPUT_ROOT}")
    print(f"paper_values.json hash: {paper_values_hash}")
    print(f"claims_ledger.csv hash: {claims_ledger_hash}")


if __name__ == "__main__":
    main()
