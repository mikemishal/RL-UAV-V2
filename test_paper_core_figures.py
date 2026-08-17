"""Core paper figure generator correctness tests -- locked guardrails.

Script-style test consistent with the project's other test files (no pytest).

Covers (per the paper-figures-core task spec):
    A. Input source is results/paper_evidence only (static source scan).
    B. Six nominal methods exactly.
    C. No PN-Kalman/Lead-Kalman.
    D. Corrected Greedy value ~75.76%.
    E. PPO ~85.39%.
    F. Lead ~83.10%.
    G. PPO nominal CI overlaps Lead (comparison conclusion consistent with 'unresolved').
    H. Acquisition conditions exactly: Lead, Escorted PPO, Full PPO.
    I. Full PPO success ~86.07%.
    J. Escorted PPO ~82.43%.
    K. Lead ~84.8%.
    L. Full PPO - Escorted PPO difference ~3.63pp.
    M. Its CI excludes zero.
    N. Escorted PPO - Lead CI includes zero.
    O. Full PPO detection time substantially below escorted/Lead.
    P. PDF outputs generated.
    Q. PNG outputs generated at >=300 DPI.
    R. No simulation imports.
    S. No PPO inference (static source scan).
    T. Figure data CSVs exactly match plotted values.

Run directly:
    python test_paper_core_figures.py
"""

from __future__ import annotations

import ast
import json
import re
from pathlib import Path

import numpy as np
import pandas as pd
from PIL import Image

PROJECT_ROOT = Path(__file__).parent
SCRIPT_PATH = PROJECT_ROOT / "experiments" / "plot_paper_core_figures.py"
EVIDENCE_ROOT = PROJECT_ROOT / "results" / "paper_evidence"
FIGURES_ROOT = PROJECT_ROOT / "results" / "paper_figures"

FORBIDDEN_TOKENS = ("SoldierEnv(", "PPO.load(", "env.step(", "env.reset(", "import stable_baselines3",
                    "from stable_baselines3", "policy.act(", ".predict(")


# ---------------------------------------------------------------------------
# A. Input source is results/paper_evidence only
# ---------------------------------------------------------------------------
def test_input_source_is_paper_evidence_only():
    text = SCRIPT_PATH.read_text()
    assert 'EVIDENCE_ROOT = PROJECT_ROOT / "results" / "paper_evidence"' in text
    # No references to raw experiment folders (results/robustness, results/final_nominal, etc.)
    forbidden_paths = ["results/robustness", "results/final_nominal", "results/corrections", "results/acquisition_ablation"]
    for p in forbidden_paths:
        assert p not in text, f"plotting script must not reference raw experiment folder {p}"


# ---------------------------------------------------------------------------
# B. Six nominal methods exactly
# ---------------------------------------------------------------------------
def test_six_nominal_methods_exact():
    df = pd.read_csv(FIGURES_ROOT / "fig_nominal_performance_data.csv")
    assert set(df["method_key"]) == {"greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman"}
    assert len(df) == 6


# ---------------------------------------------------------------------------
# C. No PN-Kalman/Lead-Kalman
# ---------------------------------------------------------------------------
def test_no_pn_kalman_lead_kalman():
    for f in FIGURES_ROOT.glob("*.csv"):
        text = f.read_text()
        assert "pn_kalman" not in text.lower().replace("kalman_greedy", "") or "kalman_greedy" in text
        assert "PN-Kalman" not in text and "Lead-Kalman" not in text
        assert "pn_kalman" not in text
        assert "lead_kalman" not in text


# ---------------------------------------------------------------------------
# D/E/F. Nominal values match frozen evidence
# ---------------------------------------------------------------------------
def test_nominal_values_match_frozen_evidence():
    df = pd.read_csv(FIGURES_ROOT / "fig_nominal_performance_data.csv")
    greedy = float(df[df["method_key"] == "greedy"]["success_rate_pct"].iloc[0])
    ppo = float(df[df["method_key"] == "ppo"]["success_rate_pct"].iloc[0])
    lead = float(df[df["method_key"] == "lead"]["success_rate_pct"].iloc[0])
    assert abs(greedy - 75.76) < 0.5
    assert abs(ppo - 85.39) < 0.5
    assert abs(lead - 83.10) < 0.5


# ---------------------------------------------------------------------------
# G. PPO nominal CI overlaps Lead (i.e. conclusion is 'unresolved', matching frozen evidence)
# ---------------------------------------------------------------------------
def test_ppo_lead_ci_overlap_consistent_with_unresolved():
    df = pd.read_csv(FIGURES_ROOT / "fig_nominal_performance_data.csv")
    ppo = df[df["method_key"] == "ppo"].iloc[0]
    lead = df[df["method_key"] == "lead"].iloc[0]
    # Overlap check: PPO's CI low <= Lead's CI high, and Lead's CI low <= PPO's CI high.
    overlap = (ppo["success_ci_low_pct"] <= lead["success_ci_high_pct"]) and (lead["success_ci_low_pct"] <= ppo["success_ci_high_pct"])
    assert overlap, "PPO and Lead CIs should visually overlap, consistent with the frozen 'CI includes 0' PPO-vs-Lead conclusion"
    comparisons = pd.read_csv(EVIDENCE_ROOT / "final_nominal_comparisons.csv")
    ppo_lead_row = comparisons[comparisons["comparison"] == "PPO - Lead"].iloc[0]
    assert ppo_lead_row["ci_low_pp"] <= 0 <= ppo_lead_row["ci_high_pp"]


# ---------------------------------------------------------------------------
# H. Acquisition conditions exactly
# ---------------------------------------------------------------------------
def test_acquisition_conditions_exact():
    df = pd.read_csv(FIGURES_ROOT / "fig_acquisition_ablation_data.csv", nrows=3)
    assert set(df["condition_display"]) == {"Lead", "Escorted PPO", "Full PPO"}


# ---------------------------------------------------------------------------
# I/J/K. Acquisition success values
# ---------------------------------------------------------------------------
def test_acquisition_success_values():
    df = pd.read_csv(FIGURES_ROOT / "fig_acquisition_ablation_data.csv", nrows=3)
    full_ppo = float(df[df["condition_display"] == "Full PPO"]["success_rate_pct"].iloc[0])
    escorted_ppo = float(df[df["condition_display"] == "Escorted PPO"]["success_rate_pct"].iloc[0])
    lead = float(df[df["condition_display"] == "Lead"]["success_rate_pct"].iloc[0])
    assert abs(full_ppo - 86.07) < 0.5
    assert abs(escorted_ppo - 82.43) < 0.5
    assert abs(lead - 84.8) < 0.5


# ---------------------------------------------------------------------------
# L/M. Full PPO - Escorted PPO difference and CI
# ---------------------------------------------------------------------------
def test_full_minus_escorted_difference_and_ci():
    text = (FIGURES_ROOT / "fig_acquisition_ablation_data.csv").read_text()
    lines = [l for l in text.splitlines() if l.startswith("Full PPO - Escorted PPO")]
    assert len(lines) == 1
    parts = lines[0].split(",")
    diff_pp, ci_low, ci_high = float(parts[1]), float(parts[2]), float(parts[3])
    assert abs(diff_pp - 3.63) < 0.1
    assert ci_low > 0, "Full PPO - Escorted PPO CI must exclude zero (significant)"
    assert ci_high > ci_low


# ---------------------------------------------------------------------------
# N. Escorted PPO - Lead CI includes zero
# ---------------------------------------------------------------------------
def test_escorted_minus_lead_ci_includes_zero():
    text = (FIGURES_ROOT / "fig_acquisition_ablation_data.csv").read_text()
    lines = [l for l in text.splitlines() if l.startswith("Escorted PPO - Lead")]
    assert len(lines) == 1
    parts = lines[0].split(",")
    ci_low, ci_high = float(parts[2]), float(parts[3])
    assert ci_low <= 0 <= ci_high


# ---------------------------------------------------------------------------
# O. Full PPO detection time substantially below escorted/Lead
# ---------------------------------------------------------------------------
def test_full_ppo_detection_time_lower():
    df = pd.read_csv(FIGURES_ROOT / "fig_acquisition_ablation_data.csv", nrows=3)
    full_ppo_t = float(df[df["condition_display"] == "Full PPO"]["mean_detection_time_seconds"].iloc[0])
    lead_t = float(df[df["condition_display"] == "Lead"]["mean_detection_time_seconds"].iloc[0])
    escorted_t = float(df[df["condition_display"] == "Escorted PPO"]["mean_detection_time_seconds"].iloc[0])
    assert full_ppo_t < lead_t - 10, "Full PPO should detect substantially earlier than Lead"
    assert full_ppo_t < escorted_t - 10
    assert abs(lead_t - escorted_t) < 0.5, "Lead and Escorted PPO should share ~equal detection timing"


# ---------------------------------------------------------------------------
# P. PDF outputs generated
# ---------------------------------------------------------------------------
def test_pdf_outputs_generated():
    for stem in ("fig_nominal_performance", "fig_acquisition_ablation"):
        path = FIGURES_ROOT / f"{stem}.pdf"
        assert path.exists(), f"{path} missing"
        assert path.stat().st_size > 1000, f"{path} suspiciously small"


# ---------------------------------------------------------------------------
# Q. PNG outputs generated at >=300 DPI
# ---------------------------------------------------------------------------
def test_png_outputs_at_300dpi():
    manifest = json.loads((FIGURES_ROOT / "core_figures_manifest.json").read_text())
    for name, info in manifest["generated_figures"].items():
        png_path = PROJECT_ROOT / info["png"]
        assert png_path.exists()
        img = Image.open(png_path)
        dpi = img.info.get("dpi", (0, 0))
        assert dpi[0] >= 299.9 and dpi[1] >= 299.9, f"{name} PNG DPI {dpi} below 300"
        expected_w_px = round(info["width_in"] * info["dpi"])
        expected_h_px = round(info["height_in"] * info["dpi"])
        assert abs(img.size[0] - expected_w_px) <= 2
        assert abs(img.size[1] - expected_h_px) <= 2


# ---------------------------------------------------------------------------
# R/S. No simulation imports / no PPO inference (static source scan)
# ---------------------------------------------------------------------------
def test_no_forbidden_tokens_in_source():
    text = SCRIPT_PATH.read_text()
    for token in FORBIDDEN_TOKENS:
        assert token not in text, f"forbidden token '{token}' found in plot_paper_core_figures.py"


def test_no_env_or_sb3_imports():
    tree = ast.parse(SCRIPT_PATH.read_text())
    forbidden_modules = {"uav_defend.envs", "stable_baselines3", "gymnasium"}
    imported = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for alias in node.names:
                imported.add(alias.name)
        elif isinstance(node, ast.ImportFrom) and node.module:
            imported.add(node.module)
    for forbidden in forbidden_modules:
        assert not any(m == forbidden or m.startswith(forbidden + ".") for m in imported), \
            f"plot_paper_core_figures.py must not import {forbidden}"


# ---------------------------------------------------------------------------
# T. Figure data CSVs exactly match plotted values (cross-check against evidence)
# ---------------------------------------------------------------------------
def test_figure_data_matches_evidence_source():
    fig_data = pd.read_csv(FIGURES_ROOT / "fig_nominal_performance_data.csv")
    evidence = pd.read_csv(EVIDENCE_ROOT / "final_nominal_paper_table.csv")
    for _, row in fig_data.iterrows():
        ev_row = evidence[evidence["method_key"] == row["method_key"]].iloc[0]
        assert abs(row["success_rate_pct"] / 100 - ev_row["success_rate"]) < 1e-9
        assert abs(row["success_ci_low_pct"] / 100 - ev_row["success_ci_low"]) < 1e-9
        assert abs(row["success_ci_high_pct"] / 100 - ev_row["success_ci_high"]) < 1e-9

    acq_evidence = pd.read_csv(EVIDENCE_ROOT / "acquisition_paper_summary.csv")
    acq_conditions = acq_evidence[acq_evidence["row_type"] == "condition_success"]
    fig_acq = pd.read_csv(FIGURES_ROOT / "fig_acquisition_ablation_data.csv", nrows=3)
    name_to_key = {"Lead": "lead", "Escorted PPO": "escorted_ppo", "Full PPO": "full_ppo"}
    for _, row in fig_acq.iterrows():
        key = name_to_key[row["condition_display"]]
        ev_row = acq_conditions[acq_conditions["condition"] == key].iloc[0]
        assert abs(row["success_rate_pct"] / 100 - ev_row["success_rate"]) < 1e-9
        assert abs(row["mean_detection_time_seconds"] - ev_row["mean_detection_time_seconds"]) < 1e-6


# ---------------------------------------------------------------------------
# Runner
# ---------------------------------------------------------------------------
if __name__ == "__main__":
    import sys
    import traceback

    tests = [(name, obj) for name, obj in list(globals().items()) if name.startswith("test_") and callable(obj)]
    tests.sort(key=lambda kv: kv[0])
    passed = failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS: {name}")
            passed += 1
        except Exception as e:
            print(f"FAIL: {name}: {e}")
            traceback.print_exc()
            failed += 1
    print(f"\n{passed} passed, {failed} failed out of {len(tests)}")
    sys.exit(1 if failed else 0)
