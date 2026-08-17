"""Paper-evidence-freeze correctness tests -- locked guardrails.

Script-style test consistent with the project's other test files (no pytest).

Covers (per the paper-evidence-freeze task spec):
    A. Six methods exactly.
    B. No PN-Kalman/Lead-Kalman in any paper-facing output.
    C. Corrected Direct Greedy nominal used.
    D. Corrected Direct Greedy detection-radius used.
    E. Invalid Greedy values rejected (not present in canonical outputs).
    F. Old nominal/detection estimator RMSE rejected.
    G. Measurement-noise corrected estimator source used.
    H. Mobility-grid supersedes old 1-D for primary mobility.
    I. Nominal seeds correct (20000-24999).
    J. All robustness seed ranges correct.
    K. No duplicate evidence IDs.
    L. No simulation imports/execution (static source scan).
    M. No environment instantiation.
    N. No model inference.
    O. All paper percentages in valid range [0,1].
    P. All CI low <= estimate <= CI high (where applicable).
    Q. All source files referenced in the registry exist.
    R. No canonical evidence marked invalid.

Run directly:
    python test_paper_evidence_freeze.py
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent
BUILDER_PATH = PROJECT_ROOT / "experiments" / "build_paper_evidence.py"
OUTPUT_ROOT = PROJECT_ROOT / "results" / "paper_evidence"

SIX_METHODS = ("greedy", "kalman_greedy", "pn", "lead", "ppo", "ppo_kalman")
FORBIDDEN_TOKENS = ("SoldierEnv(", "PPO.load(", "env.step(", "env.reset(", "import stable_baselines3",
                    "from stable_baselines3", "policy.act(")


# ---------------------------------------------------------------------------
# A. Six methods exactly
# ---------------------------------------------------------------------------
def test_six_methods_exact():
    df = pd.read_csv(OUTPUT_ROOT / "final_nominal_paper_table.csv")
    assert set(df["method_key"]) == set(SIX_METHODS)
    assert len(df) == 6


# ---------------------------------------------------------------------------
# B. No PN-Kalman/Lead-Kalman leakage
# ---------------------------------------------------------------------------
def test_no_pn_kalman_lead_kalman_leakage():
    csv_files = [f for f in OUTPUT_ROOT.glob("*.csv")]
    assert len(csv_files) > 0
    for f in csv_files:
        text = f.read_text()
        # Allow "pn_kalman"/"lead_kalman" to appear ONLY as excluded-method or invalid-reference notes
        # in the registry; forbid them as a 'method' or 'method_key' VALUE in any paper-facing table.
        df = pd.read_csv(f)
        for col in ("method", "method_key", "paper_method", "policy_instance"):
            if col in df.columns:
                values = set(str(v) for v in df[col].dropna().unique())
                assert "pn_kalman" not in values, f"{f.name} column {col} contains pn_kalman"
                assert "lead_kalman" not in values, f"{f.name} column {col} contains lead_kalman"
                assert "PN-Kalman" not in values and "Lead-Kalman" not in values


# ---------------------------------------------------------------------------
# C. Corrected Direct Greedy nominal used
# ---------------------------------------------------------------------------
def test_corrected_greedy_nominal_used():
    df = pd.read_csv(OUTPUT_ROOT / "final_nominal_paper_table.csv")
    greedy_success = float(df[df["method_key"] == "greedy"]["success_rate"].iloc[0])
    assert abs(greedy_success - 0.7576) < 0.005, f"expected corrected Greedy ~0.7576, got {greedy_success}"
    assert abs(greedy_success - 0.6794) > 0.01, "final nominal table appears to use the INVALID uncorrected Greedy value"


# ---------------------------------------------------------------------------
# D. Corrected Direct Greedy detection-radius used
# ---------------------------------------------------------------------------
def test_corrected_greedy_detection_radius_used():
    df = pd.read_csv(OUTPUT_ROOT / "detection_radius_paper_summary.csv")
    greedy = df[df["method_key"] == "greedy"].drop_duplicates(subset=["detection_radius"])
    rates = greedy["success_rate"].to_numpy(dtype=float)
    # The INVALID uncorrected Greedy is a CONSTANT 0.643 at every radius (an artifact of the escort-mode bug).
    assert not np.allclose(rates, 0.643, atol=1e-6), "detection-radius Greedy values look like the INVALID constant-0.643 artifact"
    assert rates.std() > 0.001, "corrected Greedy success should vary somewhat across detection radii"


# ---------------------------------------------------------------------------
# E. Invalid Greedy values rejected
# ---------------------------------------------------------------------------
def test_invalid_greedy_values_rejected():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    invalid_greedy_rows = registry[(registry["status"] == "invalid") & (registry["method"] == "greedy")]
    assert len(invalid_greedy_rows) >= 1, "registry must explicitly record the invalid original Greedy nominal/detection-radius rows"
    for _, r in invalid_greedy_rows.iterrows():
        assert r["paper_use"] == "none"


# ---------------------------------------------------------------------------
# F. Old nominal/detection estimator RMSE rejected
# ---------------------------------------------------------------------------
def test_old_estimator_rmse_rejected():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    invalid_estimator_rows = registry[(registry["status"] == "invalid") & (registry["result_type"] == "invalid_reference") & (registry["method"] == "estimator_rmse")]
    assert len(invalid_estimator_rows) >= 2, "expected invalid estimator-RMSE rows for both final_nominal/detection_radius-adjacent and measurement_noise old aggregation"
    experiments_flagged = set(invalid_estimator_rows["experiment"])
    assert "detection_radius" in experiments_flagged
    assert "measurement_noise" in experiments_flagged


# ---------------------------------------------------------------------------
# G. Measurement-noise corrected estimator source used
# ---------------------------------------------------------------------------
def test_measurement_noise_corrected_estimator_used():
    df = pd.read_csv(OUTPUT_ROOT / "measurement_noise_paper_summary.csv")
    assert "rmse_sample_weighted" in df.columns
    assert "theoretical_raw_rmse" in df.columns
    sub = df[df["method_key"] == "greedy"].drop_duplicates(subset=["measurement_var"])
    for _, r in sub.iterrows():
        if not pd.isna(r["theoretical_raw_rmse"]):
            assert abs(r["rmse_sample_weighted"] - r["theoretical_raw_rmse"]) < 0.05, \
                "corrected raw-measurement RMSE should closely match sqrt(3*measurement_var)"


# ---------------------------------------------------------------------------
# H. Mobility-grid supersedes old 1-D for primary mobility
# ---------------------------------------------------------------------------
def test_mobility_grid_supersedes_1d():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    superseded = registry[(registry["experiment"] == "mobility_1d") & (registry["status"] == "superseded")]
    assert len(superseded) >= 1, "registry must mark the 1-D mobility sweep as superseded for primary mobility claims"
    canonical_mobility_grid = registry[(registry["experiment"] == "mobility_grid") & (registry["status"] == "canonical")]
    assert len(canonical_mobility_grid) >= 1
    assert Path(PROJECT_ROOT / "results" / "paper_evidence" / "mobility_grid_paper_summary.csv").exists()
    assert Path(PROJECT_ROOT / "results" / "paper_evidence" / "mobility_equal_ratio_summary.csv").exists()


# ---------------------------------------------------------------------------
# I. Nominal seeds correct
# ---------------------------------------------------------------------------
def test_nominal_seeds_correct():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    nominal_rows = registry[registry["experiment"] == "final_nominal"]
    assert (nominal_rows["seed_range"] == "20000-24999").all()


# ---------------------------------------------------------------------------
# J. All robustness seed ranges correct
# ---------------------------------------------------------------------------
def test_robustness_seed_ranges_correct():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    expected = {
        "acquisition_ablation": "25000-25999",
        "detection_radius": "30000-30999",
        "measurement_noise": "31000-31999",
        "mobility_1d": "32000-32999",
        "hostile_evasion": "33000-33999",
        "maneuverability": "34000-34999",
        "mobility_grid": "35000-35499",
    }
    for exp, seed_range in expected.items():
        rows = registry[registry["experiment"] == exp]
        assert len(rows) > 0, f"no registry rows for {exp}"
        assert (rows["seed_range"] == seed_range).all(), f"seed range mismatch for {exp}"


# ---------------------------------------------------------------------------
# K. No duplicate evidence IDs
# ---------------------------------------------------------------------------
def test_no_duplicate_evidence_ids():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    assert registry["evidence_id"].duplicated().sum() == 0


# ---------------------------------------------------------------------------
# L/M/N. No simulation imports/execution; no env instantiation; no model inference (static source scan)
# ---------------------------------------------------------------------------
def test_builder_source_has_no_forbidden_tokens():
    text = BUILDER_PATH.read_text()
    for token in FORBIDDEN_TOKENS:
        assert token not in text, f"forbidden token '{token}' found in build_paper_evidence.py"


def test_builder_does_not_import_env_or_sb3():
    import ast
    tree = ast.parse(BUILDER_PATH.read_text())
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
            f"build_paper_evidence.py must not import {forbidden}"


# ---------------------------------------------------------------------------
# O. All paper percentages in valid range [0,1]
# ---------------------------------------------------------------------------
def test_all_success_rates_in_valid_range():
    for fname in ("final_nominal_paper_table.csv", "detection_radius_paper_summary.csv",
                  "measurement_noise_paper_summary.csv", "mobility_grid_paper_summary.csv",
                  "hostile_evasion_paper_summary.csv", "maneuverability_paper_summary.csv"):
        df = pd.read_csv(OUTPUT_ROOT / fname)
        for col in df.columns:
            if col == "success_rate" or col.endswith("_rate"):
                values = df[col].dropna()
                assert values.between(0.0, 1.0).all(), f"{fname}:{col} has values outside [0,1]"


# ---------------------------------------------------------------------------
# P. All CI low <= estimate <= CI high (where applicable)
# ---------------------------------------------------------------------------
def test_ci_bounds_consistent():
    df = pd.read_csv(OUTPUT_ROOT / "final_nominal_paper_table.csv")
    for _, r in df.iterrows():
        assert r["success_ci_low"] <= r["success_rate"] + 1e-9 <= r["success_ci_high"] + 1e-9, \
            f"CI does not bracket estimate for {r['method']}"

    comparisons = pd.read_csv(OUTPUT_ROOT / "final_nominal_comparisons.csv")
    for _, r in comparisons.iterrows():
        assert r["ci_low_pp"] <= r["observed_diff_pp"] + 1e-6 <= r["ci_high_pp"] + 1e-6, \
            f"CI does not bracket observed difference for {r['comparison']}"


# ---------------------------------------------------------------------------
# Q. All source files referenced in the registry exist
# ---------------------------------------------------------------------------
def test_all_registry_source_files_exist():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    missing = []
    for _, r in registry.iterrows():
        source_dir = r["source_directory"]
        source_file = r["source_filename"]
        if source_dir == "results/paper_evidence":
            continue  # self-referential cross-experiment summary, checked separately
        for fname in str(source_file).split("/"):
            path = PROJECT_ROOT / source_dir / fname
            if not path.exists():
                missing.append(str(path))
    assert not missing, f"missing registry source files: {missing}"


# ---------------------------------------------------------------------------
# R. No canonical evidence marked invalid
# ---------------------------------------------------------------------------
def test_no_canonical_evidence_marked_invalid():
    registry = pd.read_csv(OUTPUT_ROOT / "paper_evidence_registry.csv")
    # A given evidence_id must have exactly one status; canonical and invalid are mutually exclusive by construction.
    assert set(registry["status"].unique()) <= {"canonical", "supplemental", "diagnostic", "invalid", "superseded"}
    canonical_ids = set(registry[registry["status"] == "canonical"]["evidence_id"])
    invalid_ids = set(registry[registry["status"] == "invalid"]["evidence_id"])
    assert canonical_ids.isdisjoint(invalid_ids)


# ---------------------------------------------------------------------------
# Bonus: manifest no-execution guarantee is recorded and zeroed
# ---------------------------------------------------------------------------
def test_manifest_no_execution_guarantee():
    manifest = json.loads((OUTPUT_ROOT / "paper_evidence_manifest.json").read_text())
    guarantee = manifest["no_execution_guarantee"]
    assert guarantee["simulations_run"] == 0
    assert guarantee["episodes_run"] == 0
    assert guarantee["new_evaluation_seeds_used"] == 0
    assert guarantee["models_retrained"] == 0


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
