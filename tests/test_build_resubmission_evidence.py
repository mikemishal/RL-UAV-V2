"""
Tests for the resubmission evidence builder
(`experiments/build_resubmission_evidence.py`) -- proves the Tier-A/B/C
evidence hierarchy is enforced, the builder performs no seed execution, and
the key paper-facing classifications match the frozen source artifacts.

Run directly: python tests/test_build_resubmission_evidence.py
"""

from __future__ import annotations

import ast
import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import experiments.build_resubmission_evidence as b


# --- A/B: Tier-A source roots -------------------------------------------------

def test_tier_a_nominal_source_is_rmpc_final_nominal():
    assert b.TIER_A_NOMINAL_ROOT == PROJECT_ROOT / "results" / "rmpc_final_nominal"


def test_tier_a_robustness_source_is_rmpc_targeted_robustness():
    assert b.TIER_A_ROBUSTNESS_ROOT == PROJECT_ROOT / "results" / "rmpc_targeted_robustness"


def test_tier_a_sources_exist():
    b.assert_tier_a_sources_exist()  # must not raise


# --- C: RMPC tuning success data cannot become final performance data --------

def test_tier_c_tuning_success_differs_from_tier_a_rmpc_nominal():
    rmpc_config = b.load_frozen_rmpc_config()
    pv = b.build_paper_values()
    tuning_rate = rmpc_config["validation_success_rate"]
    final_rate = pv["nominal"]["families"]["RMPC"]["success_rate"]
    assert tuning_rate != final_rate
    assert abs(tuning_rate - final_rate) > 0.01


def test_paper_values_rmpc_config_excludes_performance_fields():
    pv = b.build_paper_values()
    cfg = pv["rmpc_config"]
    assert "validation_success_rate" not in cfg
    assert "validation_success_count" not in cfg


def test_tier_c_forbidden_files_never_read_by_builder():
    import inspect
    # Only the loader functions that touch Tier-C may reference selected_rmpc_config.json;
    # none of them (nor build_paper_values, which orchestrates them) may reference any
    # tuning-stage PERFORMANCE file.
    tier_c_related_source = (
        inspect.getsource(b.load_frozen_rmpc_config) + inspect.getsource(b.build_paper_values)
    )
    for forbidden in b.TIER_C_FORBIDDEN_PERFORMANCE_FILES:
        assert forbidden not in tier_c_related_source, f"builder must never read {forbidden} as performance data"


# --- D: old nominal data cannot silently replace new nominal data -------------

def test_old_nominal_never_overwrites_new_nominal_in_paper_values():
    pv = b.build_paper_values()
    old_greedy = pv["replication"]["Greedy"]["old_success_rate"]
    new_greedy = pv["nominal"]["families"]["Greedy"]["success_rate"]
    assert old_greedy != new_greedy
    assert abs(new_greedy - 0.7912) < 1e-6  # authoritative new nominal, never the old 0.7762


def test_replication_reports_both_old_and_new_separately():
    pv = b.build_paper_values()
    for fam in ("Greedy", "PN", "Lead", "PPO", "PPO-Kalman", "LR-PPO"):
        entry = pv["replication"][fam]
        assert "old_success_rate" in entry and "new_success_rate" in entry and "delta_pp" in entry


# --- E: exactly seven paper-facing controller families ------------------------

def test_exactly_seven_families_constant():
    assert b.FAMILY_ORDER == ("Greedy", "PN", "Lead", "RMPC", "PPO", "PPO-Kalman", "LR-PPO")


def test_assert_exactly_seven_families_rejects_missing():
    try:
        b.assert_exactly_seven_families({"Greedy", "PN", "Lead", "RMPC", "PPO", "PPO-Kalman"})
        raise AssertionError("expected failure for a missing family")
    except AssertionError as e:
        assert "LR-PPO" in str(e) or "7" in str(e) or "expected exactly" in str(e)


# --- F: nominal table values reproduce authoritative sources exactly ---------

def test_nominal_table_values_match_source_family_summary():
    pv = b.build_paper_values()
    expected = {"Greedy": 79.12, "PN": 75.92, "Lead": 86.76, "RMPC": 82.04,
                "PPO": 87.42, "PPO-Kalman": 83.43, "LR-PPO": 89.43}
    for fam, exp_pct in expected.items():
        actual_pct = pv["nominal"]["families"][fam]["success_rate"] * 100.0
        assert abs(actual_pct - exp_pct) < 0.01, f"{fam}: {actual_pct} != {exp_pct}"


def test_nominal_root_specific_values_match_source():
    pv = b.build_paper_values()
    expected = {"ppo_seed45": 88.86, "ppo_seed46": 89.30, "ppo_seed47": 84.10,
                "ppo_kalman_seed45": 84.84, "ppo_kalman_seed46": 79.36, "ppo_kalman_seed47": 86.08,
                "lr_ppo_seed55": 89.58, "lr_ppo_seed56": 89.38, "lr_ppo_seed57": 89.34}
    for inst_id, exp_pct in expected.items():
        actual_pct = pv["nominal"]["instances"][inst_id]["success_rate"] * 100.0
        assert abs(actual_pct - exp_pct) < 0.01, f"{inst_id}: {actual_pct} != {exp_pct}"


# --- G: robustness table values reproduce authoritative sources exactly -----

def test_robustness_matrix_matches_source():
    pv = b.build_paper_values()
    expected = {
        "strong_evasion": {"Greedy": 78.70, "PN": 76.30, "Lead": 87.70, "RMPC": 80.90,
                            "PPO": 84.83, "PPO-Kalman": 82.67, "LR-PPO": 89.60},
        "hard_mobility": {"Greedy": 52.80, "PN": 32.10, "Lead": 49.70, "RMPC": 69.30,
                           "PPO": 40.90, "PPO-Kalman": 49.00, "LR-PPO": 58.00},
    }
    for cond, fam_rates in expected.items():
        for fam, exp_pct in fam_rates.items():
            actual_pct = pv["robustness"]["matrix"][cond][fam] * 100.0
            assert abs(actual_pct - exp_pct) < 0.1, f"{cond}/{fam}: {actual_pct} != {exp_pct}"


# --- H: difference signs and CIs reproduce source artifacts -----------------

def test_nominal_difference_signs_match_source():
    pv = b.build_paper_values()
    d = pv["nominal"]["differences"]["LR-PPO - RMPC"]
    assert d["difference_pp"] > 0  # LR-PPO beats RMPC, never the reverse sign
    d2 = pv["nominal"]["differences"]["RMPC - Lead"]
    assert d2["difference_pp"] < 0  # RMPC underperforms Lead nominally


def test_difference_orientation_never_flipped_vs_raw_csv():
    raw = b.load_nominal_paired_differences().set_index("comparison")
    pv = b.build_paper_values()
    for label in raw.index:
        assert pv["nominal"]["differences"][label]["difference_pp"] == raw.loc[label, "difference_pp"]


# --- I: claims ledger has explicit tests for critical claim classifications -

def test_claims_ledger_contains_required_claim_ids():
    pv = b.build_paper_values()
    claims = b.build_claims_ledger(pv)
    assert set(claims["claim_id"]) == {"C1", "C2", "C3", "C4", "C5", "C6", "C7", "C8"}


def test_claims_ledger_c3_uses_unresolved_wording():
    pv = b.build_paper_values()
    claims = b.build_claims_ledger(pv).set_index("claim_id")
    assert "unresolved" in claims.loc["C3", "statistical_status"] or \
           claims.loc["C3", "statistical_status"] == pv["nominal"]["differences"]["LR-PPO - PPO"]["resolution"]
    assert "significantly outperformed PPO nominally" in claims.loc["C3", "prohibited_stronger_wording"]


def test_claims_ledger_prohibited_wording_never_appears_in_allowed_wording():
    pv = b.build_paper_values()
    claims = b.build_claims_ledger(pv)
    for _, row in claims.iterrows():
        for prohibited in row["prohibited_stronger_wording"].split(" | "):
            assert prohibited not in row["allowed_wording"]


# --- J/K/L: numerical winners -------------------------------------------------

def test_high_noise_winner_is_ppo():
    pv = b.build_paper_values()
    row = pv["robustness"]["matrix"]["high_measurement_noise"]
    winner = max(row, key=row.get)
    assert winner == "PPO"


def test_hard_mobility_winner_is_rmpc():
    pv = b.build_paper_values()
    row = pv["robustness"]["matrix"]["hard_mobility"]
    winner = max(row, key=row.get)
    assert winner == "RMPC"


def test_nominal_winner_is_lr_ppo():
    pv = b.build_paper_values()
    row = pv["robustness"]["matrix"]["nominal"]
    winner = max(row, key=row.get)
    assert winner == "LR-PPO"


# --- M/N/O: specific resolution classifications ------------------------------

def test_lr_ppo_vs_ppo_nominal_is_unresolved():
    pv = b.build_paper_values()
    assert pv["nominal"]["differences"]["LR-PPO - PPO"]["resolution"] == "unresolved"


def test_lr_ppo_vs_ppo_high_noise_is_resolved_negative():
    pv = b.build_paper_values()
    d = pv["robustness"]["differences"]["high_measurement_noise"]["LR-PPO - PPO"]
    assert d["resolution"] == "resolved negative"


def test_lr_ppo_vs_rmpc_hard_mobility_is_resolved_negative():
    pv = b.build_paper_values()
    d = pv["robustness"]["differences"]["hard_mobility"]["LR-PPO - RMPC"]
    assert d["resolution"] == "resolved negative"


# --- P: no seed execution occurs from the builder -----------------------------

def test_builder_never_imports_soldier_env_or_policy_registry():
    assert "SoldierEnv" not in dir(b)
    assert "get_policy" not in dir(b)
    assert "PPO" not in dir(b)
    import inspect
    header = inspect.getsource(b).split("# ====", 1)[0]  # imports live before the first section banner
    for forbidden_import in ("from uav_defend.envs import SoldierEnv",
                              "from uav_defend.policies.registry import get_policy",
                              "from stable_baselines3 import PPO",
                              "import stable_baselines3"):
        assert forbidden_import not in header


def test_builder_never_calls_env_reset_or_step():
    import inspect
    for name in ("build_paper_values", "main"):
        fn_source = inspect.getsource(getattr(b, name))
        assert ".reset(" not in fn_source
        assert ".step(" not in fn_source


# --- Table / figure generation sanity (no source values retyped) ------------

def test_nominal_performance_table_row_count():
    pv = b.build_paper_values()
    df = b.build_nominal_performance_table(pv)
    assert len(df) == 7
    assert list(df["Method"]) == list(b.FAMILY_ORDER)


def test_robustness_success_table_row_count():
    pv = b.build_paper_values()
    df = b.build_robustness_success_table(pv)
    assert len(df) == 6  # nominal + 5 conditions


def test_robustness_condensed_table_star_marks_resolved_only():
    pv = b.build_paper_values()
    df = b.build_robustness_condensed_table(pv)
    nominal_row = df[df["Condition"] == "Nominal"].iloc[0]
    assert nominal_row["LR-PPO - Lead"].endswith("*")  # resolved positive
    assert not nominal_row["LR-PPO - PPO"].endswith("*")  # unresolved


if __name__ == "__main__":
    import inspect
    module = sys.modules[__name__]
    test_fns = [obj for name, obj in inspect.getmembers(module) if name.startswith("test_") and callable(obj)]
    passed = 0
    for fn in test_fns:
        fn()
        passed += 1
        print(f"PASS {fn.__name__}")
    print(f"\n{passed}/{len(test_fns)} tests passed")
