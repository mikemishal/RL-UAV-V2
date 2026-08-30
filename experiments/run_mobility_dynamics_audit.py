"""
Mobility-dynamics audit runner (DIAGNOSTIC ONLY -- see
experiments/mobility_dynamics_audit.py for full context/caveats).

Does NOT touch the locked mobility dataset (results/robustness/mobility/) or
evaluate any publication seed. Uses ONLY diagnostic seeds 18000-18099 (plus a
read-only reanalysis of the existing immutable 90,000-row mobility CSV).

Usage:
    python experiments/run_mobility_dynamics_audit.py
"""

from __future__ import annotations

import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.mobility_dynamics_audit import (
    AUDIT_DIAGNOSTIC_SEED_OFFSET,
    AUDIT_DIAGNOSTIC_EPISODES,
    AUDIT_DIAGNOSTIC_SEEDS,
    MATCHED_TRAJECTORY_SEEDS,
    HOSTILE_SPEED_GRID,
    DEFENDER_SPEED_GRID,
    NOMINAL_DEFENDER_SPEED,
    NOMINAL_HOSTILE_SPEED,
    run_hostile_geometry_diagnostics,
    audit_initial_vertical_clipping,
    build_matched_trajectory_examples,
    run_defender_speed_diagnostics,
)

MOBILITY_SWEEP_SEED_OFFSET = 32_000
MOBILITY_SWEEP_EPISODES = 1_000
LOCKED_MOBILITY_RAW_PATH = PROJECT_ROOT / "results" / "robustness" / "mobility" / "raw_episode_results.csv"
LOCKED_MOBILITY_ROW_COUNT = 90_000

OUTPUT_ROOT = PROJECT_ROOT / "results" / "audits" / "mobility_dynamics"


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


def verify_locked_mobility_data_untouched() -> dict:
    """READ-ONLY verification -- never writes to results/robustness/mobility/."""
    if not LOCKED_MOBILITY_RAW_PATH.exists():
        raise FileNotFoundError(f"Locked mobility raw data not found: {LOCKED_MOBILITY_RAW_PATH}")
    df = pd.read_csv(LOCKED_MOBILITY_RAW_PATH)
    n_rows = len(df)
    seed_min, seed_max = int(df["eval_seed"].min()), int(df["eval_seed"].max())
    ok = (n_rows == LOCKED_MOBILITY_ROW_COUNT) and (seed_min == MOBILITY_SWEEP_SEED_OFFSET) and (
        seed_max == MOBILITY_SWEEP_SEED_OFFSET + MOBILITY_SWEEP_EPISODES - 1
    )
    assert ok, f"locked mobility data appears to have changed: rows={n_rows}, seeds={seed_min}..{seed_max}"
    return {"n_rows": n_rows, "seed_min": seed_min, "seed_max": seed_max, "unchanged": ok}


def reanalyze_existing_mobility_data() -> pd.DataFrame:
    """
    Item 18: descriptive reanalysis of the ALREADY-LOCKED, immutable 90,000-row
    mobility CSV -- detection time, post-detection intercept time, episode
    length, path length, turn/accel/climb saturation, evasion activation, per
    (v_d, v_e, method). Read-only; the source CSV is never modified.
    """
    df = pd.read_csv(LOCKED_MOBILITY_RAW_PATH)
    cols = [
        "detection_time_seconds", "post_detection_intercept_seconds", "episode_length_seconds",
        "defender_path_length", "fraction_defender_turn_saturated", "fraction_defender_accel_saturated",
        "fraction_defender_climb_saturated", "evasion_activated_episode", "fraction_steps_evasion_active",
        "detected", "success",
    ]
    rows = []
    for (v_d, v_e, method), sub in df.groupby(["defender_speed", "hostile_speed", "policy_instance"]):
        row = {"defender_speed": v_d, "hostile_speed": v_e, "mobility_ratio": v_d / v_e, "policy_instance": method}
        for c in cols:
            row[f"mean_{c}"] = float(sub[c].mean())
        rows.append(row)
    result = pd.DataFrame(rows).sort_values(["hostile_speed", "defender_speed", "policy_instance"]).reset_index(drop=True)
    return result


def main() -> int:
    print("=" * 70)
    print("MOBILITY-DYNAMICS AUDIT (diagnostic only -- no publication seeds)")
    print("=" * 70)

    print("\nVerifying locked mobility data is untouched (read-only)...")
    locked_check = verify_locked_mobility_data_untouched()
    print(json.dumps(locked_check, indent=2))

    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print("\n[18] Reanalyzing existing locked mobility data (read-only)...")
    reanalysis = reanalyze_existing_mobility_data()
    reanalysis.to_csv(OUTPUT_ROOT / "existing_mobility_reanalysis.csv", index=False)
    print(f"  {len(reanalysis)} rows written.")

    print(f"\n[7-11,13,14] Running hostile-speed geometry diagnostics "
          f"(seeds {AUDIT_DIAGNOSTIC_SEED_OFFSET}-{AUDIT_DIAGNOSTIC_SEED_OFFSET + AUDIT_DIAGNOSTIC_EPISODES - 1}, "
          f"v_e in {HOSTILE_SPEED_GRID})...")
    hostile_geom = run_hostile_geometry_diagnostics()
    hostile_geom.to_csv(OUTPUT_ROOT / "hostile_speed_geometry_diagnostics.csv", index=False)
    print(f"  {len(hostile_geom)} rows written.")

    print("\n[13] Auditing initial-velocity vertical clipping...")
    initial_clip = audit_initial_vertical_clipping()
    print(initial_clip.to_string())

    print("\n[9] Computing hostile dynamics summary (aggregated from geometry diagnostics)...")
    dyn_cols = [
        "mean_hostile_speed", "max_hostile_speed", "mean_horizontal_hostile_speed",
        "mean_abs_vertical_speed", "fraction_vertical_rate_saturated", "fraction_turn_saturated",
        "fraction_accel_saturated", "n_heading_reversals", "enemy_path_length",
        "ground_boundary_hits", "ceiling_hits", "horizontal_wall_hits",
    ]
    hostile_dyn_summary = hostile_geom.groupby("hostile_speed")[dyn_cols].mean().reset_index()
    hostile_dyn_summary = hostile_dyn_summary.merge(
        initial_clip[["hostile_speed", "fraction_initial_vz_clipped", "mean_initial_speed_after_clipping"]],
        on="hostile_speed",
    )
    frac_caught = hostile_geom.groupby("hostile_speed")["time_soldier_caught"].apply(
        lambda s: float(s.notna().mean())
    ).reset_index(name="fraction_episodes_soldier_caught_stationary_defender")
    hostile_dyn_summary = hostile_dyn_summary.merge(frac_caught, on="hostile_speed")
    hostile_dyn_summary.to_csv(OUTPUT_ROOT / "hostile_speed_dynamics_diagnostics.csv", index=False)
    print(hostile_dyn_summary.to_string())

    print(f"\n[12] Building matched same-seed trajectory examples (seeds {MATCHED_TRAJECTORY_SEEDS})...")
    matched = build_matched_trajectory_examples()
    matched.to_csv(OUTPUT_ROOT / "matched_trajectory_examples.csv", index=False)
    print(f"  {len(matched)} rows written.")

    print(f"\n[15-17] Running defender-speed diagnostics "
          f"(seeds {AUDIT_DIAGNOSTIC_SEED_OFFSET}-{AUDIT_DIAGNOSTIC_SEED_OFFSET + AUDIT_DIAGNOSTIC_EPISODES - 1}, "
          f"v_d in {DEFENDER_SPEED_GRID}, corrected Greedy)...")
    defender_diag = run_defender_speed_diagnostics(policy_name="greedy")
    defender_diag.to_csv(OUTPUT_ROOT / "defender_speed_diagnostics.csv", index=False)
    print(f"  {len(defender_diag)} rows written.")

    defender_summary_cols = [
        "mean_defender_speed", "fraction_accel_saturated", "fraction_turn_saturated",
        "fraction_climb_saturated", "cumulative_turn_angle_deg", "path_length",
        "post_detection_intercept_seconds", "closing_efficiency", "success",
    ]
    defender_summary = defender_diag.groupby("defender_speed")[defender_summary_cols].mean().reset_index()
    print("\nDefender-speed diagnostic summary (Greedy, diagnostic seeds, nominal v_e=12):")
    print(defender_summary.to_string())

    manifest = {
        "source_git_commit": _get_git_commit(),
        "source_git_branch": _get_git_branch(),
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Diagnostic-only audit of mobility-sweep non-monotonicity/inversion -- NOT publication evidence.",
        "locked_mobility_data_check": locked_check,
        "locked_mobility_data_path": str(LOCKED_MOBILITY_RAW_PATH.relative_to(PROJECT_ROOT)).replace("\\", "/"),
        "audit_diagnostic_seed_range": f"{AUDIT_DIAGNOSTIC_SEED_OFFSET}..{AUDIT_DIAGNOSTIC_SEED_OFFSET + AUDIT_DIAGNOSTIC_EPISODES - 1}",
        "matched_trajectory_seeds": list(MATCHED_TRAJECTORY_SEEDS),
        "hostile_speed_grid": list(HOSTILE_SPEED_GRID),
        "defender_speed_grid": list(DEFENDER_SPEED_GRID),
        "nominal_defender_speed": NOMINAL_DEFENDER_SPEED,
        "nominal_hostile_speed": NOMINAL_HOSTILE_SPEED,
        "stationary_defender_caveat": (
            "The 'stationary defender' hostile-geometry diagnostic places the defender at its spawn "
            "position (co-located with the soldier) and never moves it. Because the reactive-evasion "
            "mechanism triggers on enemy-to-DEFENDER distance (enemy_evasion_radius=20m), and the "
            "defender never leaves the soldier's location, this does NOT fully decouple hostile "
            "pursuit geometry from evasion pressure -- evasion still activates whenever the enemy is "
            "within 20m of the soldier, well before threat_radius=2m. Many diagnostic episodes never "
            "reach soldier_caught within max_steps as a direct consequence. This is reported "
            "transparently as a methodological limitation of this diagnostic, not fixed/tuned away."
        ),
        "no_model_files_modified": True,
        "no_publication_seed_executed": True,
    }
    (OUTPUT_ROOT / "audit_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    audit_summary = {
        "manifest": manifest,
        "existing_mobility_reanalysis": reanalysis.to_dict(orient="records"),
        "hostile_speed_dynamics_summary": hostile_dyn_summary.to_dict(orient="records"),
        "initial_vertical_clipping": initial_clip.to_dict(orient="records"),
        "defender_speed_summary": defender_summary.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "mobility_dynamics_audit.json").write_text(json.dumps(audit_summary, indent=2, default=str))

    print(f"\nAll audit outputs written to {OUTPUT_ROOT}")
    print("=" * 70)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
