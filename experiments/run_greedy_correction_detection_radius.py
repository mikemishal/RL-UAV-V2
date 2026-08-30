"""
Corrected Direct Greedy rerun -- DETECTION-RADIUS sweep (documented software-bug fix).

Reruns ONLY Direct Greedy (canonical method "greedy") on the ORIGINAL locked
robustness seeds (30000..30999) for all 6 detection-radius values, using the
corrected GreedyInterceptPolicy. All other methods' original results
(results/robustness/detection_radius/) are UNCHANGED and untouched.

Output: results/corrections/direct_greedy/detection_radius/
    correction_manifest.json
    corrected_greedy_raw.csv
    corrected_greedy_radius_summary.csv
    corrected_greedy_pairwise_comparisons.csv
    corrected_detection_summary.json
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

from uav_defend.envs import SoldierEnv
from uav_defend.policies.registry import get_policy

from experiments.robustness_detection_protocol import (
    DETECTION_RADIUS_VALUES,
    DETECTION_SWEEP_SEEDS,
    build_sweep_env_config,
)
from experiments.robustness_eval_utils import run_robustness_episode
from experiments.final_stats import wilson_interval, mcnemar_exact, paired_bootstrap_ci

OUTPUT_ROOT = PROJECT_ROOT / "results" / "corrections" / "direct_greedy" / "detection_radius"
ORIGINAL_ROOT = PROJECT_ROOT / "results" / "robustness" / "detection_radius"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def _get_git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def run_corrected_greedy(seeds: list[int], radii: list[float]) -> pd.DataFrame:
    rows = []
    for radius in radii:
        env = SoldierEnv(config=build_sweep_env_config("measurement", radius))
        policy = get_policy("greedy")
        dt = env.config.dt
        print(f"  radius={radius}...")
        for s in seeds:
            row = run_robustness_episode(env, policy, seed=s, dt=dt)
            row["sweep_variable"] = "detection_radius"
            row["detection_radius"] = radius
            row["paper_method"] = "greedy"
            row["policy_instance"] = "greedy_corrected"
            row["estimator"] = "measurement"
            row["training_seed"] = float("nan")
            rows.append(row)
        env.close()
    return pd.DataFrame(rows)


def main() -> int:
    seeds = list(DETECTION_SWEEP_SEEDS)
    radii = list(DETECTION_RADIUS_VALUES)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Rerunning corrected Direct Greedy on {len(seeds)} seeds x {len(radii)} radii...")
    df = run_corrected_greedy(seeds, radii)
    df.to_csv(OUTPUT_ROOT / "corrected_greedy_raw.csv", index=False)
    print(f"  {len(df)} rows written (expected {len(seeds) * len(radii)}).")
    assert len(df) == len(seeds) * len(radii)

    orig = pd.read_csv(ORIGINAL_ROOT / "raw_episode_results.csv")

    radius_rows = []
    comparison_rows = []
    for radius in radii:
        sub = df[df["detection_radius"] == radius].sort_values("eval_seed").reset_index(drop=True)
        n = len(sub)
        successes = int(sub["success"].sum())
        ci_lo, ci_hi = wilson_interval(successes, n)
        radius_rows.append({
            "detection_radius": radius, "method": "greedy_corrected", "n_eval_episodes": n,
            "success_rate": successes / n, "success_ci_low": ci_lo, "success_ci_high": ci_hi,
            "soldier_caught_rate": float(sub["soldier_caught"].mean()),
            "unsafe_intercept_rate": float(sub["unsafe_intercept"].mean()),
            "timeout_rate": float(sub["timeout"].mean()),
            "detection_rate": float(sub["detected"].mean()),
            "mean_episode_length_seconds": float(sub["episode_length_seconds"].mean()),
            "mean_detection_time_seconds_conditional": float(sub.loc[sub["detected"] == 1, "detection_time_seconds"].mean()),
        })

        orig_greedy = orig[(orig["detection_radius"] == radius) & (orig["policy_instance"] == "greedy")].sort_values("eval_seed").reset_index(drop=True)
        assert (orig_greedy["eval_seed"].to_numpy() == sub["eval_seed"].to_numpy()).all()
        corrected_vec = sub["success"].to_numpy(dtype=np.float64)

        # Kalman-Greedy - corrected Greedy (paired bootstrap + exact McNemar), using UNCHANGED kalman_greedy rows.
        kg_vec = orig[(orig["detection_radius"] == radius) & (orig["policy_instance"] == "kalman_greedy")].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64)
        observed, lo, hi = paired_bootstrap_ci(kg_vec, corrected_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        n10 = int(np.sum((kg_vec == 1) & (corrected_vec == 0)))
        n01 = int(np.sum((kg_vec == 0) & (corrected_vec == 1)))
        p_value = mcnemar_exact(n01=n01, n10=n10)
        comparison_rows.append({
            "detection_radius": radius, "comparison": "kalman_greedy_vs_corrected_greedy",
            "comparison_type": "fixed_paired", "observed_diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
            "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
        })

        # Descriptive: PN - corrected Greedy, Lead - corrected Greedy, PPO - corrected Greedy (equal-weight PPO).
        pn_vec = orig[(orig["detection_radius"] == radius) & (orig["policy_instance"] == "pn")].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64)
        observed, lo, hi = paired_bootstrap_ci(pn_vec, corrected_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        comparison_rows.append({
            "detection_radius": radius, "comparison": "pn_vs_corrected_greedy_descriptive",
            "comparison_type": "fixed_paired_descriptive", "observed_diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
        })

        lead_vec = orig[(orig["detection_radius"] == radius) & (orig["policy_instance"] == "lead")].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64)
        observed, lo, hi = paired_bootstrap_ci(lead_vec, corrected_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
        comparison_rows.append({
            "detection_radius": radius, "comparison": "lead_vs_corrected_greedy_descriptive",
            "comparison_type": "fixed_paired_descriptive", "observed_diff_pp": observed * 100,
            "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
        })

        ppo_df = orig[(orig["detection_radius"] == radius) & (orig["paper_method"] == "ppo")]
        ppo_success_by_seed = [
            float(ppo_df[ppo_df["training_seed"] == s]["success"].mean()) for s in (42, 43, 44)
        ]
        ppo_equal_weight = float(np.mean(ppo_success_by_seed))
        comparison_rows.append({
            "detection_radius": radius, "comparison": "ppo_vs_corrected_greedy_descriptive",
            "comparison_type": "descriptive_equal_weight_ppo",
            "observed_diff_pp": (ppo_equal_weight - corrected_vec.mean()) * 100,
            "ci_low_pp": float("nan"), "ci_high_pp": float("nan"),
        })

    radius_summary_df = pd.DataFrame(radius_rows)
    radius_summary_df.to_csv(OUTPUT_ROOT / "corrected_greedy_radius_summary.csv", index=False)
    print("\nCorrected Greedy radius summary:\n", radius_summary_df.to_string())

    comparisons_df = pd.DataFrame(comparison_rows)
    comparisons_df.to_csv(OUTPUT_ROOT / "corrected_greedy_pairwise_comparisons.csv", index=False)
    print("\nComparisons:\n", comparisons_df.to_string())

    orig_manifest = json.loads((ORIGINAL_ROOT / "experiment_manifest.json").read_text())
    manifest = {
        "original_robustness_evaluation_commit": orig_manifest.get("source_git_commit"),
        "greedy_correction_commit": _get_git_commit(),
        "reason": "Direct Greedy estimator-contract software bug",
        "original_invalid_greedy_behavior": (
            "escort-only after detection because policy requested e_hat while "
            "measurement sanitizer exposed enemy_measurement -- constant 64.3% success "
            "at every radius despite detection probability varying 0.66-0.967"
        ),
        "corrected_behavior": "pure pursuit of enemy_measurement after detection",
        "seed_range": [seeds[0], seeds[-1]],
        "number_episodes_per_radius": len(seeds),
        "detection_radius_values": radii,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (OUTPUT_ROOT / "correction_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))

    final_summary = {
        "label": "CORRECTED DIRECT GREEDY -- DETECTION-RADIUS SWEEP",
        "radius_summary": radius_summary_df.to_dict(orient="records"),
        "pairwise_comparisons": comparisons_df.to_dict(orient="records"),
    }
    (OUTPUT_ROOT / "corrected_detection_summary.json").write_text(json.dumps(final_summary, indent=2, default=str))
    print("\nManifest written:", json.dumps(manifest, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
