"""
Corrected Direct Greedy rerun -- FINAL NOMINAL seeds (documented software-bug fix).

Reruns ONLY Direct Greedy (canonical method "greedy") on the ORIGINAL locked
final-nominal seeds (20000..24999), using the corrected
GreedyInterceptPolicy (see uav_defend/policies/baseline/greedy_intercept_policy.py's
module docstring for the full bug description). All other methods'
original results (results/final_nominal/) are UNCHANGED and untouched --
only the new corrected Greedy rows are written to a separate location.

Output: results/corrections/direct_greedy/final_nominal/
    correction_manifest.json
    corrected_greedy_raw.csv
    corrected_greedy_summary.json
    corrected_nominal_comparisons.csv
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
from uav_defend.config import EnvConfig
from uav_defend.policies.registry import get_policy

from experiments.final_experiment_protocol import FINAL_NOMINAL_SEEDS
from experiments.final_eval_utils import run_final_episode
from experiments.final_stats import (
    wilson_interval,
    mcnemar_exact,
    paired_bootstrap_ci,
    hierarchical_paired_bootstrap_vs_scripted,
)

OUTPUT_ROOT = PROJECT_ROOT / "results" / "corrections" / "direct_greedy" / "final_nominal"
ORIGINAL_FINAL_NOMINAL_ROOT = PROJECT_ROOT / "results" / "final_nominal"
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816


def _get_git_commit() -> str:
    return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=PROJECT_ROOT).decode().strip()


def run_corrected_greedy(seeds: list[int]) -> pd.DataFrame:
    env = SoldierEnv(config=EnvConfig(use_kalman_tracking=False))
    policy = get_policy("greedy")
    dt = env.config.dt
    rows = []
    for s in seeds:
        row = run_final_episode(env, policy, seed=s, dt=dt)
        row["paper_method"] = "greedy"
        row["policy_instance"] = "greedy_corrected"
        row["estimator"] = "measurement"
        row["training_seed"] = float("nan")
        rows.append(row)
    env.close()
    return pd.DataFrame(rows)


def main() -> int:
    seeds = list(FINAL_NOMINAL_SEEDS)
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    print(f"Rerunning corrected Direct Greedy on {len(seeds)} final-nominal seeds "
          f"({seeds[0]}..{seeds[-1]})...")
    df = run_corrected_greedy(seeds)
    df.to_csv(OUTPUT_ROOT / "corrected_greedy_raw.csv", index=False)
    print(f"  {len(df)} rows written.")

    n = len(df)
    successes = int(df["success"].sum())
    ci_lo, ci_hi = wilson_interval(successes, n)
    corrected_summary = {
        "n_eval_episodes": n,
        "success_count": successes,
        "success_rate": successes / n,
        "success_ci_low": ci_lo,
        "success_ci_high": ci_hi,
        "soldier_caught_rate": float(df["soldier_caught"].mean()),
        "unsafe_intercept_rate": float(df["unsafe_intercept"].mean()),
        "timeout_rate": float(df["timeout"].mean()),
        "mean_episode_length_seconds": float(df["episode_length_seconds"].mean()),
        "mean_detection_time_seconds": float(df["detection_time_seconds"].mean()),
        "mean_intercept_time_seconds": float(df["intercept_time_seconds"].mean()),
    }
    (OUTPUT_ROOT / "corrected_greedy_summary.json").write_text(json.dumps(corrected_summary, indent=2, default=str))
    print("Corrected Greedy summary:", json.dumps(corrected_summary, indent=2, default=str))

    # ---- Load UNCHANGED existing rows for PPO/Kalman-Greedy/PN/Lead ----
    orig = pd.read_csv(ORIGINAL_FINAL_NOMINAL_ROOT / "raw_episode_results.csv")
    orig_greedy = orig[orig["paper_method"] == "greedy"].sort_values("eval_seed")
    old_successes = int(orig_greedy["success"].sum())
    old_n = len(orig_greedy)
    old_ci_lo, old_ci_hi = wilson_interval(old_successes, old_n)

    corrected_sorted = df.sort_values("eval_seed").reset_index(drop=True)
    assert (corrected_sorted["eval_seed"].to_numpy() == orig_greedy["eval_seed"].to_numpy()).all()
    corrected_vec = corrected_sorted["success"].to_numpy(dtype=np.float64)

    comparisons = []

    # B. PPO - corrected Greedy (hierarchical paired bootstrap; PPO retains 3 training seeds)
    ppo_df = orig[orig["paper_method"] == "ppo"]
    ppo_matrix = np.stack([
        ppo_df[ppo_df["training_seed"] == s].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64)
        for s in (42, 43, 44)
    ])
    observed, lo, hi = hierarchical_paired_bootstrap_vs_scripted(ppo_matrix, corrected_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    comparisons.append({
        "comparison": "ppo_vs_corrected_greedy", "comparison_type": "hierarchical_paired_vs_scripted",
        "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
    })

    # C. Kalman-Greedy - corrected Greedy (paired bootstrap + exact McNemar)
    kg_vec = orig[orig["paper_method"] == "kalman_greedy"].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64)
    observed, lo, hi = paired_bootstrap_ci(kg_vec, corrected_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    n10 = int(np.sum((kg_vec == 1) & (corrected_vec == 0)))
    n01 = int(np.sum((kg_vec == 0) & (corrected_vec == 1)))
    p_value = mcnemar_exact(n01=n01, n10=n10)
    comparisons.append({
        "comparison": "kalman_greedy_vs_corrected_greedy", "comparison_type": "fixed_paired",
        "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
        "n10_a_wins": n10, "n01_a_loses": n01, "mcnemar_p_value": p_value,
    })

    # D. PN - corrected Greedy (descriptive)
    pn_vec = orig[orig["paper_method"] == "pn"].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64)
    observed, lo, hi = paired_bootstrap_ci(pn_vec, corrected_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    comparisons.append({
        "comparison": "pn_vs_corrected_greedy_descriptive", "comparison_type": "fixed_paired_descriptive",
        "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
    })

    # E. Lead - corrected Greedy (descriptive)
    lead_vec = orig[orig["paper_method"] == "lead"].sort_values("eval_seed")["success"].to_numpy(dtype=np.float64)
    observed, lo, hi = paired_bootstrap_ci(lead_vec, corrected_vec, BOOTSTRAP_REPLICATES, BOOTSTRAP_SEED)
    comparisons.append({
        "comparison": "lead_vs_corrected_greedy_descriptive", "comparison_type": "fixed_paired_descriptive",
        "observed_diff_pp": observed * 100, "ci_low_pp": lo * 100, "ci_high_pp": hi * 100,
    })

    comparisons_df = pd.DataFrame(comparisons)
    comparisons_df.to_csv(OUTPUT_ROOT / "corrected_nominal_comparisons.csv", index=False)
    print("\nComparisons:\n", comparisons_df.to_string())

    orig_manifest = json.loads((ORIGINAL_FINAL_NOMINAL_ROOT / "experiment_manifest.json").read_text())
    manifest = {
        "original_final_evaluation_commit": orig_manifest.get("source_git_commit"),
        "greedy_correction_commit": _get_git_commit(),
        "reason": "Direct Greedy estimator-contract software bug",
        "original_invalid_greedy_behavior": (
            "escort-only after detection because policy requested e_hat while "
            "measurement sanitizer exposed enemy_measurement"
        ),
        "corrected_behavior": "pure pursuit of enemy_measurement after detection",
        "seed_range": [seeds[0], seeds[-1]],
        "number_episodes": len(seeds),
        "old_invalid_success_rate": old_successes / old_n,
        "old_invalid_success_ci": [old_ci_lo, old_ci_hi],
        "corrected_success_rate": successes / n,
        "corrected_success_ci": [ci_lo, ci_hi],
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "bootstrap_seed": BOOTSTRAP_SEED,
        "bootstrap_replicates": BOOTSTRAP_REPLICATES,
    }
    (OUTPUT_ROOT / "correction_manifest.json").write_text(json.dumps(manifest, indent=2, default=str))
    print("\nManifest written:", json.dumps(manifest, indent=2, default=str))

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
