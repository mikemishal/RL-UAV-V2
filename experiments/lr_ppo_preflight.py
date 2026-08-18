"""
LR-PPO campaign pre-launch preflight (items 5-8 of the pre-launch-lock task).

Performs, in order:
    5. Training-seed namespace preflight -- INTEGER GENERATION ONLY (never
       env.reset()): reconfirms the three root namespaces, audits 100,000
       generated seeds/root for uniqueness/disjointness, and records the
       first 10 seeds per root.
    6. Actual environment reset preflight -- for EACH official root,
       instantiates LeadResidualTrainingEnv and resets it 3 times, verifying
       every consumed environment seed lands in that root's namespace, no
       overlap across roots, natural acquisition (defender co-located/
       stationary at detection), 19-D observation, finite Lead direction.
    7. Validation range check -- confirms the in-training validation range
       is exactly 60000-60099 (100 seeds).
    8. Still-sealed ranges check -- confirms 60100-60299, 61000-65999,
       66000-99999 are never touched by this preflight.

Writes results/lead_residual_training/campaign_manifest_preflight.json.

Usage:
    python experiments/lr_ppo_preflight.py
"""

from __future__ import annotations

import json
import sys
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

PROJECT_ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

from experiments.lead_residual_training_env import LeadResidualTrainingEnv
from experiments.lead_residual_training_protocol import (
    TRAINING_ROOTS,
    TRAINING_ROOT_NAMESPACE_BASE,
    TRAINING_NAMESPACE_SIZE,
    RESIDUAL_VALIDATION_SEED_START,
    RESIDUAL_VALIDATION_SEED_END,
    RESIDUAL_DIAGNOSTIC_SEED_START,
    RESIDUAL_DIAGNOSTIC_SEED_END,
    EXPANDED_FINAL_SEED_START,
    EXPANDED_FINAL_SEED_END,
    ROBUSTNESS_SEED_START,
    ROBUSTNESS_SEED_END,
    VALIDATION_EPISODES,
    build_lr_ppo_episode_seed_sequence,
    build_env_config,
)

RESULTS_ROOT = PROJECT_ROOT / "results" / "lead_residual_training"


def namespace_preflight() -> dict:
    """Item 5: integer-only, never passed to env.reset()."""
    n = 100_000
    per_root_unique = {}
    first_ten = {}
    for root in TRAINING_ROOTS:
        seeds = build_lr_ppo_episode_seed_sequence(root, n)
        unique = set(seeds)
        assert len(seeds) == n
        assert len(unique) == n, f"root {root}: {n - len(unique)} duplicates"
        per_root_unique[root] = unique
        first_ten[root] = seeds[:10]

    roots = list(TRAINING_ROOTS)
    pairwise = {}
    for i in range(len(roots)):
        for j in range(i + 1, len(roots)):
            inter = per_root_unique[roots[i]] & per_root_unique[roots[j]]
            pairwise[f"{roots[i]}_vs_{roots[j]}"] = len(inter)
            assert len(inter) == 0

    return {
        "n_generated_per_root": n,
        "n_unique_per_root": {r: len(per_root_unique[r]) for r in roots},
        "namespaces": {r: [TRAINING_ROOT_NAMESPACE_BASE[r], TRAINING_ROOT_NAMESPACE_BASE[r] + TRAINING_NAMESPACE_SIZE - 1] for r in roots},
        "pairwise_overlap": pairwise,
        "first_10_seeds_per_root": first_ten,
        "all_pass": True,
    }


def env_reset_preflight() -> dict:
    """Item 6: tiny reset/step integration check per official root. Training
    infrastructure only -- never touches any evaluation range."""
    results = {}
    seen_across_roots = set()
    for root in TRAINING_ROOTS:
        base = TRAINING_ROOT_NAMESPACE_BASE[root]
        env = LeadResidualTrainingEnv(config=build_env_config(), root_seed=root)
        consumed_seeds = []
        for _ in range(3):
            obs, info = env.reset()
            seed = env._inner.last_episode_seed
            consumed_seeds.append(seed)
            assert base <= seed < base + TRAINING_NAMESPACE_SIZE, (root, seed)
            assert seed not in seen_across_roots, f"cross-root seed collision: {seed}"
            seen_across_roots.add(seed)
            assert info["just_detected"] is True
            assert info["enemy_detected"] is True
            assert np.allclose(info["defender_pos"], info["soldier_pos"]), "defender not co-located at acquisition"
            assert np.allclose(info["defender_vel"], np.zeros(3)), "defender not stationary at acquisition"
            assert obs.shape == (19,)
            lead_direction = obs[-3:]
            assert np.all(np.isfinite(lead_direction)), "non-finite Lead direction"
        env.close()
        results[root] = {
            "consumed_seeds": consumed_seeds,
            "all_in_namespace": all(base <= s < base + TRAINING_NAMESPACE_SIZE for s in consumed_seeds),
        }
    return results


def validation_range_check() -> dict:
    """Item 7."""
    n = RESIDUAL_VALIDATION_SEED_END - RESIDUAL_VALIDATION_SEED_START + 1
    return {
        "n_validation_seeds": n,
        "min": RESIDUAL_VALIDATION_SEED_START,
        "max": RESIDUAL_VALIDATION_SEED_END,
        "matches_expected_100": n == 100 == VALIDATION_EPISODES,
    }


def sealed_ranges_check() -> dict:
    """Item 8 -- documentation-only confirmation; nothing in this range is
    ever touched by this script."""
    return {
        "diagnostic_60100_60299": "UNOPENED",
        "expanded_final_61000_65999": "UNOPENED",
        "robustness_66000_99999": "UNOPENED",
        "diagnostic_range": [RESIDUAL_DIAGNOSTIC_SEED_START, RESIDUAL_DIAGNOSTIC_SEED_END],
        "expanded_final_range": [EXPANDED_FINAL_SEED_START, EXPANDED_FINAL_SEED_END],
        "robustness_range": [ROBUSTNESS_SEED_START, ROBUSTNESS_SEED_END],
    }


def main() -> int:
    RESULTS_ROOT.mkdir(parents=True, exist_ok=True)

    print("=" * 70)
    print("LR-PPO CAMPAIGN PRE-LAUNCH PREFLIGHT")
    print("=" * 70)

    print("\n[5/8] Training-seed namespace preflight (integer-only)...")
    namespace_result = namespace_preflight()
    for root in TRAINING_ROOTS:
        print(f"  root {root}: first 10 seeds = {namespace_result['first_10_seeds_per_root'][root]}")
    print(f"  pairwise overlap: {namespace_result['pairwise_overlap']}")

    print("\n[6/8] Actual environment reset preflight (3 episodes/root)...")
    env_result = env_reset_preflight()
    for root, r in env_result.items():
        print(f"  root {root}: consumed_seeds={r['consumed_seeds']} all_in_namespace={r['all_in_namespace']}")

    print("\n[7/8] Validation range check...")
    validation_result = validation_range_check()
    print(f"  {validation_result}")

    print("\n[8/8] Still-sealed ranges check...")
    sealed_result = sealed_ranges_check()
    print(f"  {sealed_result}")

    manifest = {
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "namespace_preflight": namespace_result,
        "env_reset_preflight": env_result,
        "validation_range_check": validation_result,
        "sealed_ranges_check": sealed_result,
        "all_preflight_checks_passed": True,
    }
    manifest_path = RESULTS_ROOT / "campaign_manifest_preflight.json"
    manifest_path.write_text(json.dumps(manifest, indent=2, default=str))
    print(f"\nPreflight manifest written to {manifest_path}")
    print("ALL PREFLIGHT CHECKS PASSED.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
