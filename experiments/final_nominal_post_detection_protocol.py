"""
Locked FINAL NOMINAL post-detection paper evaluation protocol -- SINGLE SOURCE
OF TRUTH for the ONE-TIME, terminal 5000-seed evaluation on the previously
reserved final-test seed range (41000..45999).

This produces the canonical nominal numbers for the revised conference paper
submission. Uses the revised five paper-facing methods only (Greedy, PN,
Lead, PPO, PPO-Kalman); Kalman-Greedy/PN-Kalman/Lead-Kalman/Escorted-PPO/
legacy-42-43-44/Random are explicitly excluded (see EXCLUDED_METHODS below).

Frozen model identities (hashes, best steps, paths) are NOT duplicated here --
they are imported from experiments/post_detection_diagnostic_protocol.py,
which is the single source of truth for the six frozen checkpoints (verified
identical against the 200-seed diagnostic and the values specified in this
final-evaluation task).

training_source_commit = 4b0a903bf801c7b439be4671ed822e2f853d72ba (unchanged)
final_nominal_evaluation_source_commit = recorded at evaluation time (see
    run_final_nominal_post_detection_evaluation.py); must be identical
    before/after the recorded campaign -- NO SOURCE MODIFICATION permitted
    once seed 41000 is opened. If a bug is discovered after opening the
    range: STOP, preserve outputs, report -- do not patch and silently rerun.

THIS IS A LOCKED TEST SET. Once opened: no model/controller/environment
parameter may change, no seed may be dropped, no statistical method may be
selected based on favorable outcomes.
"""

from __future__ import annotations

from pathlib import Path

from experiments.post_detection_diagnostic_protocol import (  # noqa: F401 -- re-exported single source of truth
    FROZEN_MODELS,
    TRAINING_SEEDS,
    TRAINING_SOURCE_COMMIT,
    PRIMARY_PAPER_METHODS,
    SCRIPTED_METHODS,
    PPO_METHODS,
    EXCLUDED_METHODS,
)

PROJECT_ROOT = Path(__file__).parent.parent

# Previously-consumed ranges that must remain untouched by this evaluation.
DIAGNOSTIC_SEED_RANGE = (40_100, 40_299)

FINAL_SEED_START = 41_000
FINAL_SEED_END = 45_999
FINAL_EPISODES = FINAL_SEED_END - FINAL_SEED_START + 1  # 5000
FINAL_SEEDS: range = range(FINAL_SEED_START, FINAL_SEED_END + 1)
assert FINAL_EPISODES == 5000, "locked protocol must cover exactly 5000 seeds"

FUTURE_ROBUSTNESS_SEED_START = 46_000

# Same bootstrap seed/replicate count used throughout this entire project
# (experiments/post_detection_diagnostic_protocol.py, final_stats.py callers) --
# fixed BEFORE seeing any final-evaluation results.
BOOTSTRAP_REPLICATES = 10_000
BOOTSTRAP_SEED = 20260816
CONFIDENCE_LEVEL = 0.95

OUTPUT_ROOT = PROJECT_ROOT / "results" / "final_nominal_post_detection_5000"

# Reference rates from the independent 200-seed diagnostic (item 29),
# recorded here BEFORE viewing any final-evaluation result, for a purely
# descriptive final-vs-diagnostic delta -- never used to alter methods/models.
DIAGNOSTIC_RATES = {
    "greedy": 0.740,
    "pn": 0.735,
    "lead": 0.880,
    ("ppo", 45): 0.885,
    ("ppo", 46): 0.885,
    ("ppo", 47): 0.815,
    ("ppo_kalman", 45): 0.855,
    ("ppo_kalman", 46): 0.765,
    ("ppo_kalman", 47): 0.850,
}

# Comparisons declared PRIMARY vs SECONDARY/descriptive BEFORE opening the
# final seed range (item 22) -- must not be reclassified after viewing results.
PRIMARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("PPO - Greedy", "ppo", "greedy"),
    ("PPO - PN", "ppo", "pn"),
    ("PPO - Lead", "ppo", "lead"),
    ("PPO-Kalman - PPO", "ppo_kalman", "ppo"),
)
SECONDARY_COMPARISONS: tuple[tuple[str, str, str], ...] = (
    ("Lead - Greedy", "lead", "greedy"),
    ("PN - Greedy", "pn", "greedy"),
    ("Lead - PN", "lead", "pn"),
)


def assert_seed_in_final_range(seed: int) -> None:
    if not (FINAL_SEED_START <= seed <= FINAL_SEED_END):
        raise ValueError(f"Seed {seed} is outside the locked final-test range {FINAL_SEED_START}..{FINAL_SEED_END}")


def assert_not_reserved_elsewhere(seed: int) -> None:
    if DIAGNOSTIC_SEED_RANGE[0] <= seed <= DIAGNOSTIC_SEED_RANGE[1]:
        raise ValueError(f"Seed {seed} is in the already-consumed diagnostic range; must not reuse here.")
    if seed >= FUTURE_ROBUSTNESS_SEED_START:
        raise ValueError(f"Seed {seed} is >= FUTURE_ROBUSTNESS_SEED_START; reserved for future robustness work.")
