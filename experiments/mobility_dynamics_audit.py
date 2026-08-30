"""
Mobility-dynamics audit -- diagnostic-only investigation of two surprising
results in the locked mobility robustness sweep (branch
feature/robustness-mobility, commit 2dff347):

  1. Defender-speed arm is non-monotonic (most methods peak near v_d=15).
  2. Hostile-speed arm is "inverted" relative to naive intuition: defender
     success INCREASES as hostile speed v_e increases.

This module is DIAGNOSTIC ONLY. It does not modify environment dynamics,
policies, or the locked mobility dataset. All seeds used here are outside
every publication/robustness range (final_nominal, acquisition_ablation,
detection_radius, measurement_noise, mobility) -- see AUDIT_DIAGNOSTIC_SEEDS.
"""

from __future__ import annotations

import numpy as np

from uav_defend.envs import SoldierEnv
from uav_defend.config import EnvConfig
from uav_defend.policies.sanitize import build_policy_info, estimator_mode_for_env

# =============================================================================
# Diagnostic seed range -- NON-PUBLICATION, disjoint from every locked range
# (validation=10000-10099, final_nominal=20000-24999, acquisition=25000-25999,
#  detection_radius=30000-30999, measurement_noise=31000-31999,
#  mobility=32000-32999, hostile_evasion=33000+, dynamics=34000+).
# =============================================================================
AUDIT_DIAGNOSTIC_SEED_OFFSET = 18_000
AUDIT_DIAGNOSTIC_EPISODES = 100
AUDIT_DIAGNOSTIC_SEEDS: range = range(
    AUDIT_DIAGNOSTIC_SEED_OFFSET, AUDIT_DIAGNOSTIC_SEED_OFFSET + AUDIT_DIAGNOSTIC_EPISODES
)

# Matched same-seed trajectory examples (a small subset of the diagnostic range).
MATCHED_TRAJECTORY_SEEDS: tuple[int, ...] = (18000, 18001, 18002)

HOSTILE_SPEED_GRID: tuple[float, ...] = (8.0, 10.0, 12.0, 14.0, 16.0)
DEFENDER_SPEED_GRID: tuple[float, ...] = (12.0, 15.0, 18.0, 21.0, 24.0)
NOMINAL_DEFENDER_SPEED = 18.0
NOMINAL_HOSTILE_SPEED = 12.0

HORIZONTAL_RADII = (2.0, 3.5, 5.0, 10.0)
ALTITUDE_THRESHOLDS = (10.0, 5.0, 3.5, 2.0)

_NOMINAL_ENV_CONFIG = EnvConfig()


def audit_env_config(defender_speed: float, hostile_speed: float) -> EnvConfig:
    """Same field-isolation guarantee as the locked mobility sweep -- only
    v_d/v_e differ from nominal; everything else (accel/turn/climb limits,
    sensing, evasion, reward) stays exactly as in the locked experiment."""
    return EnvConfig(
        v_d=defender_speed,
        v_e=hostile_speed,
    )


def stationary_defender_action() -> np.ndarray:
    """Fixed defender controller that does NOT adapt to v_e: zero command
    (defender remains at its spawn position, co-located with the soldier)."""
    return np.zeros(3, dtype=np.float32)


# =============================================================================
# Hostile-speed geometry + dynamics diagnostic (items 7-14)
# =============================================================================
def run_hostile_geometry_episode(env: SoldierEnv, seed: int) -> dict:
    """
    Run one episode with a STATIONARY defender (isolates pure hostile
    guidance/dynamics geometry from any defender interception behavior --
    per item 11, success/interception is NOT the point of this diagnostic).
    """
    dt = env.config.dt
    obs, info = env.reset(seed=seed)

    initial_hostile_altitude = float(info["enemy_pos"][2])

    horizontal_entry_time = {r: None for r in HORIZONTAL_RADII}
    horizontal_entry_altitude = {r: None for r in HORIZONTAL_RADII}
    altitude_entry_time = {a: None for a in ALTITUDE_THRESHOLDS}

    min_horizontal_dist = float("inf")
    time_min_horizontal_dist = None
    altitude_at_min_horizontal_dist = None
    range3d_at_min_horizontal_dist = None

    speeds, horiz_speeds, vert_speeds = [], [], []
    accel_sat_count = 0
    turn_sat_count = 0
    vertical_sat_count = 0
    ground_hits = 0
    ceiling_hits = 0
    wall_hits = 0
    enemy_path_length = 0.0
    prev_pos = info["enemy_pos"].copy()
    prev_heading = None
    heading_reversals = 0
    last_turn_sign = None

    time_soldier_caught = None
    step = 0
    done = False

    L = env.config.L
    H = env.config.max_altitude

    def _record_geometry(t_now, enemy_pos, soldier_pos):
        nonlocal min_horizontal_dist, time_min_horizontal_dist
        nonlocal altitude_at_min_horizontal_dist, range3d_at_min_horizontal_dist
        dx = enemy_pos[0] - soldier_pos[0]
        dy = enemy_pos[1] - soldier_pos[1]
        dz = enemy_pos[2] - soldier_pos[2]
        horiz = float(np.hypot(dx, dy))
        dist3d = float(np.sqrt(dx * dx + dy * dy + dz * dz))
        altitude = float(enemy_pos[2])
        if horiz < min_horizontal_dist:
            min_horizontal_dist = horiz
            time_min_horizontal_dist = t_now
            altitude_at_min_horizontal_dist = altitude
            range3d_at_min_horizontal_dist = dist3d
        for r in HORIZONTAL_RADII:
            if horizontal_entry_time[r] is None and horiz <= r:
                horizontal_entry_time[r] = t_now
                horizontal_entry_altitude[r] = altitude
        for a in ALTITUDE_THRESHOLDS:
            if altitude_entry_time[a] is None and altitude <= a:
                altitude_entry_time[a] = t_now

    _record_geometry(0.0, info["enemy_pos"], info["soldier_pos"])

    while not done:
        obs, reward, done, truncated, info = env.step(stationary_defender_action())
        step += 1
        t_now = step * dt

        _record_geometry(t_now, info["enemy_pos"], info["soldier_pos"])

        vel = info["enemy_vel"]
        speed = float(np.linalg.norm(vel))
        horiz_speed = float(np.hypot(vel[0], vel[1]))
        vert_speed = float(abs(vel[2]))
        speeds.append(speed)
        horiz_speeds.append(horiz_speed)
        vert_speeds.append(vert_speed)

        diag = info
        accel_sat_count += int(diag["enemy_accel_saturated"])
        turn_sat_count += int(diag["enemy_turn_saturated"])
        vertical_sat_count += int(diag["enemy_climb_saturated"])

        cur_pos = info["enemy_pos"]
        enemy_path_length += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos.copy()

        if horiz_speed > env.config.eps:
            heading = float(np.arctan2(vel[1], vel[0]))
            if prev_heading is not None:
                delta = (heading - prev_heading + np.pi) % (2 * np.pi) - np.pi
                if abs(delta) > np.radians(5.0):
                    sign = float(np.sign(delta))
                    if last_turn_sign is not None and sign != last_turn_sign:
                        heading_reversals += 1
                    last_turn_sign = sign
            prev_heading = heading

        if abs(cur_pos[2] - 0.0) < 1e-6:
            ground_hits += 1
        if abs(cur_pos[2] - H) < 1e-6:
            ceiling_hits += 1
        if abs(abs(cur_pos[0]) - L) < 1e-6 or abs(abs(cur_pos[1]) - L) < 1e-6:
            wall_hits += 1

        if info["outcome"] == "soldier_caught" and time_soldier_caught is None:
            time_soldier_caught = t_now

        if done or truncated:
            break

    n_steps = max(step, 1)
    passes_near_horizontally_but_high = bool(
        min_horizontal_dist <= 2.0 and (range3d_at_min_horizontal_dist or 0.0) > 2.0
    )

    return {
        "eval_seed": seed,
        "hostile_speed": env.config.v_e,
        "initial_hostile_altitude": initial_hostile_altitude,
        "time_min_horizontal_dist": time_min_horizontal_dist,
        "min_horizontal_dist": min_horizontal_dist,
        "altitude_at_min_horizontal_dist": altitude_at_min_horizontal_dist,
        "range3d_at_min_horizontal_dist": range3d_at_min_horizontal_dist,
        "passes_near_horizontally_but_high": passes_near_horizontally_but_high,
        **{f"time_enter_horizontal_{int(r)}m": horizontal_entry_time[r] for r in HORIZONTAL_RADII},
        **{f"altitude_at_horizontal_{int(r)}m": horizontal_entry_altitude[r] for r in HORIZONTAL_RADII},
        **{f"time_below_altitude_{a}".replace(".0", "").replace(".", "p") + "m": altitude_entry_time[a] for a in ALTITUDE_THRESHOLDS},
        "time_soldier_caught": time_soldier_caught,
        "mean_hostile_speed": float(np.mean(speeds)) if speeds else float("nan"),
        "max_hostile_speed": float(np.max(speeds)) if speeds else float("nan"),
        "mean_horizontal_hostile_speed": float(np.mean(horiz_speeds)) if horiz_speeds else float("nan"),
        "mean_abs_vertical_speed": float(np.mean(vert_speeds)) if vert_speeds else float("nan"),
        "fraction_vertical_rate_saturated": vertical_sat_count / n_steps,
        "fraction_turn_saturated": turn_sat_count / n_steps,
        "fraction_accel_saturated": accel_sat_count / n_steps,
        "n_heading_reversals": heading_reversals,
        "enemy_path_length": enemy_path_length,
        "ground_boundary_hits": ground_hits,
        "ceiling_hits": ceiling_hits,
        "horizontal_wall_hits": wall_hits,
        "n_steps": n_steps,
    }


def run_hostile_geometry_diagnostics(seeds=None) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    seeds = list(seeds) if seeds is not None else list(AUDIT_DIAGNOSTIC_SEEDS)
    rows = []
    for v_e in HOSTILE_SPEED_GRID:
        config = audit_env_config(NOMINAL_DEFENDER_SPEED, v_e)
        env = SoldierEnv(config=config)
        for s in seeds:
            rows.append(run_hostile_geometry_episode(env, seed=s))
        env.close()
    return pd.DataFrame(rows)


# =============================================================================
# Initial-velocity vertical-clipping audit (item 13) -- analytic, no rollout needed
# =============================================================================
def audit_initial_vertical_clipping(seeds=None) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    seeds = list(seeds) if seeds is not None else list(AUDIT_DIAGNOSTIC_SEEDS)
    rows = []
    for v_e in HOSTILE_SPEED_GRID:
        config = audit_env_config(NOMINAL_DEFENDER_SPEED, v_e)
        env = SoldierEnv(config=config)
        n_clipped = 0
        initial_speeds = []
        for s in seeds:
            obs, info = env.reset(seed=s)
            enemy_pos = info["enemy_pos"]
            soldier_pos = info["soldier_pos"]
            pursuit_dir = soldier_pos - enemy_pos
            norm = np.linalg.norm(pursuit_dir)
            pursuit_dir = pursuit_dir / norm if norm > env.config.eps else np.array([1.0, 0.0, 0.0])
            desired_vz = v_e * pursuit_dir[2]
            clipped = abs(desired_vz) > env.config.enemy_max_descent_rate + 1e-9
            if clipped:
                n_clipped += 1
            actual_vel = info["enemy_vel"]
            initial_speeds.append(float(np.linalg.norm(actual_vel)))
        env.close()
        rows.append({
            "hostile_speed": v_e,
            "n_episodes": len(seeds),
            "n_initial_vz_clipped": n_clipped,
            "fraction_initial_vz_clipped": n_clipped / len(seeds),
            "mean_initial_speed_after_clipping": float(np.mean(initial_speeds)),
            "max_initial_speed_after_clipping": float(np.max(initial_speeds)),
        })
    return pd.DataFrame(rows)


# =============================================================================
# Matched same-seed trajectory time series (item 12)
# =============================================================================
def build_matched_trajectory_examples(seeds=MATCHED_TRAJECTORY_SEEDS, v_e_values=(8.0, 12.0, 16.0)) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    rows = []
    for v_e in v_e_values:
        config = audit_env_config(NOMINAL_DEFENDER_SPEED, v_e)
        env = SoldierEnv(config=config)
        dt = env.config.dt
        for seed in seeds:
            obs, info = env.reset(seed=seed)
            step = 0

            def _row(t_now, info):
                enemy_pos = info["enemy_pos"]
                soldier_pos = info["soldier_pos"]
                dx, dy, dz = enemy_pos[0] - soldier_pos[0], enemy_pos[1] - soldier_pos[1], enemy_pos[2] - soldier_pos[2]
                return {
                    "seed": seed, "hostile_speed": v_e, "step": step, "time_seconds": t_now,
                    "horizontal_dist": float(np.hypot(dx, dy)),
                    "vertical_separation": float(abs(dz)),
                    "dist_3d": float(np.sqrt(dx * dx + dy * dy + dz * dz)),
                    "hostile_altitude": float(enemy_pos[2]),
                }

            rows.append(_row(0.0, info))
            done = False
            while not done:
                obs, reward, done, truncated, info = env.step(stationary_defender_action())
                step += 1
                rows.append(_row(step * dt, info))
                if done or truncated:
                    break
        env.close()
    return pd.DataFrame(rows)


# =============================================================================
# Defender-speed diagnostic (items 15-17)
# =============================================================================
def run_defender_diagnostic_episode(env: SoldierEnv, policy, seed: int) -> dict:
    dt = env.config.dt
    obs, info = env.reset(seed=seed)
    policy.reset()
    estimator_mode = estimator_mode_for_env(env)

    accel_sat = 0
    turn_sat = 0
    climb_sat = 0
    speeds = []
    path_length = 0.0
    prev_pos = info["defender_pos"].copy()
    cumulative_turn_angle = 0.0
    prev_heading = None

    detection_step = None
    intercept_step = None
    min_defender_enemy_dist_post = None
    min_defender_enemy_dist_at_detection = None

    step = 0
    done = False
    while not done:
        policy_info = build_policy_info(info, estimator_mode)
        action = policy.act(obs, policy_info)
        obs, reward, done, truncated, info = env.step(action)
        step += 1

        diag = info
        accel_sat += int(diag["defender_accel_saturated"])
        turn_sat += int(diag["defender_turn_saturated"])
        climb_sat += int(diag["defender_climb_saturated"])
        speeds.append(info["defender_speed"])

        cur_pos = info["defender_pos"]
        path_length += float(np.linalg.norm(cur_pos - prev_pos))
        prev_pos = cur_pos.copy()

        vel = info["defender_vel"]
        horiz_speed = float(np.hypot(vel[0], vel[1]))
        if horiz_speed > env.config.eps:
            heading = float(np.arctan2(vel[1], vel[0]))
            if prev_heading is not None:
                delta = (heading - prev_heading + np.pi) % (2 * np.pi) - np.pi
                cumulative_turn_angle += abs(delta)
            prev_heading = heading

        if info["enemy_detected"] and detection_step is None:
            detection_step = step
            min_defender_enemy_dist_at_detection = float(info["defender_enemy_dist"])
        if detection_step is not None:
            if min_defender_enemy_dist_post is None:
                min_defender_enemy_dist_post = info["defender_enemy_dist"]
            else:
                min_defender_enemy_dist_post = min(min_defender_enemy_dist_post, info["defender_enemy_dist"])

        if info["outcome"] == "intercepted":
            intercept_step = step

        if done or truncated:
            break

    n_steps = max(step, 1)
    post_detection_intercept_steps = (
        intercept_step - detection_step if (intercept_step is not None and detection_step is not None) else None
    )
    closing_efficiency = None
    if detection_step is not None and min_defender_enemy_dist_at_detection is not None and min_defender_enemy_dist_post is not None and path_length > 0:
        net_reduction = min_defender_enemy_dist_at_detection - min_defender_enemy_dist_post
        closing_efficiency = net_reduction / path_length

    return {
        "eval_seed": seed,
        "defender_speed": env.config.v_d,
        "outcome": info["outcome"],
        "success": 1 if info["outcome"] == "intercepted" else 0,
        "mean_defender_speed": float(np.mean(speeds)) if speeds else float("nan"),
        "fraction_accel_saturated": accel_sat / n_steps,
        "fraction_turn_saturated": turn_sat / n_steps,
        "fraction_climb_saturated": climb_sat / n_steps,
        "cumulative_turn_angle_deg": float(np.degrees(cumulative_turn_angle)),
        "path_length": path_length,
        "detection_step": detection_step,
        "post_detection_intercept_steps": post_detection_intercept_steps,
        "post_detection_intercept_seconds": (post_detection_intercept_steps * dt) if post_detection_intercept_steps is not None else None,
        "closing_efficiency": closing_efficiency,
    }


def run_defender_speed_diagnostics(policy_name: str = "greedy", seeds=None) -> "pd.DataFrame":  # noqa: F821
    import pandas as pd
    from uav_defend.policies.registry import get_policy
    seeds = list(seeds) if seeds is not None else list(AUDIT_DIAGNOSTIC_SEEDS)
    rows = []
    for v_d in DEFENDER_SPEED_GRID:
        config = audit_env_config(v_d, NOMINAL_HOSTILE_SPEED)
        env = SoldierEnv(config=config)
        policy = get_policy(policy_name)
        for s in seeds:
            row = run_defender_diagnostic_episode(env, policy, seed=s)
            row["policy"] = policy_name
            rows.append(row)
        env.close()
    return pd.DataFrame(rows)
