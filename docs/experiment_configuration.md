# UAV Interception Experiment Configuration

**Document Version:** 1.0  
**Last Updated:** 2026-03-26

This document provides a complete reference of all environment assumptions, entity behaviors, sensing models, and experiment parameters used in the UAV interception research. It is intended for thesis documentation and reproducibility.

---

## 1. Overview

### 1.1 Problem Description

The simulation models a **UAV-based soldier protection scenario** with three primary entities:

1. **Soldier** — A protected ground asset (stationary or slow-moving)
2. **Defender UAV** — A protective drone attempting to intercept the attacker
3. **Enemy UAV** — An attacking drone attempting to reach the soldier

The defender must **intercept the enemy before it reaches the soldier**. Detection occurs when the defender enters a sensing radius around the enemy, enabling tracking. The scenario presents a pursuit-evasion game with partial observability.

### 1.2 Research Objectives

This environment is used to compare three control approaches:

| Method | Description |
|--------|-------------|
| **Greedy Baseline** | Hand-designed direct pursuit controller |
| **PPO (Direct RL)** | Reinforcement learning with true enemy state observations |
| **PPO (RL-Kalman)** | Reinforcement learning with Kalman-filtered state estimates |

The research investigates whether learned policies outperform hand-designed baselines, and whether state estimation improves RL policy performance.

---

## 2. Environment Entities

### 2.1 Soldier (Protected Asset)

The soldier represents the asset to be protected. It moves with a slow Gaussian random walk.

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Position | $\mathbf{s} \in \mathbb{R}^2$ | Random in $[-L, L]^2$ | 2D position in domain |
| Speed | $v_s$ | 1.5 m/s | Movement scale (much slower than drones) |
| Threat radius | $r_{\text{threat}}$ | 2.0 m | Mission failure zone |

**Movement Model:**
```
Δs ~ N(0, v_s · dt)   (Gaussian random walk)
```

**Loss Condition:** The mission fails when the enemy enters the threat radius:
$$\|\mathbf{e} - \mathbf{s}\| \leq r_{\text{threat}}$$

---

### 2.2 Defender UAV (Controlled Agent)

The defender is the controlled agent that must intercept the enemy before it reaches the soldier.

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Initial position | $\mathbf{d}_0$ | Random in $[-L, L]^2$ | Starting position |
| Speed | $v_d$ | 18.0 m/s | Maximum velocity |
| Intercept radius | $r_{\text{intercept}}$ | 2.5 m | Neutralization distance |
| Unsafe intercept radius | $r_{\text{unsafe}}$ | 3.5 m | Collateral risk zone around soldier |

**Action Space:**
- Continuous 2D vector: $\mathbf{a} \in [-1, 1]^2$
- Interpreted as heading direction (normalized)
- Displacement per step: $\Delta \mathbf{d} = v_d \cdot \Delta t \cdot \hat{\mathbf{a}}$

**Win Condition:** Successful interception occurs when:
$$\|\mathbf{d} - \mathbf{e}\| \leq r_{\text{intercept}} \quad \text{AND} \quad \|\mathbf{e} - \mathbf{s}\| > r_{\text{unsafe}}$$

---

### 2.3 Enemy UAV (Attacker)

The enemy pursues the soldier using a **stochastic weaving pursuit** model. This makes the enemy harder to predict than constant-velocity motion.

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Initial position | $\mathbf{e}_0$ | Random in $[-L, L]^2$ | Starting position |
| Speed | $v_e$ | 12.0 m/s | Maximum velocity |
| AR(1) coefficient | $\rho$ | 0.85 | Weave bias persistence |
| Weave noise std | $\sigma_a$ | 0.5 | Lateral weave intensity |
| Heading noise std | $\sigma_e$ | 0.15 | Directional noise |
| Weave amplitude | — | 1.5 | Lateral motion multiplier |

**Weaving Pursuit Dynamics:**

1. Compute unit vector toward soldier: $\hat{\mathbf{r}} = (\mathbf{s} - \mathbf{e}) / \|\mathbf{s} - \mathbf{e}\|$
2. Perpendicular vector: $\hat{\mathbf{r}}_\perp = [-\hat{r}_y, \hat{r}_x]$
3. Update weave bias (AR(1) process): $a \leftarrow \rho \cdot a + \sigma_a \cdot \eta$, where $\eta \sim \mathcal{N}(0,1)$
4. Heading noise: $\mathbf{z} \sim \mathcal{N}(0, I_2)$
5. Unnormalized direction: $\mathbf{u}_{\text{raw}} = \hat{\mathbf{r}} + a \cdot \text{amp} \cdot \hat{\mathbf{r}}_\perp + \sigma_e \cdot \mathbf{z}$
6. Normalize and move: $\mathbf{e}_{\text{new}} = \mathbf{e} + v_e \cdot \Delta t \cdot \hat{\mathbf{u}}$

**Key Insight:** The enemy does NOT move with constant velocity. The weaving pursuit introduces:
- Lateral oscillations (AR(1) weave bias)
- Random heading perturbations
- Persistent but stochastic maneuvering

This makes the constant-velocity Kalman filter assumption a model mismatch.

---

## 3. Sensing Model

### 3.1 Detection System

The defender has a limited sensing range. The enemy is **invisible** until detected.

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Detection radius | $r_{\text{det}}$ | 15.0 m | Maximum sensing range |
| Measurement noise std | — | $\sqrt{\text{measurement\_var}}$ | Position sensor noise |

**Detection Condition:**
$$\text{detected} = \begin{cases} \text{True} & \text{if } \|\mathbf{d} - \mathbf{e}\| \leq r_{\text{det}} \\ \text{False} & \text{otherwise} \end{cases}$$

**Detection Properties:**
- Detection is **event-triggered** (occurs once when defender enters sensing radius)
- Once detected, tracking persists for the remainder of the episode
- Before detection: enemy-related observations are masked (zeros)
- After detection: noisy position measurements are available

### 3.2 Measurement Noise

Position measurements include additive Gaussian noise:
$$\mathbf{z} = \mathbf{e}_{\text{true}} + \boldsymbol{\epsilon}, \quad \boldsymbol{\epsilon} \sim \mathcal{N}(0, \sigma_m^2 I_2)$$

| Parameter | Default Value | Description |
|-----------|---------------|-------------|
| Measurement variance | 0.5 | Sensor noise power |

---

## 4. Observation Model

The observation space is a 9-dimensional normalized vector in $[-1, 1]^9$.

### 4.1 Observation Layout

| Index | Component | Description | Normalization |
|-------|-----------|-------------|---------------|
| 0-1 | $\mathbf{s}$ | Soldier position | $/L$ |
| 2-3 | $\mathbf{d}$ | Defender position | $/L$ |
| 4 | `detected` | Detection flag | 0.0 or 1.0 |
| 5-6 | $\hat{\mathbf{e}}$ | Enemy position estimate | $/L$ |
| 7-8 | $\hat{\mathbf{v}}$ | Enemy velocity estimate | $/v_e$ |

### 4.2 Direct RL Mode

When `use_kalman_tracking=False`:
- $\hat{\mathbf{e}}$ = true enemy position (after detection)
- $\hat{\mathbf{v}}$ = zeros (velocity not directly observable)
- Policy acts on **ground truth** state

### 4.3 RL-Kalman Mode

When `use_kalman_tracking=True`:
- $\hat{\mathbf{e}}$ = Kalman-filtered position estimate
- $\hat{\mathbf{v}}$ = Kalman-filtered velocity estimate
- Policy acts on **estimated** state from noisy measurements

---

## 5. Kalman Filter Configuration

### 5.1 Motion Model

The Kalman filter uses a **constant-velocity** motion model:

**State Vector:**
$$\mathbf{x} = [p_x, p_y, v_x, v_y]^T$$

**State Transition:**
$$\mathbf{x}_{k+1} = \mathbf{F} \mathbf{x}_k + \mathbf{w}_k$$

$$\mathbf{F} = \begin{bmatrix} 1 & 0 & \Delta t & 0 \\ 0 & 1 & 0 & \Delta t \\ 0 & 0 & 1 & 0 \\ 0 & 0 & 0 & 1 \end{bmatrix}$$

### 5.2 Measurement Model

Only position is observed:
$$\mathbf{z}_k = \mathbf{H} \mathbf{x}_k + \mathbf{v}_k$$

$$\mathbf{H} = \begin{bmatrix} 1 & 0 & 0 & 0 \\ 0 & 1 & 0 & 0 \end{bmatrix}$$

### 5.3 Noise Parameters

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Process variance | $\sigma_q^2$ | 1.0 | Model uncertainty (acceleration noise) |
| Measurement variance | $\sigma_r^2$ | 0.5 | Sensor noise |
| Lead time | $t_{\text{lead}}$ | 0.0 s | Prediction extrapolation time |

### 5.4 Tracking Error Metric

Tracking error measures the Euclidean distance between estimated and true enemy position:
$$\text{tracking\_error} = \|\hat{\mathbf{e}} - \mathbf{e}_{\text{true}}\|$$

Mean tracking error is computed over all post-detection timesteps in an episode.

### 5.5 Model Mismatch

**Important:** The constant-velocity Kalman model assumes linear motion, but the enemy uses stochastic weaving pursuit with:
- AR(1) lateral bias
- Random heading noise
- Non-constant velocity direction

This model mismatch may explain why RL-Kalman does not outperform Direct RL.

---

## 6. Episode Termination Conditions

### 6.1 Success (WIN)

Defender intercepts enemy safely:
$$\|\mathbf{d} - \mathbf{e}\| \leq r_{\text{intercept}} \quad \text{AND} \quad \|\mathbf{e} - \mathbf{s}\| > r_{\text{unsafe}}$$

**Reward:** +100

### 6.2 Failure — Soldier Caught (LOSS)

Enemy reaches soldier:
$$\|\mathbf{e} - \mathbf{s}\| \leq r_{\text{threat}}$$

**Reward:** -100

### 6.3 Failure — Unsafe Intercept (LOSS)

Interception occurs too close to soldier (collateral risk):
$$\|\mathbf{d} - \mathbf{e}\| \leq r_{\text{intercept}} \quad \text{AND} \quad \|\mathbf{e} - \mathbf{s}\| \leq r_{\text{unsafe}}$$

**Reward:** -150

### 6.4 Timeout (LOSS)

Episode exceeds maximum steps:
$$\text{step\_count} \geq \text{max\_steps}$$

**Reward:** -100

---

## 7. Reward Structure

### 7.1 Terminal Rewards

| Outcome | Reward |
|---------|--------|
| Safe intercept | +100 |
| Soldier caught | -100 |
| Unsafe intercept | -150 |
| Timeout | -100 |

### 7.2 Shaping Rewards (per step)

| Component | Value | Description |
|-----------|-------|-------------|
| Progress | $+5.0 \cdot \Delta d$ | Reward for closing distance to enemy |
| Time penalty | $-0.05$ | Encourages efficiency |
| Tracking improvement | $+1.0 \cdot \Delta e_{\text{track}}$ | Reward for improving tracking error (Kalman) |
| Proximity warning | $-0.5 \cdot \text{factor}$ | Penalty when enemy near soldier |

---

## 8. Simulation Parameters

### 8.1 Domain and Time

| Parameter | Symbol | Default Value | Description |
|-----------|--------|---------------|-------------|
| Domain half-size | $L$ | 50.0 m | Domain is $[-L, L]^2$ |
| Time step | $\Delta t$ | 0.5 s | Simulation timestep |
| Max episode steps | $T_{\text{max}}$ | 2000 | Maximum steps before timeout |
| Numerical epsilon | $\varepsilon$ | $10^{-8}$ | Stability constant |

### 8.2 Summary Table

| Parameter | Symbol | Value | Units |
|-----------|--------|-------|-------|
| Domain size | $2L$ | 100 | m |
| Time step | $\Delta t$ | 0.5 | s |
| Defender speed | $v_d$ | 18.0 | m/s |
| Enemy speed | $v_e$ | 12.0 | m/s |
| Soldier speed | $v_s$ | 1.5 | m/s |
| Detection radius | $r_{\text{det}}$ | 15.0 | m |
| Intercept radius | $r_{\text{intercept}}$ | 2.5 | m |
| Threat radius | $r_{\text{threat}}$ | 2.0 | m |
| Unsafe intercept radius | $r_{\text{unsafe}}$ | 3.5 | m |
| Measurement variance | $\sigma_r^2$ | 0.5 | m² |
| Process variance | $\sigma_q^2$ | 1.0 | m²/s² |
| Max episode steps | $T_{\text{max}}$ | 2000 | steps |

---

## 9. Evaluation Protocol

### 9.1 Monte Carlo Evaluation

Performance is measured via Monte Carlo simulation with fixed random seeds for reproducibility.

| Evaluation Type | Episodes | Description |
|-----------------|----------|-------------|
| Main evaluation | 1000 | Primary success rate measurement |
| Parameter sweeps (1D) | 200 per value | Sensitivity analysis |
| Speed grid (2D) | 100 per point | Joint speed space analysis |

### 9.2 Parameter Sweeps

| Sweep | Parameter Values |
|-------|------------------|
| Defender speed | 10.0, 12.0, 14.0, 16.0, 18.0, 20.0 m/s |
| Enemy speed | 8.0, 10.0, 12.0, 14.0, 16.0 m/s |
| Detection radius | 5.0, 8.0, 10.0, 12.0, 15.0 m |
| Speed grid (defender) | 10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0 m/s |
| Speed grid (enemy) | 8.0, 10.0, 12.0, 14.0, 16.0, 18.0 m/s |

### 9.3 Metrics

| Metric | Description |
|--------|-------------|
| Success rate | Fraction of episodes ending in safe intercept |
| Failure rate | Soldier caught + unsafe intercept |
| Timeout rate | Episodes exceeding max steps |
| Mean episode length | Average steps per episode |
| Mean tracking error | Kalman estimation accuracy (RL-Kalman only) |
| 95% Confidence interval | Wilson score interval for success rate |

### 9.4 Statistical Comparison

Differences between methods are tested using:
- **Z-test** for proportion comparison
- **95% confidence intervals** for success rates
- **p < 0.05** threshold for statistical significance

---

## 10. Experimental Results Summary

### 10.1 Overall Performance (1000 episodes each)

| Method | Success Rate | 95% CI | Improvement over Baseline |
|--------|-------------|--------|---------------------------|
| Greedy Baseline | 42.5% | [39.4%, 45.6%] | — |
| PPO (Direct RL) | **60.3%** | [57.3%, 63.3%] | +17.8pp ✅ |
| PPO (RL-Kalman) | 57.7% | [54.6%, 60.7%] | +15.2pp ✅ |

### 10.2 Key Findings

1. **Both RL methods significantly outperform baseline** (p < 0.001)
2. **Direct RL slightly outperforms RL-Kalman** (+2.6pp, not statistically significant)
3. **Kalman filtering does not improve performance** on this task
4. The constant-velocity model mismatch likely explains the RL-Kalman results

---

## 11. File References

### 11.1 Configuration Files

| File | Description |
|------|-------------|
| `uav_defend/config/env_config.py` | Environment parameters (EnvConfig dataclass) |
| `experiments/experiment_config.py` | Experiment parameters (sweeps, evaluation) |

### 11.2 Core Implementation

| File | Description |
|------|-------------|
| `uav_defend/envs/soldier_env.py` | Gymnasium environment |
| `uav_defend/tracking/kalman_filter.py` | EnemyKalmanFilter class |
| `uav_defend/policies/greedy.py` | Greedy baseline policy |
| `uav_defend/policies/rl_kalman.py` | PPOKalmanPolicyWrapper |

### 11.3 Experiment Scripts

| File | Description |
|------|-------------|
| `experiments/baseline/evaluate_baseline.py` | Baseline Monte Carlo evaluation |
| `experiments/rl/evaluate_rl.py` | Direct RL evaluation |
| `experiments/rl_kalman/evaluate_rl_kalman.py` | RL-Kalman evaluation |
| `experiments/comparison/compare_all_methods.py` | Three-way comparison |

---

## Appendix A: Default EnvConfig Values

```python
@dataclass
class EnvConfig:
    # Domain
    L: float = 50.0                    # Half-size of 2D domain
    
    # Time
    max_steps: int = 2000              # Maximum episode steps
    dt: float = 0.5                    # Time step (seconds)
    eps: float = 1e-8                  # Numerical stability
    
    # Speeds
    v_s: float = 1.5                   # Soldier speed
    v_e: float = 12.0                  # Enemy speed
    v_d: float = 18.0                  # Defender speed
    
    # Enemy weaving parameters
    rho: float = 0.85                  # AR(1) coefficient
    sigma_a: float = 0.5               # Weave bias noise std
    sigma_e: float = 0.15              # Heading noise std
    weave_amplitude: float = 1.5       # Lateral weave multiplier
    
    # Distance thresholds
    detection_radius: float = 15.0     # Sensing range
    intercept_radius: float = 2.5      # Interception distance
    threat_radius: float = 2.0         # Mission failure zone
    unsafe_intercept_radius: float = 3.5  # Collateral risk zone
    
    # Rewards
    reward_intercept: float = 100.0    # Win reward
    reward_soldier_caught: float = -100.0  # Loss penalty
    reward_unsafe_intercept: float = -150.0  # Unsafe intercept penalty
    reward_timeout: float = -100.0     # Timeout penalty
    reward_progress_scale: float = 5.0 # Distance progress scale
    reward_time_penalty: float = -0.05 # Per-step time penalty
    
    # Kalman tracking
    use_kalman_tracking: bool = True   # Enable Kalman filtering
    process_var: float = 1.0           # Process noise variance
    measurement_var: float = 0.5       # Measurement noise variance
    lead_time: float = 0.0             # Prediction lead time
```

---

## Appendix B: Experiment Configuration

```python
# Parameter sweeps
DEFENDER_SPEEDS = (10.0, 12.0, 14.0, 16.0, 18.0, 20.0)
ENEMY_SPEEDS = (8.0, 10.0, 12.0, 14.0, 16.0)
DETECTION_RADII = (5.0, 8.0, 10.0, 12.0, 15.0)

# 2D grid
GRID_DEFENDER_SPEEDS = (10.0, 12.0, 14.0, 16.0, 18.0, 20.0, 22.0)
GRID_ENEMY_SPEEDS = (8.0, 10.0, 12.0, 14.0, 16.0, 18.0)

# Episode counts
EVAL_EPISODES = 1000       # Main evaluation
SWEEP_EPISODES = 200       # Per parameter value
GRID_EPISODES = 100        # Per grid point

# Kalman configuration
KALMAN_CONFIG = {
    "process_var": 1.0,
    "measurement_var": 0.5,
    "lead_time": 0.0,
}
```

---

*Document generated from UAV_Defend codebase configuration files.*
