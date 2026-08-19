# Figure Captions

## Figure 1 -- Proposed method (Lead-Residual PPO) architecture
The Lead-Residual PPO (LR-PPO) controller composes a bounded angular residual
correction, learned by PPO, around a predictive Lead-guidance direction. Both
the Lead policy and the residual policy consume the same sanitized
target-state observation $s_t$. The residual policy's output $\Delta u_t$ is
clipped to a bounded angular offset $\theta_{max} = 30^\circ$ from the Lead
direction before composition and normalization; the result $u_t$ is a
commanded *direction*, not a physical turn command -- the environment
separately enforces bounded acceleration, horizontal turn rate,
climb/descent rate, and maximum speed.

## Figure 2 -- Nominal performance
Safe-interception success rate (%) at the nominal training condition
(5000 evaluation seeds). Classical-method (Greedy, PN, Lead) intervals are
Wilson 95% intervals ($n=5000$). Learned-family (PPO, LR-PPO) intervals are
hierarchical 95% intervals computed across 3 independently trained roots
(3 training roots, 5000 evaluation episodes each); PPO-Kalman is omitted from
this headline comparison and reported separately as an estimator-ablation
diagnostic (Table II). The LR-PPO nominal gain over Lead is +3.01 pp
(95% CI [2.10, 3.91]), excluding zero. The apparent LR-PPO vs. PPO gain
(+2.01 pp, 95% CI [-0.37, 5.04]) includes zero and is not a nominal
superiority claim.

## Figure 3 -- Robustness across four independent sweeps
(a) Reactive hostile evasion: safe-interception success (%) vs. reactive
evasion gain (5 tested values); the vertical dashed line marks the nominal
training gain (0.75). LR-PPO remains near 90% at the highest tested gain
(1.00) while standalone PPO declines; error bars are 95% intervals (Wilson
for Lead, hierarchical for PPO/LR-PPO).
(b) Maneuverability sensitivity: LR-PPO's advantage (percentage points) over
Lead (left) and over PPO (right) across the full 3$\times$3 grid of defender
maximum acceleration (4/8/12 m/s$^2$) and maximum horizontal turn rate
(45/90/135 deg/s); this panel plots the *hybrid advantage*, not absolute
success. The nominal cell (8 m/s$^2$, 90 deg/s) is outlined in blue; cells
with a bold black border have a 95% difference interval that excludes zero.
(c) Measurement noise: safe-interception success (%) vs. measurement-noise
standard deviation $\sigma=\sqrt{\mathrm{variance}}$ (6 tested values); the
vertical dashed line marks the nominal $\sigma=\sqrt{0.5}\approx0.707$.
PPO-Kalman (estimator ablation) is shown de-emphasized in gray; it does not
outperform PPO at any tested noise level in this frozen configuration.
(d) Detection radius: safe-interception success (%) vs. defender detection
radius (6 tested values); dashed reference lines mark the nominal detection
radius (15 m) and the (independent, fixed) hostile reactive-evasion radius
(20 m) -- the hostile has no knowledge of the defender's detection state.
All panels: Lead, PPO, and LR-PPO error bars are 95% intervals (Wilson for
Lead/scripted methods, hierarchical across 3 training roots for PPO/LR-PPO).

## Figure 4 (optional) -- Mobility operating envelope
Two matched one-dimensional speed arms sharing one nominal condition
($v_d=18$, $v_e=12$ m/s): (a) defender speed varied with hostile speed fixed
at 12 m/s; (b) hostile speed varied with defender speed fixed at 18 m/s.
Because acceleration and turn-rate limits remain fixed while speed changes,
these curves reflect speed-maneuverability coupling in the tested
constrained point-mass model, not speed alone -- they should not be
interpreted as a statement about speed in isolation. The slow-hostile
condition ($v_d=18$, $v_e=8$) is the hardest of the 9 tested mobility
conditions for every method shown.
