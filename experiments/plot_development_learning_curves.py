"""
DEVELOPMENT-ONLY learning curves (model-selection diagnostics), NOT final
paper performance figures. Uses only the in-training validation history
(seeds 40000-40099, already consumed by SuccessRateEvalCallback during
training) -- no new seeds are evaluated by this script.
"""
import json
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

root = "results/training_post_detection_400k"
seeds = (45, 46, 47)
colors = {45: "#0072B2", 46: "#D55E00", 47: "#009E73"}

fig, axes = plt.subplots(1, 2, figsize=(10, 4), constrained_layout=True)
for ax, track in zip(axes, ("direct", "kalman")):
    for seed in seeds:
        m = json.loads(open(f"{root}/{track}/seed_{seed}/run_manifest.json").read())
        xs = [e["timesteps"] for e in m["validation_history"]]
        ys = [e["success_rate"] * 100 for e in m["validation_history"]]
        ax.plot(xs, ys, marker="o", color=colors[seed], label=f"seed {seed}")
        best_step = m["best_checkpoint_timestep"]
        best_rate = m["best_validation_success_rate"] * 100
        ax.scatter([best_step], [best_rate], color=colors[seed], marker="*", s=150, zorder=5, edgecolor="black")
    ax.set_xlabel("Training timestep")
    ax.set_ylabel("Validation safe-interception rate (%)")
    ax.set_title(f"{track.capitalize()} track -- DEVELOPMENT / MODEL-SELECTION ONLY")
    ax.legend(fontsize=8)
    ax.set_ylim(0, 100)

fig.suptitle("DEVELOPMENT / MODEL-SELECTION ONLY -- NOT FINAL PAPER PERFORMANCE\n"
             "(validation seeds 40000-40099, n=100 episodes/point; stars = selected best checkpoint)",
             fontsize=9)
out_path = f"{root}/development_learning_curves.png"
fig.savefig(out_path, dpi=150)
print(f"written: {out_path}")
