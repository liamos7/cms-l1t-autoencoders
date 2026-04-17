"""
Read TensorBoard event files from a NAE training run and produce a clean
training-curve figure showing:
  - Positive / negative energy  (top panel)
  - AUC over time               (bottom panel)

Usage:
    python plot_training.py [run_dir] [--out path/to/output.pdf]

Defaults:
    run_dir : fast-ad/outputs/nae_phase2_tuned_dim20
    out     : plots/training/<run_name>.pdf  (also saved as .png)

Multiple event files in the same directory are merged in chronological order
(by wall-clock time of first event), with later files taking precedence for
overlapping steps.
"""

import argparse
import os
import sys
from collections import defaultdict

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import numpy as np
from tensorboard.backend.event_processing.event_accumulator import EventAccumulator

# ── Defaults ──────────────────────────────────────────────────────────────────

DEFAULT_RUN = "fast-ad/outputs/nae_mc_upper_bound_dim20"
OUT_DIR     = "fast-ad/plots/training"

# ── Argument parsing ──────────────────────────────────────────────────────────

parser = argparse.ArgumentParser(description="Plot NAE training curves from TensorBoard logs")
parser.add_argument("run_dir", nargs="?", default=DEFAULT_RUN,
                    help="Directory containing .tfevents files")
parser.add_argument("--out", default=None,
                    help="Output path (default: plots/training/<run_name>.pdf)")
args = parser.parse_args()

run_dir  = args.run_dir
run_name = os.path.basename(run_dir.rstrip("/"))
os.makedirs(OUT_DIR, exist_ok=True)

out_pdf = args.out or os.path.join(OUT_DIR, f"{run_name}.pdf")
out_png = out_pdf.replace(".pdf", ".png")

# ── Load & merge event files ──────────────────────────────────────────────────

def load_run(run_dir):
    """
    Collect all .tfevents files, sort by the wall_time of their first event,
    and merge scalar series — later files override earlier ones for duplicate
    steps (restart / resume scenario).
    """
    event_files = sorted(
        [os.path.join(run_dir, f) for f in os.listdir(run_dir)
         if f.startswith("events.out.tfevents")],
    )
    if not event_files:
        sys.exit(f"No .tfevents files found in {run_dir}")

    # sort by first wall_time so we merge in chronological order
    def first_wall_time(path):
        ea = EventAccumulator(path)
        ea.Reload()
        tags = ea.Tags()["scalars"]
        if not tags:
            return float("inf")
        return ea.Scalars(tags[0])[0].wall_time

    event_files = sorted(event_files, key=first_wall_time)
    print(f"Found {len(event_files)} event file(s) in {run_dir}")

    # step → value dict per tag, later files override
    merged: dict[str, dict[int, float]] = defaultdict(dict)

    for path in event_files:
        ea = EventAccumulator(path)
        ea.Reload()
        for tag in ea.Tags()["scalars"]:
            for ev in ea.Scalars(tag):
                merged[tag][ev.step] = ev.value
        print(f"  Loaded {os.path.basename(path):55s}  "
              f"tags: {len(ea.Tags()['scalars'])}")

    # convert to sorted numpy arrays
    series: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tag, step_val in merged.items():
        steps = np.array(sorted(step_val.keys()))
        vals  = np.array([step_val[s] for s in steps])
        series[tag] = (steps, vals)
        print(f"    {tag:<40}  {len(steps)} points  "
              f"steps {steps[0]:,}–{steps[-1]:,}")

    return series


print(f"\nLoading run: {run_dir}")
series = load_run(run_dir)

# ── Extract series ────────────────────────────────────────────────────────────

def get(tag):
    if tag not in series:
        sys.exit(f"Tag '{tag}' not found. Available: {list(series.keys())}")
    return series[tag]

steps_e,   pos_e   = get("energy/pos_energy_")
_,         neg_e   = get("energy/neg_energy_")
steps_auc, auc_val = get("roc_auc_")

# ── Style ─────────────────────────────────────────────────────────────────────

plt.rcParams.update({
    "font.family":       "sans-serif",
    "font.size":         12,
    "axes.titlesize":    13,
    "axes.labelsize":    12,
    "xtick.labelsize":   10,
    "ytick.labelsize":   10,
    "legend.fontsize":   11,
    "axes.spines.top":   False,
    "axes.spines.right": False,
    "axes.grid":         True,
    "grid.alpha":        0.25,
    "grid.linestyle":    "--",
})

BLUE   = "#4363d8"
RED    = "#e6194b"
PURPLE = "#911eb4"
GREY   = "0.55"

# ── Plot ──────────────────────────────────────────────────────────────────────

fig, axes = plt.subplots(2, 1, figsize=(9, 8), layout="constrained",
                         gridspec_kw={"height_ratios": [1.4, 1]})

# ── Panel 1: energies ─────────────────────────────────────────────────────────

ax1 = axes[0]
ax1.plot(steps_e / 1e3, pos_e, color=RED,    linewidth=2,   label="Positive energy (data)",    marker="o", markersize=4)
ax1.plot(steps_e / 1e3, neg_e, color=BLUE,   linewidth=2,   label="Negative energy (noise)",   marker="s", markersize=4)
ax1.set_ylabel("Energy")
ax1.set_title(f"Training curves — {run_name}")
ax1.legend(loc="best", framealpha=0.9, edgecolor="0.8")
ax1.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax1.set_xlabel("")
ax1.tick_params(labelbottom=False)   # shared x with lower panel

# mark best AUC step on the energy panel too
best_step = steps_auc[np.argmax(auc_val)]
ax1.axvline(best_step / 1e3, color=GREY, linewidth=1, linestyle=":", alpha=0.8)
ax1.text(best_step / 1e3, 1.0, "  best AUC", fontsize=8, color=GREY,
         va="bottom", transform=ax1.get_xaxis_transform())

# ── Panel 2: AUC ─────────────────────────────────────────────────────────────

ax2 = axes[1]
ax2.plot(steps_auc / 1e3, auc_val, color=PURPLE, linewidth=2,
         marker="o", markersize=4, label="ROC AUC (validation)")

# highlight best point
best_auc = auc_val.max()
ax2.scatter([best_step / 1e3], [best_auc], color=PURPLE, s=80, zorder=5,
            edgecolors="white", linewidths=1.5)
ax2.annotate(f"  best: {best_auc:.4f}\n  step {best_step:,}",
             xy=(best_step / 1e3, best_auc),
             xytext=(10, -18), textcoords="offset points",
             fontsize=9, color=PURPLE,
             arrowprops=dict(arrowstyle="-", color=PURPLE, lw=0.8))

ax2.axvline(best_step / 1e3, color=GREY, linewidth=1, linestyle=":", alpha=0.8)
ax2.set_ylabel("ROC AUC")
ax2.set_xlabel("Training step (thousands)")
ax2.xaxis.set_major_formatter(ticker.FuncFormatter(lambda x, _: f"{x:.0f}k"))
ax2.set_ylim(max(0, auc_val.min() - 0.02), min(1.0, auc_val.max() + 0.02))
ax2.legend(loc="lower right", framealpha=0.9, edgecolor="0.8")

# ── Save ─────────────────────────────────────────────────────────────────────

fig.savefig(out_pdf, bbox_inches="tight")
fig.savefig(out_png, bbox_inches="tight", dpi=150)
plt.close(fig)
print(f"\nSaved:\n  {out_pdf}\n  {out_png}")
