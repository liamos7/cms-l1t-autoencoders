import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from mpl_toolkits.axes_grid1 import make_axes_locatable
from pathlib import Path

LABEL_MAP = {
    "glugluhtogg":         r"$gg \to H \to \gamma\gamma$",
    "glugluhtotautau":     r"$gg \to H \to \tau\tau$",
    "hto2longlivedto4b":   r"$H \to 2LL \to 4b$",
    "singleneutrino":      "Single Neutrino",
    "suep":                "SUEP",
    "tt":                  r"$t\bar{t}$",
    "vbfhto2b":            r"VBF $H \to bb$",
    "vbfhtotautau":        r"VBF $H \to \tau\tau$",
    "zb":                  "Zero Bias",
    "zprimetotautau":      r"$Z' \to \tau\tau$",
    "zz":                  r"$ZZ$",
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 11,
    "axes.titlesize": 12,
    "axes.labelsize": 9,
    "xtick.labelsize": 7,
    "ytick.labelsize": 7,
    "figure.dpi": 150,
})

CMAP = "inferno"
# Target percentile of total_et to pick a representative event (not too quiet, not saturated)
TARGET_TOTAL_ET_PERCENTILE = 75


def pick_event(h5path):
    """Return the event whose total_et is closest to TARGET_TOTAL_ET_PERCENTILE."""
    with h5py.File(h5path, "r") as f:
        total_et = np.array(f["total_et"])
        target = np.percentile(total_et, TARGET_TOTAL_ET_PERCENTILE)
        idx = int(np.argmin(np.abs(total_et - target)))
        event = f["et_regions"][idx]   # (18, 14)
    return event


def plot_et_regions(h5_files, output_path):
    n_files = len(h5_files)
    n_cols = int(np.ceil(np.sqrt(n_files)))
    n_rows = int(np.ceil(n_files / n_cols))

    fig, axes = plt.subplots(
        n_rows, n_cols,
        figsize=(3.8 * n_cols, 3.5 * n_rows),
        constrained_layout=True,
    )
    axes = np.array(axes).flatten()

    events = {Path(p).stem: pick_event(p) for p in h5_files}

    for ax, path in zip(axes, h5_files):
        stem = Path(path).stem
        event = events[stem]   # (18, 14)
        label = LABEL_MAP.get(stem, stem)

        vmax = float(event.max())
        # Power-norm with gamma<1 compresses the high end and lifts the low end,
        # making sparse deposits visible without a log scale blowing up zeros.
        norm = mcolors.PowerNorm(gamma=0.4, vmin=0, vmax=max(vmax, 1))

        im = ax.imshow(
            event.T,           # (14, 18): phi on y-axis, eta on x-axis
            origin="lower",
            aspect="auto",
            cmap=CMAP,
            norm=norm,
            interpolation="nearest",
        )

        ax.set_title(label, pad=5)
        ax.set_xlabel(r"$i\eta$", labelpad=2)
        ax.set_ylabel(r"$i\phi$", labelpad=2)

        ax.set_xticks(range(0, 18, 2))
        ax.set_yticks(range(0, 14, 2))
        ax.set_xticklabels(range(1, 19, 2))
        ax.set_yticklabels(range(1, 15, 2))

        # Colorbar attached directly to its subplot — no overlap
        divider = make_axes_locatable(ax)
        cax = divider.append_axes("right", size="5%", pad=0.05)
        cb = fig.colorbar(im, cax=cax)
        # Show a few representative tick values in GeV
        nice_ticks = [t for t in [0, 25, 100, 250, 500, 1000] if t <= vmax]
        if nice_ticks:
            cb.set_ticks(nice_ticks)
            cb.set_ticklabels([str(t) for t in nice_ticks])
        cb.set_label("ET [GeV]", fontsize=7, labelpad=4)
        cb.ax.tick_params(labelsize=6)

    for ax in axes[n_files:]:
        ax.set_visible(False)

    fig.suptitle(r"ET Regions", fontsize=14)
    fig.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {output_path}")


if __name__ == "__main__":
    h5_dir = Path("h5_files")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    h5_files = sorted(h5_dir.glob("*.h5"))
    if not h5_files:
        raise FileNotFoundError(f"No .h5 files found in {h5_dir}")

    plot_et_regions(h5_files, output_dir / "et_regions_sample.png")
