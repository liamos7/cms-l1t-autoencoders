import h5py
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
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

# Qualitative palette with 11 well-separated colors
COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#fabed4", "#469990", "#dcbeff",
]

OBSERVABLES = {
    "nPV": {
        "xlabel": "Number of Primary Vertices",
        "bins": np.linspace(0, 150, 75),
        "logy": False,
    },
    "total_et": {
        "xlabel": "Total ET [GeV]",
        "bins": np.linspace(0, 2000, 80),
        "logy": True,
    },
    "first_jet_eta": {
        "xlabel": r"Leading Jet $\eta$",
        "bins": np.linspace(-5, 5, 40),
        "logy": False,
    },
    "first_jet_et": {
        "xlabel": "Leading Jet ET [GeV]",
        "bins": np.linspace(0, 500, 50),
        "logy": True,
    },
    "student_score": {
        "xlabel": "Student Score",
        "bins": np.linspace(0, 256, 80),
        "logy": True,
    },
    "teacher_score": {
        "xlabel": "Teacher Score",
        "bins": np.linspace(0, 128, 80),
        "logy": True,
    },
    "ht": {
        "xlabel": "HT [GeV]",
        "bins": np.linspace(0, 2000, 80),
        "logy": True,
    },
}

plt.rcParams.update({
    "font.family": "sans-serif",
    "font.size": 13,
    "axes.titlesize": 15,
    "axes.labelsize": 14,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 11,
    "figure.dpi": 150,
})


def plot_observable(obs_key, obs_cfg, config_list, output_dir):
    fig, ax = plt.subplots(figsize=(11, 6))
    bins = obs_cfg["bins"]
    logy = obs_cfg.get("logy", False)

    for i, item in enumerate(config_list):
        stem = item["stem"]
        filepath = item["path"]
        color = COLORS[i % len(COLORS)]
        label = LABEL_MAP.get(stem, stem)

        if not Path(filepath).exists():
            print(f"Warning: File not found - {filepath}")
            continue

        try:
            with h5py.File(filepath, "r") as f:
                if obs_key not in f:
                    print(f"  Skipping {stem}: key '{obs_key}' not found")
                    continue
                data = np.array(f[obs_key]).flatten()

            # Clip to bin range so overflow doesn't pile into last bin
            data = data[(data >= bins[0]) & (data <= bins[-1])]

            ax.hist(
                data, bins=bins, density=True,
                histtype="step", linewidth=1.8,
                color=color, label=label,
            )

        except Exception as e:
            print(f"Failed to process {stem} for {obs_key}: {e}")

    if logy:
        ax.set_yscale("log")
        ax.yaxis.set_major_formatter(ticker.LogFormatterSciNotation())
        # Raise the y-floor so a massive spike at zero doesn't compress the rest
        ylim_bottom, ylim_top = ax.get_ylim()
        if ylim_top > 0 and ylim_bottom > 0:
            ax.set_ylim(bottom=max(ylim_bottom, ylim_top * 1e-5))

    ax.set_xlabel(obs_cfg["xlabel"])
    ax.set_ylabel("Probability Density")
    ax.set_title(obs_cfg["xlabel"])
    ax.grid(True, alpha=0.25, linestyle="--")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    # Place legend outside the axes to the right so it never overlaps data
    legend = ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.01, 1),
        borderaxespad=0,
        framealpha=0.85,
        edgecolor="0.7",
        handlelength=1.5,
        labelspacing=0.4,
    )

    fig.tight_layout()
    out = Path(output_dir) / f"{obs_key}.png"
    fig.savefig(out, dpi=300, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


if __name__ == "__main__":
    h5_dir = Path("h5_files")
    output_dir = Path("plots")
    output_dir.mkdir(exist_ok=True)

    config_list = [
        {"stem": p.stem, "path": str(p)}
        for p in sorted(h5_dir.glob("*.h5"))
    ]

    for obs_key, obs_cfg in OBSERVABLES.items():
        print(f"Plotting {obs_key}...")
        plot_observable(obs_key, obs_cfg, config_list, output_dir)
