'''
For HDF5 data, plot t-SNE for event latent representations, and plot latent entry correlations with observables.
'''

import os
import numpy as np
import h5py
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import mplhep as hep
from sklearn.manifold import TSNE

hep.style.use("CMS")

H5_DIR = "/scratch/network/lo8603/thesis/fast-ad/data/h5_files"
TSNE_DIR = "plots/tsne"
CORR_DIR = "plots/correlations"

os.makedirs(TSNE_DIR, exist_ok=True)
os.makedirs(CORR_DIR, exist_ok=True)

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

COLORS = [
    "#e6194b", "#3cb44b", "#4363d8", "#f58231", "#911eb4",
    "#42d4f4", "#f032e6", "#bfef45", "#469990", "#dcbeff", "#800000",
]

SAMPLES = [
    "glugluhtotautau",
    "hto2longlivedto4b",
    "singleneutrino",
    "suep",
    "tt",
    "vbfhto2b",
    "vbfhtotautau",
    "zb",
    "zprimetotautau",
    "zz",
]

CORR_SAMPLES = ["glugluhtogg"] + SAMPLES
ALL_SAMPLES  = ["glugluhtogg"] + SAMPLES


def plot_latent_tsne_with_observables(name, nmax=5000):
    with h5py.File(f"{H5_DIR}/{name}.h5", "r") as f:
        latent_space   = f["teacher_latent"][:nmax]
        student_scores = f["student_score"][:nmax]
        energy         = f["total_et"][:nmax]
        pileup         = f["nPV"][:nmax].astype(np.float32)
        first_jet_et   = f["first_jet_et"][:nmax]
        first_jet_eta  = f["first_jet_eta"][:nmax]
        ht             = f["ht"][:nmax]

    print(f"  Running t-SNE for {name}...")
    tsne = TSNE(n_components=2, perplexity=30, learning_rate="auto", init="pca", random_state=42)
    latent_2d = tsne.fit_transform(latent_space)

    panels = [
        (energy,         "Total Energy",  "viridis"),
        (student_scores, "Student Score", "plasma"),
        (pileup,         "Pileup (nPV)",  "inferno"),
        (first_jet_et,   "First Jet ET",  "magma"),
        (ht,             "HT",            "cividis"),
        (first_jet_eta,  "First Jet Eta", "coolwarm"),
    ]

    pretty_title = LABEL_MAP.get(name, name)

    fig, axes = plt.subplots(2, 3, figsize=(16, 10), layout="constrained")
    fig.suptitle(pretty_title, fontsize=15)

    for ax, (values, label, cmap) in zip(axes.flat, panels):
        sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1],
                        c=values, cmap=cmap, s=3, alpha=0.5, linewidths=0,
                        rasterized=True)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046, shrink=0.85)
        cb.set_label(label, fontsize=10)
        cb.ax.tick_params(labelsize=8)
        cb.outline.set_linewidth(0.5)
        ax.set_title(label, fontsize=11, pad=5)
        ax.set_xlabel("t-SNE 1", fontsize=10)
        ax.set_ylabel("t-SNE 2", fontsize=10)
        ax.tick_params(labelsize=8)
        ax.xaxis.set_major_locator(plt.MaxNLocator(5))
        ax.yaxis.set_major_locator(plt.MaxNLocator(5))
        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)

    out = os.path.join(TSNE_DIR, f"tsne_{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


OBS_LABELS = ["Energy", "Student Score", "Pileup (nPV)", "First Jet ET", "HT", "First Jet Eta"]


def _load_observables(name, nmax=5000):
    with h5py.File(f"{H5_DIR}/{name}.h5", "r") as f:
        latents        = f["teacher_latent"][:nmax]
        student_scores = f["student_score"][:nmax]
        energy         = f["total_et"][:nmax]
        pileup         = f["nPV"][:nmax].astype(np.float32)
        first_jet_et   = f["first_jet_et"][:nmax]
        first_jet_eta  = f["first_jet_eta"][:nmax]
        ht             = f["ht"][:nmax]
    obs = np.stack([energy, student_scores, pileup, first_jet_et, ht, first_jet_eta], axis=1)
    return latents, obs


def _corr_matrix(latents, obs):
    """Returns (n_obs, n_latent) Pearson correlation matrix."""
    n_latent = latents.shape[1]
    n_obs    = obs.shape[1]
    # vectorised: zscore both, then dot
    def _zscore(x):
        s = x.std(axis=0, keepdims=True)
        s[s == 0] = 1
        return (x - x.mean(axis=0, keepdims=True)) / s
    lz = _zscore(latents)   # (N, n_latent)
    oz = _zscore(obs)       # (N, n_obs)
    mat = (oz.T @ lz) / len(latents)  # (n_obs, n_latent)
    return mat


def _symmax(mat):
    """Return a symmetric colormap limit based on the 98th percentile of |r|."""
    return float(np.percentile(np.abs(mat), 98))


def plot_latent_correlations(name, nmax=5000):
    latents, obs = _load_observables(name, nmax)
    mat = _corr_matrix(latents, obs)  # (n_obs, n_latent)

    n_obs, n_latent = mat.shape
    vlim = max(0.1, _symmax(mat))

    # transpose so observables are rows (y) and latent dims are columns (x)
    fig_w = max(8, n_latent * 0.13 + 2.5)
    fig_h = n_obs * 0.55 + 1.8
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))

    im = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                   interpolation="nearest")
    cb = fig.colorbar(im, ax=ax, fraction=0.03, pad=0.02, shrink=0.8)
    cb.set_label("Pearson r", fontsize=11)
    cb.ax.tick_params(labelsize=9)

    # x: latent dims — tick every 5
    step = max(1, n_latent // 20 * 5)  # round to nearest 5
    xticks = np.arange(0, n_latent, step)
    ax.set_xticks(xticks)
    ax.set_xticklabels(xticks, fontsize=8)
    ax.set_xlabel("Latent dim", fontsize=11)

    # y: observables
    ax.set_yticks(np.arange(n_obs))
    ax.set_yticklabels(OBS_LABELS, fontsize=10)
    ax.set_ylabel("Observable", fontsize=11)

    ax.set_title(f"Latent–observable correlations: {LABEL_MAP.get(name, name)}", fontsize=12, pad=8)

    # annotate vmin/vmax used
    ax.text(1.0, -0.08, f"clim ±{vlim:.2f}", transform=ax.transAxes,
            fontsize=8, ha="right", color="gray")

    out = os.path.join(CORR_DIR, f"correlations_{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_combined_correlations(samples=None, nmax=5000):
    """Grid of per-dataset heatmaps with a single shared colorbar."""
    if samples is None:
        samples = CORR_SAMPLES

    valid, mats = [], []
    for name in samples:
        if not os.path.exists(f"{H5_DIR}/{name}.h5"):
            print(f"  Skipping {name}: file not found")
            continue
        latents, obs = _load_observables(name, nmax)
        mats.append(_corr_matrix(latents, obs))
        valid.append(name)

    # shared symmetric colormap limit across all datasets
    all_vals = np.concatenate([m.ravel() for m in mats])
    vlim = max(0.1, float(np.percentile(np.abs(all_vals), 98)))

    n_obs, n_latent = mats[0].shape
    ncols = 4
    nrows = (len(valid) + ncols - 1) // ncols

    panel_w = max(3.0, n_latent * 0.055 + 1.0)
    panel_h = n_obs * 0.38 + 0.8
    fig, axes = plt.subplots(nrows, ncols,
                             figsize=(panel_w * ncols + 0.8, panel_h * nrows + 0.8),
                             gridspec_kw={"hspace": 0.55, "wspace": 0.25})
    axes_flat = np.array(axes).flat

    im_ref = None
    for ax, name, mat in zip(axes_flat, valid, mats):
        im_ref = ax.imshow(mat, aspect="auto", cmap="RdBu_r", vmin=-vlim, vmax=vlim,
                           interpolation="nearest")
        step = max(1, (n_latent // 4 // 5) * 5) if n_latent > 8 else 1
        xticks = np.arange(0, n_latent, step)
        ax.set_xticks(xticks)
        ax.set_xticklabels(xticks, fontsize=6)
        ax.set_xlabel("Latent dim", fontsize=7)
        ax.set_yticks(np.arange(n_obs))
        ax.set_yticklabels(OBS_LABELS, fontsize=7)
        ax.set_title(LABEL_MAP.get(name, name), fontsize=8, pad=3)

    for ax in list(axes_flat)[len(valid):]:
        ax.set_visible(False)

    # single shared colorbar on the right
    if im_ref is not None:
        cbar_ax = fig.add_axes([1.01, 0.15, 0.015, 0.7])
        cb = fig.colorbar(im_ref, cax=cbar_ax)
        cb.set_label("Pearson r", fontsize=11)
        cb.ax.tick_params(labelsize=9)
        fig.text(0.5, 1.005, f"Latent–observable correlations (all datasets)  —  clim ±{vlim:.2f}",
                 ha="center", fontsize=12)

    out = os.path.join(CORR_DIR, "correlations_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_combined_tsne(samples=ALL_SAMPLES, nmax=2000):
    """Run t-SNE on teacher_latent pooled across all datasets, colour by type."""
    all_latents = []
    all_labels  = []

    for name in samples:
        path = f"{H5_DIR}/{name}.h5"
        if not os.path.exists(path):
            print(f"  Skipping {name}: file not found")
            continue
        with h5py.File(path, "r") as f:
            latent = f["teacher_latent"][:nmax]
        all_latents.append(latent)
        all_labels.extend([name] * len(latent))
        print(f"  Loaded {name}: {len(latent):,} events")

    latents_arr = np.concatenate(all_latents, axis=0)
    labels_arr  = np.array(all_labels)

    print(f"  Running t-SNE on {len(latents_arr):,} events × {latents_arr.shape[1]} dims...")
    tsne = TSNE(n_components=2, perplexity=40, learning_rate="auto",
                init="pca", random_state=42, n_jobs=-1)
    coords = tsne.fit_transform(latents_arr)

    fig, ax = plt.subplots(figsize=(11, 7))

    unique_names = [s for s in samples if s in set(labels_arr)]
    for i, name in enumerate(unique_names):
        mask = labels_arr == name
        ax.scatter(
            coords[mask, 0], coords[mask, 1],
            s=4, alpha=0.5, linewidths=0,
            color=COLORS[i % len(COLORS)],
            label=LABEL_MAP.get(name, name),
            rasterized=True,
        )

    ax.set_xlabel("t-SNE 1", fontsize=12)
    ax.set_ylabel("t-SNE 2", fontsize=12)
    ax.set_title("t-SNE of teacher latent space (all datasets)", fontsize=13, pad=10)
    ax.xaxis.set_major_locator(plt.MaxNLocator(6))
    ax.yaxis.set_major_locator(plt.MaxNLocator(6))
    ax.tick_params(labelsize=10)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    ax.legend(
        loc="upper left",
        bbox_to_anchor=(1.02, 1),
        borderaxespad=0,
        markerscale=5,
        framealpha=0.95,
        edgecolor="0.8",
        fontsize=11,
        handlelength=1.5,
        handleheight=1.5,
    )

    fig.tight_layout()
    out = os.path.join(TSNE_DIR, "tsne_combined.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("=== Combined t-SNE ===")
    plot_combined_tsne()

    print("\n=== Per-sample t-SNE plots ===")
    for name in SAMPLES:
        print(f"Processing {name}...")
        plot_latent_tsne_with_observables(name)

    print("\n=== Correlation heatmaps (per dataset) ===")
    for name in CORR_SAMPLES:
        print(f"Processing {name}...")
        plot_latent_correlations(name)

    print("\n=== Combined correlation heatmap ===")
    plot_combined_correlations()

    print("\nDone.")
