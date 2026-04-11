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

    fig, axes = plt.subplots(2, 3, figsize=(15, 9),
                             gridspec_kw={"hspace": 0.35, "wspace": 0.05})
    fig.suptitle(name, fontsize=15, y=1.01)

    for ax, (values, label, cmap) in zip(axes.flat, panels):
        sc = ax.scatter(latent_2d[:, 0], latent_2d[:, 1], c=values, cmap=cmap, s=2, alpha=0.6)
        cb = fig.colorbar(sc, ax=ax, pad=0.02, fraction=0.046)
        cb.set_label(label, fontsize=11)
        cb.ax.tick_params(labelsize=9)
        ax.set_title(label, fontsize=12, pad=4)
        ax.set_xlabel("t-SNE 1", fontsize=11)
        ax.set_ylabel("t-SNE 2", fontsize=11)
        ax.tick_params(labelsize=9)

    out = os.path.join(TSNE_DIR, f"tsne_{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


def plot_latent_correlations(name, nmax=5000):
    with h5py.File(f"{H5_DIR}/{name}.h5", "r") as f:
        latents        = f["teacher_latent"][:nmax]
        student_scores = f["student_score"][:nmax]
        energy         = f["total_et"][:nmax]
        pileup         = f["nPV"][:nmax].astype(np.float32)
        first_jet_et   = f["first_jet_et"][:nmax]
        first_jet_eta  = f["first_jet_eta"][:nmax]
        ht             = f["ht"][:nmax]

    observables = [
        (energy,         "Energy"),
        (student_scores, "Student Score"),
        (pileup,         "Pileup (nPV)"),
        (first_jet_et,   "First Jet ET"),
        (ht,             "HT"),
        (first_jet_eta,  "First Jet Eta"),
    ]

    n_latent = latents.shape[1]
    xs = np.arange(n_latent)

    fig, axes = plt.subplots(2, 3, figsize=(14, 7),
                             gridspec_kw={"hspace": 0.45, "wspace": 0.35})
    fig.suptitle(f"Latent–observable correlations: {name}", fontsize=13, y=1.02)

    for ax, (obs, label) in zip(axes.flat, observables):
        corrs = np.array([np.corrcoef(obs, latents[:, i])[0, 1] for i in xs])
        ax.scatter(xs, corrs, s=8, linewidths=0)
        ax.axhline(0, color="gray", lw=0.6, ls="--")
        ax.set_title(label, fontsize=12, pad=3)
        ax.set_xlabel("Latent index", fontsize=10)
        ax.set_ylabel("Pearson r", fontsize=10)
        ax.tick_params(labelsize=9)

    # hide x-label on top row to reduce clutter
    for ax in axes[0]:
        ax.set_xlabel("")

    out = os.path.join(CORR_DIR, f"correlations_{name}.pdf")
    fig.savefig(out, bbox_inches="tight")
    fig.savefig(out.replace(".pdf", ".png"), bbox_inches="tight", dpi=150)
    plt.close(fig)
    print(f"  Saved {out}")


if __name__ == "__main__":
    print("=== t-SNE plots ===")
    for name in SAMPLES:
        print(f"Processing {name}...")
        plot_latent_tsne_with_observables(name)

    print("\n=== Correlation plots ===")
    for name in CORR_SAMPLES:
        print(f"Processing {name}...")
        plot_latent_correlations(name)

    print("\nDone.")
