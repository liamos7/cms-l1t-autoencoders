"""
rocs.py — AE vs NAE ROC curves
===============================
Uses ONLY the 20% validation split for both background (ZB) and all signal
processes. Reproduces the exact stratified split from datasets.py.

Memory-efficient: loads and scores one class at a time, discards raw data
before moving to the next. Uses contiguous chunk reads from HDF5 instead
of fancy indexing (which is extremely slow for large index arrays).
"""
import os
import torch
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import h5py
from pathlib import Path
from collections import OrderedDict
from sklearn.metrics import roc_curve, auc
from sklearn.model_selection import train_test_split

from fastad.models.__init__ import get_cicada_ae, get_cicada_nae_with_energy

sns.set_theme(style='whitegrid', context='notebook', font_scale=1.1)

# ── Config ────────────────────────────────────────────────────────────────────
DATA_DIR = Path("./data/h5_files/")
AE_PATH  = "outputs/ae_phase1_sigmoid_dim20/model_best.pkl"
NAE_PATH = "outputs/nae_phase2_fixed_dim20/model_best.pkl"
DEVICE   = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
BATCH    = 1024
OUT_DIR  = "plots/rocs"
os.makedirs(OUT_DIR, exist_ok=True)

# ── Class ordering (must match datasets.py exactly) ───────────────────────────
CLASS_ORDER = OrderedDict([
    ("zb",                0),
    ("glugluhtotautau",   1),
    ("glugluhtogg",       2),
    ("hto2longlivedto4b", 3),
    ("singleneutrino",    4),
    ("suep",              5),
    ("tt",                6),
    ("vbfhto2b",          7),
    ("vbfhtotautau",      8),
    ("zprimetotautau",    9),
    ("zz",               10),
])

PRETTY_NAMES = {
    "glugluhtotautau":   "GluGluH→ττ",
    "glugluhtogg":       "GluGluH→gg",
    "hto2longlivedto4b": "H→2LL→4b",
    "singleneutrino":    "Single Neutrino",
    "suep":              "SUEP",
    "tt":                "tt̄",
    "vbfhto2b":          "VBF H→bb",
    "vbfhtotautau":      "VBF H→ττ",
    "zprimetotautau":    "Z'→ττ",
    "zz":                "ZZ",
}


# ── Compute val-split indices per class ───────────────────────────────────────
def get_val_indices_per_class(data_dir):
    """
    Reproduce the exact stratified split from datasets.py on indices only.
    Returns OrderedDict: process_name -> sorted array of local row indices.
    Peak memory: ~2 * N_total * 4 bytes (int32 index + label arrays).
    """
    # 1. File sizes (no data loaded)
    sizes = OrderedDict()
    for process in CLASS_ORDER:
        path = data_dir / f"{process}.h5"
        if path.exists():
            with h5py.File(path, "r") as f:
                sizes[process] = f["et_regions"].shape[0]
        else:
            sizes[process] = 0

    # 2. Build label array, split indices only
    y_all = np.concatenate([
        np.full(s, lab, dtype=np.int32)
        for (p, lab), s in zip(CLASS_ORDER.items(), sizes.values()) if s > 0
    ])
    N = len(y_all)
    idx_all = np.arange(N, dtype=np.int32)
    train_size = int(0.8 * N)
    _, idx_val = train_test_split(
        idx_all, train_size=train_size, stratify=y_all,
        random_state=42, shuffle=True,
    )
    del idx_all, y_all

    # 3. Map to per-file local indices
    result = OrderedDict()
    offset = 0
    for process in CLASS_ORDER:
        s = sizes[process]
        if s == 0:
            continue
        mask = (idx_val >= offset) & (idx_val < offset + s)
        local = np.sort(idx_val[mask] - offset)
        result[process] = local
        offset += s
    del idx_val

    return result


# ── Score specific rows from an HDF5 file ─────────────────────────────────────
def load_and_score(model, filepath, row_indices):
    """
    Read only the val-split rows from an HDF5 file and score them.
    Uses contiguous chunk reads + boolean masking (fast) instead of
    h5py fancy indexing with a large index array (extremely slow).
    """
    with h5py.File(filepath, "r") as f:
        total = f["et_regions"].shape[0]

    # Boolean mask for which rows to keep
    keep = np.zeros(total, dtype=bool)
    keep[row_indices] = True

    norm_factor = np.float32(np.log1p(255))
    scores = []

    with h5py.File(filepath, "r") as f:
        ds = f["et_regions"]
        CHUNK = 50_000  # contiguous read size
        for start in range(0, total, CHUNK):
            end = min(start + CHUNK, total)
            chunk_keep = keep[start:end]
            if not chunk_keep.any():
                continue
            # One contiguous read, then filter in RAM — much faster than fancy indexing
            raw = ds[start:end][chunk_keep]
            for i in range(0, len(raw), BATCH):
                arr = raw[i:i+BATCH].astype(np.float32)
                np.log1p(arr, out=arr)
                arr /= norm_factor
                t = torch.from_numpy(arr).unsqueeze(1).to(DEVICE)
                with torch.no_grad():
                    s = model.predict(t)
                scores.append(s.cpu().numpy())
                del t
            del raw

    return np.concatenate(scores) if scores else np.array([], dtype=np.float32)


# ── Plotting helpers ──────────────────────────────────────────────────────────
def safe_fname(name):
    return (name.replace(' ', '_').replace('→', 'to')
                .replace("'", '').replace('τ', 'tau').replace('ℓ', 'l')
                .replace('̄', ''))

def save_fig(fig, stem):
    fig.savefig(f'{OUT_DIR}/{stem}.png', dpi=200, bbox_inches='tight')
    fig.savefig(f'{OUT_DIR}/{stem}.pdf', bbox_inches='tight')
    print(f"  Saved: {OUT_DIR}/{stem}")
    plt.close(fig)


# ══════════════════════════════════════════════════════════════════════════════
#  MAIN
# ══════════════════════════════════════════════════════════════════════════════

# 1. Load models
print("Loading models...")
model_ae = get_cicada_ae()
ae_ckpt = torch.load(AE_PATH, map_location=DEVICE, weights_only=False)
model_ae.load_state_dict(ae_ckpt['model_state'])
model_ae.to(DEVICE).eval()

model_nae = get_cicada_nae_with_energy(NAE_PATH).to(DEVICE).eval()

# 2. Compute val indices (lightweight — just int arrays)
print("\nComputing val split indices...")
val_indices = get_val_indices_per_class(DATA_DIR)
for p, idx in val_indices.items():
    print(f"  {PRETTY_NAMES.get(p, p)}: {len(idx)} val events")

# 3. Score each class one at a time (only one file in RAM at once)
print("\nScoring...")
bg_ae, bg_nae = None, None
signal_results = OrderedDict()

for process, local_idx in val_indices.items():
    pretty = PRETTY_NAMES.get(process, process)
    filepath = DATA_DIR / f"{process}.h5"
    print(f"  {pretty} ({len(local_idx)} events)...", end=" ", flush=True)

    s_ae  = load_and_score(model_ae,  filepath, local_idx)
    s_nae = load_and_score(model_nae, filepath, local_idx)
    print("done")

    if CLASS_ORDER[process] == 0:
        bg_ae, bg_nae = s_ae, s_nae
    else:
        signal_results[pretty] = (s_ae, s_nae)

print(f"\nBackground (ZB val): {len(bg_ae)} events")
for name, (s, _) in signal_results.items():
    print(f"  {name}: {len(s)} events")


# ── 4. Per-signal ROC plots (AE vs NAE) ───────────────────────────────────────
print("\nPlotting per-signal ROC curves...")
palette = sns.color_palette("tab10", len(signal_results))
auc_cache = {}

for (name, (s_ae, s_nae)), color in zip(signal_results.items(), palette):
    fig, ax = plt.subplots(figsize=(7, 6))
    y_true = np.concatenate([np.zeros(len(bg_ae)), np.ones(len(s_ae))])

    fpr_ae,  tpr_ae,  _ = roc_curve(y_true, np.concatenate([bg_ae,  s_ae]))
    fpr_nae, tpr_nae, _ = roc_curve(y_true, np.concatenate([bg_nae, s_nae]))
    auc_ae  = auc(fpr_ae,  tpr_ae)
    auc_nae = auc(fpr_nae, tpr_nae)
    auc_cache[name] = (auc_ae, auc_nae)

    ax.plot(fpr_ae,  tpr_ae,  linestyle='--', linewidth=1.8, color=color, alpha=0.7,
            label=f'AE  (AUC = {auc_ae:.3f})')
    ax.plot(fpr_nae, tpr_nae, linestyle='-',  linewidth=2.2, color=color,
            label=f'NAE (AUC = {auc_nae:.3f})')
    ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
    ax.set_xlabel('False Positive Rate (Zero Bias)')
    ax.set_ylabel('True Positive Rate')
    ax.set_title(f'ROC — {name}')
    ax.legend()
    plt.tight_layout(pad=1.5)
    save_fig(fig, f'roc_{safe_fname(name)}')


# ── 5. Combined: NAE only, all signals ────────────────────────────────────────
print("\nPlotting combined NAE ROC...")
fig, ax = plt.subplots(figsize=(10, 8))
for (name, (_, s_nae)), color in zip(signal_results.items(), palette):
    y_true = np.concatenate([np.zeros(len(bg_nae)), np.ones(len(s_nae))])
    fpr, tpr, _ = roc_curve(y_true, np.concatenate([bg_nae, s_nae]))
    ax.plot(fpr, tpr, linewidth=2, color=color,
            label=f'{name} (AUC = {auc(fpr, tpr):.3f})')
ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
ax.set_xlabel('False Positive Rate (Zero Bias)')
ax.set_ylabel('True Positive Rate')
ax.set_title('NAE — All Signals (val split only)')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout(pad=1.5)
save_fig(fig, 'roc_nae_all')


# ── 6. Combined: AE vs NAE, all signals ──────────────────────────────────────
print("\nPlotting combined AE vs NAE ROC...")
fig, ax = plt.subplots(figsize=(10, 8))
for (name, (s_ae, s_nae)), color in zip(signal_results.items(), palette):
    y_true = np.concatenate([np.zeros(len(bg_ae)), np.ones(len(s_ae))])
    fpr_ae,  tpr_ae,  _ = roc_curve(y_true, np.concatenate([bg_ae,  s_ae]))
    fpr_nae, tpr_nae, _ = roc_curve(y_true, np.concatenate([bg_nae, s_nae]))
    ax.plot(fpr_ae,  tpr_ae,  linestyle='--', linewidth=1.4, color=color, alpha=0.5)
    ax.plot(fpr_nae, tpr_nae, linestyle='-',  linewidth=2.0, color=color, label=name)
ax.plot([0, 1], [0, 1], 'k--', alpha=0.2, linewidth=1)
ax.set_xlabel('False Positive Rate (Zero Bias)')
ax.set_ylabel('True Positive Rate')
ax.set_title('AE (dashed) vs NAE (solid) — All Signals (val split only)')
ax.legend(fontsize=9, loc='lower right')
plt.tight_layout(pad=1.5)
save_fig(fig, 'roc_ae_vs_nae_all')


# ── 7. Summary table ──────────────────────────────────────────────────────────
print("\n" + "=" * 60)
print(f"{'Signal':<22} {'AE AUC':>8} {'NAE AUC':>9} {'Δ':>8}")
print("-" * 60)
for name in signal_results:
    a_ae, a_nae = auc_cache[name]
    delta = a_nae - a_ae
    print(f"{name:<22} {a_ae:>8.4f} {a_nae:>9.4f} {delta:>+8.4f}")
print("=" * 60)

print("\nDone.")