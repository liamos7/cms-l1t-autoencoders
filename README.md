# Real-Time Autoencoder Anomaly Detection Methods at the Large Hadron Collider

**Princeton University Senior Thesis — Liam J. O'Shaughnessy (2025–2026)**  
Advised by Professor Isobel R. Ojalvo

This repository contains all code, trained model checkpoints, and generated figures for a senior thesis investigating anomaly detection at the CMS Level-1 trigger. The project has two main contributions:

1. **Latent space interpretability analysis** of the CICADA teacher autoencoder — characterizing what physical information the 80-dimensional bottleneck encodes using t-SNE, lasso regression, and event classifiers.
2. **Energy-Based Model (EBM) training** of a Normalized Autoencoder (NAE) with Langevin Monte Carlo negative sampling, designed to suppress outlier reconstruction and improve anomaly sensitivity.

The thesis PDF is included in this repository: [`O'Shaughnessy_Liam_Senior_Thesis.pdf`](O'Shaughnessy_Liam_Senior_Thesis.pdf).

---

## Background

[CICADA](https://arxiv.org/abs/2411.19506) is an autoencoder-based anomaly detection algorithm deployed at the CMS Layer-1 Calorimeter trigger. It compresses 18×14 calorimeter trigger tower images into an 80-dimensional latent space and assigns each event a reconstruction error score. Events that deviate from Zero Bias (minimum-bias) training data receive high scores and can trigger readout without presupposing the form of new physics.

This thesis evaluates the teacher autoencoder's learned representations and proposes an energy-based enhancement via a Normalized Autoencoder (NAE) trained using contrastive divergence with Langevin Monte Carlo (LMC) negative sampling. The NAE is trained on ten BSM holdout signal classes (ggH→γγ, ggH→ττ, tt̄, VBF H→ττ, VBF H→bb, Z′→ττ, ZZ, SUEP, H→2LL→4b, Single Neutrino) against a Zero Bias background.

---

## Key Results

| Model | Signal (worst) | Signal (best) | Notes |
|-------|----------------|---------------|-------|
| CICADA teacher-v1.0.0 | Single Neutrino: 0.622 | Z′→ττ: 0.976 | Deployed baseline |
| Improved AE (PyTorch) | Single Neutrino: 0.972 | tt̄: 0.9998 | Phase 1, d=20 |
| LMC NAE | Single Neutrino: 0.975 | tt̄: 0.9998 | Phase 2, d=20 |
| Oracle NAE (upper bound) | Single Neutrino: 0.980 | tt̄: 1.000 | MC negatives |

Key findings from the latent space analysis:
- Total E_T is encoded with near-perfect fidelity (R² ≈ 0.995).
- Pileup (n_PV) is weakly but non-negligibly encoded (R² ≈ 0.018, ~55× less than E_T).
- Jet kinematics (leading jet η) are essentially absent (R² ≈ 0.037).
- Latent dimension z₁₇ dominates both E_T and teacher score prediction by lasso coefficient magnitude, despite ranking far below z₆₁ in pairwise correlations — a key multivariate finding.
- AUC is robust to latent dimension choice across d = 10–80 for most signals.

---

## Repository Structure

```
cms-l1t-autoencoders/
├── fast-ad/                        # Main ML project (NAE/AE training + evaluation)
│   ├── fastad/                     # Core library
│   │   ├── models/
│   │   │   ├── modules.py          # Encoder/decoder building blocks
│   │   │   ├── teachers.py         # AE and NAEWithEnergyTraining model classes
│   │   │   └── __init__.py         # Model factory functions
│   │   ├── datasets.py             # CICADA dataset loader, 80/10/10 stratified split
│   │   ├── trainers.py             # Training loop (Phase 1: loss, Phase 2: AUC)
│   │   └── loggers.py              # TensorBoard integration
│   ├── train-teacher.py            # Main training entry point
│   ├── ae_vs_nae_rocs.py           # AE vs NAE ROC comparison
│   ├── eval_latent_dim_rocs.py     # Latent dimension sweep evaluation
│   ├── nae_mc_oracle_rocs.py       # Oracle upper-bound experiment
│   ├── plot_training.py            # TensorBoard → clean training curves
│   ├── autoresearch/               # Autonomous hyperparameter search agent
│   ├── data/                       # Data processing scripts + HDF5 files (gitignored)
│   └── outputs/                    # Model checkpoints + TensorBoard logs
├── correlations.py                 # t-SNE + observable correlation heatmaps
├── lasso_analysis.py               # Lasso regression on CICADA latent space
├── train_latent_classifier.py      # Latent-space event classifiers (XGBoost + MLP)
├── train_et_regions_classifier.py  # Raw ET-region baseline classifiers
├── slurm_scripts/                  # Adroit HPC job submission scripts
├── plots/                          # Generated figures (latent analysis)
└── requirements_backup.txt         # Full conda environment snapshot
```

---

## Setup

### Prerequisites

- Python 3.10+
- CUDA-capable GPU (training was run on Princeton Adroit, node `adroit-h11g1`)
- Conda recommended

### Environment

```bash
conda create -n cms-ad python=3.10
conda activate cms-ad
pip install torch==2.5.1+cu121 torchvision==0.20.1+cu121 --index-url https://download.pytorch.org/whl/cu121
pip install numpy scikit-learn xgboost h5py tensorboard tensorboardx matplotlib
```

Key package versions used in the thesis:

| Package | Version |
|---------|---------|
| PyTorch | 2.5.1+cu121 |
| NumPy | 2.2.6 |
| scikit-learn | 1.7.2 |
| XGBoost | 3.2.0 |
| h5py | 3.14.0 |
| TensorBoard | 2.20.0 |

### Data

The 11 HDF5 files (`data/h5_files/`) are **not committed** to this repository due to size. They are derived from CICADANtuples ROOT files available on CERN EOS at:

```
/eos/cms/store/group/phys_exotica/axol1tl/CICADANtuples/
```

To reproduce the HDF5 files from scratch:

1. **Skim ROOT files** (on CERN lxplus/EOS):
   ```bash
   python fast-ad/data/skim-inputs-mp.py
   ```

2. **Convert to HDF5** (with corrected uniform random sampling to avoid nPV bias):
   ```bash
   python fast-ad/data/process_to_hdf5.py
   ```

3. **Verify nPV distribution**:
   ```bash
   python fast-ad/data/check_sampling_npv.py
   ```

Each HDF5 file contains the following fields:

| Field | Shape | Description |
|-------|-------|-------------|
| `et_regions` | (N, 18, 14) | Raw calorimeter trigger tower images |
| `teacher_latent` | (N, 80) | CICADA teacher-v1.0.0 latent encodings |
| `teacher_score` | (N,) | Teacher reconstruction error (anomaly score) |
| `total_et` | (N,) | Scalar sum of all trigger tower E_T [GeV] |
| `nPV` | (N,) | Number of primary vertices (pileup proxy) |
| `first_jet_eta` | (N,) | Leading jet pseudorapidity |
| `ht` | (N,) | Scalar sum of jet transverse momenta [GeV] |

**Important**: The Zero Bias HDF5 file was originally built from sequential ROOT file reads, producing a biased nPV distribution (the Gaussian peak disappearing). The corrected pipeline (`process_to_hdf5.py`) uses uniform random sampling. See Appendix A of the thesis for details.

---

## Training

### Phase 1 — Standard Autoencoder

```bash
python fast-ad/train-teacher.py \
    --dataset CICADA \
    --model AE \
    --data-root-path /path/to/data/h5_files/ \
    --latent-dim 20 \
    --epochs 100 \
    -o outputs/ae_phase1_dim20/
```

The Phase 1 AE uses Adam (lr=1e-4, weight decay=1e-4), validates every 10,000 steps, and saves the checkpoint with the lowest validation reconstruction loss.

### Phase 2 — Normalized Autoencoder (Energy-Based)

```bash
python fast-ad/train-teacher.py \
    --dataset CICADA \
    --model NAEWithEnergyTraining \
    --data-root-path /path/to/data/h5_files/ \
    --load-pretrained-path outputs/ae_phase1_dim20/model_best.pkl \
    --latent-dim 20 \
    --epochs 50 \
    -o outputs/nae_phase2_dim20/
```

Phase 2 loads the Phase 1 checkpoint, projects latent codes onto the unit sphere S¹⁹, and trains with contrastive divergence loss using Langevin Monte Carlo negative sampling. Best checkpoint is selected by **validation AUC** (not loss). See `fast-ad/fastad/models/teachers.py` for the full NAE implementation and documented bug fixes.

### Oracle Upper Bound

To establish the theoretical performance ceiling, replace Langevin sampling with real MC signal events as negatives:

```bash
python fast-ad/train-teacher.py \
    --dataset CICADA \
    --model NAEWithEnergyTraining \
    --data-root-path /path/to/data/h5_files/ \
    --load-pretrained-path outputs/ae_phase1_dim20/model_best.pkl \
    --latent-dim 20 \
    --use-mc-negatives \
    -o outputs/nae_mc_oracle_dim20/
```

### Monitoring

```bash
tensorboard --logdir outputs/nae_phase2_dim20/
```

### HPC (Princeton Adroit)

All training runs use the SLURM scripts in `slurm_scripts/`. They load modules `anaconda3/2024.10` and `cudatoolkit/12.6` and activate the conda env at `/scratch/network/lo8603/thesis/conda/envs/myenv`. Update the paths for your environment.

```bash
sbatch slurm_scripts/run_training.sh        # Phase 1 + Phase 2 training
sbatch slurm_scripts/train_latent_dim_sweep.sh  # Latent dimension sweep
```

---

## Evaluation

### AE vs NAE ROC Comparison

```bash
python fast-ad/ae_vs_nae_rocs.py \
    --data-root-path /path/to/data/h5_files/ \
    --ae-path outputs/ae_phase1_dim20/model_best.pkl \
    --nae-path outputs/nae_phase2_dim20/model_best.pkl \
    --latent-dim 20
```

Outputs per-signal ROC curves with bootstrapped 95% CIs to `fast-ad/plots/rocs_test/`.

### Latent Dimension Sweep

```bash
python fast-ad/eval_latent_dim_rocs.py \
    --data-root-path /path/to/data/h5_files/ \
    --sweep-dir outputs/latent_dim_variation/
```

### Oracle Evaluation

```bash
python fast-ad/nae_mc_oracle_rocs.py \
    --data-root-path /path/to/data/h5_files/ \
    --nae-path outputs/nae_mc_oracle_dim20/model_best.pkl
```

---

## Latent Space Analysis

### t-SNE + Observable Correlations

```bash
python correlations.py --data-root-path /path/to/data/h5_files/
```

Outputs t-SNE embeddings colored by observable (E_T, student score, n_PV, leading jet η, H_T) to `plots/tsne/`, and Pearson correlation heatmaps to `plots/correlations/`.

### Lasso Regression

```bash
python lasso_analysis.py --data-root-path /path/to/data/h5_files/
```

Runs 5-fold cross-validated lasso regression of each observable (E_T, teacher score, n_PV, first jet η, H_T) on the 80-dimensional latent space. Produces regularization paths, active sets, and cumulative R² curves in `plots/lasso/`.

### Event Classifiers

```bash
# Latent-space classifier (XGBoost + MLP)
python train_latent_classifier.py --data-root-path /path/to/data/h5_files/

# Raw ET-region classifier (baseline comparison)
python train_et_regions_classifier.py --data-root-path /path/to/data/h5_files/
```

---

## Autoresearch Framework

The `fast-ad/autoresearch/` directory contains an autonomous hyperparameter search agent (inspired by [Karpathy's autoresearch](https://github.com/karpathy/autoresearch)). An LLM agent iteratively modifies `train.py` to maximize `best_val_auc × stability`, running 10–15 minute experiments on Adroit.

```bash
# Launch agent session (keeps GPU node alive for interactive agent)
sbatch fast-ad/autoresearch/run_autoresearch.sh
```

The agent modifies only `train.py` (hyperparameters, LMC loops, replay buffer, loss). `evaluate.py` is read-only and scores each run.

---

## Data Split

All results use a **three-way stratified split** of Zero Bias data (80% train / 10% validation / 10% test) with fixed seeds (SEED_80_20=42, SEED_50_50=43). This eliminates checkpoint-selection bias from using the same set for both early stopping and ROC reporting. All AUC values reported in the thesis are on the held-out test split.

```
SEED_80_20 = 42  # reproduces original 80/20 split; Phase 1 checkpoints remain valid
SEED_50_50 = 43  # splits remaining 20% into val/test
```

---

## NAE Training Instabilities

Energy-based training is notoriously unstable. Appendix B of the thesis documents seven specific bug classes encountered and fixed:

1. Missing gradient normalization in latent Langevin chain
2. Improper clamping pulling off-manifold negatives back into data range
3. Replay buffer initialized with real ZB encodings rather than uniform sphere samples
4. Energy regularization applied to both positive and negative samples
5. Temperature coupling contrastive loss and regularization
6. Divergence-skip logic suppressing corrective gradients
7. Best checkpoint selected by validation loss rather than validation AUC

The final hyperparameter configuration is in Appendix B, Table B.1.

---

## Citation

If you use this code or build on this work, please cite:

```
@thesis{oshaughnessy2026cicada,
  author  = {O'Shaughnessy, Liam J.},
  title   = {Real-Time Autoencoder Anomaly Detection Methods at the Large Hadron Collider},
  school  = {Princeton University},
  year    = {2026},
  type    = {Senior Thesis},
  department = {Department of Physics}
}
```

This work builds on:
- [CICADA](https://arxiv.org/abs/2411.19506): Gandrakota et al., "Real-time Anomaly Detection at the L1 Trigger of CMS Experiment" (2024)
- [NAE](https://arxiv.org/abs/2105.05735): Yoon, Noh, Park, "Autoencoding Under Normalization Constraints" (2021)
- [fast-ad](https://github.com/ligerlac/fast-ad): Lino Gerlach's fast anomaly detection codebase

---

## License

For research and educational use. Contact [lo8603@princeton.edu](mailto:lo8603@princeton.edu) for other uses.
