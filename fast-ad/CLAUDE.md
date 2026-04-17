# Thesis: Anomaly Detection for CICADA L1 Trigger

## Project
Senior thesis at Princeton. Anomaly detection for CMS/LHC CICADA trigger system.
Goal: outperform CICADA autoencoder at detecting BSM physics (SUEP, GGH2TT, etc) against zero-bias background.

## Environment
- Compute: adroit-h11g1 GPU node (NOT the login node where claude runs)
- Conda env: /scratch/network/lo8603/thesis/conda/envs/myenv
- Data: data/h5_files/
- Key scripts: nae-autoresearch/train.py, nae-autoresearch/evaluate.py
- Agent instructions: nae-autoresearch/program.md
- To submit jobs: bash nae-autoresearch/run_autoresearch.sh

## Structure
- fastad/: core modules (datasets.py, modules.py, teachers.py, etc)
- nae-autoresearch/: NAE training + evaluation + autoresearch loop
- outputs/: results
- rocs.py, train-teacher.py: analysis scripts

## Current focus
NAE Phase 2 energy-based training with Langevin Monte Carlo in 80-dim spherical latent space.
Hyperparameter tuning ongoing. See nae-autoresearch/program.md for full context and known failure modes.

## Constraints
- Claude Code runs on LOGIN node (adroit5), not GPU node
- Never run training directly — always submit via run_autoresearch.sh or sbatch


# NAE EBM Training Fixes, proposed by extended Opus

## Summary of Changes

### Fix #1 — `modules.py` — CicadaDecoder sigmoid output
**Problem:** CicadaDecoder had no output activation, so decoded values were unbounded.
Positive data lives in [0,1] after log-transform, but negative samples decoded from
off-manifold latent codes could be anywhere (-0.3, 2.1, etc.), making the energy
surface noisy and the contrastive signal unreliable.

**Change:** Added `nn.Sigmoid()` as the final layer of `CicadaDecoder.net`.

**⚠️ CRITICAL: You must retrain Phase 1 with this change.**
The old Phase 1 checkpoint was trained without sigmoid — its decoder weights learned
to output raw values that happen to land near [0,1] for ZB data. Loading those weights
into a model with sigmoid will squash the outputs through sigmoid(sigmoid-range-values),
distorting the energy surface. Retrain Phase 1 from scratch with the sigmoid-equipped
decoder, then use that new checkpoint for Phase 2.

### Fix #2 — `teachers.py` — Only regularize negative energy
**Problem:** `gamma * (E_pos² + E_neg²)` regularized both positive and negative energy.
Regularizing positive energy fights the reconstruction objective (which wants low pos energy).

**Change:** `gamma * E_neg²` — only penalize negative energy from diverging, per Yoon et al. Section 6.1.

### Fix #3 — `teachers.py` — Gradient normalization in data-space Langevin
**Problem:** Latent chain normalized gradients to unit norm before stepping (good), but
data chain used raw gradients. With unbounded decoder output (fix #1) or steep energy
landscapes, raw gradients could produce enormous steps, making the data chain unstable.

**Change:** Added per-sample gradient normalization in the data-space Langevin loop,
matching the latent chain pattern.

### Fix #4 — `teachers.py` — Temperature decoupled from regularization
**Problem:** `loss = (cd_loss + reg_loss) / T + l2_loss` — temperature divided both the
contrastive term and the regularization, silently coupling T and gamma.

**Change:** `loss = cd_loss / T + reg_loss + l2_loss` — temperature only scales the
MLE gradient term.

### Fix #5 — `teachers.py` — Better NaN/Inf handling
**Problem:** NaN loss returned `{'loss': 0.0}` and silently skipped backward, potentially
corrupting Adam momentum/variance estimates.

**Change:** Logs a warning with energy values, explicitly zeros gradients, and returns
`float('nan')` so downstream logging can detect the issue.

### Fix #6 — `trainers.py` + `train-teacher.py` — Best model by AUC
**Problem:** Best model was selected by lowest validation loss (ZB reconstruction error),
which can decrease even as contrastive discrimination (AUC) degrades.

**Change:** Added `best_model_metric` parameter to `BaseTrainer`. For `NAEWithEnergyTraining`,
automatically uses `'auc'` (higher is better); for standard AE, uses `'loss'` (lower is better).

## Recommended Training Sequence

```bash
# 1. Retrain Phase 1 WITH sigmoid decoder
python train-teacher.py \
  --dataset CICADA --model AE \
  --data-root-path ./data/h5_files/ \
  --latent-dim 20 --epochs 100 \
  -o ./outputs/ae_phase1_sigmoid_dim20/

# 2. Run Phase 2 with fixed code
python train-teacher.py \
  --dataset CICADA --model NAEWithEnergyTraining \
  --data-root-path ./data/h5_files/ \
  --load-pretrained-path ./outputs/ae_phase1_sigmoid_dim20/model_best.pkl \
  --latent-dim 20 --epochs 50 \
  -o ./outputs/nae_phase2_fixed_dim20/
```

## Autoresearch Sweep Parameters (after fixes applied)

Once the code fixes are in, these are the most impactful hyperparameters to sweep:

| Parameter      | Range                 | Why                                      |
|---------------|-----------------------|------------------------------------------|
| `neg_lambda`  | [1.0, 2.0, 5.0]      | Controls contrastive push strength       |
| `gamma`       | [1e-3, 1e-2, 1e-1]   | Neg energy regularization weight         |
| `z_steps`     | [15, 30, 50]          | Latent chain length                      |
| `z_step_size` | [0.001, 0.005, 0.01]  | Latent chain step size                   |
| `x_steps`     | [5, 10, 20]           | Data chain length                        |
| `x_step_size` | [0.001, 0.005, 0.01]  | Data chain step size (now normalized)    |