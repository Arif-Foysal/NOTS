# NOTS-NIDS

**Nash-Optimized Topological Shields: A Game-Theoretic Persistent Homology Framework for Adversarially Robust Network Intrusion Detection**

> Target venue: IEEE Transactions on Information Forensics and Security (T-IFS)

## Quick Start

```bash
pip install -r requirements.txt
```

## Repository Structure

| Directory | Purpose |
|---|---|
| `config.py` | All hyperparameters and paths |
| `preprocessing/` | Data loading, cleaning, scaling, windowing |
| `topology/` | UMAP reduction, Ripser filtration, persistence diagrams, Wasserstein distance |
| `game_theory/` | Nash equilibrium solver and runtime sampler |
| `detector/` | Baseline builder, NOTS detector, adaptive baseline |
| `adversarial/` | White-box (FGSM) and black-box (surrogate RF) attacks |
| `baselines/` | RF, Kitsune, LUCID comparison baselines |
| `experiments/` | Experiments 1–5 (baseline, white-box, black-box, ablation, efficiency) |
| `evaluation/` | Metrics, theorem validator, paper figure plotter |
| `notebooks/` | Colab/Kaggle notebooks for setup, exploration, and running experiments |
| `data/` | Dataset download scripts and instructions |
| `results/` | Output directory for CSVs and figures |

## Datasets

See `data/README.md` for download instructions for CICIDS-2017, UNSW-NB15, and NSL-KDD.

## Running Experiments

Use `notebooks/03_run_experiments.ipynb` on Google Colab (free T4 GPU) or Kaggle (free P100).

## Citation

If you use this code, please cite the accompanying paper.
