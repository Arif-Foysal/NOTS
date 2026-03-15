# Running NOTS on Google Colab — Step by Step

---

## Prerequisites

- A Google account (for Google Drive + Colab)
- A Kaggle account (for automatic dataset download — free at kaggle.com)

---

## Step 1: Get Your Kaggle API Token

1. Go to **https://www.kaggle.com/settings**
2. Scroll to **API** section → click **Create New Token**
3. Copy the token string (starts with `KGAT_...`)
4. In Colab, click the **key icon** (left sidebar) → Add a secret:
   - Name: `KAGGLE_API_TOKEN`
   - Value: paste your token
   - Toggle notebook access **on**

> This lets the notebook automatically download the CICIDS-2017 dataset (~500 MB).

---

## Step 2: Upload Code to Google Drive

You already have `nots_nids.zip`. Upload it to the root of your Google Drive (`My Drive/`).

> [!NOTE]
> The projection method is now **PCA** by default (instead of UMAP) to ensure theoretical stability. You can change this to `umap` in `config.py` if needed.

---

## Step 3: Open Google Colab

1. Go to **https://colab.research.google.com**
2. Click **File → New Notebook**
3. **Switch to GPU**: Runtime → Change runtime type → **T4 GPU** → Save

---

## Step 4: Run These 4 Cells

### Cell 1 — Mount Drive, Unzip Code, Auto-Download Dataset

```python
# Mount Drive
from google.colab import drive
drive.mount('/content/drive')

# Unzip code
!unzip -qo /content/drive/MyDrive/nots_nids.zip -d /content/drive/MyDrive/

# Setup
import sys, os
REPO_DIR = '/content/drive/MyDrive/nots_nids'
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# Auto-download CICIDS-2017 from Kaggle (skips if already present)
from data.download import ensure_cicids2017
DATA_DIR = ensure_cicids2017('data/cicids2017')
print(f"Dataset ready at: {DATA_DIR}")
```

### Cell 2 — Install Dependencies

```python
%%bash
pip install -q ripser persim pot umap-learn cvxpy scikit-learn pandas numpy \
    matplotlib seaborn joblib torch torchvision tqdm memory-profiler
pip install -q cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com 2>/dev/null || \
    echo "cuML not available — using CPU UMAP (this is fine)"
```

### Cell 3 — Run All 7 Experiments (~45–90 min)

```python
import sys, os, time, random, logging
import numpy as np
import torch

REPO_DIR = '/content/drive/MyDrive/nots_nids'
sys.path.insert(0, REPO_DIR)
os.chdir(REPO_DIR)

# Reproducibility
np.random.seed(42); random.seed(42); torch.manual_seed(42)
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(message)s')

from config import Config
cfg = Config()

# ── 1. Load ─────────────────────────────────────────────────────────────
print("=" * 60); print("STEP 1/8: Loading CICIDS-2017..."); print("=" * 60)
from data.download import ensure_cicids2017
DATA_DIR = ensure_cicids2017('data/cicids2017')
from preprocessing.loader import load_cicids2017
df, label_col = load_cicids2017(DATA_DIR)
print(f"✅ Loaded: {df.shape}")

# ── 2. Preprocess ───────────────────────────────────────────────────────
print("\nSTEP 2/8: Preprocessing...")
from preprocessing.cleaner import clean_dataframe, encode_labels
from preprocessing.scaler import fit_scaler, apply_scaler
from preprocessing.windowing import create_windows
from sklearn.model_selection import train_test_split

df_clean, dropped = clean_dataframe(df, label_col)
df_clean, label_map = encode_labels(df_clean, label_col)
numeric_cols = df_clean.select_dtypes(include=[np.number]).columns.tolist()
feature_cols = [c for c in numeric_cols if c not in ['label_int', 'is_attack']]

idx = np.arange(len(df_clean))
idx_train, idx_temp = train_test_split(idx, test_size=0.4, random_state=42,
                                        stratify=df_clean['is_attack'])
idx_val, idx_test = train_test_split(idx_temp, test_size=0.5, random_state=42,
                                      stratify=df_clean.iloc[idx_temp]['is_attack'])

df_train = df_clean.iloc[idx_train].reset_index(drop=True)
df_val   = df_clean.iloc[idx_val].reset_index(drop=True)
df_test  = df_clean.iloc[idx_test].reset_index(drop=True)

X_scaled, scaler = fit_scaler(df_train[feature_cols].values)
df_train[feature_cols] = X_scaled
df_val[feature_cols]   = apply_scaler(df_val[feature_cols].values, scaler)
df_test[feature_cols]  = apply_scaler(df_test[feature_cols].values, scaler)

train_windows = create_windows(df_train, feature_cols, window_size=cfg.WINDOW_SIZE)
val_windows   = create_windows(df_val,   feature_cols, window_size=cfg.WINDOW_SIZE)
test_windows  = create_windows(df_test,  feature_cols, window_size=cfg.WINDOW_SIZE)
print(f"✅ Windows: train={len(train_windows)}, val={len(val_windows)}, test={len(test_windows)}")

# ── 3. Fit NOTS ─────────────────────────────────────────────────────────
print("\nSTEP 3/8: Fitting NOTS detector (~5-15 min)...")
from detector.nots_detector import NOTSDetector
detector = NOTSDetector(cfg)
detector.fit(train_windows, val_windows, feature_cols=feature_cols)
print(f"✅ ε_min={detector.epsilon_min:.6f}, τ={detector.tau:.6f}")
detector.save(cfg.RESULTS_DIR)

# ── 4. Exp 1: Baseline ─────────────────────────────────────────────────
print("\nSTEP 4/8: Experiment 1 — Baseline detection...")
from experiments.exp1_baseline import run_experiment_1
from baselines.rf_baseline import RFBaseline
from baselines.kitsune_baseline import KitsuneBaseline

X_train_bl = df_train[feature_cols].values; y_train_bl = df_train['is_attack'].values
X_test_bl  = df_test[feature_cols].values;  y_test_bl  = df_test['is_attack'].values

rf = RFBaseline(random_state=42); rf.fit(X_train_bl, y_train_bl)
kitsune = KitsuneBaseline(random_state=42)
kitsune.fit(X_train_bl[y_train_bl == 0],
            X_val=df_val[feature_cols].values, y_val=df_val['is_attack'].values)

exp1 = run_experiment_1(detector, test_windows, {'RF': rf, 'Kitsune': kitsune},
                        cfg, X_test=X_test_bl, y_test=y_test_bl, label_map=label_map)
print("✅ Exp 1 saved")

# ── 5. Exp 2: White-box ────────────────────────────────────────────────
print("\nSTEP 5/8: Experiment 2 — White-box attack (~15-30 min)...")
from experiments.exp2_whitebox import run_experiment_2
exp2 = run_experiment_2(detector, test_windows, cfg)
print("🔬", exp2['theorem_validation']['summary'])

# ── 6. Exp 3: Black-box ────────────────────────────────────────────────
print("\nSTEP 6/8: Experiment 3 — Black-box attack...")
from experiments.exp3_blackbox import run_experiment_3
exp3 = run_experiment_3(detector, train_windows, test_windows, cfg)

# ── 7. Exp 4: Ablation ─────────────────────────────────────────────────
print("\nSTEP 7/10: Experiment 4 — Ablation study...")
from experiments.exp4_ablation import run_experiment_4
exp4 = run_experiment_4(train_windows, val_windows, test_windows, cfg,
                        full_detector=detector)

# ── 8. Exp 5: Efficiency ───────────────────────────────────────────────
print("\nSTEP 8/10: Experiment 5 — Efficiency benchmarks...")
from experiments.exp5_efficiency import run_experiment_5
exp5 = run_experiment_5(detector, cfg)

# ── 9. Exp 6: Adaptive ─────────────────────────────────────────────────
print("\nSTEP 9/10: Experiment 6 — Adaptive robustness...")
from experiments.exp6_adaptive import run_experiment_6
exp6 = run_experiment_6(detector, test_windows, cfg)

# ── 10. Exp 7: Sensitivity ─────────────────────────────────────────────
print("\nSTEP 10/10: Experiment 7 — Hyperparameter sensitivity...")
from experiments.exp7_sensitivity import run_experiment_7
exp7 = run_experiment_7(train_windows, val_windows, test_windows, cfg)

print("\n" + "=" * 60)
print("🎉 ALL 7 EXPERIMENTS COMPLETE!")
print("=" * 60)
```

### Cell 4 — View Results & Generate Figures

```python
import pandas as pd
from evaluation.plotter import plot_ablation_bars, plot_efficiency_table, plot_wasserstein_vs_delta

print("=" * 60); print("RESULTS SUMMARY"); print("=" * 60)
for name in ['exp1_results','exp2_results','exp3_results','exp4_ablation','exp5_efficiency', 'exp6_adaptive', 'exp7_sensitivity']:
    path = f'results/{name}.csv'
    if os.path.exists(path):
        print(f"\n--- {name} ---")
        print(pd.read_csv(path).to_string(index=False))

# Generate paper figures
fig_dir = cfg.FIGURES_DIR
plot_ablation_bars(exp4, f'{fig_dir}/fig_ablation')
if 'table' in exp5: plot_efficiency_table(exp5['table'], f'{fig_dir}/fig_efficiency')
if 'sweep' in exp2:
    plot_wasserstein_vs_delta({'NOTS': exp2['sweep']}, detector.epsilon_min,
                              f'{fig_dir}/fig_dr_vs_delta')

print(f"\n📊 Figures saved to: {fig_dir}")
print("   → fig_ablation.pdf, fig_efficiency.pdf, fig_dr_vs_delta.pdf")
```

---

## That's It — 4 Cells Total

| Cell | What it does | Time |
|---|---|---|
| 1 | Mount Drive + unzip + download dataset | ~3 min |
| 2 | Install libraries | ~2 min |
| 3 | Run all 7 experiments | ~45-90 min |
| 4 | View results + generate figures | ~1 min |

---

## Where Are My Results?

After running, check `My Drive/nots_nids/results/`:

```
results/
├── exp1_results.csv         <- Baseline detection rates
├── exp2_results.csv         <- White-box attack + theorem validation
├── exp3_results.csv         <- Black-box attack results
├── exp4_ablation.csv        <- Ablation study (4 variants)
├── exp5_efficiency.csv      <- Throughput benchmarks
├── exp6_adaptive.csv        <- Adaptive baseline robustness
├── exp7_sensitivity.csv     <- Hyperparameter sensitivity
└── figures/
    ├── fig_ablation.pdf     <- Paper-ready figure
    ├── fig_dr_vs_delta.pdf  <- Theorem validation figure
    └── fig_efficiency.pdf   <- Efficiency table figure
```

---

## Troubleshooting

| Problem | Solution |
|---|---|
| "Kaggle credentials not found" | Click key icon in Colab sidebar → add `KAGGLE_API_TOKEN` secret with your token (starts with `KGAT_...`). Get it from kaggle.com → Settings → API → Create New Token |
| "No CSV files found" | Dataset may be in a nested folder — the code handles this automatically |
| Session disconnects | Results save after each experiment. Reload: `detector = NOTSDetector(cfg); detector.load(cfg.RESULTS_DIR)` |
| Out of memory | Reduce `WINDOW_SIZE` in `config.py` from 500 to 200 |
| "No GPU" | Runtime → Change runtime type → T4 GPU |
