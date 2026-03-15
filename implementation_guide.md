You are implementing the complete experimental codebase for a research paper titled:

**"Nash-Optimized Topological Shields: A Game-Theoretic Persistent Homology Framework for Adversarially Robust Network Intrusion Detection"**

Target venue: IEEE Transactions on Information Forensics and Security (T-IFS).

You will produce a complete, modular, well-commented Python codebase organized as
a research repository. Every file must be production-quality — clean imports, type
hints, docstrings, and reproducible random seeds throughout. The code must run
on Google Colab (free T4 GPU) and Kaggle (free P100) with no paid dependencies.

---

## Repository structure to create

```
nots_nids/
├── README.md
├── requirements.txt
├── config.py                  # All hyperparameters and paths in one place
├── data/
│   ├── download.py            # Dataset download scripts
│   └── README.md              # Instructions for CICIDS-2017, UNSW-NB15, NSL-KDD
├── preprocessing/
│   ├── __init__.py
│   ├── loader.py              # Load and merge CSV files per dataset
│   ├── cleaner.py             # Handle NaN, Inf, duplicates, label normalization
│   ├── scaler.py              # MinMaxScaler fit on train, applied to val/test
│   └── windowing.py           # Sliding window → point clouds in R^d
├── topology/
│   ├── __init__.py
│   ├── umap_reducer.py        # UMAP dimensionality reduction R^80 → R^5
│   ├── ripser_filtration.py   # Vietoris-Rips filtration via Ripser
│   ├── persistence.py         # Persistence diagram utilities and Betti numbers
│   └── wasserstein.py         # Wasserstein distance via POT library
├── game_theory/
│   ├── __init__.py
│   ├── nash_solver.py         # LP formulation and Nash equilibrium S* via CVXPY
│   └── sampler.py             # Runtime feature subset sampling from S*
├── detector/
│   ├── __init__.py
│   ├── baseline_builder.py    # Construct D_norm from benign training windows
│   ├── nots_detector.py       # Main NOTS detection class (full pipeline)
│   └── adaptive_baseline.py   # EWM baseline update with poisoning resistance
├── adversarial/
│   ├── __init__.py
│   ├── whitebox.py            # FGSM L-inf perturbation (white-box attack Exp 2)
│   └── blackbox.py            # Surrogate RF + transfer attack (black-box Exp 3)
├── baselines/
│   ├── __init__.py
│   ├── kitsune_baseline.py    # Kitsune anomaly detector reimplementation
│   ├── rf_baseline.py         # Random Forest NIDS baseline
│   └── lucid_baseline.py      # LUCID CNN baseline (simplified)
├── experiments/
│   ├── __init__.py
│   ├── exp1_baseline.py       # Experiment 1: baseline detection (no attack)
│   ├── exp2_whitebox.py       # Experiment 2: white-box adversarial sweep
│   ├── exp3_blackbox.py       # Experiment 3: black-box transfer attack
│   ├── exp4_ablation.py       # Experiment 4: ablation study (4 variants)
│   └── exp5_efficiency.py     # Experiment 5: throughput and latency benchmarks
├── evaluation/
│   ├── __init__.py
│   ├── metrics.py             # DR, FPR, F1, precision, recall per attack class
│   ├── theorem_validator.py   # Check empirical DR vs analytical e_min bound
│   └── plotter.py             # All figures for the paper (matplotlib)
├── notebooks/
│   ├── 00_setup_colab.ipynb   # Colab setup: Drive mount, installs, GPU check
│   ├── 01_data_exploration.ipynb
│   ├── 02_topology_demo.ipynb # Visual demo of persistence diagrams
│   ├── 03_run_experiments.ipynb  # Orchestrates all 5 experiments
│   └── 04_paper_figures.ipynb # Reproduces all tables and figures for paper
└── results/
    └── .gitkeep
```

---

## Technical specification — implement exactly as described

### 1. config.py

Create a single `Config` dataclass with the following fields and defaults:

```
RANDOM_SEED = 42
WINDOW_SIZE = 500          # N flows per point cloud
UMAP_N_COMPONENTS = 5      # R^80 → R^5
UMAP_N_NEIGHBORS = 15
UMAP_MIN_DIST = 0.1
RIPSER_MAX_DIM = 1         # Compute beta_0 and beta_1 only
RIPSER_MAX_EDGE = 2.0      # Maximum filtration radius
WASSERSTEIN_P = 2          # 2-Wasserstein distance
DELTA_VALUES = [0.01, 0.05, 0.10]   # Adversarial perturbation sweep
ALPHA_EWM = 0.05           # Baseline EWM update rate
TAU_MULTIPLIER = 1.0       # tau = e_min * TAU_MULTIPLIER
TRAIN_RATIO = 0.60
VAL_RATIO = 0.20
TEST_RATIO = 0.20
BATCH_SIZE_WINDOWS = 50    # Process this many windows before saving checkpoint
RESULTS_DIR = "results/"
FIGURES_DIR = "results/figures/"

CICIDS_LABEL_COL = " Label"
CICIDS_BENIGN_LABEL = "BENIGN"

FEATURE_COLS = None        # Set to None = use all 78 numeric columns auto-detected
N_FEATURE_SUBSETS = 20     # Number of feature subsets in Nash LP
```

### 2. preprocessing/loader.py

Implement `load_cicids2017(data_dir)`:
- CICIDS-2017 is provided as multiple CSVs (one per day/attack type). Load all CSVs
  from data_dir, concatenate them into a single DataFrame.
- Strip whitespace from all column names.
- Keep the Label column. All other columns must be numeric.
- Print a class distribution table showing count and percentage per label.
- Return: `(df, label_col)` tuple.

Implement `load_unsw_nb15(data_dir)`:
- UNSW-NB15 has train and test CSV files plus a features CSV describing column names.
- Load train and test separately, concatenate, return with label column = 'attack_cat'
  (string) and 'label' (binary 0/1).

Implement `load_nsl_kdd(data_dir)`:
- NSL-KDD has KDDTrain+.txt and KDDTest+.txt.
- Map attack types to 4 categories: DoS, Probe, R2L, U2R. Normal = BENIGN.
- Return standard DataFrame format.

All loaders must:
- Handle encoding errors gracefully (some CICIDS files have encoding issues).
- Drop rows where Label is NaN.
- Log the shape and class distribution after loading.

### 3. preprocessing/cleaner.py

Implement `clean_dataframe(df, label_col)`:
- Replace `inf` and `-inf` with NaN.
- Drop columns with more than 50% NaN values.
- Fill remaining NaN with column median.
- Drop exact duplicate rows.
- Remove columns with zero variance.
- For CICIDS-2017: normalize label strings (strip whitespace, title case).
- Return cleaned df and list of dropped columns.

Implement `encode_labels(df, label_col)`:
- Map string labels to integers. BENIGN = 0, all attacks = 1+ (multiclass).
- Also create binary column 'is_attack' (0=benign, 1=any attack).
- Return df with new columns 'label_int' and 'is_attack', plus the label_map dict.

### 4. preprocessing/scaler.py

Implement `fit_scaler(X_train)` and `apply_scaler(X, scaler)`:
- Use MinMaxScaler from sklearn.
- CRITICAL: fit ONLY on training data, apply to val and test.
- Save scaler to disk with joblib for reproducibility.
- Return scaled array and scaler object.

### 5. preprocessing/windowing.py

Implement `create_windows(df, feature_cols, label_col, window_size, step_size=None)`:
- step_size defaults to window_size // 2 (50% overlap).
- Each window is a (window_size × len(feature_cols)) matrix = one point cloud.
- Window label = 'attack' if ANY flow in the window has is_attack=1, else 'benign'.
- Window attack_type = the majority attack type in the window (for multiclass eval).
- Return list of dicts: {'points': np.array shape (N, d), 'label': str, 'attack_type': str}

### 6. topology/umap_reducer.py

Implement `UMAPReducer` class:
- `__init__(n_components, n_neighbors, min_dist, random_state, use_gpu=False)`
- `fit(X)`: fit UMAP on training point clouds. If use_gpu=True and cuml is available,
  use `cuml.manifold.UMAP`, else fall back to standard `umap.UMAP`.
- `transform(X)`: project new point clouds into the learned low-dimensional space.
- `fit_transform(X)`: combined.
- Save and load methods using joblib.
- Add try/except: if cuml import fails, log a warning and fall back to CPU UMAP silently.

### 7. topology/ripser_filtration.py

Implement `compute_persistence_diagram(point_cloud, max_dim=1, max_edge=2.0)`:
- Use `ripser.ripser(points, maxdim=max_dim, thresh=max_edge)`.
- Return a dict: {'dgm_0': array of (birth, death) for H0,
                  'dgm_1': array of (birth, death) for H1}
- Filter out points where death == inf (replace with max finite death + small epsilon).
- Handle edge case: if point_cloud has fewer than 5 points, return empty diagrams with
  a warning.

Implement `compute_betti_numbers(dgm, epsilon)`:
- At filtration scale epsilon, count how many H0 and H1 features are alive.
- beta_0 = number of (b,d) pairs in dgm_0 where b <= epsilon < d
- beta_1 = number of (b,d) pairs in dgm_1 where b <= epsilon < d
- Return (beta_0, beta_1).

### 8. topology/wasserstein.py

Implement `wasserstein_distance(dgm1, dgm2, p=2)`:
- Use `persim.wasserstein` from the persim library.
- Compute separately for H0 and H1 diagrams.
- Return combined distance: W_total = W_H0 + W_H1.
- Handle empty diagram edge case: if either diagram is empty, return a large
  sentinel distance (e.g., 999.0) so the detector fires.
- Add timing decorator to log computation time.

Implement `compute_wasserstein_trajectory(windows, D_norm, p=2)`:
- Given a list of live windows and D_norm, return array of W distances.
- Used for plotting the Wasserstein distance over time for paper figures.

### 9. game_theory/nash_solver.py

Implement `solve_nash_equilibrium(feature_importance_matrix, n_subsets)`:

This is the core game theory component. Implement as follows:

- `feature_importance_matrix` is shape (n_subsets × n_features): for each candidate
  feature subset omega_i, the detection payoff U(omega_i, H*) against the worst-case
  attacker. Estimate this from validation data as: for subset omega_i, run NOTS with
  only those features and record mean Wasserstein distance on attack windows.
- Formulate the Nash LP using CVXPY:
  ```
  Variables: p (probability vector over subsets, length n_subsets)
  Objective: maximize v (the guaranteed minimum payoff)
  Constraints:
    for each attacker response j: sum_i p[i] * U[i,j] >= v
    sum(p) == 1
    p >= 0
  ```
- Solve with `cp.Problem(cp.Maximize(v), constraints).solve(solver=cp.ECOS)`.
- Return: `{'S_star': p.value,  # Nash-optimal distribution over subsets
             'epsilon_min': v.value,  # Guaranteed detection floor
             'subset_indices': list of feature index lists per subset}`
- Log the Nash equilibrium value (epsilon_min) — this becomes tau in the detector.

### 10. game_theory/sampler.py

Implement `NashSampler` class:
- `__init__(S_star, subset_indices, random_state=42)`
- `sample()`: draw one feature subset index according to S_star distribution.
  Return the list of feature indices for that subset.
- `sample_batch(n)`: draw n subsets (for parallel processing).
- This runs at O(1) at inference time — just a np.random.choice call.

### 11. detector/baseline_builder.py

Implement `build_baseline(benign_windows, umap_reducer, n_baseline_windows=50)`:
- Take first n_baseline_windows benign windows from training set.
- For each: project to R^5 via UMAP, compute persistence diagram via Ripser.
- Average persistence diagrams: use the Frechet mean approximation.
  (Practical approximation: collect all (birth, death) points from all baseline
  diagrams, cluster with k-means where k = expected number of persistent features,
  return cluster centers as D_norm.)
- Return D_norm as a dict {'dgm_0': array, 'dgm_1': array}.
- Save D_norm to disk as .npy files.

### 12. detector/nots_detector.py

Implement `NOTSDetector` class — the main integration:

```python
class NOTSDetector:
    def __init__(self, config: Config):
        self.config = config
        self.umap_reducer = None
        self.nash_sampler = None
        self.D_norm = None
        self.tau = None           # Detection threshold = epsilon_min

    def fit(self, train_windows, val_windows):
        # Step 1: Fit UMAP on all training point clouds
        # Step 2: Build D_norm from benign training windows
        # Step 3: On val set, estimate feature payoff matrix U
        # Step 4: Solve Nash LP → get S_star and epsilon_min
        # Step 5: Set self.tau = epsilon_min * config.TAU_MULTIPLIER
        # Step 6: Initialize NashSampler with S_star
        # Log all steps with timing

    def detect(self, window):
        # Step 1: Sample feature subset omega from NashSampler
        # Step 2: Project window[:,omega] through UMAP
        # Step 3: Compute persistence diagram D_live
        # Step 4: Compute W = wasserstein_distance(D_norm, D_live)
        # Step 5: Return {'alert': W >= self.tau, 'W': W, 'omega': omega}

    def detect_batch(self, windows):
        # Run detect() on a list of windows, return list of result dicts

    def save(self, path):
        # Serialize all components to path using joblib

    def load(self, path):
        # Load from path
```

### 13. detector/adaptive_baseline.py

Implement `AdaptiveBaseline` class:
- `__init__(D_norm_initial, alpha=0.05, tau)`
- `update(D_live, W_current)`:
  - If W_current < tau / 2: update baseline.
    D_norm_new = blend(D_norm, D_live, alpha).
    Blending: merge the two diagram point sets, re-cluster to original size.
  - If W_current >= tau / 2: do NOT update (poisoning resistance). Log skip.
  - Return updated D_norm.
- Track update history (timestamp, W, updated: bool) for analysis.

### 14. adversarial/whitebox.py

Implement `fgsm_perturb_flow(x, epsilon, wasserstein_grad_fn)`:

The white-box attack for Exp 2. Since the Wasserstein distance is not easily
differentiable w.r.t. individual flow features, use a numerical gradient approach:

```python
def compute_numerical_gradient(x, epsilon_w, detector, feature_mask):
    """
    Numerically estimate gradient of W w.r.t. x for features in feature_mask.
    For each feature i in feature_mask:
        grad[i] = (W(x + h*e_i) - W(x - h*e_i)) / (2h)
    where h = 1e-4
    """

def fgsm_attack(x, delta_max, detector, n_steps=10, step_size=None):
    """
    Iterative FGSM (PGD-style) L-inf attack.
    step_size defaults to delta_max / n_steps.
    At each step: x = clip(x + step_size * sign(grad), x_original - delta_max,
                               x_original + delta_max)
    Also clip to [0, 1] since features are MinMax scaled.
    Return perturbed x and the W values at each step (for convergence plot).
    """
```

Implement `run_whitebox_sweep(test_windows, detector, delta_values)`:
- For each delta in delta_values, perturb all attack windows and run detection.
- Return dict: {delta: {'DR': float, 'FPR': float, 'W_values': array}}

### 15. adversarial/blackbox.py

Implement black-box transfer attack for Exp 3:

```python
def train_surrogate_model(X_train, y_train):
    """Train a Random Forest surrogate on the same training data."""

def craft_blackbox_adversarial(X_attack, surrogate_model, delta_max):
    """
    Use the surrogate RF's feature importances to craft perturbations.
    Perturb features with highest importance in the direction that
    reduces the surrogate's attack probability.
    Return perturbed X.
    """

def run_blackbox_experiment(test_windows, detector, surrogate_model, delta_max=0.10):
    """
    Craft adversarial flows using surrogate, package into windows,
    run NOTS detector, return DR and FPR.
    """
```

### 16. baselines/rf_baseline.py

Implement `RFBaseline` class:
- Standard Random Forest classifier on flow-level features (not windows).
- `fit(X_train, y_train)`: train with n_estimators=100, class_weight='balanced'.
- `predict(X_test)`: return binary predictions.
- `evaluate(X_test, y_test)`: return dict with DR, FPR, F1 per class.
- This is your primary comparison baseline.

### 17. baselines/kitsune_baseline.py

Implement a simplified Kitsune-style anomaly detector:
- Kitsune uses an ensemble of Autoencoders. Implement a simplified version:
  - Train one Autoencoder per attack class cluster (use K-Means to find clusters
    in benign traffic, k=10).
  - At test time, reconstruction error = anomaly score.
  - Alert if max reconstruction error across ensemble > threshold (set on val set).
- Use PyTorch for the autoencoders (available free on Colab/Kaggle).
- Keep architecture small: input → 32 → 16 → 8 → 16 → 32 → input.

### 18. experiments/exp1_baseline.py

```python
def run_experiment_1(detector, test_windows, baselines, config):
    """
    Experiment 1: Baseline detection with no adversarial perturbation.

    Steps:
    1. Run NOTSDetector.detect_batch(test_windows)
    2. Run each baseline on same test data
    3. Compute metrics for each method × each attack class
    4. Save results to results/exp1_results.csv
    5. Print summary table to console
    6. Return results dict for plotting

    Metrics: DR (detection rate), FPR (false positive rate),
             F1, Precision, Recall, per attack class and overall.
    """
```

### 19. experiments/exp2_whitebox.py

```python
def run_experiment_2(detector, test_windows, config):
    """
    Experiment 2: White-box adversarial attack sweep.

    Steps:
    1. For each delta in config.DELTA_VALUES:
       a. Run fgsm_attack on all attack windows
       b. Run NOTSDetector on perturbed windows
       c. Record DR, FPR, mean W, e_min comparison
    2. THEOREM VALIDATION: assert empirical DR >= epsilon_min for all delta.
       If this fails, log a CRITICAL warning — it indicates a proof gap.
    3. Save results to results/exp2_results.csv
    4. Return dict: {delta: metrics}
    """
```

### 20. experiments/exp3_blackbox.py

```python
def run_experiment_3(detector, train_windows, test_windows, config):
    """
    Experiment 3: Black-box transfer attack.

    Steps:
    1. Extract flow-level features from training windows
    2. Train surrogate RF model (blackbox.train_surrogate_model)
    3. Craft adversarial flows using surrogate at delta_max=0.10
    4. Package into windows, run NOTS detection
    5. Compare DR to Exp 1 baseline DR (measures hardness of black-box vs white-box)
    6. Save to results/exp3_results.csv
    """
```

### 21. experiments/exp4_ablation.py

```python
def run_experiment_4(train_windows, val_windows, test_windows, config):
    """
    Experiment 4: Ablation study with 4 variants.

    Variant A — Full NOTS: TDA + Nash game theory (standard detector)
    Variant B — TDA only: Use TDA with fixed feature set (no Nash sampling,
                use all features every time)
    Variant C — Game theory only: Use Nash sampling but replace TDA with
                statistical anomaly score (Mahalanobis distance on raw features)
    Variant D — Plain ML: Standard Random Forest (same as rf_baseline)

    For each variant, run on the same test_windows.
    Report DR and FPR for each. The contribution of each component =
    (Full NOTS DR) - (variant without that component DR).

    Save to results/exp4_ablation.csv
    """
```

### 22. experiments/exp5_efficiency.py

```python
def run_experiment_5(detector, config):
    """
    Experiment 5: Computational efficiency benchmarks.

    Benchmark each pipeline stage separately:
    - UMAP projection: time per window at N=100, 500, 1000
    - Ripser filtration: time per window at N=100, 500, 1000
    - Wasserstein distance: time per window
    - Nash sampling: time per sample (should be microseconds)
    - End-to-end: flows per second at each N

    Also benchmark:
    - Memory usage (tracemalloc) at each N
    - Approximate vs exact Wasserstein (compare accuracy vs speed)

    Use timeit with n_repeats=100 for stable estimates.
    Save raw benchmark data to results/exp5_efficiency.csv
    Print a formatted table: rows = N values, cols = stages.
    """
```

### 23. evaluation/metrics.py

Implement the following functions:

```python
def compute_detection_rate(y_true, y_pred):
    """DR = TP / (TP + FN) for binary classification."""

def compute_fpr(y_true, y_pred):
    """FPR = FP / (FP + TN)."""

def compute_per_class_metrics(y_true, y_pred, attack_types):
    """
    For each unique attack type, compute DR, FPR, F1, Precision, Recall.
    Return a pandas DataFrame with attack types as rows.
    """

def compute_full_metrics(results_list, windows):
    """
    Given list of detector output dicts and original windows,
    compute all metrics. Return comprehensive metrics dict.
    """
```

### 24. evaluation/theorem_validator.py

```python
def validate_epsilon_min_bound(exp2_results, epsilon_min):
    """
    The most important validation in the paper.

    For each delta value, check that empirical DR >= epsilon_min.
    If DR < epsilon_min for any delta, this is a THEOREM VIOLATION.

    Return:
    {
      'bound_holds': bool,  # True if DR >= epsilon_min for ALL delta values
      'violations': list,   # List of (delta, empirical_DR, epsilon_min) tuples where bound fails
      'margin': float,      # min(empirical_DR - epsilon_min) across all deltas
      'summary': str        # Human-readable verdict
    }

    Also compute: for what maximum delta does the bound still hold?
    This is the practical security claim of the paper.
    """
```

### 25. evaluation/plotter.py

Generate all figures needed for the IEEE T-IFS paper. Use matplotlib with
`plt.style.use('seaborn-v0_8-whitegrid')` and set `rcParams` for IEEE
double-column format (figure width = 3.5 inches for single column,
7.0 inches for double column, font size 9pt).

Implement:

```python
def plot_detection_rate_comparison(exp1_results, save_path):
    """
    Grouped bar chart: x-axis = attack classes, y-axis = DR.
    One bar group per method (NOTS, Kitsune, RF, LUCID).
    Include error bars if multiple runs available.
    Figure: double-column width, 3.5 inch height.
    """

def plot_wasserstein_vs_delta(exp2_results, epsilon_min, save_path):
    """
    Line plot: x-axis = delta values, y-axis = Detection Rate.
    One line per method. Horizontal dashed line at y = epsilon_min.
    Annotate: "Theorem bound (e_min)" on the dashed line.
    This is the key theorem validation figure.
    """

def plot_persistence_diagrams(D_norm, D_live_benign, D_live_attack, save_path):
    """
    3-panel figure showing:
    Left: D_norm (baseline)
    Center: D_live for benign traffic (should be close to D_norm)
    Right: D_live for attack traffic (should be far from D_norm)
    Plot H0 as circles, H1 as triangles. Include diagonal line.
    """

def plot_wasserstein_timeseries(W_trajectory, labels, tau, save_path):
    """
    Time series: x-axis = window index, y-axis = W distance.
    Color background red where label=attack, white where benign.
    Horizontal dashed line at tau. Shows how the detector triggers.
    """

def plot_ablation_bars(exp4_results, save_path):
    """
    Horizontal bar chart: y-axis = variant name, x-axis = DR.
    Highlight Full NOTS bar. Show contribution of each component.
    """

def plot_efficiency_table(exp5_results, save_path):
    """
    Render the efficiency results as a formatted table figure
    suitable for inclusion in the paper.
    """

def plot_roc_curves(all_results, save_path):
    """
    ROC curves for all methods on the same axes.
    Include AUC values in the legend.
    """
```

### 26. notebooks/00_setup_colab.ipynb

Create a notebook with these cells:

Cell 1 — Mount Drive and check GPU:
```python
from google.colab import drive
drive.mount('/content/drive')
import torch
print(f"GPU: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'None — switch to GPU runtime'}")
import subprocess
result = subprocess.run(['nvidia-smi'], capture_output=True, text=True)
print(result.stdout)
```

Cell 2 — Install all dependencies:
```bash
%%bash
pip install -q ripser persim pot umap-learn cvxpy scikit-learn pandas numpy \
    matplotlib seaborn joblib torch torchvision tqdm memory-profiler
# Try cuML (GPU UMAP) — only works on Colab with T4/A100
pip install -q cudf-cu11 cuml-cu11 --extra-index-url=https://pypi.nvidia.com 2>/dev/null || \
    echo "cuML not available, using CPU UMAP"
```

Cell 3 — Clone/setup repo:
```python
import sys
sys.path.insert(0, '/content/drive/MyDrive/nots_nids')
from config import Config
cfg = Config()
print("Config loaded. Results will save to:", cfg.RESULTS_DIR)
```

Cell 4 — Quick sanity check (run a tiny persistence diagram):
```python
import numpy as np
from ripser import ripser
from persim import wasserstein
points = np.random.rand(50, 5)
dgms = ripser(points, maxdim=1)['dgms']
print(f"H0 features: {len(dgms[0])}, H1 features: {len(dgms[1])}")
print("Ripser working correctly.")
```

### 27. notebooks/03_run_experiments.ipynb

Create a notebook that orchestrates all 5 experiments sequentially with
clear section headers, timing, and checkpointing. Structure:

- Section 1: Load config and data
- Section 2: Preprocess and window
- Section 3: Fit NOTS detector (fit() call)
- Section 4: Run Exp 1 (baseline)
- Section 5: Run Exp 2 (white-box) — print theorem validation verdict
- Section 6: Run Exp 3 (black-box)
- Section 7: Run Exp 4 (ablation)
- Section 8: Run Exp 5 (efficiency)
- Section 9: Generate all paper figures

After each experiment, save results to Drive immediately with:
```python
import pandas as pd
pd.DataFrame(results).to_csv(f'/content/drive/MyDrive/nots_nids/results/expN_results.csv')
print("Saved to Drive.")
```

---

## Critical implementation requirements

1. **Reproducibility**: Set `np.random.seed(42)`, `torch.manual_seed(42)`,
   `random.seed(42)` at the top of every script and notebook. Pass
   `random_state=42` to every sklearn and UMAP call.

2. **Logging**: Use Python's `logging` module (not print) in all module files.
   Notebooks may use print. Log level INFO by default, DEBUG for topology steps.

3. **Checkpointing**: Every experiment must save intermediate results after every
   BATCH_SIZE_WINDOWS windows. If the Colab session dies, the experiment must be
   resumable from the last checkpoint without restarting from scratch.

4. **Empty diagram handling**: Ripser sometimes returns empty diagrams for very
   uniform point clouds. Every function that receives a persistence diagram must
   gracefully handle the empty case without crashing.

5. **Memory management**: For large datasets (CICIDS-2017 has 2.8M rows), do NOT
   load everything into RAM at once. Use chunked loading with pandas `chunksize`
   parameter. Process and window each chunk, save window metadata, then discard chunk.

6. **The epsilon_min assertion**: In exp2_whitebox.py, after computing empirical
   DR at each delta, add:
   ```python
   if empirical_dr < epsilon_min - 0.02:  # 2% tolerance for numerical noise
       logging.critical(
           f"THEOREM BOUND VIOLATED at delta={delta}: "
           f"empirical DR={empirical_dr:.4f} < epsilon_min={epsilon_min:.4f}. "
           f"Review proof or implementation."
       )
   ```
   This is the scientific integrity check — do not remove it.

7. **Figure quality**: All figures must be saved as both `.pdf` (for LaTeX
   inclusion in paper) and `.png` at 300 DPI. Use `bbox_inches='tight'`.

8. **requirements.txt**: Pin exact versions:
   ```
   ripser==0.6.8
   persim==0.3.7
   pot==0.9.3
   umap-learn==0.5.6
   cvxpy==1.4.2
   scikit-learn==1.4.0
   pandas==2.1.4
   numpy==1.26.3
   matplotlib==3.8.2
   seaborn==0.13.1
   joblib==1.3.2
   torch==2.1.2
   tqdm==4.66.1
   memory-profiler==0.61.0
   ```

---

## What to implement first (priority order)

If you implement in this order, you can run partial experiments at each step:

1. `config.py` + `requirements.txt`
2. `preprocessing/` (all 4 files)
3. `topology/` (all 4 files)
4. `detector/baseline_builder.py` + `detector/nots_detector.py`
5. `game_theory/` (both files)
6. `evaluation/metrics.py` + `evaluation/theorem_validator.py`
7. `experiments/exp1_baseline.py`
8. `adversarial/whitebox.py` + `experiments/exp2_whitebox.py`
9. `baselines/` (all 3 files)
10. `experiments/exp3_blackbox.py` + `exp4_ablation.py` + `exp5_efficiency.py`
11. `evaluation/plotter.py`
12. All notebooks

---

## Dataset download instructions to include in data/README.md

```
CICIDS-2017 (PRIMARY — use this first):
URL: https://www.unb.ca/cic/datasets/ids-2017.html
Download: "MachineLearningCSV.zip" (pre-extracted features, ~500MB)
NOT the raw PCAPs (50GB). Extract to data/cicids2017/

UNSW-NB15 (SECONDARY):
URL: https://research.unsw.edu.au/projects/unsw-nb15-dataset
Files needed: UNSW_NB15_training-set.csv, UNSW_NB15_testing-set.csv
Extract to data/unsw_nb15/

NSL-KDD (COMPARISON BASELINE ONLY):
URL: https://www.unb.ca/cic/datasets/nsl.html
Files: KDDTrain+.txt, KDDTest+.txt
Extract to data/nsl_kdd/
```

---

Begin implementation now. Create every file in the repository structure above.
For each file, implement the full working code — not stubs, not placeholders,
not "TODO" comments. Every function must be complete and runnable.
Start with config.py, then proceed in the priority order listed above.
```