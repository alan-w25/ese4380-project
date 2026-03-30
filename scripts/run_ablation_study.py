"""
Comprehensive Ablation Study: Regime-Switching Moment Evolution Operators
Sprint 1 - Task 4 & 5: Verify models and run ablation across regime counts

This script runs regime-switching models with R=1,2,3,4 regimes on all datasets:
- Synthetic: OU, Double-well, CIR
- Real-world: S&P 500, ABIDE fMRI

Outputs comprehensive results table for manuscript.
"""

import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from sklearn.linear_model import Ridge
import numpy.linalg as la
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =======================
# REGIME MODEL FUNCTIONS
# =======================

def fit_regime_model(M, n_regimes=3, alpha=1e-3, random_state=0):
    """
    Fit a K-means + per-regime linear operator model on a moment time series.
    """
    M = np.asarray(M)
    T, K = M.shape

    # one-step pairs
    X = M[:-1]   # m_t
    Y = M[1:]    # m_{t+1}

    # K-means clustering on current state m_t
    kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
    regime_labels = kmeans.fit_predict(X)

    # per-regime linear operators
    operators = []
    for r in range(n_regimes):
        idx = np.where(regime_labels == r)[0]
        X_r, Y_r = X[idx], Y[idx]

        if len(idx) < 2:  # not enough samples
            print(f"Warning: Regime {r} has only {len(idx)} samples")
            # use global mean as fallback
            A_r = np.zeros((K, K))
            b_r = np.mean(Y, axis=0) if len(Y) > 0 else np.zeros(K)
        else:
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(X_r, Y_r)
            A_r = reg.coef_       # (K, K)
            b_r = reg.intercept_  # (K,)

        operators.append((A_r, b_r))

    spectral_radii = [max(abs(la.eigvals(A))) for (A, _) in operators]

    model = {
        "kmeans": kmeans,
        "operators": operators,
        "regime_labels": regime_labels,
        "spectral_radii": spectral_radii,
        "n_regimes": n_regimes,
        "regime_counts": [np.sum(regime_labels == r) for r in range(n_regimes)]
    }
    return model


def step_forward(m_t, model):
    """One-step prediction given current moment vector m_t."""
    kmeans = model["kmeans"]
    operators = model["operators"]

    regime = kmeans.predict(m_t.reshape(1, -1))[0]
    A_r, b_r = operators[regime]
    m_next = A_r @ m_t + b_r
    return m_next, regime


def rollout(M, model, h=10, start_idx=0):
    """Closed-loop rollout for horizon h."""
    M = np.asarray(M)
    m_hat = M[start_idx].copy()
    preds = [m_hat]
    regimes = []

    for _ in range(h):
        m_hat, r = step_forward(m_hat, model)
        preds.append(m_hat)
        regimes.append(r)

    return np.array(preds), np.array(regimes)


def multi_horizon_metrics(M, model, horizons=[1, 5, 10], n_starts=50, seed=0, train_frac=0.7):
    """
    Compute MSE, RMSE, NRMSE for multiple forecast horizons on test set.
    """
    M = np.asarray(M)
    T, K = M.shape
    T_train = int(train_frac * T)

    # Only evaluate on test set
    rng = np.random.default_rng(seed)

    results = []
    for h in horizons:
        # Sample starting points from test set
        max_start = T - h - 1
        min_start = T_train

        if max_start <= min_start:
            continue

        starts = rng.integers(low=min_start, high=max_start, size=n_starts)

        errs = []
        for s in starts:
            try:
                preds, _ = rollout(M, model, h=h, start_idx=s)
                true_seg = M[s:s+h+1]
                # Only count error at horizon h (not intermediate steps)
                err = np.mean((preds[h] - true_seg[h])**2)
                errs.append(err)
            except:
                continue

        if len(errs) == 0:
            continue

        mse = float(np.mean(errs))
        rmse = np.sqrt(mse)

        # NRMSE: normalize by std of test set
        test_std = np.std(M[T_train:])
        nrmse = rmse / (test_std + 1e-8)

        results.append({
            "horizon": h,
            "MSE": mse,
            "RMSE": rmse,
            "NRMSE": nrmse
        })

    return pd.DataFrame(results)


# =======================
# DATA LOADING
# =======================

def load_all_datasets(data_dir):
    """Load all moment datasets."""
    datasets = {}

    # Synthetic datasets
    for name in ['OU Moments', 'Double Well Moments', 'Cir Moments']:
        path = data_dir / f"{name}.csv"
        if path.exists():
            df = pd.read_csv(path, header=None)
            datasets[name.replace(' Moments', '')] = df.values
            print(f"Loaded {name}: shape {df.values.shape}")

    # Real-world datasets
    sp_path = data_dir / "sp500_moments.csv"
    if sp_path.exists():
        df = pd.read_csv(sp_path)
        moment_cols = [c for c in df.columns if c.startswith('m')]
        if len(moment_cols) > 0:
            datasets['SP500'] = df[moment_cols].values
            print(f"Loaded S&P 500: shape {datasets['SP500'].shape}")

    # ABIDE (combined healthy + AD, or use healthy only)
    abide_healthy_path = data_dir / "abide_healthy_moments_100.csv"
    if abide_healthy_path.exists():
        df = pd.read_csv(abide_healthy_path)
        moment_cols = [c for c in df.columns if c.startswith('m')]
        if len(moment_cols) > 0:
            datasets['ABIDE'] = df[moment_cols].values
            print(f"Loaded ABIDE: shape {datasets['ABIDE'].shape}")

    return datasets


# =======================
# ABLATION STUDY
# =======================

def run_ablation_study(datasets, regime_counts=[1, 2, 3, 4], alpha=1e-3, train_frac=0.7):
    """
    Run ablation study across different numbers of regimes.
    """
    all_results = []

    for dataset_name, M in datasets.items():
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset_name}")
        print(f"{'='*60}")

        T, K = M.shape
        T_train = int(train_frac * T)
        M_train = M[:T_train]

        for n_regimes in regime_counts:
            print(f"\n  Fitting {n_regimes} regime(s)...")

            try:
                # Fit model on training data
                model = fit_regime_model(M_train, n_regimes=n_regimes,
                                        alpha=alpha, random_state=42)

                # Get spectral radii
                max_spectral = max(model['spectral_radii'])
                min_spectral = min(model['spectral_radii'])
                stable_count = sum(1 for rho in model['spectral_radii'] if rho < 1.0)

                # Get regime distribution
                regime_counts_dist = model['regime_counts']
                min_regime_size = min(regime_counts_dist)
                max_regime_size = max(regime_counts_dist)

                # Compute multi-horizon metrics on full data (will use test set)
                metrics_df = multi_horizon_metrics(M, model,
                                                   horizons=[1, 5, 10],
                                                   n_starts=50,
                                                   train_frac=train_frac)

                # Add metadata to each row
                for _, row in metrics_df.iterrows():
                    result = {
                        'Dataset': dataset_name,
                        'n_regimes': n_regimes,
                        'horizon': row['horizon'],
                        'MSE': row['MSE'],
                        'RMSE': row['RMSE'],
                        'NRMSE': row['NRMSE'],
                        'max_spectral_radius': max_spectral,
                        'min_spectral_radius': min_spectral,
                        'n_stable_regimes': stable_count,
                        'min_regime_size': min_regime_size,
                        'max_regime_size': max_regime_size
                    }
                    all_results.append(result)

                print(f"    h=1:  MSE={metrics_df[metrics_df.horizon==1]['MSE'].values[0]:.3e}, "
                      f"NRMSE={metrics_df[metrics_df.horizon==1]['NRMSE'].values[0]:.3f}")
                print(f"    Spectral radii: {model['spectral_radii']}")
                print(f"    Regime sizes: {regime_counts_dist}")

            except Exception as e:
                print(f"    Error: {e}")
                continue

    return pd.DataFrame(all_results)


# =======================
# MAIN EXECUTION
# =======================

if __name__ == "__main__":
    # Set paths
    project_dir = Path(__file__).parent
    data_dir = project_dir / "data"

    print("="*60)
    print("REGIME-SWITCHING ABLATION STUDY")
    print("="*60)

    # Load datasets
    print("\nLoading datasets...")
    datasets = load_all_datasets(data_dir)

    if len(datasets) == 0:
        print("ERROR: No datasets loaded!")
        exit(1)

    print(f"\nLoaded {len(datasets)} datasets: {list(datasets.keys())}")

    # Run ablation study
    print("\n" + "="*60)
    print("Running ablation study (R=1,2,3,4)...")
    print("="*60)

    results_df = run_ablation_study(datasets, regime_counts=[1, 2, 3, 4])

    # Save results
    output_path = project_dir / "ablation_results.csv"
    results_df.to_csv(output_path, index=False)
    print(f"\n{'='*60}")
    print(f"Results saved to: {output_path}")
    print(f"{'='*60}")

    # Display summary
    print("\n" + "="*60)
    print("SUMMARY: Best NRMSE by Dataset")
    print("="*60)

    for dataset in results_df['Dataset'].unique():
        df_subset = results_df[results_df['Dataset'] == dataset]
        for h in [1, 5, 10]:
            h_subset = df_subset[df_subset['horizon'] == h]
            if len(h_subset) > 0:
                best_idx = h_subset['NRMSE'].idxmin()
                best_row = h_subset.loc[best_idx]
                print(f"{dataset:12s} h={h:2d}: "
                      f"R={best_row['n_regimes']:.0f} → "
                      f"NRMSE={best_row['NRMSE']:.4f} "
                      f"(ρ_max={best_row['max_spectral_radius']:.3f})")

    print("\nAblation study complete!")
