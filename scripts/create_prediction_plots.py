"""
Create actual vs. predicted plots for HMM and K-means models.
Generates visualizations for different datasets and horizons.
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.linear_model import Ridge
from sklearn.cluster import KMeans
from hmmlearn import hmm
import warnings
warnings.filterwarnings('ignore')

# ============================================================================
# Data loading functions (reuse from existing scripts)
# ============================================================================

def load_ou_process():
    """Load OU process data."""
    df = pd.read_csv("data/OU Moments.csv", header=None)
    return df.values

def load_double_well():
    """Load double-well data."""
    df = pd.read_csv("data/Double Well Moments.csv", header=None)
    return df.values

def load_cir():
    """Load CIR data."""
    df = pd.read_csv("data/Cir Moments.csv", header=None)
    return df.values

def load_sp500():
    """Load SP500 data."""
    df = pd.read_csv("data/sp500_moments.csv")
    moment_cols = [c for c in df.columns if c.startswith('m')]
    return df[moment_cols].values

def load_abide():
    """Load ABIDE data."""
    df = pd.read_csv("data/abide_healthy_moments_100.csv")
    moment_cols = [c for c in df.columns if c.startswith('m')]
    return df[moment_cols].values

# ============================================================================
# K-means model
# ============================================================================

def fit_kmeans_regime_model(M, n_regimes=3, alpha=1e-3, random_state=0):
    """Fit K-means based regime-switching model."""
    M = np.asarray(M)
    T, K = M.shape

    kmeans = KMeans(n_clusters=n_regimes, random_state=random_state, n_init=20)
    labels = kmeans.fit_predict(M)

    X = M[:-1]
    Y = M[1:]
    z = labels[:-1]

    operators = []
    for r in range(n_regimes):
        mask = (z == r)
        if np.sum(mask) < 2:
            A_r = np.zeros((K, K))
            b_r = np.mean(Y, axis=0) if len(Y) > 0 else np.zeros(K)
        else:
            X_r = X[mask]
            Y_r = Y[mask]
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(X_r, Y_r)
            A_r = reg.coef_
            b_r = reg.intercept_
        operators.append((A_r, b_r))

    return {
        "kmeans": kmeans,
        "operators": operators,
        "n_regimes": n_regimes,
        "type": "kmeans"
    }

def step_forward_kmeans(m_t, model):
    """One-step prediction using K-means."""
    kmeans = model["kmeans"]
    operators = model["operators"]

    r = kmeans.predict(m_t.reshape(1, -1))[0]
    A_r, b_r = operators[r]
    m_next = A_r @ m_t + b_r

    return m_next

# ============================================================================
# HMM model
# ============================================================================

def fit_hmm_regime_model(M, n_regimes=3, alpha=1e-3, random_state=0, n_iter=100):
    """Fit HMM-based regime-switching model."""
    M = np.asarray(M)
    T, K = M.shape

    hmm_model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        init_params="stmc",
        params="stmc"
    )
    hmm_model.fit(M)

    posteriors = hmm_model.predict_proba(M)

    X = M[:-1]
    Y = M[1:]
    gamma = posteriors[:-1]

    operators = []
    for r in range(n_regimes):
        weights = gamma[:, r]
        eff_n = np.sum(weights)

        if eff_n < 2:
            A_r = np.zeros((K, K))
            b_r = np.mean(Y, axis=0) if len(Y) > 0 else np.zeros(K)
        else:
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(X, Y, sample_weight=weights)
            A_r = reg.coef_
            b_r = reg.intercept_

        operators.append((A_r, b_r))

    return {
        "hmm_model": hmm_model,
        "operators": operators,
        "n_regimes": n_regimes,
        "type": "hmm"
    }

def step_forward_hmm(m_t, model):
    """One-step prediction using HMM soft assignment."""
    hmm_model = model["hmm_model"]
    operators = model["operators"]

    probs = hmm_model.predict_proba(m_t.reshape(1, -1))[0]

    m_next = np.zeros_like(m_t)
    for r, (A_r, b_r) in enumerate(operators):
        m_next += probs[r] * (A_r @ m_t + b_r)

    return m_next

# ============================================================================
# Multi-step prediction
# ============================================================================

def multi_step_forecast(m_init, model, horizon):
    """Multi-step forecast for h steps."""
    step_fn = step_forward_hmm if model["type"] == "hmm" else step_forward_kmeans

    m_t = m_init.copy()
    trajectory = [m_t]

    for _ in range(horizon):
        m_t = step_fn(m_t, model)
        trajectory.append(m_t)

    return np.array(trajectory)

# ============================================================================
# Plotting functions
# ============================================================================

def create_prediction_plots(dataset_name, M, model_type, n_regimes, horizon):
    """Create actual vs predicted plots for a given configuration."""
    # Split data
    split_idx = int(0.7 * len(M))
    M_train = M[:split_idx]
    M_test = M[split_idx:]

    # Train model
    print(f"Training {model_type.upper()} (R={n_regimes}) on {dataset_name}...")
    if model_type == "kmeans":
        model = fit_kmeans_regime_model(M_train, n_regimes=n_regimes)
    else:
        model = fit_hmm_regime_model(M_train, n_regimes=n_regimes)

    # Generate predictions on test set
    predictions = []
    actuals = []

    for t in range(len(M_test) - horizon):
        m_init = M_test[t]
        pred_trajectory = multi_step_forecast(m_init, model, horizon)
        m_pred = pred_trajectory[-1]  # h-step ahead prediction
        m_actual = M_test[t + horizon]

        predictions.append(m_pred)
        actuals.append(m_actual)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Create plots
    fig, axes = plt.subplots(2, 2, figsize=(12, 10))
    fig.suptitle(f'{dataset_name} - {model_type.upper()} (R={n_regimes}, h={horizon})\nActual vs Predicted Moments',
                 fontsize=14, fontweight='bold')

    moments = ['m1', 'm2', 'm3', 'm4']

    for i, ax in enumerate(axes.flat):
        # Scatter plot
        ax.scatter(actuals[:, i], predictions[:, i], alpha=0.5, s=20)

        # Perfect prediction line
        min_val = min(actuals[:, i].min(), predictions[:, i].min())
        max_val = max(actuals[:, i].max(), predictions[:, i].max())
        ax.plot([min_val, max_val], [min_val, max_val], 'r--', linewidth=2, label='Perfect prediction')

        # Calculate RMSE
        rmse = np.sqrt(np.mean((actuals[:, i] - predictions[:, i])**2))

        ax.set_xlabel(f'Actual {moments[i]}', fontsize=11)
        ax.set_ylabel(f'Predicted {moments[i]}', fontsize=11)
        ax.set_title(f'{moments[i]} (RMSE: {rmse:.4f})', fontsize=12)
        ax.legend()
        ax.grid(True, alpha=0.3)

    plt.tight_layout()

    # Save plot
    filename = f"plots/pred_{dataset_name}_{model_type}_R{n_regimes}_h{horizon}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

def create_time_series_plot(dataset_name, M, model_type, n_regimes, horizon, num_steps=50):
    """Create time series comparison plot."""
    # Split data
    split_idx = int(0.7 * len(M))
    M_train = M[:split_idx]
    M_test = M[split_idx:]

    # Train model
    print(f"Training {model_type.upper()} (R={n_regimes}) on {dataset_name} for time series...")
    if model_type == "kmeans":
        model = fit_kmeans_regime_model(M_train, n_regimes=n_regimes)
    else:
        model = fit_hmm_regime_model(M_train, n_regimes=n_regimes)

    # Generate predictions
    predictions = []
    actuals = []

    for t in range(min(num_steps, len(M_test) - horizon)):
        m_init = M_test[t]
        pred_trajectory = multi_step_forecast(m_init, model, horizon)
        m_pred = pred_trajectory[-1]
        m_actual = M_test[t + horizon]

        predictions.append(m_pred)
        actuals.append(m_actual)

    predictions = np.array(predictions)
    actuals = np.array(actuals)

    # Create time series plot
    fig, axes = plt.subplots(4, 1, figsize=(14, 10))
    fig.suptitle(f'{dataset_name} - {model_type.upper()} (R={n_regimes}, h={horizon})\nTime Series: Actual vs Predicted',
                 fontsize=14, fontweight='bold')

    moments = ['m1', 'm2', 'm3', 'm4']
    time_steps = np.arange(len(actuals))

    for i, ax in enumerate(axes):
        ax.plot(time_steps, actuals[:, i], 'b-', linewidth=2, label='Actual', alpha=0.7)
        ax.plot(time_steps, predictions[:, i], 'r--', linewidth=2, label='Predicted', alpha=0.7)

        ax.set_ylabel(moments[i], fontsize=11)
        ax.legend(loc='upper right')
        ax.grid(True, alpha=0.3)

        if i == len(axes) - 1:
            ax.set_xlabel('Time step', fontsize=11)

    plt.tight_layout()

    # Save plot
    filename = f"plots/timeseries_{dataset_name}_{model_type}_R{n_regimes}_h{horizon}.png"
    plt.savefig(filename, dpi=150, bbox_inches='tight')
    print(f"Saved: {filename}")
    plt.close()

# ============================================================================
# Main execution
# ============================================================================

if __name__ == "__main__":
    import os
    os.makedirs("plots", exist_ok=True)

    # Dataset configurations
    datasets = {
        'OU': load_ou_process(),
        'Double-well': load_double_well(),
        'CIR': load_cir(),
        'SP500': load_sp500(),
        'ABIDE': load_abide()
    }

    # Load best configurations from ablation results
    kmeans_results = pd.read_csv("ablation_results.csv")
    hmm_results = pd.read_csv("ablation_results_hmm.csv")

    # Standardize names
    kmeans_results['Dataset'] = kmeans_results['Dataset'].replace({
        'Double Well': 'Double-well',
        'Cir': 'CIR'
    })
    hmm_results['dataset'] = hmm_results['dataset'].replace({
        'Double Well': 'Double-well',
        'Cir': 'CIR'
    })

    # Create plots for selected configurations
    configurations = [
        ('OU', 'kmeans', 1, 1),
        ('OU', 'hmm', 1, 1),
        ('Double-well', 'kmeans', 1, 1),
        ('Double-well', 'hmm', 1, 1),
        ('CIR', 'kmeans', 1, 5),
        ('CIR', 'hmm', 1, 5),
        ('SP500', 'kmeans', 1, 1),
        ('SP500', 'hmm', 2, 10),
        ('ABIDE', 'kmeans', 1, 1),
        ('ABIDE', 'hmm', 1, 1),
    ]

    print("=" * 80)
    print("Creating Actual vs Predicted Plots")
    print("=" * 80)

    for dataset_name, model_type, n_regimes, horizon in configurations:
        print(f"\n{dataset_name} - {model_type.upper()} (R={n_regimes}, h={horizon})")
        print("-" * 80)

        M = datasets[dataset_name]

        # Create scatter plots
        create_prediction_plots(dataset_name, M, model_type, n_regimes, horizon)

        # Create time series plots
        create_time_series_plot(dataset_name, M, model_type, n_regimes, horizon)

    print("\n" + "=" * 80)
    print("All plots created successfully!")
    print("=" * 80)
