"""
HMM-based Regime-Switching Moment Evolution Operators
Soft regime assignment using Hidden Markov Models

This script implements and evaluates regime-switching models with SOFT assignment
via HMMs, as an alternative to hard K-means clustering.

Key differences from K-means approach:
1. Soft membership: P(z_t = r | observations) instead of hard labels
2. Temporal smoothness: Transition probabilities discourage rapid switching
3. Training: EM algorithm (Baum-Welch) for joint estimation

Outputs results for comparison with K-means baseline.
"""

import numpy as np
import pandas as pd
from hmmlearn import hmm
from sklearn.linear_model import Ridge
import numpy.linalg as la
from pathlib import Path
import warnings
warnings.filterwarnings('ignore')

# =======================
# HMM REGIME MODEL FUNCTIONS
# =======================

def fit_hmm_regime_model(M, n_regimes=3, alpha=1e-3, random_state=0, n_iter=100):
    """
    Fit an HMM-based regime-switching model with soft assignments.

    Approach:
    1. Fit Gaussian HMM on moment sequences to get regime posteriors
    2. Use soft assignments for weighted per-regime Ridge regression
    3. Learn operators A^(r), b^(r) via weighted least squares

    Parameters:
    -----------
    M : array (T, K)
        Moment time series
    n_regimes : int
        Number of regimes
    alpha : float
        Ridge regularization parameter
    random_state : int
        Random seed
    n_iter : int
        Maximum EM iterations

    Returns:
    --------
    model : dict
        - hmm_model: trained HMM
        - operators: list of (A^(r), b^(r))
        - posteriors: soft regime assignments (T-1, n_regimes)
        - spectral_radii: list of spectral radii
        - transition_matrix: regime transition probabilities
    """
    M = np.asarray(M)
    T, K = M.shape

    # Fit Gaussian HMM on moment vectors
    # Using "full" covariance to capture moment dependencies
    hmm_model = hmm.GaussianHMM(
        n_components=n_regimes,
        covariance_type="full",
        n_iter=n_iter,
        random_state=random_state,
        init_params="stmc",  # initialize starts, transitions, means, covs
        params="stmc"  # update all parameters
    )

    # Fit HMM
    hmm_model.fit(M)

    # Get posterior probabilities: P(z_t = r | m_{1:T})
    posteriors = hmm_model.predict_proba(M)  # (T, n_regimes)

    # Build one-step prediction pairs
    X = M[:-1]   # m_t
    Y = M[1:]    # m_{t+1}
    gamma = posteriors[:-1]  # posteriors for X (T-1, n_regimes)

    # Fit per-regime operators using soft assignments
    operators = []
    for r in range(n_regimes):
        # Weighted Ridge regression: weights = P(z_t = r)
        weights = gamma[:, r]  # (T-1,)

        # Check effective sample size
        eff_n = np.sum(weights)

        if eff_n < 2:
            print(f"Warning: Regime {r} has effective sample size {eff_n:.1f}")
            # Fallback to global mean
            A_r = np.zeros((K, K))
            b_r = np.mean(Y, axis=0) if len(Y) > 0 else np.zeros(K)
        else:
            # Weighted Ridge: minimize sum_t w_t ||Y_t - A X_t - b||^2 + alpha ||A||^2
            # Use sklearn with sample_weight
            reg = Ridge(alpha=alpha, fit_intercept=True)
            reg.fit(X, Y, sample_weight=weights)
            A_r = reg.coef_       # (K, K)
            b_r = reg.intercept_  # (K,)

        operators.append((A_r, b_r))

    # Compute spectral radii
    spectral_radii = [max(abs(la.eigvals(A))) for (A, _) in operators]

    # Get transition matrix
    transition_matrix = hmm_model.transmat_

    # Compute effective regime counts (sum of posterior probabilities)
    regime_counts = np.sum(gamma, axis=0)

    model = {
        "hmm_model": hmm_model,
        "operators": operators,
        "posteriors": gamma,  # (T-1, n_regimes)
        "spectral_radii": spectral_radii,
        "transition_matrix": transition_matrix,
        "n_regimes": n_regimes,
        "regime_counts": regime_counts,
        "type": "hmm"
    }

    return model


def step_forward_hmm(m_t, model, return_probs=False):
    """
    One-step prediction using soft HMM assignment.

    Prediction: m_{t+1} = sum_r P(z_t=r | m_t) * (A^(r) m_t + b^(r))

    Parameters:
    -----------
    m_t : array (K,)
        Current moment vector
    model : dict
        HMM regime model
    return_probs : bool
        If True, also return regime probabilities

    Returns:
    --------
    m_next : array (K,)
        Predicted next moment
    (optional) probs : array (n_regimes,)
        Regime probabilities
    """
    hmm_model = model["hmm_model"]
    operators = model["operators"]

    # Get posterior probabilities for current state
    # Note: HMM predict_proba expects (n_samples, n_features)
    probs = hmm_model.predict_proba(m_t.reshape(1, -1))[0]  # (n_regimes,)

    # Soft prediction: weighted average over regimes
    m_next = np.zeros_like(m_t)
    for r, (A_r, b_r) in enumerate(operators):
        m_next += probs[r] * (A_r @ m_t + b_r)

    if return_probs:
        return m_next, probs
    return m_next


def rollout_hmm(M, model, h=10, start_idx=0):
    """
    Closed-loop HMM rollout for horizon h.

    Returns:
    --------
    preds : array (h+1, K)
        Predicted trajectory including initial state
    regime_probs : array (h, n_regimes)
        Soft regime probabilities at each step
    """
    M = np.asarray(M)
    m_hat = M[start_idx].copy()
    preds = [m_hat]
    regime_probs = []

    for _ in range(h):
        m_hat, probs = step_forward_hmm(m_hat, model, return_probs=True)
        preds.append(m_hat)
        regime_probs.append(probs)

    return np.array(preds), np.array(regime_probs)


def multi_horizon_metrics_hmm(M, model, horizons=[1, 5, 10], n_starts=50, seed=0, train_frac=0.7):
    """
    Compute MSE, RMSE, NRMSE for HMM model at multiple forecast horizons.

    Similar to K-means version but uses soft predictions.
    """
    M = np.asarray(M)
    T, K = M.shape

    # Train/test split
    train_end = int(T * train_frac)
    test_start = train_end

    results = {}

    np.random.seed(seed)

    for h in horizons:
        mse_list = []

        # Sample random starting points in test set
        max_start = T - h - 1
        if max_start <= test_start:
            print(f"Warning: Not enough test data for horizon {h}")
            continue

        start_indices = np.random.randint(test_start, max_start, size=n_starts)

        for start_idx in start_indices:
            # Rollout from start_idx for h steps
            preds, _ = rollout_hmm(M, model, h=h, start_idx=start_idx)

            # Compare predictions to ground truth
            true_traj = M[start_idx:start_idx+h+1]  # (h+1, K)

            # MSE over trajectory (excluding initial state)
            mse = np.mean((preds[1:] - true_traj[1:])**2)
            mse_list.append(mse)

        # Aggregate metrics
        mse_avg = np.mean(mse_list)
        rmse_avg = np.sqrt(mse_avg)

        # NRMSE: normalize by std of test set
        test_std = np.std(M[test_start:])
        nrmse = rmse_avg / test_std if test_std > 0 else np.inf

        results[h] = {
            'mse': mse_avg,
            'rmse': rmse_avg,
            'nrmse': nrmse
        }

    return results


# =======================
# DATA LOADING
# =======================

def load_dataset(name):
    """Load moment data for a given dataset."""
    data_dir = Path("data")

    if name == "OU":
        # Synthetic datasets have no header
        df = pd.read_csv(data_dir / "OU Moments.csv", header=None)
        M = df.values
    elif name == "Double-well":
        # DoubleWell has header
        df = pd.read_csv(data_dir / "DoubleWell_moments.csv")
        moment_cols = [c for c in df.columns if c.startswith('m')]
        M = df[moment_cols].values
    elif name == "CIR":
        df = pd.read_csv(data_dir / "Cir Moments.csv", header=None)
        M = df.values
    elif name == "SP500":
        df = pd.read_csv(data_dir / "sp500_moments.csv")
        moment_cols = [c for c in df.columns if c.startswith('m')]
        M = df[moment_cols].values
    elif name == "ABIDE":
        df = pd.read_csv(data_dir / "abide_healthy_moments_100.csv")
        moment_cols = [c for c in df.columns if c.startswith('m')]
        M = df[moment_cols].values
    else:
        raise ValueError(f"Unknown dataset: {name}")

    return M


# =======================
# ABLATION STUDY
# =======================

def run_hmm_ablation():
    """
    Run ablation study with HMM regime-switching models.

    Tests R=1,2,3,4 regimes on all datasets at horizons h=1,5,10.
    """
    datasets = ["OU", "Double-well", "CIR", "SP500", "ABIDE"]
    regime_counts = [1, 2, 3, 4]
    horizons = [1, 5, 10]

    results = []

    for dataset in datasets:
        print(f"\n{'='*60}")
        print(f"Dataset: {dataset}")
        print(f"{'='*60}")

        # Load data
        M = load_dataset(dataset)
        T, K = M.shape
        print(f"Shape: T={T}, K={K}")

        # Split
        train_end = int(T * 0.7)
        M_train = M[:train_end]

        for R in regime_counts:
            print(f"\n  R={R} regimes:")

            try:
                # Fit HMM model on training data
                model = fit_hmm_regime_model(
                    M_train,
                    n_regimes=R,
                    alpha=1e-3,
                    random_state=42,
                    n_iter=100
                )

                # Compute metrics on full dataset
                metrics = multi_horizon_metrics_hmm(
                    M, model,
                    horizons=horizons,
                    n_starts=50,
                    seed=42,
                    train_frac=0.7
                )

                # Extract spectral radius
                rho_max = max(model['spectral_radii'])

                # Extract transition matrix stability
                trans_mat = model['transition_matrix']
                trans_entropy = -np.sum(trans_mat * np.log(trans_mat + 1e-10), axis=1).mean()

                # Regime counts
                regime_counts_eff = model['regime_counts']
                min_regime_size = regime_counts_eff.min()

                print(f"    ρ_max = {rho_max:.3f}")
                print(f"    Transition entropy = {trans_entropy:.3f}")
                print(f"    Min regime size = {min_regime_size:.1f}")

                # Store results for each horizon
                for h in horizons:
                    if h in metrics:
                        results.append({
                            'dataset': dataset,
                            'R': R,
                            'horizon': h,
                            'mse': metrics[h]['mse'],
                            'rmse': metrics[h]['rmse'],
                            'nrmse': metrics[h]['nrmse'],
                            'rho_max': rho_max,
                            'transition_entropy': trans_entropy,
                            'min_regime_size': min_regime_size,
                            'type': 'HMM'
                        })
                        print(f"    h={h}: NRMSE={metrics[h]['nrmse']:.4f}")

            except Exception as e:
                print(f"    ERROR: {e}")
                continue

    # Save results
    df_results = pd.DataFrame(results)
    df_results.to_csv("ablation_results_hmm.csv", index=False)
    print(f"\n{'='*60}")
    print(f"HMM ablation complete. Results saved to ablation_results_hmm.csv")
    print(f"Total experiments: {len(df_results)}")
    print(f"{'='*60}")

    return df_results


if __name__ == "__main__":
    print("HMM-based Regime-Switching Ablation Study")
    print("==========================================\n")

    # Check if hmmlearn is available
    try:
        import hmmlearn
        print(f"hmmlearn version: {hmmlearn.__version__}")
    except ImportError:
        print("ERROR: hmmlearn not installed. Install with: pip install hmmlearn")
        exit(1)

    # Run ablation
    df_results = run_hmm_ablation()

    # Display summary
    print("\n\nSummary Statistics:")
    print(df_results.groupby(['dataset', 'R'])[['nrmse', 'rho_max']].mean())
