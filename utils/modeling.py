import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge
from sklearn.preprocessing import PolynomialFeatures
import matplotlib.pyplot as plt

def schatten_Lk_normalize(moment_df, moment_cols, moment_orders, train_frac=0.7, eps=1e-8):
    """Normalize moments by their L_k norm over the training set."""
    T = len(moment_df)
    T_train = int(train_frac * T)

    norms = {}
    normed = moment_df.copy()

    for col, k in zip(moment_cols, moment_orders):
        v = normed[col].values.astype(float)
        train_v = v[:T_train]
        # Avoid k=0 issues if standardizing non-moment data; using abs sum for stability
        Lk = np.power(np.sum(np.abs(train_v) ** k), 1.0 / k) if k != 0 else 1.0
        if Lk < eps:
            Lk = 1.0
        norms[col] = Lk
        normed[col] = v / Lk

    return normed, norms

def compute_multi_horizon_metrics(M_true, preds, T_train, horizons=(1, 5, 10)):
    """Compute MSE, RMSE, NRMSE for specific forecast horizons."""
    M_true = np.asarray(M_true)
    preds  = np.asarray(preds)
    T, k = M_true.shape
    L, k_pred = preds.shape
    
    rows = []
    for h in horizons:
        start_j = h - 1
        if start_j >= L: continue

        js = np.arange(start_j, L)
        t_targets = T_train + 1 + js
        
        mask = t_targets < T
        js = js[mask]
        t_targets = t_targets[mask]
        
        if len(js) == 0: continue

        Y_hat  = preds[js]
        Y_true = M_true[t_targets]

        mse   = ((Y_true - Y_hat) ** 2).mean()
        rmse  = np.sqrt(mse)
        std_y = Y_true.std()
        nrmse = rmse / (std_y + 1e-8)

        rows.append({"horizon": h, "MSE": mse, "RMSE": rmse, "NRMSE": nrmse})

    return pd.DataFrame(rows)

# ================= VAR(2) Model =================

def fit_var2_ridge(M_norm, ridge_alpha=1e-2, train_frac=0.7):
    T, k = M_norm.shape
    T_train = int(train_frac * T)
    
    X_train, Y_train = [], []
    for t in range(1, T_train - 1):
        feat = np.concatenate([M_norm[t], M_norm[t - 1]]) 
        Y_train.append(M_norm[t + 1])
        X_train.append(feat)

    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(X_train, Y_train)
    return model.coef_, T_train

def recursive_forecast_var2(M_norm, A, T_train):
    T, k = M_norm.shape
    preds = []
    m_prev = M_norm[T_train - 1]
    m_curr = M_norm[T_train]

    for t in range(T_train, T - 1):
        feat = np.concatenate([m_curr, m_prev])
        y_pred = A @ feat
        preds.append(y_pred)
        m_prev, m_curr = m_curr, y_pred

    return np.array(preds)

def run_var_baseline(df_moments, moment_cols, name="VAR(2)", train_frac=0.7, ridge_alpha=1e-2, debug=False, real_world=False):
    print(f"\n===== Running {name} Baseline =====")
    moment_orders = list(range(1, len(moment_cols) + 1))
    
    # Normalize
    if real_world:
        norm_df, norms = robust_generalized_mean_normalize(df_moments, moment_cols, moment_orders, train_frac)
    else:
        norm_df, norms = schatten_Lk_normalize(df_moments, moment_cols, moment_orders, train_frac)
        
    M_norm = norm_df[moment_cols].values
    M_true = df_moments[moment_cols].values
    
    if debug:
        print(f"DEBUG: M_true shape: {M_true.shape}, M_norm shape: {M_norm.shape}")
        print(f"DEBUG: M_norm head:\n{M_norm[:5]}")
        print(f"DEBUG: M_norm min: {M_norm.min():.4f}, max: {M_norm.max():.4f}")

    # Fit & Forecast
    A, T_train = fit_var2_ridge(M_norm, train_frac=train_frac, ridge_alpha=ridge_alpha)
    preds_norm = recursive_forecast_var2(M_norm, A, T_train)
    
    if debug:
        print(f"DEBUG: preds_norm shape: {preds_norm.shape}")
        if preds_norm.size > 0:
             print(f"DEBUG: preds_norm min: {preds_norm.min():.4f}, max: {preds_norm.max():.4f}")
    
    # Un-normalize
    preds_true = np.zeros_like(preds_norm)
    for j, col in enumerate(moment_cols):
        preds_true[:, j] = preds_norm[:, j] * norms[col]
        
    # Metrics
    return compute_multi_horizon_metrics(M_true, preds_true, T_train)

# ================= Polynomial Ridge Model =================

def fit_poly_baseline(M_norm, degree=2, ridge_alpha=1.0, train_frac=0.7):
    T, k = M_norm.shape
    T_train = int(train_frac * T)
    
    X_train_raw = M_norm[:T_train-1]
    Y_train     = M_norm[1:T_train]
    
    # Scaling to avoid overflow
    scale_vec = np.max(np.abs(X_train_raw), axis=0)
    scale_vec[scale_vec == 0] = 1.0
    
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    X_train = poly.fit_transform(X_train_raw / scale_vec)
    
    model = Ridge(alpha=ridge_alpha, fit_intercept=False)
    model.fit(X_train, Y_train)
    
    return model, poly, T_train, scale_vec

def recursive_forecast_poly(M_norm, model, poly, T_train, scale_vec):
    T, k = M_norm.shape
    preds = []
    prev = M_norm[T_train-1].copy()
    
    for t in range(T_train, T - 1):
        x_poly = poly.transform((prev / scale_vec).reshape(1, -1))
        next_pred = model.predict(x_poly)[0]
        preds.append(next_pred)
        prev = next_pred
        
    return np.array(preds)
        
def run_poly_baseline(df_moments, moment_cols, name="Poly Ridge", train_frac=0.7, degree=2, ridge_alpha=1.0, debug=False, real_world=False):
    print(f"\n===== Running {name} Baseline =====")
    moment_orders = list(range(1, len(moment_cols) + 1))
    
    if real_world:
        norm_df, norms = robust_generalized_mean_normalize(df_moments, moment_cols, moment_orders, train_frac)
    else:
        norm_df, norms = schatten_Lk_normalize(df_moments, moment_cols, moment_orders, train_frac)
        
    M_norm = norm_df[moment_cols].values
    M_true = df_moments[moment_cols].values
    
    if debug:
        print(f"DEBUG: M_true shape: {M_true.shape}, M_norm shape: {M_norm.shape}")
        print(f"DEBUG: M_norm head:\n{M_norm[:5]}")
        print(f"DEBUG: M_norm min: {M_norm.min():.4f}, max: {M_norm.max():.4f}")

    model, poly, T_train, scale_vec = fit_poly_baseline(M_norm, train_frac=train_frac, degree=degree, ridge_alpha=ridge_alpha)
    preds_norm = recursive_forecast_poly(M_norm, model, poly, T_train, scale_vec)
    
    if debug:
        print(f"DEBUG: preds_norm shape: {preds_norm.shape}")
        if preds_norm.size > 0:
             print(f"DEBUG: preds_norm min: {preds_norm.min():.4f}, max: {preds_norm.max():.4f}")

    preds_true = np.zeros_like(preds_norm)
    for j, col in enumerate(moment_cols):
        preds_true[:, j] = preds_norm[:, j] * norms[col]

    return compute_multi_horizon_metrics(M_true, preds_true, T_train)
    
def robust_generalized_mean_normalize(moment_df, moment_cols, moment_orders, train_frac=0.7, eps=1e-8):
    """
    Normalize moments using the Generalized Mean to ensure comparable scales.
    Scale Factor s_k = ( (1/T) * sum(|m|^k) )^(1/k)
    """
    T = len(moment_df)
    T_train = int(train_frac * T)

    norms = {}
    normed = moment_df.copy()

    for col, k in zip(moment_cols, moment_orders):
        v = normed[col].values.astype(float)
        train_v = v[:T_train]
        
        # --- THE FIX: DIVIDE BY T_train INSIDE THE ROOT ---
        # Old (Schatten): np.power(np.sum(np.abs(train_v) ** k), 1.0 / k)
        # New (Robust):   np.power(np.mean(np.abs(train_v) ** k), 1.0 / k)
        
        # Calculate Generalized Mean (Power Mean) of the training data
        if k == 0:
            s_k = 1.0
        else:
            s_k = np.power(np.mean(np.abs(train_v) ** k), 1.0 / k)
        
        # Safety check for zeroes
        if s_k < eps:
            s_k = 1.0
            
        norms[col] = s_k
        normed[col] = v / s_k

    return normed, norms