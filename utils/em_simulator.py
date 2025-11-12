import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats
import pandas as pd

def eu_simulator(f_t, g_t, W_t, x0, T, dt):
    """
    Simulate an SDE using the Euler-Maruyama method.
    
    Parameters:
    f_t: function - the drift function f(X_t)
    g_t: function - the diffusion function g(X_t)
    W_t: array-like - the Wiener process increments
    x0: float - initial condition
    T: float - total time to simulate
    dt: float - time step size
    
    Returns:
    X: array - simulated values of X_t at each time step
    """
    N = int(T / dt)  # number of time steps
    X = np.zeros(N + 1)
    X[0] = x0
    
    for i in range(1, N + 1):
        t = i * dt
        dW = W_t[i-1]  # Wiener increment for this time step
        X[i] = X[i-1] + f_t(X[i-1]) * dt + g_t(X[i-1]) * dW
        
    return X 

def make_ou_drift(mu, theta):
    """Return f_t(x) = theta * (mu - x) for an OU process."""
    return lambda x: theta * (mu - x)  

def make_ou_diffusion(sigma):
    """Return g_t(x) = sigma (state-independent diffusion for OU)."""
    return lambda x: sigma 

def wiener_increments(N, dt, rng=None):
    """Generate N i.i.d. Wiener increments ~ N(0, dt)."""
    rng = np.random.default_rng(rng)  
    return rng.normal(loc=0.0, scale=np.sqrt(dt), size=N)  

def generate_ou_series(mu, theta, sigma, x0, T, dt, seed=None, method="euler"):
    """
    Generate a single OU path X_t using eu simulator
    """
    N = int(T / dt)                               
    W = wiener_increments(N, dt, rng=seed)      
    f_t = make_ou_drift(mu, theta)               
    g_t = make_ou_diffusion(sigma)               
    X = eu_simulator(f_t, g_t, W, x0, T, dt)     
    return X                            

def generate_ou_paths(mu, theta, sigma, x0, T, dt, n_series, seed=None):
    """
    Generate multiple independent OU paths (shape: n_series x (N+1)).
    """
    N = int(T / dt)                                   
    rng = np.random.default_rng(seed)                 
    X_panel = np.empty((n_series, N + 1), float)      
    f_t = make_ou_drift(mu, theta)                    
    g_t = make_ou_diffusion(sigma)                    
    for i in range(n_series):                         
        W = rng.normal(0.0, np.sqrt(dt), size=N)      
        xi0 = x0                                      
        X_panel[i] = eu_simulator(f_t, g_t, W, xi0, T, dt)  
    return X_panel                                    # return all paths

def save_ou_paths_to_csv(X_panel, filename):
    """Save the generated OU paths to a CSV file."""
    df = pd.DataFrame(X_panel)
    df.to_csv(filename, index=False)