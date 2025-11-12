import numpy as np 
import matplotlib.pyplot as plt 
import scipy.stats as stats

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