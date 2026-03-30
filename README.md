# Learning Dynamic Operators for Moment Evolution in Stochastic Systems

([Project Writeup (PDF)](./Moment%20Forecasting%20Writeup.pdf))

This project studies moment forecasting in stochastic systems by learning regime-switching linear operators for the evolution of low-order moments, rather than learning an instantaneous closure map.

## Overview
The framework models moment dynamics with regime-specific linear operators, enabling piecewise-stationary dynamics while preserving interpretability. The project evaluates forecasting performance, dynamical stability, and moment realizability across simulated SDEs, S&P 500 cross-sectional returns, and ABIDE fMRI time series.

## Methods
- Regime-switching linear operators for moment evolution
- K-means clustering and per-regime Ridge regression
- Multi-step forecasting at different horizons
- Stability analysis via spectral radius
- Realizability checks via Hankel PSD conditions

## Repository Structure
- `data/`: datasets and processed inputs
- `notebooks/`: experiments and analysis
- `utils/`: helper functions and model code
- `plots/`: generated figures
- `tables/`: result tables
- `outputs/`: saved outputs

## Results
See the writeup for the full methodology, experiments, ablations, and results.
