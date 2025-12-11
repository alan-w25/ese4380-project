# Sprint 1 Summary: Core Methodology Complete

## Overview
Sprint 1 focused on establishing the core methodology for regime-switching moment evolution operators and validating the approach across all datasets. **All tasks completed successfully.**

---

## Tasks Completed ✅

### 1. **Proposed Models Section** ✅
**File:** `manuscript.tex` (lines 687-760)

Added comprehensive section including:
- Mathematical formulation of regime-switching operators
- K-means clustering procedure for regime identification
- Per-regime operator learning via Ridge regression
- Multi-step forecasting and regime transitions
- Model selection criteria (choosing R)
- Stability and physical constraints discussion

**Key Equations:**
- Regime dynamics: `m_{t+1} = A^(z_t) m_t + b^(z_t) + ε_t`
- Per-regime training: Ridge regression with Frobenius norm regularization
- Regime assignment: `z_t = argmin_r ||m_t - μ_r||`

---

### 2. **Formalized Regime-Switching Specification** ✅
**Included in manuscript section above**

- Clearly distinguished K-means (hard assignment) vs HMM (probabilistic)
- Documented design choice rationale: efficiency, interpretability, stability
- Specified training procedure with mathematical detail
- Explained model selection criteria (R ∈ {1,2,3,4})

---

### 3. **Fixed Lag Feature Bug** ✅
**File:** `notebooks/ESE_5380_Regime_Models.ipynb` (cell 15)

- Completed the `fit_regime_model_lags()` function
- Added proper call to `build_lagged_features(M, p)`
- Fixed K-means clustering on current states
- Implemented companion matrix for spectral radius calculation with lags
- Function now supports arbitrary lag orders p

---

### 4. **Verified Regime Model on All Datasets** ✅
**Script:** `run_ablation_study.py`

Created comprehensive testing framework:
- Loads all 5 datasets (OU, Double-well, CIR, S&P 500, ABIDE)
- Implements complete regime-switching pipeline
- Handles edge cases (small regimes, numerical issues)
- Computes spectral radii for stability analysis
- Tracks regime distribution statistics

---

### 5. **Ran Complete Ablation Study** ✅
**Output:** `ablation_results.csv`

**Scope:**
- 5 datasets × 4 regime counts (R=1,2,3,4) × 3 horizons (h=1,5,10)
- Total: 60 model configurations

**Key Findings:**

#### Synthetic Datasets (Clean, Well-Behaved):
- **OU Process:** R=3 best for h=1 (NRMSE=0.223), but R=1 better for h≥5 due to stability
- **Double-well:** R=4 best for h=1 (NRMSE=0.164), R=1 for longer horizons
- **CIR Process:** R=1 optimal across all horizons (NRMSE=0.008-0.028)

#### Real-World Datasets (Challenging):
- **S&P 500:** R=1 consistently best (NRMSE≈0.85), all models very stable (ρ_max<0.4)
  - Market returns highly unpredictable, regime-switching doesn't help
- **ABIDE fMRI:** R=1 only stable option (NRMSE≈0.72)
  - Small sample size (176 timesteps) causes instability with R>1
  - Spectral radii explode to 9.1 with R=4

#### Critical Insight:
**Multi-regime models improve short-term accuracy but sacrifice long-term stability.**
- Models with R>1 often have ρ_max > 1 (unstable)
- Unstable regimes cause catastrophic divergence at h=10 (MSE > 1000x baseline)

---

### 6. **Generated Publication-Ready Tables** ✅
**Files:**
- `tables/regime_ablation_full.tex` (comprehensive results)
- `tables/regime_ablation_compact.tex` (summary comparison)

**Table 1: Full Ablation Results**
- All datasets × regime counts × horizons
- Metrics: MSE, RMSE, NRMSE, spectral radius, stability count
- Best NRMSE per dataset/horizon highlighted in bold

**Table 2: Compact Comparison**
- Best-performing regime count per dataset
- Easy-to-scan summary for manuscript

---

## Files Created/Modified

### New Files:
1. `run_ablation_study.py` - Complete experimental pipeline
2. `generate_results_table.py` - LaTeX table generator
3. `ablation_results.csv` - Raw numerical results
4. `tables/regime_ablation_full.tex` - Comprehensive LaTeX table
5. `tables/regime_ablation_compact.tex` - Summary LaTeX table
6. `SPRINT1_SUMMARY.md` - This file

### Modified Files:
1. `manuscript.tex` - Added ~75 lines of "Proposed Models" section
2. `notebooks/ESE_5380_Regime_Models.ipynb` - Fixed lag feature bug

---

## Integration into Manuscript

### Immediate Next Steps for Paper:
1. **Include tables in Results section:**
   ```latex
   \input{tables/regime_ablation_compact}
   ```

2. **Add discussion of key findings:**
   - Trade-off between short-term accuracy and long-term stability
   - Why R=1 dominates for real-world data
   - Spectral radius as stability diagnostic

3. **Reference the methodology:**
   - Proposed Models section is complete and publication-ready
   - Equations are numbered and can be referenced
   - Clear algorithmic description

---

## Sprint 1 Performance Metrics

**Code Quality:**
- All scripts run successfully without errors
- Proper error handling for edge cases
- Reproducible results (fixed random seeds)

**Manuscript Quality:**
- Mathematically rigorous formulation
- Clear algorithmic descriptions
- Proper LaTeX formatting

**Experimental Rigor:**
- 60 model configurations tested
- Multi-horizon evaluation (1, 5, 10 steps)
- Both synthetic and real-world validation

---

## What's Next: Sprint 2 Preview

Based on instructor feedback, Sprint 2 should address:

### High Priority:
1. **Add Limitations Section** - Discuss sensitivity to K, curse of dimensionality, regime fragmentation
2. **Restructure Introduction** - Add comparison table (instantaneous closure vs dynamic operators)
3. **Create Pipeline Diagram** - Visual flowchart of methodology
4. **Add Trajectory Visualizations** - Plot predicted vs true moment trajectories

### Medium Priority:
5. **Clarify Hankel Realizability** - Add numerical example, explain multi-step difficulty
6. **Expand evaluation** - Cross-regime generalization experiments
7. **Implement stability-constrained training** - Enforce ρ(A) < 1 during optimization

---

## Sprint 1 Status: ✅ COMPLETE

All 6 core methodology tasks completed successfully. The paper now has:
- ✅ Complete mathematical formulation
- ✅ Working implementation across all datasets
- ✅ Comprehensive experimental validation
- ✅ Publication-ready results tables

**Ready to proceed to Sprint 2 when you are.**
