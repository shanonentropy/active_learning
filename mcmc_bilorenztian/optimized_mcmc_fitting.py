"""
Optimized MCMC fitting code for pyro_odmr_gaussian.ipynb
This script contains efficiency-improved versions of the main fitting loop.
Choose the version that best fits your needs (1, 2A, 2B, or 3)
"""

import torch
import pyro
import numpy as np
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_value
from scipy.stats.mstats import mquantiles
from bilorentzian_model import model


def F_np(x_in, A, X, Amp, G1, G2):
    """Vectorized bilorentzian function for posterior prediction"""
    A_reshaped = A[None, :]
    X_reshaped = X[None, :]
    B_reshaped = A_reshaped + X_reshaped
    Amp_reshaped = Amp[None, :]
    G1_reshaped = G1[None, :]
    G2_reshaped = G2[None, :]
    x_in_reshaped = x_in[:, None]
    
    F = (Amp_reshaped) * (0.5 * G1_reshaped) / ((x_in_reshaped - A_reshaped)**2 + (0.5 * G1_reshaped)**2) \
        + (Amp_reshaped) * (0.5 * G2_reshaped) / ((x_in_reshaped - B_reshaped)**2 + (0.5 * G2_reshaped)**2)
    return F


# ============================================================================
# VERSION 1: Maximum Speed (Single Chain, No Multiprocessing)
# Expected: 40-60% faster than num_chains=2
# Trade-off: No chain diagnostics (R-hat)
# ============================================================================

def fit_mcmc_v1_fastest(x_scale, y_esr, temps, num_samples=10, warmup_steps=10):
    """
    Fastest version: Single chain execution with JIT compilation
    """
    init_vals = {
        "A": torch.tensor(50.0),
        "X": torch.tensor(8.0),
        "gamma1": torch.tensor(8.0),
        "amp": torch.tensor(3.0),
        "var": torch.tensor(0.05),
    }
    
    # Enable JIT compilation for faster gradient computation
    kernel = NUTS(model, jit_compile=True, 
                  init_strategy=init_to_value(values=init_vals),
                  ignore_jit_warnings=True, max_tree_depth=5)
    
    # Pre-compute all data tensors outside loop
    data_tensors = {}
    for j in range(y_esr.shape[1]):
        y_vals = y_esr.iloc[20:, j].values
        data_tensors[j] = (
            torch.tensor(x_scale[20:], dtype=torch.float64),
            torch.tensor(y_vals, dtype=torch.float64)
        )
    
    results = {key: [] for key in ['A', 'X', 'B', 'gamma1', 'amp', 'X_var', 'A_var', 'gamma1_var', 'amp_var']}
    
    for j in range(y_esr.shape[1]):
        print(f"\n--- Slice {j}/{y_esr.shape[1]-1} ---")
        pyro.clear_param_store()
        
        data_j = data_tensors[j]
        
        # Single chain execution
        posterior = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, 
                        num_chains=1, disable_progbar=True)
        posterior.run(data_j)
        
        hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
        
        # Collect results
        A_ = hmc_samples['A']
        X_ = hmc_samples['X']
        results['A'].append(A_.mean())
        results['X'].append(X_.mean())
        results['B'].append((A_ + X_).mean())
        results['gamma1'].append(hmc_samples['gamma1'].mean())
        results['amp'].append(hmc_samples['amp'].mean())
        results['A_var'].append(A_.var())
        results['X_var'].append(X_.var())
        results['gamma1_var'].append(hmc_samples['gamma1'].var())
        results['amp_var'].append(hmc_samples['amp'].var())
        
        pyro.clear_param_store()
    
    return results


# ============================================================================
# VERSION 2A: Balanced Speed with Diagnostics (Sequential Dual Chains)
# Expected: 30-40% faster than parallel num_chains=2
# Trade-off: Parallel processing disabled, but R-hat available
# ============================================================================

def fit_mcmc_v2a_balanced(x_scale, y_esr, temps, num_samples=10, warmup_steps=10):
    """
    Balanced version: 2 chains run sequentially in same process (no fork overhead)
    Keeps diagnostic benefits without multiprocessing overhead
    """
    init_vals_base = {
        "A": torch.tensor(50.0),
        "X": torch.tensor(8.0),
        "gamma1": torch.tensor(8.0),
        "amp": torch.tensor(3.0),
        "var": torch.tensor(0.05),
    }
    
    # JIT compiled kernel
    kernel = NUTS(model, jit_compile=True,
                  init_strategy=init_to_value(values=init_vals_base),
                  ignore_jit_warnings=True, max_tree_depth=5)
    
    # Pre-compute data tensors
    data_tensors = {}
    for j in range(y_esr.shape[1]):
        y_vals = y_esr.iloc[20:, j].values
        data_tensors[j] = (
            torch.tensor(x_scale[20:], dtype=torch.float64),
            torch.tensor(y_vals, dtype=torch.float64)
        )
    
    results = {key: [] for key in ['A', 'X', 'B', 'gamma1', 'amp', 'X_var', 'A_var', 'gamma1_var', 'amp_var']}
    
    for j in range(y_esr.shape[1]):
        print(f"\n--- Slice {j}/{y_esr.shape[1]-1} ---")
        pyro.clear_param_store()
        
        data_j = data_tensors[j]
        
        # 2 chains with NO multiprocessing (mp_context=None disables fork)
        posterior = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                        num_chains=2, mp_context=None, disable_progbar=True)
        posterior.run(data_j)
        
        hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
        
        A_ = hmc_samples['A']
        X_ = hmc_samples['X']
        results['A'].append(A_.mean())
        results['X'].append(X_.mean())
        results['B'].append((A_ + X_).mean())
        results['gamma1'].append(hmc_samples['gamma1'].mean())
        results['amp'].append(hmc_samples['amp'].mean())
        results['A_var'].append(A_.var())
        results['X_var'].append(X_.var())
        results['gamma1_var'].append(hmc_samples['gamma1'].var())
        results['amp_var'].append(hmc_samples['amp'].var())
        
        pyro.clear_param_store()
    
    return results


# ============================================================================
# VERSION 2B: Model Optimization (Requires bilorentzian_model.py modification)
# Expected: 20-25% faster (combined with other optimizations)
# Trade-off: Model code must be changed
# ============================================================================

# NOTE: This requires modifying bilorentzian_model.py to use Independent Normal
# See EFFICIENCY_ANALYSIS.md for details on the model change


# ============================================================================
# VERSION 3: Production Setup (All optimizations combined)
# Expected: 60-80% faster overall
# ============================================================================

def fit_mcmc_v3_production(x_scale, y_esr, temps, num_samples=15, warmup_steps=15):
    """
    Production version: Single chain with all optimizations
    - JIT compilation enabled
    - Reduced tree depth
    - Pre-computed data tensors
    - Optimized initialization
    """
    init_vals = {
        "A": torch.tensor(50.0),
        "X": torch.tensor(8.0),
        "gamma1": torch.tensor(8.0),
        "amp": torch.tensor(3.0),
        "var": torch.tensor(0.05),
    }
    
    # Optimized kernel: JIT compiled, smaller tree depth
    kernel = NUTS(model, jit_compile=True,
                  init_strategy=init_to_value(values=init_vals),
                  ignore_jit_warnings=True, max_tree_depth=4)
    
    # Pre-compute ALL data tensors at once
    print("Pre-computing data tensors...")
    data_tensors = {}
    for j in range(y_esr.shape[1]):
        y_vals = y_esr.iloc[20:, j].values
        data_tensors[j] = (
            torch.tensor(x_scale[20:], dtype=torch.float64),
            torch.tensor(y_vals, dtype=torch.float64)
        )
    
    results = {key: [] for key in ['A', 'X', 'B', 'gamma1', 'amp', 'X_var', 'A_var', 'gamma1_var', 'amp_var']}
    
    print(f"Starting MCMC for {y_esr.shape[1]} data slices...")
    for j in range(y_esr.shape[1]):
        print(f"  [{j+1}/{y_esr.shape[1]}] ", end="", flush=True)
        pyro.clear_param_store()
        
        data_j = data_tensors[j]
        
        # Single chain, single process, optimized
        posterior = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps,
                        num_chains=1, disable_progbar=True)
        posterior.run(data_j)
        
        hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
        
        A_ = hmc_samples['A']
        X_ = hmc_samples['X']
        results['A'].append(A_.mean())
        results['X'].append(X_.mean())
        results['B'].append((A_ + X_).mean())
        results['gamma1'].append(hmc_samples['gamma1'].mean())
        results['amp'].append(hmc_samples['amp'].mean())
        results['A_var'].append(A_.var())
        results['X_var'].append(X_.var())
        results['gamma1_var'].append(hmc_samples['gamma1'].var())
        results['amp_var'].append(hmc_samples['amp'].var())
        
        pyro.clear_param_store()
    
    print("\nDone!")
    return results


# ============================================================================
# USAGE EXAMPLE
# ============================================================================
if __name__ == "__main__":
    # After your data loading (from the notebook):
    # temps, x_scale, y_esr are all available
    
    # Choose your version:
    # results = fit_mcmc_v1_fastest(x_scale, y_esr, temps)
    # results = fit_mcmc_v2a_balanced(x_scale, y_esr, temps)
    results = fit_mcmc_v3_production(x_scale, y_esr, temps)
    
    print("See EFFICIENCY_ANALYSIS.md for detailed comparisons and recommendations")
