"""
Bilorentzian Batch Models for Pyro MCMC Inference

This module defines Pyro probabilistic models for:
1. Batch calibration: Learn shared parameters (alpha, beta, gamma1, amp) 
   over multiple observations at different temperatures
2. Batch prediction: Infer temperatures for multiple observations using 
   fixed learned calibration parameters

Models use vectorized computation with Pyro plates for efficient MCMC sampling.

Author: ODMR Analysis Pipeline
Date: 2026-05-11
"""

import torch
import pyro
import pyro.distributions as dist
from typing import Dict, Optional


# Bilorentzian parameters (from domain knowledge)
OFFSET_AB = 8.420054307219287  # Fixed offset between peak centers B - A (GHz)
GAMMA_DEFAULT = 8.020510711744828  # Default Lorentzian width (GHz)

# Prior specifications
PRIORS = {
    'alpha': {'mean': -0.077, 'std': 0.01},  # Temperature sensitivity (K^-1)
    'beta': {'mean': 70., 'std': 10.0},  # Temperature intercept (GHz)
    'gamma1': {'mean': 8.02, 'std': 1.0},  # Lorentzian width (GHz)
    'amp': {'mean': 3., 'std': 0.25},  # Amplitude
    'var': {'scale': 0.1},  # Observation noise variance (HalfNormal scale)
    'T': {'low': 250., 'high': 330.},  # Temperature bounds (K)
}


def batch_calibration_model(batch_data: Dict[str, torch.Tensor]) -> None:
    """
    Pyro model for learning bilorentzian calibration parameters over a batch.
    
    This model learns shared parameters (alpha, beta, gamma1, amp, var) that apply
    to all observations in the batch. Each observation is at a different but known
    temperature, allowing the model to learn how the bilorentzian peak centers
    vary with temperature.
    
    The bilorentzian function is:
        F(x) = amp * (0.5*gamma1) / ((x - A)^2 + (0.5*gamma1)^2)
             + amp * (0.5*gamma1) / ((x - B)^2 + (0.5*gamma1)^2)
    
    where:
        A = T * alpha + beta  (temperature-dependent peak 1 center)
        B = A + OFFSET_AB     (fixed offset for peak 2 center)
    
    Args:
        batch_data: Dict with keys:
            - 'x': frequency axis [num_frequencies] (double tensor)
            - 'temperatures': temperature for each observation [batch_size] (double tensor)
            - 'y_observations': observed spectra [batch_size, num_frequencies] (double tensor)
    
    Notes:
        - Parameters (alpha, beta, gamma1, amp, var) are sampled OUTSIDE the plate
          (shared across all observations in batch)
        - Observations and computed functions are sampled INSIDE the plate
          (per-observation likelihood)
        - Broadcasting strategy: x [num_freqs, 1] × A [1, batch_size] → [num_freqs, batch_size]
    
    Example:
        >>> from batch_processor import BatchMCMCProcessor
        >>> processor = BatchMCMCProcessor(x_axis, temperatures, y_obs, batch_size=3)
        >>> batch_container = processor.get_batch_container([0, 1, 2])
        >>> batch_data = batch_container.get_calibration_data_dict()
        >>> # kernel = NUTS(batch_calibration_model, ...)
        >>> # mcmc = MCMC(kernel, num_samples=2000)
        >>> # mcmc.run(batch_data)
    """
    # Extract data from dict
    x = batch_data['x']  # [num_frequencies]
    temperatures = batch_data['temperatures']  # [batch_size]
    y_observations = batch_data['y_observations']  # [batch_size, num_frequencies]
    
    batch_size = temperatures.shape[0]
    num_freqs = x.shape[0]
    
    # =========================================================================
    # Sample shared parameters (OUTSIDE plate - one sample per parameter)
    # =========================================================================
    
    # Temperature sensitivity parameter (slope of A vs T)
    alpha = pyro.sample("alpha", 
                       dist.Normal(PRIORS['alpha']['mean'], 
                                 PRIORS['alpha']['std']))
    
    # Temperature intercept parameter (intercept of A vs T)
    beta = pyro.sample("beta",
                      dist.Normal(PRIORS['beta']['mean'],
                                PRIORS['beta']['std']))
    
    # Lorentzian width (same for both peaks)
    gamma1 = pyro.sample("gamma1",
                        dist.Normal(PRIORS['gamma1']['mean'],
                                  PRIORS['gamma1']['std']))
    
    # Amplitude (same for both peaks)
    amp = pyro.sample("amp",
                     dist.Normal(PRIORS['amp']['mean'],
                               PRIORS['amp']['std']))
    
    # Observation noise variance
    var = pyro.sample("var",
                     dist.HalfNormal(PRIORS['var']['scale']))
    
    # =========================================================================
    # Vectorized computation over batch (INSIDE plate)
    # =========================================================================
    
    with pyro.plate("batch", batch_size):
        # Compute peak centers for each observation
        # A = T * alpha + beta, shape: [batch_size]
        A = temperatures * alpha + beta
        B = A + OFFSET_AB
        
        # Reshape for broadcasting with frequency axis
        # x: [num_freqs] → [num_freqs, 1]
        # A, B: [batch_size] → [1, batch_size]
        x_reshaped = x.unsqueeze(-1)  # [num_freqs, 1]
        A_reshaped = A.unsqueeze(0)   # [1, batch_size]
        B_reshaped = B.unsqueeze(0)   # [1, batch_size]
        
        # Compute bilorentzian function
        # Result shape: [num_freqs, batch_size]
        gamma_half = 0.5 * gamma1
        
        peak1 = (amp * gamma_half) / ((x_reshaped - A_reshaped)**2 + gamma_half**2)
        peak2 = (amp * gamma_half) / ((x_reshaped - B_reshaped)**2 + gamma_half**2)
        F = peak1 + peak2  # [num_freqs, batch_size]
        
        # Transpose for observation model: [batch_size, num_freqs]
        F_mean = F.transpose(0, 1)
        
        # Observation model: each observation is a multivariate normal
        # with mean F and covariance var * I
        covariance = var * torch.eye(num_freqs, dtype=x.dtype, device=x.device)
        
        pyro.sample("obs",
                   dist.MultivariateNormal(F_mean, covariance),
                   obs=y_observations)


def batch_prediction_model(batch_data: Dict[str, torch.Tensor],
                          learned_params: Dict[str, torch.Tensor]) -> None:
    """
    Pyro model for predicting temperatures using fixed calibration parameters.
    
    This model uses parameters learned from batch_calibration_model to infer
    temperatures for new observations where the temperature is unknown.
    
    Args:
        batch_data: Dict with keys:
            - 'x': frequency axis [num_frequencies]
            - 'y_observations': observed spectra [batch_size, num_frequencies]
        
        learned_params: Dict with keys:
            - 'alpha': learned temperature sensitivity [scalar or distribution]
            - 'beta': learned temperature intercept [scalar or distribution]
            - 'gamma1': learned Lorentzian width [scalar or distribution]
            - 'amp': learned amplitude [scalar or distribution]
            - Optional: 'var': learned noise variance [scalar or distribution]
    
    Notes:
        - alpha, beta, gamma1, amp are FIXED to learned values
        - Temperature T is sampled per observation (INSIDE plate)
        - var can be fixed or re-learned depending on learned_params
    
    Example:
        >>> # After calibration
        >>> cal_results = processor.run_batch_calibration(...)
        >>> learned = cal_results['posterior_means']
        >>> learned_params = {k: torch.tensor(v) for k, v in learned.items()}
        >>> 
        >>> # Prediction on new observations
        >>> pred_results = processor.run_batch_prediction(
        ...     batch_indices=[3, 4, 5],
        ...     model_fn=batch_prediction_model,
        ...     learned_params=learned_params
        ... )
    """
    # Extract data from dict
    x = batch_data['x']  # [num_frequencies]
    y_observations = batch_data['y_observations']  # [batch_size, num_frequencies]
    
    batch_size = y_observations.shape[0]
    num_freqs = x.shape[0]
    
    # =========================================================================
    # Extract learned parameters (FIXED, not sampled)
    # =========================================================================
    
    alpha = learned_params['alpha']  # Scalar
    beta = learned_params['beta']    # Scalar
    gamma1 = learned_params['gamma1']  # Scalar
    amp = learned_params['amp']      # Scalar
    
    # =========================================================================
    # Sample variance (can be fixed or re-learned)
    # =========================================================================
    
    if 'var' in learned_params:
        # Use learned variance (fix it)
        var = learned_params['var']
    else:
        # Re-learn variance
        var = pyro.sample("var",
                         dist.HalfNormal(PRIORS['var']['scale']))
    
    # =========================================================================
    # Sample temperatures per observation (INSIDE plate)
    # =========================================================================
    
    with pyro.plate("batch", batch_size):
        # Sample temperature for each observation
        T = pyro.sample("T",
                       dist.Uniform(PRIORS['T']['low'],
                                  PRIORS['T']['high']))  # [batch_size]
        
        # Compute peak centers using fixed parameters and sampled temperatures
        A = T * alpha + beta  # [batch_size]
        B = A + OFFSET_AB
        
        # Reshape for broadcasting
        x_reshaped = x.unsqueeze(-1)  # [num_freqs, 1]
        A_reshaped = A.unsqueeze(0)   # [1, batch_size]
        B_reshaped = B.unsqueeze(0)   # [1, batch_size]
        
        # Compute bilorentzian function
        gamma_half = 0.5 * gamma1
        
        peak1 = (amp * gamma_half) / ((x_reshaped - A_reshaped)**2 + gamma_half**2)
        peak2 = (amp * gamma_half) / ((x_reshaped - B_reshaped)**2 + gamma_half**2)
        F = peak1 + peak2  # [num_freqs, batch_size]
        
        # Transpose for observation model: [batch_size, num_freqs]
        F_mean = F.transpose(0, 1)
        
        # Observation model
        covariance = var * torch.eye(num_freqs, dtype=x.dtype, device=x.device)
        
        pyro.sample("obs",
                   dist.MultivariateNormal(F_mean, covariance),
                   obs=y_observations)


def bilorentzian_function(x: torch.Tensor,
                         A: torch.Tensor,
                         B: torch.Tensor,
                         amp: torch.Tensor,
                         gamma1: torch.Tensor) -> torch.Tensor:
    """
    Compute bilorentzian function values.
    
    Utility function (not a Pyro model) for computing the bilorentzian function
    for given parameters. Useful for posterior predictive checks and visualization.
    
    Args:
        x: Frequency axis [num_frequencies]
        A: Peak 1 center [... shape]
        B: Peak 2 center [... shape]
        amp: Amplitude [... shape]
        gamma1: Lorentzian width [... shape]
    
    Returns:
        Function values with shape [num_frequencies, *batch_shape]
    
    Example:
        >>> x_scale = torch.linspace(0, 100, 500)
        >>> A = torch.tensor(45.)
        >>> B = torch.tensor(53.)
        >>> F = bilorentzian_function(x_scale, A, B, amp=3.0, gamma1=8.0)
        >>> plt.plot(x_scale.numpy(), F.numpy())
    """
    gamma_half = 0.5 * gamma1
    
    # Handle broadcasting
    x_reshaped = x.unsqueeze(-1)  # [num_freqs, 1]
    
    peak1 = (amp * gamma_half) / ((x_reshaped - A)**2 + gamma_half**2)
    peak2 = (amp * gamma_half) / ((x_reshaped - B)**2 + gamma_half**2)
    
    return peak1 + peak2


def get_default_priors() -> Dict:
    """
    Get default prior specifications for all parameters.
    
    Returns:
        Dict with prior hyperparameters for each parameter
    
    Example:
        >>> priors = get_default_priors()
        >>> print(priors['alpha'])
        {'mean': -0.077, 'std': 0.01}
    """
    return PRIORS.copy()


def set_priors(new_priors: Dict) -> None:
    """
    Update prior specifications globally.
    
    Args:
        new_priors: Dict with updated prior values
    
    Example:
        >>> set_priors({
        ...     'alpha': {'mean': -0.08, 'std': 0.015},
        ...     'beta': {'mean': 72., 'std': 15.0}
        ... })
    
    Notes:
        - This modifies the module-level PRIORS dict
        - Affects all subsequent model calls
    """
    global PRIORS
    PRIORS.update(new_priors)
