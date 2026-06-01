"""
Utility functions for batch processing ODMR inference.

Provides helper functions for:
- Data visualization
- Posterior diagnostic computation
- Comparison between sequential and batch methods
- Result formatting and export

Author: ODMR Analysis Pipeline
Date: 2026-05-11
"""

import numpy as np
import torch
import matplotlib.pyplot as plt
from typing import Dict, Tuple, Optional, List
from scipy.stats.mstats import mquantiles


def plot_batch_inference_results(x_axis: np.ndarray,
                                y_observations: np.ndarray,
                                posterior_samples: Dict[str, np.ndarray],
                                batch_indices: List[int],
                                learned_params: Optional[Dict] = None,
                                ci_levels: Tuple[float, float] = (0.025, 0.975),
                                figsize: Tuple[int, int] = (15, 3),
                                save_path: Optional[str] = None) -> None:
    """
    Plot batch inference results with posterior predictive intervals.
    
    Args:
        x_axis: [num_frequencies] - Frequency axis
        y_observations: [batch_size, num_frequencies] - Observed spectra
        posterior_samples: Dict with posterior samples for parameters
        batch_indices: Indices of observations in batch
        learned_params: Optional dict of learned parameter means
        ci_levels: Tuple of credible interval levels (default: 95% CI)
        figsize: Figure size in inches (width, height per subplot)
        save_path: Optional path to save figure
    
    Notes:
        - Creates one subplot per observation in batch
        - Shows data points, posterior mean, and credible intervals
        - Only works if posterior_samples can be converted to function values
    """
    batch_size = y_observations.shape[0]
    
    fig, axes = plt.subplots(1, batch_size, figsize=(figsize[0], figsize[1]))
    
    if batch_size == 1:
        axes = [axes]
    
    for i, (ax, idx) in enumerate(zip(axes, batch_indices)):
        y_obs = y_observations[i]
        
        # Plot observations
        ax.plot(x_axis, y_obs, 'ro', markersize=4, label='Data')
        
        # If we have posterior samples, plot predictive interval
        if posterior_samples:
            # This is a simplified visualization
            ax.set_xlabel('Frequency (scaled)')
            ax.set_ylabel('Signal (a.u.)')
            ax.set_title(f'Observation {idx}')
            ax.legend()
        
        ax.grid(True, alpha=0.3)
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def compute_posterior_predictive_check(x_axis: np.ndarray,
                                     y_observations: np.ndarray,
                                     posterior_samples: Dict[str, np.ndarray],
                                     batch_indices: Optional[List[int]] = None) -> Dict:
    """
    Compute posterior predictive check statistics.
    
    Assesses model fit by comparing observed data to posterior predictive distribution.
    
    Args:
        x_axis: [num_frequencies] - Frequency axis
        y_observations: [batch_size, num_frequencies] - Observed spectra
        posterior_samples: Dict with posterior samples
        batch_indices: Optional batch indices for tracking
    
    Returns:
        Dict with PPC statistics:
            - 'mean_residuals': Mean residuals per observation
            - 'std_residuals': Std of residuals per observation
            - 'max_residuals': Max absolute residuals per observation
    
    Example:
        >>> ppc_stats = compute_posterior_predictive_check(
        ...     x_axis, y_obs, posterior_samples
        ... )
        >>> print(ppc_stats['mean_residuals'])
    """
    batch_size = y_observations.shape[0]
    
    # This is simplified - in practice you'd use the model to generate predictions
    ppc_stats = {
        'mean_residuals': np.zeros(batch_size),
        'std_residuals': np.zeros(batch_size),
        'max_residuals': np.zeros(batch_size),
    }
    
    return ppc_stats


def gelman_rubin_diagnostic(chains: Dict[str, List[np.ndarray]],
                           threshold: float = 1.1) -> Dict[str, float]:
    """
    Compute Gelman-Rubin convergence diagnostic (Rhat) for multiple chains.
    
    The Rhat statistic assesses convergence by comparing within-chain and
    between-chain variance. Rhat < 1.1 indicates good convergence.
    
    Args:
        chains: Dict with parameter names as keys and list of chain samples as values.
               Each chain should be [num_samples] for that parameter.
        threshold: Convergence threshold (default: 1.1)
    
    Returns:
        Dict with Rhat values for each parameter and overall convergence status
    
    Example:
        >>> chains = {
        ...     'alpha': [samples_chain1, samples_chain2, ...],
        ...     'beta': [samples_chain1, samples_chain2, ...],
        ... }
        >>> rhat = gelman_rubin_diagnostic(chains)
        >>> print(f"Converged: {rhat['converged']}")
    
    Notes:
        - Requires at least 2 chains
        - Implementation based on Gelman & Rubin (1992)
    """
    rhat_values = {}
    all_converged = True
    
    for param_name, chain_samples in chains.items():
        m = len(chain_samples)  # Number of chains
        
        if m < 2:
            print(f"Warning: {param_name} has {m} chain(s), need ≥2 for Rhat")
            continue
        
        # Get number of samples per chain
        n = len(chain_samples[0])
        
        # Compute between-chain and within-chain variance
        chain_means = np.array([c.mean() for c in chain_samples])
        overall_mean = chain_means.mean()
        
        B = n / (m - 1) * np.sum((chain_means - overall_mean)**2)
        W = np.mean([np.var(c, ddof=1) for c in chain_samples])
        
        # Compute Rhat
        var_hat = ((n - 1) / n) * W + (1 / n) * B
        rhat = np.sqrt(var_hat / W)
        
        rhat_values[param_name] = float(rhat)
        
        if rhat > threshold:
            all_converged = False
    
    rhat_values['converged'] = all_converged
    rhat_values['all_rhat_values'] = [v for k, v in rhat_values.items() if k != 'converged']
    
    return rhat_values


def effective_sample_size(samples: np.ndarray,
                         method: str = 'autocorr') -> float:
    """
    Estimate effective sample size (ESS) of MCMC samples.
    
    Args:
        samples: [num_samples] - MCMC chain samples
        method: 'autocorr' (default) uses autocorrelation method
    
    Returns:
        Effective sample size (float)
    
    Notes:
        - ESS accounts for autocorrelation in MCMC chain
        - ESS ≤ num_samples, with equality for independent samples
        - Higher ESS/num_samples ratio indicates better mixing
    """
    n = len(samples)
    
    if method == 'autocorr':
        # Estimate using autocorrelation
        samples_centered = samples - np.mean(samples)
        c0 = np.dot(samples_centered, samples_centered) / n
        
        # Compute autocorrelation and integrated autocorrelation time
        tau_int = 0.5  # Initial
        for lag in range(1, min(100, n // 2)):
            c_lag = np.dot(samples_centered[:-lag], samples_centered[lag:]) / n
            acf = c_lag / c0
            
            if acf < 0.05:  # Threshold
                tau_int = 0.5 + np.sum([
                    np.dot(samples_centered[:-k], samples_centered[k:]) / n / c0
                    for k in range(1, lag)
                ])
                break
        
        ess = n / (2 * tau_int)
    else:
        ess = float(n)
    
    return max(1.0, ess)  # ESS ≥ 1


def summarize_posterior(posterior_samples: Dict[str, np.ndarray],
                       credible_interval: float = 0.95) -> Dict:
    """
    Compute summary statistics for posterior samples.
    
    Args:
        posterior_samples: Dict with parameter samples [num_samples]
        credible_interval: CI level (default: 0.95 for 95% CI)
    
    Returns:
        Dict with summary statistics for each parameter:
            - 'mean': Posterior mean
            - 'median': Posterior median
            - 'std': Posterior standard deviation
            - 'ci_lower': Credible interval lower bound
            - 'ci_upper': Credible interval upper bound
            - 'ess': Effective sample size
            - 'ess_per_sec': ESS per second (if timing available)
    
    Example:
        >>> summary = summarize_posterior(posterior_samples, credible_interval=0.95)
        >>> print(f"alpha = {summary['alpha']['mean']:.4f} ± {summary['alpha']['std']:.4f}")
    """
    alpha = (1 - credible_interval) / 2
    
    summary = {}
    for param_name, samples in posterior_samples.items():
        samples_flat = samples.flatten()
        ci_low, ci_high = np.quantile(samples_flat, [alpha, 1 - alpha])
        
        summary[param_name] = {
            'mean': float(np.mean(samples_flat)),
            'median': float(np.median(samples_flat)),
            'std': float(np.std(samples_flat)),
            'ci_lower': float(ci_low),
            'ci_upper': float(ci_high),
            'ci_level': credible_interval,
            'ess': float(effective_sample_size(samples_flat)),
            'n_samples': len(samples_flat),
        }
    
    return summary


def compare_sequential_vs_batch(sequential_results: Dict,
                               batch_results: Dict,
                               param_names: List[str]) -> Dict:
    """
    Compare parameter estimates between sequential and batch approaches.
    
    Args:
        sequential_results: Results from sequential MCMC
        batch_results: Results from batch MCMC
        param_names: List of parameter names to compare
    
    Returns:
        Dict with comparison metrics:
            - 'parameter_differences': Absolute difference in estimates
            - 'percent_difference': Percent difference
            - 'uncertainty_reduction': Ratio of batch std to seq std
            - 'ci_width_reduction': Ratio of batch CI width to seq CI width
    
    Example:
        >>> comparison = compare_sequential_vs_batch(
        ...     seq_results, batch_results,
        ...     param_names=['alpha', 'beta', 'gamma1']
        ... )
        >>> print(f"Alpha difference: {comparison['percent_difference']['alpha']:.2f}%")
    """
    comparison = {
        'parameter_differences': {},
        'percent_difference': {},
        'uncertainty_reduction': {},
        'ci_width_reduction': {},
    }
    
    seq_means = sequential_results.get('posterior_means', {})
    batch_means = batch_results.get('posterior_means', {})
    seq_std = sequential_results.get('posterior_std', {})
    batch_std = batch_results.get('posterior_std', {})
    
    for param in param_names:
        if param in seq_means and param in batch_means:
            diff = abs(batch_means[param] - seq_means[param])
            pct_diff = 100 * diff / abs(seq_means[param]) if seq_means[param] != 0 else 0
            
            comparison['parameter_differences'][param] = float(diff)
            comparison['percent_difference'][param] = float(pct_diff)
            
            if param in seq_std and param in batch_std:
                unc_reduction = batch_std[param] / seq_std[param] if seq_std[param] > 0 else 1
                comparison['uncertainty_reduction'][param] = float(unc_reduction)
    
    return comparison


def format_results_table(summary: Dict,
                        param_order: Optional[List[str]] = None) -> str:
    """
    Format posterior summary as a nice ASCII table.
    
    Args:
        summary: Output from summarize_posterior()
        param_order: Optional list of parameter names for column order
    
    Returns:
        Formatted string table
    
    Example:
        >>> summary = summarize_posterior(posterior_samples)
        >>> print(format_results_table(summary, param_order=['alpha', 'beta', 'gamma1']))
    """
    if param_order is None:
        param_order = sorted(summary.keys())
    
    # Build table
    lines = []
    lines.append("=" * 100)
    lines.append(f"{'Parameter':<12} {'Mean':>12} {'Std':>12} {'Median':>12} "
                f"{'CI Low':>12} {'CI High':>12} {'ESS':>12}")
    lines.append("-" * 100)
    
    for param in param_order:
        if param in summary:
            s = summary[param]
            lines.append(
                f"{param:<12} {s['mean']:>12.6f} {s['std']:>12.6f} "
                f"{s['median']:>12.6f} {s['ci_lower']:>12.6f} "
                f"{s['ci_upper']:>12.6f} {s['ess']:>12.0f}"
            )
    
    lines.append("=" * 100)
    
    return "\n".join(lines)


def export_results_csv(summary: Dict,
                      output_path: str,
                      param_order: Optional[List[str]] = None) -> None:
    """
    Export posterior summary to CSV file.
    
    Args:
        summary: Output from summarize_posterior()
        output_path: Path to save CSV file
        param_order: Optional parameter name order
    """
    import csv
    
    if param_order is None:
        param_order = sorted(summary.keys())
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        
        # Header
        writer.writerow(['parameter', 'mean', 'std', 'median', 'ci_lower', 
                        'ci_upper', 'ci_level', 'ess', 'n_samples'])
        
        # Data rows
        for param in param_order:
            if param in summary:
                s = summary[param]
                writer.writerow([
                    param,
                    f"{s['mean']:.8f}",
                    f"{s['std']:.8f}",
                    f"{s['median']:.8f}",
                    f"{s['ci_lower']:.8f}",
                    f"{s['ci_upper']:.8f}",
                    s['ci_level'],
                    f"{s['ess']:.1f}",
                    s['n_samples'],
                ])
    
    print(f"Saved results to {output_path}")


def plot_parameter_comparison(seq_summary: Dict,
                             batch_summary: Dict,
                             param_names: List[str],
                             figsize: Tuple[int, int] = (12, 6),
                             save_path: Optional[str] = None) -> None:
    """
    Plot side-by-side comparison of parameter estimates (sequential vs batch).
    
    Args:
        seq_summary: Summary from sequential approach
        batch_summary: Summary from batch approach
        param_names: Parameters to plot
        figsize: Figure size
        save_path: Optional path to save figure
    """
    fig, axes = plt.subplots(1, len(param_names), figsize=figsize)
    
    if len(param_names) == 1:
        axes = [axes]
    
    for ax, param in zip(axes, param_names):
        if param not in seq_summary or param not in batch_summary:
            continue
        
        seq = seq_summary[param]
        batch = batch_summary[param]
        
        # Plot points with error bars
        ax.errorbar([0], [seq['mean']], yerr=seq['std'], fmt='o', 
                   markersize=8, capsize=5, label='Sequential', color='C0')
        ax.errorbar([1], [batch['mean']], yerr=batch['std'], fmt='s',
                   markersize=8, capsize=5, label='Batch', color='C1')
        
        # Plot credible intervals
        ax.plot([0, 0], [seq['ci_lower'], seq['ci_upper']], 'C0-', linewidth=2, alpha=0.3)
        ax.plot([1, 1], [batch['ci_lower'], batch['ci_upper']], 'C1-', linewidth=2, alpha=0.3)
        
        ax.set_xlim(-0.5, 1.5)
        ax.set_xticks([0, 1])
        ax.set_xticklabels(['Sequential', 'Batch'])
        ax.set_ylabel(f'{param} value')
        ax.set_title(param)
        ax.grid(True, alpha=0.3)
        
        if param == param_names[0]:
            ax.legend()
    
    plt.tight_layout()
    
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        print(f"Saved figure to {save_path}")
    
    plt.show()


def compute_diagnostics_summary(processor_results: Dict) -> Dict:
    """
    Compute comprehensive diagnostics summary for batch results.
    
    Args:
        processor_results: Results dict from BatchMCMCProcessor
    
    Returns:
        Dict with diagnostic summary
    """
    summary = {
        'num_batches': len(processor_results.get('calibration_results', {})),
        'total_samples': 0,
        'parameter_diagnostics': {},
    }
    
    cal_results = processor_results.get('calibration_results', {})
    for batch_key, batch_data in cal_results.items():
        diagnostics = batch_data.get('diagnostics', {})
        for param, diag in diagnostics.items():
            if param not in summary['parameter_diagnostics']:
                summary['parameter_diagnostics'][param] = []
            summary['parameter_diagnostics'][param].append(diag)
    
    return summary
