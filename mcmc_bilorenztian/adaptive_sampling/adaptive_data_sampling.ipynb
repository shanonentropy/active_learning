"""
Here we implement Adaptive Data Point Sampling for Bi-Lorentzian MCMC fitting

This script  selects data points for model evaluation based on signal amplitude.
Instead of using all 100 frequency points, it:
- Samples densely where the bi-Lorentzian function has significant amplitude (ie. amplitude > threshold- set by user)
- Samples sparsely where signal is nearly zero (non-informative regions)

Theory:
- Non-informative regions (low signal) add noise but add little information
    - the low region can be seen as putting a limit on the noise 
    - and how heavy the tails of the lorentzian/gaussian funcs are  
- Sampling fewer points in these regions reduces computational burden
    - as seen in the notebook 66% of the data points are in the low_signal region
    - dropping these points leads to > 2X speedup
- While sampling denser in high-signal regions improves parameter constraints

Usage:
    from adaptive_data_sampling import AdaptiveDataSampler

    sampler = AdaptiveDataSampler(x_scale, y_esr, hmc_samples)
    indices, adaptive_data = sampler.create_adaptive_sample(
        amplitude_threshold=0.1,
        high_signal_fraction=0.8,
        low_signal_fraction=0.2
    )
    to see the the basic idea behine the adaptive sampler see test_case_adaptive_sampler.ipynb
"""

import numpy as np
import torch
from typing import Tuple, Dict, List
import matplotlib.pyplot as plt


class AdaptiveDataSampler:
    """
    Intelligently samples data points based on bi-Lorentzian signal amplitude.
    
    Parameters:
    -----------
    x_ : array-like
        Frequency axis (e.g., scaled to 0-100)
    y_esr : array-like or pd.DataFrame
        ESR measurements, shape (n_points, n_samples)
    hmc_samples : dict
        Posterior samples from MCMC with keys: 'A', 'X', 'B', 'gamma1', 'amp', 'var'
        Each value should be a 1D array of posterior samples
    """
    
    def __init__(self, x_scale, y_esr, hmc_samples):
        self.x_scale = np.asarray(x_scale).flatten()
        self.y_esr = np.asarray(y_esr).flatten() if hasattr(y_esr, 'values') else np.asarray(y_esr).flatten()
        self.n_points = len(self.x_scale)
        
        # Convert posterior samples to numpy if needed
        self.hmc_samples = {}
        for key, val in hmc_samples.items():
            if isinstance(val, torch.Tensor):
                self.hmc_samples[key] = val.detach().cpu().numpy()
            else:
                self.hmc_samples[key] = np.asarray(val)
    
    def bilorentzian(self, f: np.ndarray, A, X, amp, gamma1):
        """
        Evaluate bi-Lorentzian function.
        
        Supports both scalar and array-valued parameters for flexible use.
        
        F(x_) = amp * (0.5*gamma1) / ((f-A)^2 + (0.5*gamma1)^2)
             + amp * (0.5*gamma1) / ((f-B)^2 + (0.5*gamma1)^2)
        
        where B = A + X
        
        Parameters:
        -----------
        x_ : np.ndarray
            Frequency array, shape (n_points,)
        A, X, amp, gamma1 : float or np.ndarray
            Lorentzian parameters. If array-valued with shape (n_samples,),
            will be reshaped to (n_samples, 1) for proper broadcasting.
            Result will have shape (n_samples, n_points).
        
        Returns:
        --------
        np.ndarray
            Bi-Lorentzian evaluated at f. Shape depends on inputs:
            - Scalar parameters: shape (n_points,)
            - Array parameters shape (N,): shape (N, n_points)
        """
        # Convert inputs to numpy arrays
        A = np.asarray(A)
        X = np.asarray(X)
        amp = np.asarray(amp)
        gamma1 = np.asarray(gamma1)
        f = np.asarray(f)
        
        # Reshape 1D arrays to column vectors for broadcasting
        # This allows (N, 1) to broadcast with (M,) -> (N, M)
        if A.ndim == 1:
            A = A[:, np.newaxis]
        if X.ndim == 1:
            X = X[:, np.newaxis]
        if amp.ndim == 1:
            amp = amp[:, np.newaxis]
        if gamma1.ndim == 1:
            gamma1 = gamma1[:, np.newaxis]
        
        B = A + X
        
        lorentz1 = (0.5 * gamma1) / ((f - A)**2 + (0.5 * gamma1)**2)
        lorentz2 = (0.5 * gamma1) / ((f - B)**2 + (0.5 * gamma1)**2)
        
        result = amp * (lorentz1 + lorentz2)
        
        # If all inputs were scalar, squeeze the result back to 1D
        if result.shape[0] == 1:
            result = result.squeeze(axis=0)
        
        return result
    
    def compute_signal_amplitude(self) -> np.ndarray:
        """
        Compute mean signal amplitude across all posterior samples.
        
        Returns:
        --------
        amplitude : array of shape (n_points,)
            Mean absolute amplitude at each frequency point
        """
        A_samples = self.hmc_samples['A']
        X_samples = self.hmc_samples['X']
        amp_samples = self.hmc_samples['amp']
        gamma1_samples = self.hmc_samples['gamma1']
        
        n_samples = len(A_samples)
        amplitudes = np.zeros((n_samples, self.n_points))
        
        for i in range(n_samples):
            amplitudes[i, :] = self.bilorentzian(
                self.x_scale,
                A_samples[i],
                X_samples[i],
                amp_samples[i],
                gamma1_samples[i]
            )
        
        # Return mean amplitude (absolute value to capture both positive and negative deviations)
        return np.abs(amplitudes).mean(axis=0)
    
    def identify_signal_regions(self, amplitude_threshold: float = 0.1) -> Tuple[np.ndarray, np.ndarray]:
        """
        Identify high-signal and low-signal regions.
        
        Parameters:
        -----------
        amplitude_threshold : float
            Threshold for classifying point as "high signal"
            Points with amplitude > threshold are considered informative
        
        Returns:
        --------
        high_signal_idx : array
            Indices where amplitude > threshold
        low_signal_idx : array
            Indices where amplitude <= threshold
        """
        amplitudes = self.compute_signal_amplitude()
        
        high_signal_idx = np.where(amplitudes > amplitude_threshold)[0]
        low_signal_idx = np.where(amplitudes <= amplitude_threshold)[0]
        
        return high_signal_idx, low_signal_idx, amplitudes
    
    def create_adaptive_sample(
        self,
        amplitude_threshold: float = 0.3,
        high_signal_fraction: float = 1.0,
        low_signal_fraction: float = 0.3,
        seed: int = 42
    ) -> Tuple[np.ndarray, Tuple]:
        """
        Create adaptive sampling by selecting points based on signal amplitude.
        
        Parameters:
        -----------
        amplitude_threshold : float
            Threshold for identifying high-signal regions
        high_signal_fraction : float (0-1)
            Fraction of high-signal points to keep (1.0 = keep all)
        low_signal_fraction : float (0-1)
            Fraction of low-signal points to keep (< 1.0 = subsample)
        seed : int
            Random seed for reproducibility
        
        Returns:
        --------
        selected_indices : array
            Indices of selected data points, sorted
        adaptive_data : tuple
            (x_adaptive, y_adaptive) with selected points
        statistics : dict
            Statistics about the sampling
        """
        np.random.seed(seed)
        
        # Identify signal regions
        high_signal_idx, low_signal_idx, amplitudes = self.identify_signal_regions(amplitude_threshold)
        
        # Subsample based on fractions
        high_selected = np.random.choice(
            high_signal_idx,
            size=max(1, int(len(high_signal_idx) * high_signal_fraction)),
            replace=False
        )
        
        low_selected = np.random.choice(
            low_signal_idx,
            size=max(1, int(len(low_signal_idx) * low_signal_fraction)),
            replace=False
        )
        
        # Combine and sort
        selected_indices = np.sort(np.concatenate([high_selected, low_selected]))
        
        # Create adaptive data
        x_adaptive = self.x_scale[selected_indices]
        y_adaptive = self.y_esr[selected_indices]
        
        # Compute statistics
        stats = {
            'original_n_points': len(self.x_scale),
            'selected_n_points': len(selected_indices),
            'reduction_ratio': len(self.x_scale) / len(selected_indices),
            'high_signal_total': len(high_signal_idx),
            'high_signal_selected': len(high_selected),
            'low_signal_total': len(low_signal_idx),
            'low_signal_selected': len(low_selected),
            'amplitude_threshold': amplitude_threshold,
        }
        
        return selected_indices, (x_adaptive, y_adaptive), stats
    
    def plot_sampling_strategy(
        self,
        amplitude_threshold: float = 0.1,
        high_signal_fraction: float = 1.0,
        low_signal_fraction: float = 0.3,
        figsize: Tuple = (14, 5)
    ):
        """
        Visualize the adaptive sampling strategy.
        
        Shows:
        - Left: Signal amplitude profile with threshold
        - Middle: Original vs. selected points
        - Right: Data with sampling overlay
        """
        # Compute signal amplitude
        high_signal_idx, low_signal_idx, amplitudes = self.identify_signal_regions(amplitude_threshold)
        
        # Get adaptive sample
        selected_indices, _, stats = self.create_adaptive_sample(
            amplitude_threshold=amplitude_threshold,
            high_signal_fraction=high_signal_fraction,
            low_signal_fraction=low_signal_fraction
        )
        
        fig, axes = plt.subplots(1, 3, figsize=figsize)
        
        # Plot 1: Signal amplitude
        axes[0].plot(self.x_scale, amplitudes, 'b-', linewidth=2, label='Signal Amplitude')
        axes[0].axhline(amplitude_threshold, color='r', linestyle='--', label=f'Threshold = {amplitude_threshold}')
        axes[0].fill_between(self.x_scale, 0, amplitudes, where=(amplitudes > amplitude_threshold),
                            alpha=0.3, color='green', label='High Signal')
        axes[0].fill_between(self.x_scale, 0, amplitudes, where=(amplitudes <= amplitude_threshold),
                            alpha=0.3, color='red', label='Low Signal')
        axes[0].set_xlabel('Frequency')
        axes[0].set_ylabel('Mean Amplitude')
        axes[0].set_title('Bi-Lorentzian Amplitude Profile')
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
        
        # Plot 2: Sampling statistics
        labels = ['High\nSignal\nTotal', 'High\nSignal\nSelected', 'Low\nSignal\nTotal', 'Low\nSignal\nSelected']
        values = [stats['high_signal_total'], stats['high_signal_selected'],
                 stats['low_signal_total'], stats['low_signal_selected']]
        colors = ['green', 'darkgreen', 'red', 'darkred']
        axes[1].bar(labels, values, color=colors, alpha=0.7)
        axes[1].set_ylabel('Number of Points')
        axes[1].set_title('Sampling Distribution')
        axes[1].grid(True, alpha=0.3, axis='y')
        
        # Add text annotations
        for i, v in enumerate(values):
            axes[1].text(i, v + 1, str(v), ha='center', va='bottom', fontweight='bold')
        
        # Plot 3: Data points overlay
        axes[2].plot(self.x_scale, self.y_esr, 'o-', color='lightgray', label='All Points', markersize=4, alpha=0.5)
        axes[2].plot(self.x_scale[selected_indices], self.y_esr[selected_indices], 'ro', 
                    label='Selected Points', markersize=6, alpha=0.8)
        axes[2].set_xlabel('Frequency')
        axes[2].set_ylabel('ESR Signal')
        axes[2].set_title(f"Data Sampling (n={stats['selected_n_points']}/{stats['original_n_points']})")
        axes[2].legend()
        axes[2].grid(True, alpha=0.3)
        
        plt.tight_layout()
        return fig, axes, stats
    
    def print_sampling_report(
        self,
        amplitude_threshold: float = 0.1,
        high_signal_fraction: float = 1.0,
        low_signal_fraction: float = 0.3
    ):
        """Print detailed sampling statistics."""
        _, _, stats = self.create_adaptive_sample(
            amplitude_threshold=amplitude_threshold,
            high_signal_fraction=high_signal_fraction,
            low_signal_fraction=low_signal_fraction
        )
        
        print("\n" + "="*70)
        print("ADAPTIVE DATA SAMPLING REPORT")
        print("="*70)
        print(f"\nOriginal Configuration:")
        print(f"  - Total frequency points: {stats['original_n_points']}")
        print(f"\nAmplitude Threshold: {stats['amplitude_threshold']}")
        print(f"\nSignal Region Breakdown:")
        print(f"  High Signal (amp > threshold):")
        print(f"    - Total points: {stats['high_signal_total']}")
        print(f"    - Selected: {stats['high_signal_selected']} ({high_signal_fraction*100:.0f}%)")
        print(f"\n  Low Signal (amp ≤ threshold):")
        print(f"    - Total points: {stats['low_signal_total']}")
        print(f"    - Selected: {stats['low_signal_selected']} ({low_signal_fraction*100:.0f}%)")
        print(f"\nAdaptive Configuration:")
        print(f"  - Selected frequency points: {stats['selected_n_points']}")
        print(f"  - Data reduction: {stats['reduction_ratio']:.2f}x fewer points")
        print(f"  - Computational speedup estimate: {stats['reduction_ratio']:.2f}x faster")
        print("\n" + "="*70 + "\n")


# Example usage and helper functions
def example_usage_with_notebook_data():
    """
    Example showing how to use this with data from the Pyro notebook.
    
    This assumes you have:
    - x_scale: frequency axis
    - y_esr: ESR data (one column)
    - hmc_samples: dictionary from posterior.get_samples()
    """
    # Code snippet for notebook:
    """
    from adaptive_data_sampling import AdaptiveDataSampler
    
    # After running MCMC for one frequency slice
    hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
    
    # Create sampler
    sampler = AdaptiveDataSampler(x_scale, y_esr.iloc[:, j].values, hmc_samples)
    
    # Generate adaptive sampling strategy
    selected_idx, adaptive_data, stats = sampler.create_adaptive_sample(
        amplitude_threshold=0.1,      # Points with |F| > 0.1 are "informative"
        high_signal_fraction=1.0,     # Keep all high-signal points
        low_signal_fraction=0.3       # Keep only 30% of low-signal points
    )
    
    # Visualize
    fig, axes, stats = sampler.plot_sampling_strategy(
        amplitude_threshold=0.1,
        high_signal_fraction=1.0,
        low_signal_fraction=0.3
    )
    plt.show()
    
    # Print report
    sampler.print_sampling_report(
        amplitude_threshold=0.1,
        high_signal_fraction=1.0,
        low_signal_fraction=0.3
    )
    
    # Use adaptive data for next iteration of fitting
    x_adaptive, y_adaptive = adaptive_data
    
    # Convert to tensors for Pyro model
    data_adaptive = (
        torch.tensor(x_adaptive, dtype=torch.float64),
        torch.tensor(y_adaptive, dtype=torch.float64)
    )
    
    # Run MCMC with adaptive data
    posterior_adaptive = MCMC(kernel, num_samples=100, warmup_steps=100, num_chains=1)
    posterior_adaptive.run(data_adaptive)
    """
    pass


if __name__ == "__main__":
    print(__doc__)
    print("\nFor usage examples, see example_usage_with_notebook_data() or README below:")
    print("""
    ============================================================================
    QUICK START GUIDE
    ============================================================================
    
    Step 1: Run your standard MCMC loop on one or all frequency slices
    Step 2: Extract posterior samples:
        hmc_samples = {k: v.detach().cpu().numpy() 
                      for k, v in posterior.get_samples().items()}
    
    Step 3: Create sampler:
        from adaptive_data_sampling import AdaptiveDataSampler
        sampler = AdaptiveDataSampler(x_scale, y_esr_column, hmc_samples)
    
    Step 4: Generate adaptive sample:
        selected_idx, (x_adapt, y_adapt), stats = sampler.create_adaptive_sample(
            amplitude_threshold=0.1,     # Amplitude threshold
            high_signal_fraction=1.0,    # Keep all high-signal points
            low_signal_fraction=0.3      # Keep 30 percent of low-signal points
        )
    
    Step 5: Visualize:
        fig, axes, stats = sampler.plot_sampling_strategy()
        plt.show()
        
        sampler.print_sampling_report()
    
    Step 6: Use adaptive data in new MCMC run:
        data_adapt = (torch.tensor(x_adapt), torch.tensor(y_adapt))
        posterior_adapt = MCMC(kernel, ...).run(data_adapt)
    
    ============================================================================
    KEY PARAMETERS EXPLAINED
    ============================================================================
    
    amplitude_threshold (0.1):
        - Points where |F(f)| > threshold are "high signal"
        - Adjust based on your noise level
        - Higher threshold = more aggressive downsampling
    
    high_signal_fraction (1.0):
        - Fraction of high-signal points to retain
        - 1.0 = keep all informative points
        - Can reduce if computational cost is still high
    
    low_signal_fraction (0.3):
        - Fraction of low-signal (noisy) points to retain
        - 0.3 = keep only 30% of non-informative points
        - Reduces computational cost while preserving information
    
    ============================================================================
    """)
