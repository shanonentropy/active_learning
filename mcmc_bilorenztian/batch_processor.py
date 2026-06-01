"""
Batch Processing for Bilorentzian ODMR MCMC Inference

This module provides classes and utilities for learning bilorentzian model parameters
over multiple observations simultaneously using Pyro MCMC, rather than sequentially.

Classes:
    BatchDataContainer: Manages data for batch inference
    BatchMCMCProcessor: Orchestrates batch MCMC workflow

Author: ODMR Analysis Pipeline
Date: 2026-05-11
"""

import torch
import numpy as np
import pyro
from pyro.infer import MCMC, NUTS, HMC
from pyro.infer.autoguide import init_to_value
from pathlib import Path
from typing import Dict, List, Tuple, Optional, Union
import json
import warnings


class BatchDataContainer:
    """
    Container for managing batch data for MCMC inference.
    
    Handles data extraction, tensor conversion, and batch formatting for
    bilorentzian model inference.
    
    Attributes:
        x_axis: Frequency axis (shared across all observations)
        temperatures: Temperature for each observation
        y_observations: Spectra data for each observation
        batch_indices: Indices of observations included in this batch
        batch_size: Number of observations in batch
        num_frequencies: Number of frequency points per spectrum
    """
    
    def __init__(self, 
                 x_axis: Union[np.ndarray, torch.Tensor],
                 temperature_array: Union[np.ndarray, torch.Tensor],
                 y_observations: Union[np.ndarray, torch.Tensor],
                 batch_indices: List[int],
                 dtype: torch.dtype = torch.float64):
        """
        Initialize batch data container.
        
        Args:
            x_axis: [num_frequencies] - Frequency axis (shared across batch)
            temperature_array: [num_observations] - Temperature for each observation
            y_observations: [num_observations, num_frequencies] - Spectra data
            batch_indices: List of observation indices to include in batch
            dtype: PyTorch data type (default: float64 for precision)
        
        Raises:
            ValueError: If batch_indices are invalid or data shapes don't match
            IndexError: If batch_indices exceed array bounds
        """
        # Convert to numpy if needed
        if isinstance(x_axis, torch.Tensor):
            x_axis = x_axis.cpu().numpy()
        if isinstance(temperature_array, torch.Tensor):
            temperature_array = temperature_array.cpu().numpy()
        if isinstance(y_observations, torch.Tensor):
            y_observations = y_observations.cpu().numpy()
        
        # Validate shapes
        if len(x_axis.shape) != 1:
            raise ValueError(f"x_axis must be 1D, got shape {x_axis.shape}")
        if len(temperature_array.shape) != 1:
            raise ValueError(f"temperature_array must be 1D, got shape {temperature_array.shape}")
        if len(y_observations.shape) != 2:
            raise ValueError(f"y_observations must be 2D, got shape {y_observations.shape}")
        if y_observations.shape[0] != temperature_array.shape[0]:
            raise ValueError(
                f"y_observations batch size {y_observations.shape[0]} "
                f"doesn't match temperatures {temperature_array.shape[0]}"
            )
        if y_observations.shape[1] != x_axis.shape[0]:
            raise ValueError(
                f"y_observations frequencies {y_observations.shape[1]} "
                f"doesn't match x_axis {x_axis.shape[0]}"
            )
        
        # Validate batch indices
        max_idx = len(temperature_array)
        invalid_indices = [i for i in batch_indices if i < 0 or i >= max_idx]
        if invalid_indices:
            raise IndexError(
                f"Invalid batch indices {invalid_indices}. "
                f"Valid range: [0, {max_idx-1}]"
            )
        
        # Store raw data
        self.x_axis_np = x_axis.astype(np.float64)
        self.temperatures_np = temperature_array.astype(np.float64)
        self.y_observations_np = y_observations.astype(np.float64)
        self.batch_indices = list(batch_indices)
        self.dtype = dtype
        
        # Extract batch data
        self.temps_batch_np = self.temperatures_np[self.batch_indices]
        self.y_batch_np = self.y_observations_np[self.batch_indices, :]
        
        # Compute dimensions
        self.batch_size = len(self.batch_indices)
        self.num_frequencies = len(self.x_axis_np)
        
        # Store as tensors (lazy conversion)
        self._x_tensor = None
        self._temps_tensor = None
        self._y_tensor = None
    
    @property
    def x_tensor(self) -> torch.Tensor:
        """Get x_axis as PyTorch tensor."""
        if self._x_tensor is None:
            self._x_tensor = torch.tensor(self.x_axis_np, dtype=self.dtype)
        return self._x_tensor
    
    @property
    def temps_tensor(self) -> torch.Tensor:
        """Get batch temperatures as PyTorch tensor."""
        if self._temps_tensor is None:
            self._temps_tensor = torch.tensor(self.temps_batch_np, dtype=self.dtype)
        return self._temps_tensor
    
    @property
    def y_tensor(self) -> torch.Tensor:
        """Get batch observations as PyTorch tensor."""
        if self._y_tensor is None:
            self._y_tensor = torch.tensor(self.y_batch_np, dtype=self.dtype)
        return self._y_tensor
    
    def get_batch_data_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get batch data formatted as dictionary for model input.
        
        Returns:
            dict with keys:
                - 'x': frequency axis [num_frequencies]
                - 'temperatures': temperatures [batch_size]
                - 'y_observations': spectra [batch_size, num_frequencies]
                - 'batch_indices': original indices in full dataset
        
        Example:
            >>> batch_data_dict = container.get_batch_data_dict()
            >>> calibration_model(batch_data_dict)
        """
        return {
            'x': self.x_tensor,
            'temperatures': self.temps_tensor,
            'y_observations': self.y_tensor,
            'batch_indices': self.batch_indices,
        }
    
    def get_calibration_data_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get data dict specifically for calibration model (removes temperatures from sampling).
        
        Returns:
            dict with keys: 'x', 'temperatures', 'y_observations'
        """
        return {
            'x': self.x_tensor,
            'temperatures': self.temps_tensor,
            'y_observations': self.y_tensor,
        }
    
    def get_prediction_data_dict(self) -> Dict[str, torch.Tensor]:
        """
        Get data dict for prediction model (removes temperatures, only has spectra).
        
        Returns:
            dict with keys: 'x', 'y_observations'
        """
        return {
            'x': self.x_tensor,
            'y_observations': self.y_tensor,
        }
    
    def __len__(self) -> int:
        """Return batch size (number of observations)."""
        return self.batch_size
    
    def __repr__(self) -> str:
        """String representation of batch container."""
        return (
            f"BatchDataContainer(batch_size={self.batch_size}, "
            f"num_frequencies={self.num_frequencies}, "
            f"indices={self.batch_indices})"
        )
    
    def summary(self) -> Dict:
        """
        Get summary statistics of batch data.
        
        Returns:
            dict with summary info (shapes, ranges, etc.)
        """
        return {
            'batch_size': self.batch_size,
            'num_frequencies': self.num_frequencies,
            'batch_indices': self.batch_indices,
            'temperature_range': (float(self.temps_batch_np.min()), 
                                 float(self.temps_batch_np.max())),
            'y_range': (float(self.y_batch_np.min()), 
                       float(self.y_batch_np.max())),
            'y_mean': float(self.y_batch_np.mean()),
            'y_std': float(self.y_batch_np.std()),
        }


class BatchMCMCProcessor:
    """
    Orchestrator for batch MCMC inference on bilorentzian model.
    
    Manages workflow for learning calibration parameters over batches and
    predicting temperatures using learned parameters.
    
    Attributes:
        x_axis: Frequency axis
        temperatures: Temperature for each observation
        y_observations: Spectra data
        batch_size: Size of batches (configurable)
        results: Dictionary storing inference results
    """
    
    def __init__(self,
                 x_axis: Union[np.ndarray, torch.Tensor],
                 temperatures: Union[np.ndarray, torch.Tensor],
                 y_observations: Union[np.ndarray, torch.Tensor],
                 batch_size: int = 3,
                 seed: int = 42,
                 device: str = 'cpu'):
        """
        Initialize batch MCMC processor.
        
        Args:
            x_axis: [num_frequencies] - Frequency axis
            temperatures: [num_observations] - Temperature for each observation
            y_observations: [num_observations, num_frequencies] - Spectra data
            batch_size: Number of observations per batch (default: 3)
            seed: Random seed for reproducibility
            device: 'cpu' or 'cuda' (default: 'cpu')
        
        Raises:
            ValueError: If data shapes are incompatible
        """
        # Convert to numpy and store
        if isinstance(x_axis, torch.Tensor):
            x_axis = x_axis.cpu().numpy()
        if isinstance(temperatures, torch.Tensor):
            temperatures = temperatures.cpu().numpy()
        if isinstance(y_observations, torch.Tensor):
            y_observations = y_observations.cpu().numpy()
        
        self.x_axis = np.array(x_axis, dtype=np.float64)
        self.temperatures = np.array(temperatures, dtype=np.float64)
        self.y_observations = np.array(y_observations, dtype=np.float64)
        self.batch_size = batch_size
        self.seed = seed
        self.device = device
        
        # Validate data
        if len(self.x_axis.shape) != 1:
            raise ValueError(f"x_axis must be 1D")
        if len(self.temperatures.shape) != 1:
            raise ValueError(f"temperatures must be 1D")
        if len(self.y_observations.shape) != 2:
            raise ValueError(f"y_observations must be 2D")
        
        # Set PyTorch dtype
        torch.set_default_dtype(torch.float64)
        
        # Initialize results storage
        self.calibration_results = {}
        self.prediction_results = {}
        self.batch_indices_history = []
        self.metadata = {
            'num_observations': len(self.temperatures),
            'num_frequencies': len(self.x_axis),
            'batch_size': batch_size,
            'seed': seed,
        }
    
    def create_batches(self,
                      num_observations: Optional[int] = None,
                      overlap: bool = False,
                      step: int = 1) -> List[List[int]]:
        """
        Create batch indices for processing.
        
        Args:
            num_observations: Number of observations to use (None = all)
            overlap: If True, use sliding window; if False, contiguous batches
            step: Step size for sliding window (only used if overlap=True)
        
        Returns:
            List of lists, each sublist contains indices for one batch
        
        Example:
            >>> processor = BatchMCMCProcessor(x, temps, y, batch_size=3)
            >>> batches = processor.create_batches(num_observations=6, overlap=False)
            >>> # Returns [[0, 1, 2], [3, 4, 5]]
        """
        if num_observations is None:
            num_observations = len(self.temperatures)
        
        num_observations = min(num_observations, len(self.temperatures))
        available_indices = list(range(num_observations))
        
        batches = []
        
        if overlap:
            # Sliding window approach
            for start_idx in range(0, len(available_indices) - self.batch_size + 1, step):
                batch = available_indices[start_idx:start_idx + self.batch_size]
                batches.append(batch)
        else:
            # Contiguous batches
            for start_idx in range(0, len(available_indices), self.batch_size):
                batch = available_indices[start_idx:start_idx + self.batch_size]
                if len(batch) > 0:  # Include even if smaller than batch_size
                    batches.append(batch)
        
        self.batch_indices_history = batches
        return batches
    
    def get_batch_container(self, batch_indices: List[int]) -> BatchDataContainer:
        """
        Create a BatchDataContainer for given indices.
        
        Args:
            batch_indices: List of observation indices
        
        Returns:
            BatchDataContainer instance
        
        Example:
            >>> batch = processor.get_batch_container([0, 1, 2])
            >>> print(batch)
            BatchDataContainer(batch_size=3, num_frequencies=100, indices=[0, 1, 2])
        """
        return BatchDataContainer(
            self.x_axis,
            self.temperatures,
            self.y_observations,
            batch_indices
        )
    
    def run_batch_calibration(self,
                             batch_indices: List[int],
                             model_fn,
                             num_samples: int = 2000,
                             warmup_steps: int = 200,
                             init_vals: Optional[Dict] = None,
                             max_tree_depth: int = 6) -> Dict:
        """
        Run MCMC calibration on a batch.
        
        Args:
            batch_indices: Indices of observations to use for calibration
            model_fn: Pyro model function (must accept batch_data_dict)
            num_samples: Number of MCMC samples (default: 2000)
            warmup_steps: Number of warmup/burn-in steps (default: 200)
            init_vals: Dictionary of initialization values for parameters
            max_tree_depth: NUTS max tree depth (default: 6)
        
        Returns:
            dict with keys:
                - 'posterior_samples': {param_name: [num_samples]}
                - 'posterior_means': {param_name: scalar}
                - 'posterior_std': {param_name: scalar}
                - 'batch_indices': indices used
                - 'diagnostics': MCMC diagnostics
        
        Example:
            >>> from bilorentzian_batch_models import batch_calibration_model
            >>> results = processor.run_batch_calibration(
            ...     batch_indices=[0, 1, 2],
            ...     model_fn=batch_calibration_model,
            ...     num_samples=2000
            ... )
            >>> print(results['posterior_means']['alpha'])
        """
        # Get batch container
        batch_container = self.get_batch_container(batch_indices)
        batch_data = batch_container.get_calibration_data_dict()
        
        # Set up initialization
        if init_vals is None:
            init_vals = {}
        
        # Clear Pyro parameter store
        pyro.clear_param_store()
        
        # Create kernel
        kernel = NUTS(
            model_fn,
            jit_compile=True,
            init_strategy=init_to_value(values=init_vals) if init_vals else None,
            ignore_jit_warnings=True,
            max_tree_depth=max_tree_depth
        )
        
        # Run MCMC
        posterior = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
        posterior.run(batch_data)
        
        # Extract samples
        posterior_samples = {
            k: v.detach().cpu().numpy() 
            for k, v in posterior.get_samples().items()
        }
        
        # Compute statistics
        posterior_means = {k: float(v.mean()) for k, v in posterior_samples.items()}
        posterior_std = {k: float(v.std()) for k, v in posterior_samples.items()}
        
        # Get diagnostics
        diagnostics = self._compute_diagnostics(posterior_samples)
        
        # Store results
        batch_key = f"batch_{len(self.calibration_results)}"
        self.calibration_results[batch_key] = {
            'posterior_samples': posterior_samples,
            'posterior_means': posterior_means,
            'posterior_std': posterior_std,
            'batch_indices': batch_indices,
            'diagnostics': diagnostics,
        }
        
        return self.calibration_results[batch_key]
    
    def run_batch_prediction(self,
                            batch_indices: List[int],
                            model_fn,
                            learned_params: Dict,
                            num_samples: int = 2000,
                            warmup_steps: int = 200,
                            init_vals: Optional[Dict] = None,
                            max_tree_depth: int = 6) -> Dict:
        """
        Run MCMC prediction on a batch using fixed calibration parameters.
        
        Args:
            batch_indices: Indices of observations to predict temperatures for
            model_fn: Pyro model function (must accept batch_data_dict and learned_params)
            learned_params: Dict of fixed calibration parameters
            num_samples: Number of MCMC samples (default: 2000)
            warmup_steps: Number of warmup steps (default: 200)
            init_vals: Initialization values
            max_tree_depth: NUTS max tree depth
        
        Returns:
            dict with keys:
                - 'posterior_samples': {param_name: [num_samples]}
                - 'posterior_means': {param_name: scalar}
                - 'posterior_std': {param_name: scalar}
                - 'temperatures': predicted temperatures for batch
                - 'batch_indices': indices used
        """
        # Get batch container
        batch_container = self.get_batch_container(batch_indices)
        batch_data = batch_container.get_prediction_data_dict()
        
        # Convert learned_params to tensors
        learned_params_tensor = {
            k: torch.tensor(v, dtype=torch.float64) if not isinstance(v, torch.Tensor) else v
            for k, v in learned_params.items()
        }
        
        # Set up initialization
        if init_vals is None:
            init_vals = {}
        
        # Clear Pyro parameter store
        pyro.clear_param_store()
        
        # Create kernel
        kernel = NUTS(
            lambda data: model_fn(data, learned_params_tensor),
            jit_compile=True,
            init_strategy=init_to_value(values=init_vals) if init_vals else None,
            ignore_jit_warnings=True,
            max_tree_depth=max_tree_depth
        )
        
        # Run MCMC
        posterior = MCMC(kernel, num_samples=num_samples, warmup_steps=warmup_steps, num_chains=1)
        posterior.run(batch_data)
        
        # Extract samples
        posterior_samples = {
            k: v.detach().cpu().numpy()
            for k, v in posterior.get_samples().items()
        }
        
        # Compute statistics
        posterior_means = {k: float(v.mean()) for k, v in posterior_samples.items()}
        posterior_std = {k: float(v.std()) for k, v in posterior_samples.items()}
        
        # Store results
        batch_key = f"batch_{len(self.prediction_results)}"
        self.prediction_results[batch_key] = {
            'posterior_samples': posterior_samples,
            'posterior_means': posterior_means,
            'posterior_std': posterior_std,
            'batch_indices': batch_indices,
            'temperatures': posterior_means.get('T', None),
        }
        
        return self.prediction_results[batch_key]
    
    @staticmethod
    def _compute_diagnostics(posterior_samples: Dict[str, np.ndarray]) -> Dict:
        """
        Compute MCMC diagnostics (Gelman-Rubin, ESS, etc.).
        
        Args:
            posterior_samples: {param_name: [num_samples]}
        
        Returns:
            dict with diagnostic metrics
        """
        diagnostics = {}
        for param_name, samples in posterior_samples.items():
            diagnostics[param_name] = {
                'mean': float(np.mean(samples)),
                'std': float(np.std(samples)),
                'autocorr_lag1': float(_compute_autocorr(samples, lag=1)),
                'effective_sample_size': float(_compute_ess(samples)),
            }
        return diagnostics
    
    def save_results(self, output_path: Union[str, Path]) -> None:
        """
        Save calibration and prediction results to JSON.
        
        Args:
            output_path: Path to save results file
        """
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Convert numpy arrays to lists for JSON serialization
        results_to_save = {
            'metadata': self.metadata,
            'calibration_results': self._make_json_serializable(self.calibration_results),
            'prediction_results': self._make_json_serializable(self.prediction_results),
            'batch_indices_history': self.batch_indices_history,
        }
        
        with open(output_path, 'w') as f:
            json.dump(results_to_save, f, indent=2)
    
    @staticmethod
    def _make_json_serializable(obj):
        """Recursively convert numpy arrays and tensors to Python lists."""
        if isinstance(obj, dict):
            return {k: BatchMCMCProcessor._make_json_serializable(v) for k, v in obj.items()}
        elif isinstance(obj, (list, tuple)):
            return [BatchMCMCProcessor._make_json_serializable(v) for v in obj]
        elif isinstance(obj, np.ndarray):
            return obj.tolist()
        elif isinstance(obj, (np.integer, np.floating)):
            return obj.item()
        else:
            return obj


def _compute_autocorr(x: np.ndarray, lag: int = 1) -> float:
    """
    Compute autocorrelation at given lag.
    
    Args:
        x: 1D array of samples
        lag: Lag to compute autocorrelation (default: 1)
    
    Returns:
        Autocorrelation value
    """
    x = np.asarray(x).squeeze()
    x = x - np.mean(x)
    c0 = np.dot(x, x) / len(x)
    c_lag = np.dot(x[:-lag], x[lag:]) / len(x)
    return c_lag / c0


def _compute_ess(x: np.ndarray, threshold: float = 0.05) -> float:
    """
    Estimate Effective Sample Size using autocorrelation.
    
    Args:
        x: 1D array of samples
        threshold: Autocorr threshold for stopping (default: 0.05)
    
    Returns:
        Effective sample size estimate
    """
    n = len(x)
    autocorr = _compute_autocorr(x, lag=1)
    
    if autocorr >= threshold:
        # Estimate integrated autocorrelation time
        tau_int = 0.5  # Initial guess
        for lag in range(1, min(100, n // 2)):
            acf = _compute_autocorr(x, lag=lag)
            if acf < threshold:
                tau_int = 0.5 + np.sum([_compute_autocorr(x, lag=k) for k in range(1, lag)])
                break
        return n / (2 * tau_int)
    else:
        return float(n)
