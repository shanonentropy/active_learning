import importlib
import torch
import pyro
import pyro.distributions as dist
import numpy as np

import odmrsimulator


def model(odmrsimulator, data_):
    """Top-level model function for Pyro MCMC. Picklable for Windows multiprocessing.
    Expects `data_` to be (T, (x_obs, y_obs))."""
    T, data = data_[0], data_[1]
    x_obs, y_obs = data[0], data[1]
    RF_sweep = np.linspace(2720, 2990, 100)
    PL_vs_T = np.zeros(len(RF_sweep))
    BNV = 0.0

    D_0 = pyro.sample("D_0", dist.Normal(2877, 1.)).double()
    alpha = pyro.sample("alpha", dist.Normal(-0.072, 0.0005)).double()
    E_rad_s = pyro.sample("E_rad_s", dist.Normal(5.0, 1.0)).double()
    temp = pyro.sample("temp", dist.Uniform(243, 343)).double()
    gamma = pyro.sample("gamma", dist.Normal(1.0, 0.2)).double()
    Amp = pyro.sample("Amp", dist.Normal(0.9, 0.1)).double()
    T1_0 = pyro.sample("T1_0", dist.Normal(500e-6, 50e-6)).double() + 1e-8
    T2_0 = pyro.sample("T2_0", dist.Normal(1e-6, 0.1e-6)).double() + 1e-8
    rabi_rate = pyro.sample("rabi_rate", dist.Normal(3.0, 0.5)).double() + 1e-8
    var = pyro.sample("var", dist.HalfNormal(scale=0.1)).double()

    # convert to native floats
    def to_float(v):
        if isinstance(v, torch.Tensor):
            return float(v.detach().cpu().item())
        return float(v)

    D_0_f = to_float(D_0)
    alpha_f = to_float(alpha)
    E_rad_s_f = to_float(E_rad_s)
    temp_f = to_float(temp)
    gamma_f = to_float(gamma)
    Amp_f = to_float(Amp)
    T1_0_f = to_float(T1_0)
    T2_0_f = to_float(T2_0)
    rabi_rate_f = to_float(rabi_rate)
    var_f = to_float(var)

    # clamp same safe ranges as notebook
    D_0_f = float(np.clip(D_0_f, 2876.9, 2879.0))
    alpha_f = float(np.clip(alpha_f, -1.0, -0.06))
    E_rad_s_f = float(np.clip(E_rad_s_f, 0.0, 1e9))
    temp_f = float(np.clip(temp_f, 77.0, 400.0))
    gamma_f = float(np.clip(gamma_f, -10.0, 10.0))
    Amp_f = float(np.clip(Amp_f, 1e-6, 10.0))
    T1_0_f = float(np.clip(T1_0_f, 1e-9, 1e9))
    T2_0_f = float(np.clip(T2_0_f, 1e-9, 1e9))
    rabi_rate_f = float(np.clip(rabi_rate_f, 1e-9, 1e6))
    var_f = float(np.clip(var_f, 1e-12, 1.0))

    importlib.reload(odmrsimulator)
    odmrsimulator.D0 = D_0_f
    odmrsimulator.T0 = 300.0
    odmrsimulator.alpha = alpha_f
    odmrsimulator.T1_0 = T1_0_f
    odmrsimulator.beta = D_0_f
    odmrsimulator.T2_0 = T2_0_f
    odmrsimulator.gamma = gamma_f
    odmrsimulator.D_MHz = D_0_f
    odmrsimulator.D_rad_s = D_0_f * 2 * np.pi * 2
    odmrsimulator.E_rad_s = E_rad_s_f

    for i, RF_freq in enumerate(RF_sweep):
        PL_vs_T[i] = odmrsimulator.NV_ODMR(BNV, RF_freq, rabi_rate_f, temp_f)

    loc = torch.tensor(PL_vs_T).double()
    try:
        cov = (var_f * torch.eye(y_obs.shape[0]).double())
    except Exception:
        cov = (torch.tensor(var_f).double() * torch.eye(y_obs.shape[0]).double())

    obs = y_obs
    if not isinstance(obs, torch.Tensor):
        obs = torch.tensor(obs).double()

    pyro.sample("obs", dist.MultivariateNormal(loc, covariance_matrix=cov), obs=obs)
