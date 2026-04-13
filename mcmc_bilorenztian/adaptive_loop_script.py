############ import modules ########

#torch modules
import torch
import pyro
import pyro.distributions as dist
from pyro.infer import MCMC, NUTS
from pyro.infer.autoguide import init_to_value
# backend modules
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from scipy.stats.mstats import mquantiles
from sklearn.preprocessing import MinMaxScaler
import time
# Import adaptive sampling utility
from adaptive_data_sampling import AdaptiveDataSampler
# Import the model
from bilorentzian_model_optimized import model
#set default precision 
torch.set_default_dtype(torch.float64)
print("Libraries imported successfully!")

######### data loading and preprocessing #########

# Temperature array
temps_ = ['25', '30', '35', '40', '45', '50', '45', '40', '35', '30', '25', '20']
temps = np.array(temps_, dtype=float)

# Load ODMR data
fpath = './cycle1'
df_ = pd.read_csv(fpath, sep=',', header=0)
df = df_.iloc[0:, :-1]
df.drop(columns=['25 C-lower power', '15', '10', '10.1', '-30', '-20'], inplace=True)

# Extract data
x_esr = df.frequency.values
y_esr = df.iloc[:, 2:]

# Scale x-axis to 0-100
sc = MinMaxScaler()
x_scale = sc.fit_transform(x_esr.reshape(-1, 1)).flatten() * 100

# Normalize y data (baseline subtraction, sign flip, normalization)
y_esr = y_esr.apply(lambda x: x - x[:10].mean())
y_esr = -1 * y_esr
y_esr = y_esr.apply(lambda x: x / x.max())

print(f"Data loaded successfully!")
print(f"  - Frequency points: {len(x_scale)}")
print(f"  - Temperature measurements: {len(temps)}")
print(f"  - Temperatures: {temps}")

############# Define bi-Lorentzian function (numpy version for plotting purposes) ###############
def F_np(x_in, A, X, Amp, G1, G2):
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

# Setup MCMC kernel
init_vals = {
    "A": torch.tensor(50.0),
    "X": torch.tensor(8.0),
    "gamma1": torch.tensor(8.0),
    "amp": torch.tensor(3.0),
    "var": torch.tensor(0.05),
}

kernel = NUTS(
    model,
    jit_compile=True,
    init_strategy=init_to_value(values=init_vals),
    ignore_jit_warnings=True,
    max_tree_depth=5
)  #

# Run MCMC on first temperature slice with FULL data
print("Running initial MCMC on first temperature slice (full data for adaptive sampling)...")
j_init = 0
y_init = y_esr.iloc[:, j_init].values

data_full = (
    torch.tensor(x_scale, dtype=torch.float64),
    torch.tensor(y_init, dtype=torch.float64)
)

start_time = time.time()
pyro.clear_param_store()
posterior_init = MCMC(
    kernel,
    num_samples=100,
    warmup_steps=100,
    num_chains=1,
    disable_progbar=False,
   
) #  initial_params=init_vals
posterior_init.run(data_full)
time_init = time.time() - start_time

# Extract samples
hmc_samples_init = {
    k: v.detach().cpu().numpy()
    for k, v in posterior_init.get_samples().items()
}

print(f"Initial MCMC completed in {time_init:.2f} seconds")
print(f"Posterior samples shape: {[(k, v.shape) for k, v in hmc_samples_init.items()]}")

 # Create sampler
sampler = AdaptiveDataSampler(x_scale, y_esr.iloc[:, j_init].values, hmc_samples_init)

# Generate adaptive sampling strategy
selected_idx, adaptive_data, stats = sampler.create_adaptive_sample(
    amplitude_threshold=0.1,     
    high_signal_fraction=1.0,    
    low_signal_fraction=0.1      
)


# Visualize
fig, axes, stats = sampler.plot_sampling_strategy(
    amplitude_threshold=0.1,
    high_signal_fraction=1.0,
    low_signal_fraction=0.3
)
plt.show()

# Pre-compute constants outside the loop for efficiency
# Note: Kernel must be recreated per iteration (Pyro state management with changing data shapes)
#       but init_vals and result containers are computed once

# Define initial values once (same for all iterations)
init_vals = {
    "A": torch.tensor(50.0),
    "X": torch.tensor(8.0),
    "gamma1": torch.tensor(8.0),
    "amp": torch.tensor(3.0),
    "var": torch.tensor(0.05),
}

# Initialize result containers
idx, amp_vals, gamma1_vals, A_freq, B_freq = [], [], [], [], []
amp_var, gamma1_var, A_freq_var, B_freq_var = [], [], [], []

print("Setup complete: init values pre-computed (kernel will be created per-iteration)")

start_time = time.time()
for num in range(0, y_esr.shape[1]):
  # Generate adaptive sampling strategy based on current temperature slice
  sampler = AdaptiveDataSampler(x_scale, y_esr.iloc[:, num].values, hmc_samples_init)
  selected_idx, adaptive_data, stats = sampler.create_adaptive_sample(
    amplitude_threshold=0.1,     
    high_signal_fraction=1.0,    
    low_signal_fraction=0.1      
    )
  
  # Extract paired adaptive x and y (compatible sizes)
  x_adaptive, y_adaptive = adaptive_data
  
  # Convert to MATCHED tensors for Pyro model
  # x_adaptive and y_adaptive have the same length from create_adaptive_sample()
  data_adaptive = (
    torch.tensor(x_adaptive, dtype=torch.float64),
    torch.tensor(y_adaptive, dtype=torch.float64)
    )
  
  print(f"  [{num+1}/{y_esr.shape[1]}]", end="", flush=True)
  
  # Clear parameter store and create fresh kernel for each iteration
  # (data shape changes between iterations due to adaptive sampling)
  pyro.clear_param_store()
  kernel_iter = NUTS(
    model,
    jit_compile=True,
    init_strategy=init_to_value(values=init_vals),
    ignore_jit_warnings=True,
    max_tree_depth=5
  )
  
  # Run MCMC with adaptive data
  posterior_adaptive = MCMC(kernel_iter, num_samples=100, warmup_steps=100, num_chains=1)
  posterior_adaptive.run(data_adaptive)
  
  # Extract samples
  hmc_samples = {
        k: v.detach().cpu().numpy()
        for k, v in posterior_adaptive.get_samples().items()
    }
  print(f"MCMC completed in {time_init:.2f} seconds")
  print(f"Posterior samples shape: {[(k, v.shape) for k, v in hmc_samples.items()]}")
  
  A_ = hmc_samples['A']  # Shape: (100,) for single chain
  X_ = hmc_samples['X']
  B_ = (A_ + X_)
  amp_ = hmc_samples['amp']
  gamma1_ = hmc_samples['gamma1']
  gamma2_ = hmc_samples['gamma1']  # Since gamma2 = gamma1 in the model
  var = hmc_samples['var']
  
  # Evaluate model on full x_scale for visualization and quantiles
  F = F_np(x_scale, A_, X_, amp_, gamma1_, gamma2_)
  qs = mquantiles(F.T, [0.025, 0.975], axis=0)
  F_mean = F.mean(axis=1)
  
  # Accumulate results
  idx.append(num)
  A_freq.append(A_.mean())
  B_freq.append(B_.mean())
  gamma1_vals.append(gamma1_.mean())
  amp_vals.append(amp_.mean())
  A_freq_var.append(A_.var())
  gamma1_var.append(gamma1_.var())
  amp_var.append(amp_.var())
  
  print('#################')
  plt.fill_between(x_scale.flatten(), qs[0], qs[1], alpha=0.7, color="#7A68A6");
  plt.plot(x_scale, F_mean)
  plt.plot(x_adaptive, y_adaptive, 'ro'); # plotting the adaptive data subset
  plt.xlabel('frequency axis')
  plt.title('Posterior distribution for the function given distributions for all parameters');
  plt.show()
  print('#################')

print(f'total time processing all data is {(time.time() -  start_time):.3f}')

print('converting data to dataframe') 
df_ = pd.DataFrame(zip(temps, A_freq, B_freq, gamma1_vals), columns=['temp', 'a_freq', 'b_freq', 'gamma1'])

df_.to_csv('./cycle1_map')

from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error

A_var = df_.a_freq.values.reshape(-1,1)
df_.temp = df_.temp +273.15
x_train, x_test, y_train, y_test = train_test_split(A_var, df_.temp, test_size=0.3, random_state=42)

ln = LinearRegression()

ln.fit(x_train, y_train)
temp_pred = ln.predict(x_test)

plt.plot(y_test, temp_pred, 'x'); plt.plot(y_train, ln.predict(x_train), 'ro')
plt.legend(['testing', 'training'])
mse = mean_squared_error(y_test, temp_pred)
print(f'Testing RMSE:{np.sqrt(mse):.4f}')
plt.xlabel('Measured Temp (C)'); plt.ylabel('Predicted Temp (C)'); plt.title('Linear Regression Model for Temperature Prediction');
