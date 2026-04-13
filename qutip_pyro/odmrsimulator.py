'''This modules containst the necessary functions to simulate the NV center ODMR spectra using the Lindblad master equation approach.
The helper functions '''

# import packages
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import gridspec
import matplotlib.colors as mcolors
from matplotlib.pyplot import figure, show, grid, tight_layout
plt.rcParams.update({'font.sans-serif':'DejaVu Sans'})

# Make the Jupyter Notebook fill 90 percent of the screen (nerd_mode)
from IPython.display import display, HTML
display(HTML("<style>:root { --jp-notebook-max-width: 90% !important; }</style>"))

# import qutip for quantum calculations
from qutip import *
from helper import *
import sys


# ################################ Global constants (units and defaults)  ###########################
# Default values - can be overridden by setting module attributes (e.g., odmrsimulator.B = 300)

# Temperature-dependent D parameters
D0 = 2870.0  # MHz (zero-field splitting at T0)
T0 = 300.0   # K (reference temperature)
alpha = -0.074  # MHz/K (temperature coefficient)

# T1 parameters
T1_0 = 1e6  # us (T1 at T0)
beta = 0.01  # 1/K (T1 temperature coefficient)

# T2 parameters
T2_0 = 1e3  # us (T2 at T0)
gamma = 0.005  # 1/K (T2 temperature coefficient)

# Magnetic field (Gauss)
B = 0.0

# Strain parameters
E_rad_s = 2 * np.pi * 2.5e6 * 2  # rad/s (transverse strain splitting, default ~10 MHz)
strain_angle = np.pi / 4  # radians (strain axis angle in xy plane)

# ####################################################################################################


# evolution of state while optically pumping (outputs the expectation value of the expect_op)
def evolve_optical_pumping(rho_0, H, t_end, num_pts, c_op_list, expect_op):
  tlist = np.linspace(0, t_end, num_pts)
  output = mesolve(H, rho_0, tlist, c_op_list, [expect_op])

  return output

# evolution of state while optically pumping (outputs the density matrices / state list)
def evolve_optical_pumping_ds(rho_0, H, t_end, num_pts, c_op_list):
  tlist = np.linspace(0, t_end, num_pts)
  output = mesolve(H, rho_0, tlist, c_op_list)

  return output

''' Evolve the system to get the polarized state under optical pumping (laser on). 
Condition is set to laser on for 3 us with 1000 time steps '''
output = evolve_optical_pumping_ds(rho_0, H_0, 3, 1000, cops)
rho_polarized_0 = output.states[-1]

# now we set the condition to laser off by turning off the laser pumping collapse operators
cops_noLaser = [cop_01, cop_12, cop_20, cop_10, cop_21, cop_02, cop_30, cop_41, cop_52, cop_36, cop_46, cop_56, cop_60, cop_61, cop_62]

# taking the polarized state from the previous step we time evolve for 10 microseconds in 1000 steps with laser off to get the relaxed state
# note we are computing the full density matrix evolution here
rho_relaxed = evolve_optical_pumping_ds(rho_polarized_0, H_0, 10, 10000, cops_noLaser)
rho_relaxed_0 = rho_relaxed.states[-1]

''' Define PL operator to calculate photoluminescence signal 
Note: The actual PL signal will depend on several factors such as:
collection efficiency, optical throughput,quantum efficiency of detector
losses,fiber coupling, detector dead time etc 

here we assume only 0.4% of the florescence photons enter the fiber and if the SPD avg eff is 50% then overall detection efficiency is 0.2%
assuming we are not deadtime limited and detector is not saturated our estimated rate is 0.002 '''


collection_rate = 0.2E-3

PL_op = collection_rate*rate_eg*(es_p1*es_p1.dag()+es_0*es_0.dag()+es_m1*es_m1.dag())

# Temperature-dependent models and ODMR function (with optional strain params)

def D_of_T(T, D0=None, T0=None, alpha=None):
  """Linear model for zero-field splitting (MHz) vs temperature (K).
  If `D0`, `T0` or `alpha` are not provided, the function will read the
  current module-level values so the optimizer func can update those
  globals at runtime and have the change take effect immediately.
  this was added becuase otherwise temp_simulator never reflected local changes
  """
  module = sys.modules[__name__]
  if D0 is None:
    D0 = getattr(module, 'D0', 2870.0)  # Falls back to default if attribute missing
  if T0 is None:
    T0 = getattr(module, 'T0', 300.0)  # Falls back to default if attribute missing
  if alpha is None:
    alpha = getattr(module, 'alpha', -0.074)  # Falls back to default if attribute missing
  return D0 + alpha * (T - T0)


def T1_of_T(T, T1_0=None, T0=None, beta=None):
  """Phenomenological T1 (microseconds) vs T.
  If parameters are omitted the function reads module-level defaults so they
  can be changed at runtime (useful for fitting/optimization).
  """
  module = sys.modules[__name__]
  if T1_0 is None:
    T1_0 = getattr(module, 'T1_0', 1e6)
  if T0 is None:
    T0 = getattr(module, 'T0', 300.0)
  if beta is None:
    beta = getattr(module, 'beta', 0.01)
  return T1_0 * np.exp(-beta * (T - T0))


def T2_of_T(T, T2_0=None, T0=None, gamma=None):
  """Phenomenological T2 (microseconds) vs T.
  Reads module-level defaults when parameters are not provided so optimizers
  can modify those globals and changes are picked up immediately.
  """
  module = sys.modules[__name__]
  if T2_0 is None:
    T2_0 = getattr(module, 'T2_0', 1e3)
  if T0 is None:
    T0 = getattr(module, 'T0', 300.0)
  if gamma is None:
    gamma = getattr(module, 'gamma', 0.005)
  return max(1.0, T2_0 * np.exp(-gamma * (T - T0)))


def NV_ODMR(BNV, RF_freq, rabi_rate, T=300.0, E_rad_s_param=None, strain_angle_param=None):
  """Temperature-aware NV_ODMR with optional strain coupling.
  If `E_rad_s_param` or `strain_angle_param` are not provided, the function uses
  the global `E_rad_s` and `strain_angle` defined in the constants cell.

  Returns PL readout (scalar) for the given RF frequency and temperature.
  """
  # zero-field splitting at temperature T (MHz)
  D = D_of_T(T)

  # Use provided strain params or fall back to module attributes
  module = sys.modules[__name__]
  if E_rad_s_param is None:
    E_used = getattr(module, 'E_rad_s', 2 * np.pi * 2.5e6 * 2)
  else:
    E_used = E_rad_s_param

  if strain_angle_param is None:
    angle_used = getattr(module, 'strain_angle', np.pi/4)
  else:
    angle_used = strain_angle_param

  # convert strain from rad/s to MHz to match D/Delta units used below
  E_MHz = E_used / (2 * np.pi * 1e6)

  # Frequency detunings (MHz)
  Delta_p = (D + 2.8 * BNV) - RF_freq
  Delta_m = (D - 2.8 * BNV) - RF_freq   # B field in Gauss
  Omega_m = rabi_rate
  Omega_p = rabi_rate

  # strain coupling mixes the m=+1 and m=-1 ground states; include as off-diagonal term
  strain_coupling = (np.exp(2j * angle_used) * gs_p1 * gs_m1.dag()
                     + np.exp(-2j * angle_used) * gs_m1 * gs_p1.dag())

  H_rf = 2 * np.pi * (Delta_p * gs_p1 * gs_p1.dag()
                      + Delta_m * gs_m1 * gs_m1.dag()
                      + Omega_p / 2 * gs_0 * gs_p1.dag()
                      + Omega_p / 2 * gs_p1 * gs_0.dag()
                      + Omega_m / 2 * gs_0 * gs_m1.dag()
                      + Omega_m / 2 * gs_m1 * gs_0.dag()
                      + E_MHz * strain_coupling
                      + 4.708E8 * es_p1 * es_p1.dag()
                      + 4.708E8 * es_0 * es_0.dag()
                      + 4.708E8 * es_m1 * es_m1.dag()
                      + 0.5 * 4.708E8 * shelf_state * shelf_state.dag())

  # Dynamic T1/T2 rates (units: 1/us)
  T1_us = T1_of_T(T)
  rate_T1_dyn = 1.0 / T1_us
  T2_us = T2_of_T(T)
  rate_deph = 1.0 / T2_us

  # Reconstruct T1-related collapse operators with temperature-dependent rates
  cop_01_dyn = np.sqrt(rate_T1_dyn) * gs_m1 * gs_0.dag()
  cop_12_dyn = np.sqrt(rate_T1_dyn) * gs_p1 * gs_m1.dag()
  cop_20_dyn = np.sqrt(rate_T1_dyn) * gs_0 * gs_p1.dag()
  cop_10_dyn = cop_01_dyn.dag()
  cop_21_dyn = cop_12_dyn.dag()
  cop_02_dyn = cop_20_dyn.dag()

  # Pure dephasing on ground states (phenomenological)
  cop_phi0 = np.sqrt(rate_deph) * gs_0 * gs_0.dag()
  cop_phim1 = np.sqrt(rate_deph) * gs_m1 * gs_m1.dag()
  cop_phip1 = np.sqrt(rate_deph) * gs_p1 * gs_p1.dag()

  # Build dynamic collapse list: use radiative/ISC/shelve operators already defined globally,
  # but replace the static T1 operators with the dynamic versions and add dephasing.
  dynamic_cops = [cop_03, cop_14, cop_25, cop_30, cop_41, cop_52, cop_36, cop_46, cop_56, cop_60, cop_61, cop_62,
                  cop_01_dyn, cop_12_dyn, cop_20_dyn, cop_10_dyn, cop_21_dyn, cop_02_dyn,
                  cop_phi0, cop_phim1, cop_phip1]

  readout = evolve_optical_pumping(rho_relaxed_0, H_rf, 1, 1000, dynamic_cops, PL_op)
  return readout.expect[0][-1]


