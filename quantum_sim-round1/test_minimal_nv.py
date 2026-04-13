import importlib
import traceback

try:
    import odmrsimulator
    importlib.reload(odmrsimulator)

    # Set safe, clamped parameter values (match the ranges used in the notebook)
    odmrsimulator.D0 = 2877.0
    odmrsimulator.T0 = 300.0
    odmrsimulator.alpha = -0.07
    odmrsimulator.T1_0 = 500e-6
    odmrsimulator.T2_0 = 1e-6
    odmrsimulator.gamma = 1.0
    odmrsimulator.D_MHz = 2877.0
    odmrsimulator.D_rad_s = odmrsimulator.D_MHz * 2 * 3.141592653589793 * 2
    odmrsimulator.E_rad_s = None

    # Minimal inputs for NV_ODMR
    BNV = 0.0
    RF_freq = 2877.0  # near D0
    rabi_rate = 3.0
    temp = 300.0

    print('Calling NV_ODMR with:', BNV, RF_freq, rabi_rate, temp)
    val = odmrsimulator.NV_ODMR(BNV, RF_freq, rabi_rate, temp)
    print('NV_ODMR result:', val)

    # Also test helper rate functions
    print('D_of_T:', odmrsimulator.D_of_T(temp))
    print('T1_of_T:', odmrsimulator.T1_of_T(temp))
    print('T2_of_T:', odmrsimulator.T2_of_T(temp))

except Exception:
    traceback.print_exc()
    raise
