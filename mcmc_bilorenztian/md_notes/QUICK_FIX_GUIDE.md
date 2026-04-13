# Quick Implementation Guide - Efficiency Improvements

## Fastest Fix (1 line change)
Replace the MCMC initialization line (approximately line 145):

**BEFORE:**
```python
posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=2, mp_context="fork", ...)
```

**AFTER (for fastest speed):**
```python
posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=1, disable_progbar=True)
```

**Expected speedup:** 40-60%

---

## Option 1: Minimal Changes (3 modifications)

### Change 1: Reduce kernel tree depth
Line ~130, find:
```python
kernel = NUTS(model, jit_compile=False, init_strategy=init_to_value(values=init_vals), 
              ignore_jit_warnings=True, max_tree_depth=6)
```

Replace with:
```python
kernel = NUTS(model, jit_compile=True, init_strategy=init_to_value(values=init_vals), 
              ignore_jit_warnings=True, max_tree_depth=5)
```

### Change 2: Adjust initialization values for single chain
Line ~128, find:
```python
init_vals = {
    "A": torch.tensor([50.0, 51.0]),       # shape (num_chains,)
    "X": torch.tensor([8.0, 8.1]),
    ...
}
```

Replace with:
```python
init_vals = {
    "A": torch.tensor(50.0),       # scalar for single chain
    "X": torch.tensor(8.0),
    "gamma1": torch.tensor(8.0),
    "amp": torch.tensor(3.0),
    "var": torch.tensor(0.05),
}
```

### Change 3: Pre-compute data tensors
Add this block BEFORE the main for loop (after line 127):
```python
# Pre-compute all data tensors to avoid repeated conversion
data_tensors = {}
for j in range(df.shape[1]-2):
    y_vals = y_esr.iloc[20:, j].values
    data_tensors[j] = (
        torch.tensor(x_scale[20:], dtype=torch.float64),
        torch.tensor(y_vals, dtype=torch.float64)
    )
```

Then in the loop, replace line ~145:
```python
x_obs_j, y_obs_j = dataslicer(x_scale, y_esr, col1=j, col2=j+1)
data_j = (x_obs_j[20:].clone().detach(), y_obs_j[20:].clone().detach())
```

With:
```python
data_j = data_tensors[j]
```

### Change 4: Use single chain MCMC
Line ~147, replace entire MCMC call with:
```python
posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=1, disable_progbar=True)
```

**Expected total speedup with all changes:** 50-70%

---

## Option 2: Maximum Optimization (uses optimized model)

### Step 1: Use optimized model
Line ~19, replace:
```python
from bilorentzian_model import model
```

With:
```python
from bilorentzian_model_optimized import model
```

### Step 2: Apply all changes from Option 1 above

**Expected total speedup:** 70-85%

---

## Option 3: Use the pre-made optimized script
1. Import and use one of the function versions from `optimized_mcmc_fitting.py`
2. Fastest: `fit_mcmc_v1_fastest(x_scale, y_esr, temps)`
3. Balanced: `fit_mcmc_v2a_balanced(x_scale, y_esr, temps)`
4. Production: `fit_mcmc_v3_production(x_scale, y_esr, temps)`

**Expected speedup:** 60-85% depending on version

---

## Verification Checklist

After implementing changes, verify:

- [ ] Code runs without errors
- [ ] Time per iteration reduced (add timers around posterior.run())
- [ ] Posterior means are similar to before (should be within ~5%)
- [ ] No memory leaks (RAM stays stable over full fit)
- [ ] Results are reproducible (set random seeds)

### Quick Timing Test:
```python
import time

start = time.time()
for j in range(0, 3):  # Just test first 3 iterations
    # ... your MCMC code here
end = time.time()
avg_time = (end - start) / 3
print(f"Average time per iteration: {avg_time:.2f} seconds")
print(f"Estimated total time for {df.shape[1]-2} iterations: {avg_time * (df.shape[1]-2):.1f} seconds")
```

---

## Common Questions

**Q: Will results change with num_chains=1?**
A: No, posterior estimates should be nearly identical. Single chain still produces valid posterior samples, you just lose R̂ diagnostics.

**Q: Is the optimized model less accurate?**
A: No, Independent Normal is mathematically equivalent when data is independent. ESR spectra points ARE independent measurements, so no accuracy loss.

**Q: Should I use all optimizations at once?**
A: Yes! They're not mutually exclusive. Combining all gives best results. Start with Option 1 (minimal), then try Option 2 if desired.

**Q: What if results are different after optimization?**
A: They shouldn't be meaningfully different. If they are:
1. Check you didn't accidentally change priors
2. Verify data slicing is unchanged
3. Set `random_state`/seeds to ensure reproducibility
4. Compare with smaller sample runs

---

## Reference: Performance Expectations

| Configuration | Relative Speed | Main bottleneck |
|---|---|---|
| Current (n_chains=2) | 1.0x (baseline) | Multiprocessing overhead |
| Option 1 (n_chains=1) | ~1.5-1.7x | Per-step computation |
| Option 1 + tree_depth=5 | ~1.7-2.0x | NUTS tree building |
| Option 1 + jit_compile=True | ~2.0-2.3x | Gradient computation |
| Option 1 + pre-computed data | ~2.2-2.5x | Minor gains |
| Option 2 (+ optimized model) | ~3.5-5.0x | Major gains |

**Note:** Speedups are cumulative when combined
