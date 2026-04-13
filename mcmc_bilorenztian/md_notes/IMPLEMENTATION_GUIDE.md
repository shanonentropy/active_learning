# Implementation & Testing Protocol

## Overview
This document guides you through implementing the efficiency improvements and verifying they work.

---

## Executive Summary

**Current Performance:** ~2 min/iteration with `num_chains=2` (too slow)

**Target Performance:** ~30-40 sec/iteration with optimizations (3-4x faster)

**Easiest Fix:** Change `num_chains=2` → `num_chains=1` + enable JIT compilation (40-60% speedup, 1-line changes)

**Maximum Speedup:** Use all optimizations including optimized model (60-85% speedup overall)

---

## Implementation Path

### Option A: Minimal Changes (Recommended for First Try)
**Time to implement:** 10 minutes  
**Expected speedup:** 50-70%  
**Risk level:** Very low

1. Change line 130: `jit_compile=False` → `jit_compile=True`
2. Change line 130: `max_tree_depth=6` → `max_tree_depth=5`
3. Change line 128: Update init_vals from list to scalar format
4. Add data pre-computation block before loop
5. Change MCMC call: `num_chains=2, mp_context="fork"` → `num_chains=1`

**See:** CODE_COMPARISONS.md for exact code

---

### Option B: Maximum Optimization (If Option A is successful)
**Time to implement:** 15 minutes  
**Expected additional speedup:** +15-25%  
**Risk level:** Low

Everything in Option A, plus:
- Switch to `bilorentzian_model_optimized.py`
- This changes likelihood from MultivariateNormal → Independent Normal

**Validation:** Run parallel fits and compare posteriors (should be nearly identical)

---

## Testing Protocol

### Phase 1: Baseline Measurement (Before changes)
```python
# Add this before your MCMC loop
import time
import json

times_baseline = []

for j in range(0, min(5, df.shape[1]-2)):  # Test first 5 iterations
    print(f"Baseline Iteration {j}...", end="", flush=True)
    
    start = time.time()
    # [YOUR CURRENT MCMC CODE HERE]
    pyro.clear_param_store()
    x_obs_j, y_obs_j = dataslicer(x_scale, y_esr, col1=j, col2=j+1)
    data_j = (x_obs_j[20:].clone().detach(), y_obs_j[20:].clone().detach())
    posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=2, 
                     mp_context="fork", disable_progbar=True, initial_params=init_vals)
    posterior.run(data_j)
    # [REST OF CODE]
    elapsed = time.time() - start
    
    times_baseline.append(elapsed)
    print(f" {elapsed:.2f}s")

avg_baseline = sum(times_baseline) / len(times_baseline)
print(f"\n{'='*50}")
print(f"BASELINE: {avg_baseline:.2f}s per iteration")
print(f"{'='*50}\n")

# Save for later comparison
with open('baseline_times.json', 'w') as f:
    json.dump({'times': times_baseline, 'average': avg_baseline}, f)
```

### Phase 2: Apply Option A Changes
1. Make all 5 changes listed in "Option A" above
2. Run the same test code again (modify to test optimized version)

```python
times_optimized = []

for j in range(0, min(5, df.shape[1]-2)):
    print(f"Optimized Iteration {j}...", end="", flush=True)
    
    start = time.time()
    # [YOUR NEW OPTIMIZED MCMC CODE]
    # ... (single chain, etc.)
    elapsed = time.time() - start
    
    times_optimized.append(elapsed)
    print(f" {elapsed:.2f}s")

avg_optimized = sum(times_optimized) / len(times_optimized)
speedup = avg_baseline / avg_optimized

print(f"\n{'='*50}")
print(f"OPTIMIZED: {avg_optimized:.2f}s per iteration")
print(f"SPEEDUP: {speedup:.2f}x faster")
print(f"{'='*50}\n")
```

### Phase 3: Validate Results
```python
# Compare posteriors from baseline vs optimized runs
import pandas as pd
import numpy as np

# After running both versions, compare:
# - Were different chains selected? (should use same first chain)
# - Are posterior means within 5%?
# - Are posterior stds similar?

comparison = pd.DataFrame({
    'metric': ['A_mean', 'X_mean', 'gamma1_mean', 'A_std', 'X_std', 'gamma1_std'],
    'baseline': [
        results_baseline['A'][0], results_baseline['X'][0], results_baseline['gamma1'][0],
        np.std(results_baseline['A']), np.std(results_baseline['X']), np.std(results_baseline['gamma1'])
    ],
    'optimized': [
        results_optimized['A'][0], results_optimized['X'][0], results_optimized['gamma1'][0],
        np.std(results_optimized['A']), np.std(results_optimized['X']), np.std(results_optimized['gamma1'])
    ]
})

comparison['percent_diff'] = (
    (comparison['optimized'] - comparison['baseline']) / comparison['baseline']
).abs() * 100

print(comparison)
print(f"\nAll differences < 5%? {(comparison['percent_diff'] < 5).all()}")
```

---

## Success Criteria

✅ **Performance:**
- Speedup ≥ 3x per iteration (from ~120s to <40s)
- Stable iteration times (no variance > 20%)

✅ **Accuracy:**
- Posterior means within 5% of baseline
- Standard deviations similar
- Chain visually looks similar

✅ **Stability:**
- No crashes or errors
- Memory usage stable (not growing)
- Reproducible results

---

## Troubleshooting

### Problem: Results are different after optimization
**Check:**
1. Did you accidentally change the priors? Check init_vals format matches model
2. Is data slicing identical? (should use same 20: cutoff)
3. Are you comparing same random seed? Use `torch.manual_seed(0)` before MCMC
4. Did you swap model file? (can cause different results)

**Resolution:** Re-compare with original code using exact same seed

---

### Problem: Code runs but posteriors are unchanged
**Likely cause:** Changes didn't apply correctly

**Check:**
1. Print kernel settings: `print(kernel); print(init_vals)`
2. Verify MCMC call shows `num_chains=1`
3. Check data_tensors is being used (not recomputing)

---

### Problem: Some iterations are much slower than others
**Likely cause:** Adaptive step size warmup

**Normal behavior:** First iteration after `pyro.clear_param_store()` slower (rebuilds adaptation)

**Check:** Enable detailed timing inside iteration:
```python
start_warmup = time.time()
posterior.run(data_j)
elapsed_warmup = time.time() - start_warmup

print(f"Warmup time: {elapsed_warmup:.2f}s")
```

---

### Problem: JIT compilation not helping (or slowing things down)
**Try:** Set `jit_compile=False` for small datasets (<500 pts)
- JIT has overhead that only pays off with large data
- ESR spectra: ~1000 pts usually
- If data is smaller, toggle back to False

---

## Rollback

If anything goes wrong, rolling back is simple:
1. Keep original notebook saved
2. Comment out optimized code, uncomment original
3. Change `num_chains=1` back to `num_chains=2`
4. Run with original model

All changes are backward compatible!

---

## Performance Expectations Timeline

| Stage | Time/Iteration | Note |
|-------|---|---|
| Start | ~120s | num_chains=2 with multiprocessing |
| After Option A | ~40-50s | 2.5-3x faster |
| After Option B | ~30-40s | 3-4x faster |
| After reaching 5-10 chains? | ~15-25s | Running many fits in parallel at iteration level |

---

## Next Steps (After Success)

Once optimizations are working:

1. **Run full dataset** - Complete all temperature points
2. **Collect timing** - How long does full fit take now?
3. **Explore parameter space** - Can now afford to test more priors/settings
4. **Add diagnostics** - Can afford R-hat calculation post-hoc due to saved samples

---

## Questions?

Refer to:
- Detailed analysis: **EFFICIENCY_ANALYSIS.md**
- Code examples: **CODE_COMPARISONS.md**
- Quick reference: **QUICK_FIX_GUIDE.md**
- Pre-built code: **optimized_mcmc_fitting.py**
