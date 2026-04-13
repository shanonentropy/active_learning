# Implementation Checklist - Step by Step

## Pre-Implementation

- [ ] **Backup your notebook** - Save a copy before making changes
  ```
  Copy: pyro_odmr_gaussian.ipynb → pyro_odmr_gaussian_backup.ipynb
  ```

- [ ] **Note current performance** - Time 3 iterations before changes
  ```python
  import time
  times = []
  for j in range(3):
      start = time.time()
      # your MCMC code
      times.append(time.time() - start)
  baseline_avg = sum(times) / len(times)
  print(f"Baseline: {baseline_avg:.1f} sec/iteration")
  ```

- [ ] **Understand current settings** - Print kernel info
  ```python
  print(f"jit_compile: {kernel.jit_compile}")
  print(f"max_tree_depth: {kernel.max_tree_depth}")
  print(f"init_vals type: {type(init_vals['A'])}")
  ```

---

## Phase 1: Quick Fixes (Choose ONE)

### Option 1A: Minimal (Just do standard tuning)
**Expected speedup: ~2.5x | Time: 10 minutes**

#### Step 1.1: Enable JIT compilation
File: `pyro_odmr_gaussian.ipynb`, Line ~130
```diff
  kernel = NUTS(model, 
-              jit_compile=False,
+              jit_compile=True,
               init_strategy=init_to_value(values=init_vals),
```
- [ ] Change `jit_compile=False` → `jit_compile=True`
- [ ] Verify change with: `print(kernel.jit_compile)`

#### Step 1.2: Reduce tree depth
File: `pyro_odmr_gaussian.ipynb`, Line ~130 (same line)
```diff
               ignore_jit_warnings=True,
-              max_tree_depth=6)
+              max_tree_depth=5)
```
- [ ] Change `max_tree_depth=6` → `max_tree_depth=5`
- [ ] Verify change with: `print(kernel.max_tree_depth)`

#### Step 1.3: Update init values format
File: `pyro_odmr_gaussian.ipynb`, Line ~128
```diff
  init_vals = {
-    "A": torch.tensor([50.0, 51.0]),
-    "X": torch.tensor([8.0, 8.1]),
-    "gamma1": torch.tensor([8.0, 8.2]),
-    "amp": torch.tensor([3.0, 3.1]),
-    "var": torch.tensor([0.05, 0.06]),
+    "A": torch.tensor(50.0),
+    "X": torch.tensor(8.0),
+    "gamma1": torch.tensor(8.0),
+    "amp": torch.tensor(3.0),
+    "var": torch.tensor(0.05),
  }
```
- [ ] Convert all tensor values to scalars (remove the lists)
- [ ] Verify: `print(init_vals['A'].shape)` should be empty `torch.Size([])`

#### Step 1.4: Add data pre-computation
File: `pyro_odmr_gaussian.ipynb`, Insert NEW CELL after line ~138
```python
# Pre-compute all data tensors to avoid repeated conversion
print("Pre-computing data tensors...")
data_tensors = {}
for j in range(df.shape[1]-2):
    y_vals = y_esr.iloc[20:, j].values
    data_tensors[j] = (
        torch.tensor(x_scale[20:], dtype=torch.float64),
        torch.tensor(y_vals, dtype=torch.float64)
    )
print(f"Pre-computed {len(data_tensors)} data slices")
```
- [ ] Insert new cell
- [ ] Run cell to verify no errors
- [ ] Should print: "Pre-computed 12 data slices" (or similar)

#### Step 1.5: Update MCMC call - Remove multiprocessing
File: `pyro_odmr_gaussian.ipynb`, Line ~147 (in the for loop)
```diff
  for j in range(0, df.shape[1]-2):
    print(j)
    pyro.clear_param_store()
-   x_obs_j, y_obs_j = dataslicer(x_scale, y_esr, col1=j, col2=j+1)
-   data_j = (x_obs_j[20:].clone().detach(), y_obs_j[20:].clone().detach())
+   data_j = data_tensors[j]
    posterior = MCMC(kernel, num_samples=10, warmup_steps=10, 
-                    num_chains=2, mp_context="fork", disable_progbar=True,
-                    initial_params=init_vals)
+                    num_chains=1, disable_progbar=True)
```
- [ ] Replace data fetching with `data_j = data_tensors[j]`
- [ ] Change `num_chains=2, mp_context="fork"` → `num_chains=1`
- [ ] Remove `initial_params=init_vals` (not needed for single chain)
- [ ] Verify: Should only have `num_chains=1` in MCMC call

- [ ] **VERIFICATION CHECKPOINT**
  ```python
  # Run first 3 iterations to test
  for j in range(3):
      # your MCMC code
  ```
  - Should run without errors
  - Should be noticeably faster

---

### Option 1B: Manual Copy-Paste (Use pre-built code)
**Expected speedup: ~3.8x | Time: 5 minutes**

- [ ] Copy entire MCMC section from `optimized_mcmc_fitting.py`
- [ ] Delete old MCMC loop and supporting init code
- [ ] Paste new code from `fit_mcmc_v3_production()` function
- [ ] Adjust variable names to match your data
- [ ] Run and compare

---

## Phase 2: Validation (20 minutes)

### Step 2.1: Time the optimized version
File: `pyro_odmr_gaussian.ipynb`, Create timing cell
```python
import time
times_optimized = []

print("Testing optimized code (first 3 iterations)...")
for j in range(0, min(3, df.shape[1]-2)):
    start = time.time()
    pyro.clear_param_store()
    
    # Your MCMC code here
    data_j = data_tensors[j]
    posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=1, disable_progbar=True)
    posterior.run(data_j)
    
    elapsed = time.time() - start
    times_optimized.append(elapsed)
    print(f"  Iteration {j}: {elapsed:.1f}s")

avg_optimized = sum(times_optimized) / len(times_optimized)
print(f"\nOptimized average: {avg_optimized:.1f} sec/iteration")
```
- [ ] Run this cell
- [ ] Note the times
- [ ] Calculate speedup: `baseline_avg / avg_optimized`

### Step 2.2: Verify results haven't changed
File: `pyro_odmr_gaussian.ipynb`, Create comparison cell
```python
import numpy as np

# Compare posteriors from first iteration
print("Comparing results...")
print(f"A_mean optimization: {A_freq[0]:.4f}")
print(f"X_mean optimization: {X_freq[0]:.4f}")
print(f"gamma1_mean optimization: {gamma1_vals[0]:.4f}")

# These should match baseline results within ~5%
# If baseline results were saved, compare here
```
- [ ] Results should be similar to before (within 5%)
- [ ] If very different, check that model file is correct

### Step 2.3: Run full pipeline
- [ ] Comment out plotting cells (takes time)
- [ ] Run FULL notebook end-to-end
- [ ] Time total execution
- [ ] Compare to baseline timing

---

## Phase 3: Optional - Model Optimization (10 min, +25% speedup)

### Step 3.1: Switch to optimized model
File: `pyro_odmr_gaussian.ipynb`, Line ~19
```diff
- from bilorentzian_model import model
+ from bilorentzian_model_optimized import model
```
- [ ] Change import to use optimized model
- [ ] Verify file exists: `ls bilorentzian_model_optimized.py`

### Step 3.2: Test compatibility
- [ ] Run first 3 iterations
- [ ] Compare results to previous version
- [ ] Should be nearly identical (same posteriors)

### Step 3.3: Re-run timing test
- [ ] Run timing code again
- [ ] Should see additional 20-25% speedup
- [ ] Total speedup should now be 3.5-4x

---

## Phase 4: Final Verification

### Step 4.1: Full accuracy check
- [ ] Run on all temperatures
- [ ] Create comparison table:
  ```python
  comparison = pd.DataFrame({
      'temperature': temps,
      'a_freq': A_freq,
      'b_freq': B_freq
  })
  print(comparison)
  ```
- [ ] Results should look reasonable
- [ ] No NaNs or infinities

### Step 4.2: Check memory usage
- [ ] Monitor RAM during full run
- [ ] Memory should NOT grow unbounded
- [ ] Should stabilize at some level

### Step 4.3: Reproducibility
- [ ] Run twice with same code
- [ ] Results should be identical (no randomness)
- [ ] Or set `torch.manual_seed(0)` to ensure reproducibility

---

## Phase 5: Cleanup

- [ ] Remove timing debug cells (optional)
- [ ] Keep backup file "just in case"
- [ ] Update any documentation about runtime
- [ ] Celebrate 3-4x speedup! 🎉

---

## Troubleshooting Checklist

### Issue: Code crashes immediately
- [ ] Verify all syntax changes were correct
- [ ] Check parentheses/brackets are balanced
- [ ] Run one cell at a time to find error location

### Issue: Results are NaN
- [ ] Check data_tensors was computed correctly
- [ ] Print first data slice: `print(data_tensors[0])`
- [ ] Verify init_vals are reasonable numbers

### Issue: Still slow (not much speedup)
- [ ] Verify changes actually applied:
  ```python
  print(f"jit_compile: {kernel.jit_compile}")
  print(f"max_tree_depth: {kernel.max_tree_depth}")
  ```
- [ ] Check num_chains actually 1:
  ```python
  posterior = MCMC(..., num_chains=???, ...)  # What does this say?
  ```
- [ ] Profile to find real bottleneck:
  ```python
  import cProfile
  cProfile.run('your_code()', sort='cumulative')
  ```

### Issue: Different results after optimization
- [ ] Set seed before running both versions
- [ ] Use exact same data slices
- [ ] Check model file hasn't changed
- [ ] Differences > 10% suggest a bug - investigate

### Issue: Getting very large posterior uncertainties
- [ ] Check data scaling is unchanged
- [ ] Verify likelihood function (if using optimized)
- [ ] Run with original model to confirm

---

## Success Criteria ✓

All of these should be true:

- [ ] Code runs without errors
- [ ] Timing shows 2.5-4x speedup
- [ ] Posterior means within 5% of baseline
- [ ] Memory stable during run
- [ ] Chain traces look reasonable
- [ ] No NaNs/infinities in results

---

## Documentation to Update

After success, consider documenting:

- [ ] New baseline timing per iteration
- [ ] Total time for full 12-temperature fit
- [ ] Which optimizations were used
- [ ] Any limitations or notes

---

## Next Steps

After optimization complete:

1. **Use saved posteriors** - You now have fast posterior samples
2. **Increase sampling** - Can now afford more samples (20 instead of 10)
3. **Add diagnostics** - Can run R̂ diagnostics post-hoc on samples
4. **Explore parameters** - Try different priors/models with new speed
5. **Run more datasets** - Can fit other cycles/sensors faster

---

## Estimated Timeline

| Phase | Duration | Cumulative |
|-------|----------|-----------|
| Pre-implementation (backup + baseline) | 5 min | 5 min |
| Phase 1: Quick fixes | 10 min | 15 min |
| Phase 2: Validation | 15 min | 30 min |
| Phase 3: Model opt (optional) | 10 min | 40 min |
| Phase 4: Final checks | 10 min | 50 min |
| Total | ~50 min | Ready to run! |

**Then:** 24 min → 6-7 min for full dataset (18 min saved per run!) ⏱️

---

## Questions?

Refer to:
- **Want to understand:** EFFICIENCY_ANALYSIS.md
- **Want code examples:** CODE_COMPARISONS.md
- **Want detailed testing:** IMPLEMENTATION_GUIDE.md
- **Want quick summary:** OPTIMIZATION_SUMMARY.md
