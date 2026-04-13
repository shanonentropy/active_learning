# Code Comparison: Before vs After

## Complete MCMC Loop Refactoring

### BEFORE (Current - Slow)
```python
init_vals = {
    "A": torch.tensor([50.0, 51.0]),       # shape (num_chains,)
    "X": torch.tensor([8.0, 8.1]),
    "gamma1": torch.tensor([8.0, 8.2]),
    "amp": torch.tensor([3.0, 3.1]),
    "var": torch.tensor([0.05, 0.06]),
}

kernel = NUTS(model, jit_compile=False, init_strategy=init_to_value(values=init_vals), 
              ignore_jit_warnings=True, max_tree_depth=6)

for j in range(0, df.shape[1]-2):
  print(j)
  pyro.clear_param_store()
  x_obs_j, y_obs_j = dataslicer(x_scale, y_esr, col1=j, col2=j+1)
  data_j = (x_obs_j[20:].clone().detach(), y_obs_j[20:].clone().detach())
  # Use the model callable and pass data to MCMC.run
  posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=2, 
                   mp_context="fork", disable_progbar=True, initial_params=init_vals)
  posterior.run(data_j)
  hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
  # ... rest of processing
```

**Problems:**
- ❌ `num_chains=2` with `mp_context="fork"` = multiprocessing overhead
- ❌ `jit_compile=False` = slower gradients
- ❌ `max_tree_depth=6` = expensive NUTS trees
- ❌ Tensor conversion in loop (x_obs_j, y_obs_j)
- ❌ Multiple `.clone().detach()` calls per iteration

---

### AFTER (Optimized - Fast)
```python
init_vals = {
    "A": torch.tensor(50.0),           # scalar for single chain
    "X": torch.tensor(8.0),
    "gamma1": torch.tensor(8.0),
    "amp": torch.tensor(3.0),
    "var": torch.tensor(0.05),
}

kernel = NUTS(model, jit_compile=True, init_strategy=init_to_value(values=init_vals), 
              ignore_jit_warnings=True, max_tree_depth=5)

# Pre-compute all data tensors OUTSIDE loop
print("Pre-computing data tensors...")
data_tensors = {}
for j in range(df.shape[1]-2):
    y_vals = y_esr.iloc[20:, j].values
    data_tensors[j] = (
        torch.tensor(x_scale[20:], dtype=torch.float64),
        torch.tensor(y_vals, dtype=torch.float64)
    )

for j in range(0, df.shape[1]-2):
  print(f"  [{j+1}/{df.shape[1]-2}]", end="", flush=True)
  pyro.clear_param_store()
  
  data_j = data_tensors[j]  # Pre-computed, zero-copy access
  
  # Single chain, no multiprocessing
  posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=1, 
                   disable_progbar=True)
  posterior.run(data_j)
  hmc_samples = {k: v.detach().cpu().numpy() for k, v in posterior.get_samples().items()}
  # ... rest of processing (unchanged)
```

**Improvements:**
- ✅ `num_chains=1` = no multiprocessing overhead
- ✅ `jit_compile=True` = compiled gradients
- ✅ `max_tree_depth=5` = smaller, cheaper trees
- ✅ Data tensors pre-computed once
- ✅ No redundant clones/detaches
- ✅ Progress indicator shows batch progress

**Expected speedup: 50-70%**

---

## Advanced: With Optimized Model

### Replace:
```python
from bilorentzian_model import model
```

### With:
```python
from bilorentzian_model_optimized import model
```

### Model Changes (for reference):
```python
# BEFORE (slow): O(N^3) matrix operations
pyro.sample("obs", dist.MultivariateNormal(F, var * torch.eye(data[1].shape[0])), obs=data[1])

# AFTER (fast): O(N) element-wise operations
with pyro.plate("data", data[0].size(0)):
    pyro.sample("obs", dist.Normal(F, torch.sqrt(var)), obs=data[1])
```

**Expected additional speedup: 20-25%**
**Combined speedup: 60-85%**

---

## Step-by-Step Refactoring

### 1. Fix Imports (line ~19)
```diff
- from bilorentzian_model import model
+ from bilorentzian_model_optimized import model  # optional for extra speed
```

### 2. Fix Kernel Initialization (line ~130)
```diff
  kernel = NUTS(model, 
-              jit_compile=False,
+              jit_compile=True,
               init_strategy=init_to_value(values=init_vals),
               ignore_jit_warnings=True, 
-              max_tree_depth=6)
+              max_tree_depth=5)
```

### 3. Fix Init Values (line ~128)
```diff
  init_vals = {
-    "A": torch.tensor([50.0, 51.0]),       # shape (num_chains,)
-    "X": torch.tensor([8.0, 8.1]),
-    "gamma1": torch.tensor([8.0, 8.2]),
-    "amp": torch.tensor([3.0, 3.1]),
-    "var": torch.tensor([0.05, 0.06]),
+    "A": torch.tensor(50.0),               # scalar for single chain
+    "X": torch.tensor(8.0),
+    "gamma1": torch.tensor(8.0),
+    "amp": torch.tensor(3.0),
+    "var": torch.tensor(0.05),
  }
```

### 4. Add Pre-computation (before loop, line ~140)
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
```

### 5. Update Loop (line ~140+)
```diff
  for j in range(0, df.shape[1]-2):
-   print(j)
+   print(f"  [{j+1}/{df.shape[1]-2}]", end="", flush=True)
    pyro.clear_param_store()
-   x_obs_j, y_obs_j = dataslicer(x_scale, y_esr, col1=j, col2=j+1)
-   data_j = (x_obs_j[20:].clone().detach(), y_obs_j[20:].clone().detach())
+   data_j = data_tensors[j]
    posterior = MCMC(kernel, num_samples=10, warmup_steps=10, 
-                    num_chains=2, mp_context="fork", disable_progbar=True, 
-                    initial_params=init_vals)
+                    num_chains=1, disable_progbar=True)
```

### 6. Remove dataslicer function (optional, no longer needed)
If not used elsewhere, can delete the `dataslicer` function definition.

---

## Validation

After changes, run this test:
```python
import time

# Run first 3 iterations
times = []
for j in range(0, min(3, df.shape[1]-2)):
    start = time.time()
    # ... MCMC code
    elapsed = time.time() - start
    times.append(elapsed)
    print(f"Iteration {j}: {elapsed:.2f}s")

avg = sum(times) / len(times)
total_est = avg * (df.shape[1]-2)
print(f"\nAverage: {avg:.2f}s/iteration")
print(f"Estimated total: {total_est:.1f}s ({total_est/60:.1f} minutes)")
```

---

## Files Provided

1. **EFFICIENCY_ANALYSIS.md** - Detailed analysis of bottlenecks
2. **QUICK_FIX_GUIDE.md** - Quick implementation steps
3. **optimized_mcmc_fitting.py** - 3 pre-built optimized versions
4. **bilorentzian_model_optimized.py** - Model with faster likelihood
5. **CODE_COMPARISONS.md** - This file (before/after examples)
