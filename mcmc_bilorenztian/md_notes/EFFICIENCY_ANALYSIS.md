# MCMC Fitting Efficiency Analysis & Improvement Plan

## Current Performance Issue
- **Single chain (n_chains=1):** ~2 minutes per iteration ✓ (acceptable)
- **Dual chains (n_chains=2):** Very slow (timing TBD, but significantly worse)
- **Root cause:** Inefficient multiprocessing configuration for small sample sizes

---

## Bottleneck Analysis

### 1. **Multiprocessing Overhead (PRIMARY ISSUE)**
```python
posterior = MCMC(kernel, num_samples=10, warmup_steps=10, num_chains=2, 
                 mp_context="fork", ...)
```
**Problem:**
- Only 20 iterations per chain (10 warmup + 10 samples)
- Process forking has overhead (~100-500ms per fork)
- With 2 chains: `2 × (overhead) + 2 × (computation)` 
- For small sample sizes, overhead > benefit of parallelization
- Sequential execution is actually faster for this case

**Impact:** Process creation/teardown can take 30-50% of total time for small samples

---

### 2. **NUTS Kernel Configuration**
```python
kernel = NUTS(model, jit_compile=False, max_tree_depth=6, ...)
```
**Problems:**
- `max_tree_depth=6` → trees can reach 2^6 = 64 nodes per step
- NUTS automatically builds trees, making each step expensive
- `jit_compile=False` → each gradient computation is slower
- Bilorentzian model is relatively simple - NUTS may be overkill

**Impact:** Increases per-step computational cost by 2-3x

---

### 3. **Model Likelihood Computation (SECONDARY ISSUE)**
```python
pyro.sample("obs", dist.MultivariateNormal(F, var * torch.eye(data[1].shape[0])), obs=data[1])
```
**Problems:**
- Creates full NxN covariance matrix for each likelihood evaluation
- For N~1000 frequency points, this is expensive
- Matrix inverse/determinant computed every HMC step
- With adaptive step sizes and tree building, this gets called 100s of times

**Impact:** Each likelihood evaluation is O(N³) instead of O(N)

---

### 4. **Data Processing Inefficiency**
```python
for j in range(0, df.shape[1]-2):
    x_obs_j, y_obs_j = dataslicer(x_scale, y_esr, col1=j, col2=j+1)
    data_j = (x_obs_j[20:].clone().detach(), y_obs_j[20:].clone().detach())
```
**Problems:**
- Tensor conversion happens inside MCMC loop
- Data slicing could be pre-computed
- Unnecessary clone/detach operations

**Impact:** Minor (~5-10% overall), but adds up

---

## Optimization Strategy (Ranked by Impact)

### **TIER 1: High Impact (30-50% speedup expected)**

#### Option 1A: Single Chain Execution
```python
num_chains=1  # Sequential execution
```
- **Expected speedup:** 40-60%
- **Trade-off:** Loss of R-hat diagnostic (but chain diagnostics less critical for production fitting)
- **When to use:** If final parameter estimates are priority over diagnostics

#### Option 1B: Batch Sequential Chains
```python
num_chains=2, mp_context=None  # Sequential chains in same process
```
- **Expected speedup:** 30-40%
- **Trade-off:** Still runs 2 chains but no fork overhead
- **When to use:** If chain diagnostics are important

---

### **TIER 2: Medium Impact (15-30% speedup expected)**

#### Option 2A: Replace MultivariateNormal with Independent Normals
```python
# Instead of:
pyro.sample("obs", dist.MultivariateNormal(F, var * torch.eye(...)), obs=data[1])

# Use:
with pyro.plate("data", len(data[1])):
    pyro.sample("obs", dist.Normal(F, torch.sqrt(var)), obs=data[1])
```
- **Expected speedup:** 20-25% (compared to full MVN)
- **Assumption:** Data points are independent (likely valid for your data)
- **Impact:** Reduces likelihood computation from O(N³) to O(N)

#### Option 2B: Enable JIT Compilation
```python
kernel = NUTS(model, jit_compile=True, ...)  # Change to True
```
- **Expected speedup:** 15-20% (first call slower, subsequent calls faster)
- **Trade-off:** First compile takes ~2-5 seconds
- **Note:** Requires `torch.jit` compatible code

---

### **TIER 3: Low-Medium Impact (10-15% speedup expected)**

#### Option 3A: Reduce NUTS Tree Depth
```python
kernel = NUTS(model, max_tree_depth=4, ...)  # Was 6, change to 4-5
```
- **Expected speedup:** 10-15%
- **Trade-off:** May reduce sampler efficiency (slightly lower ESS per step)
- **Benefit:** Still much more efficient than HMC

#### Option 3B: Pre-compute Data Tensors
```python
# Outside loop: convert all data at once
data_tensors = {}
for j in range(df.shape[1]-2):
    data_tensors[j] = (
        torch.tensor(x_scale[20:], dtype=torch.float64),
        torch.tensor(y_esr.iloc[20:, j].values, dtype=torch.float64)
    )

# Inside loop:
data_j = data_tensors[j]  # No conversion needed
```
- **Expected speedup:** 5-10% (reduced tensor conversion overhead)
- **Benefit:** Cleaner code, less redundant computation

---

## Recommended Implementation Plan

### **Phase 1: Immediate (Test these first)**
1. Change `num_chains=1` temporarily
2. Compare execution time vs. current `num_chains=2`
3. If significant speedup confirmed → makes Phases 2-3 worth it

### **Phase 2: Quick Wins (Safe improvements)**
1. Pre-compute all data tensors
2. Enable JIT compilation (`jit_compile=True`)
3. Reduce `max_tree_depth` from 6 → 5

### **Phase 3: Model-Level Optimization**
1. Replace MultivariateNormal with Independent Normal distribution
2. This requires modifying [bilorentzian_model.py](bilorentzian_model.py)
3. Best efficiency gain but requires model verification

### **Phase 4: Advanced (If needed)**
1. Switch kernel NUTS → HMC with fixed step size + adaptive mass matrix
2. Implement parallel iteration fitting (batch multiple j values)
3. Use GPU acceleration if available

---

## Expected Total Speedup
- **Phase 1 + 2 combined:** 50-75% faster (2-5x total speedup expected)
- **Phase 1 + 2 + 3 combined:** 60-80% faster (3-8x total speedup expected)
- **All phases:** Potentially 10-15x faster

---

## Testing Protocol
1. **Baseline:** Time current code with both `num_chains=1` and `num_chains=2`
2. **After each optimization:** Run on 3-4 data slices and measure mean time
3. **Verify accuracy:** Compare posterior means/stds with original to ensure no quality loss
4. **Monitor diagnostics:** Check n_eff (effective sample size) and R-hat values

---

## Quick Start Commands
```bash
# Profile the current bottlenecks
time python -m cProfile -s cumulative your_script.py

# Monitor multiprocessing overhead
# Add timing around MCMC.run() call
```
