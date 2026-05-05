# PSO Algorithm Optimization Analysis & Plan

## Critical Issues Identified Across All 6 Models

### 1. **ZDT4 Completely Fails (IGD > 10, HV = 0)** — All Models
- ZDT4 has 10 variables with massive multimodal landscape (x2..x10 ∈ [-5, 5])
- **Root Cause**: No mutation operator → swarm gets trapped in local optima
- **Fix**: Add polynomial mutation with adaptive probability

### 2. **ZDT2 Inconsistent (IGD ~0.4-0.6)** — Most Models
- Concave Pareto front makes convergence harder
- **Root Cause**: Poor velocity clamping + archive too small
- **Fix**: Velocity clamping at 20% of range + larger archive (300) + more generations

### 3. **WFG Problems Have High IGD** — All Models
- WFG suite is inherently harder for PSO-based methods
- **Root Cause**: k=4 is too small for 3-objective problems (should be ≥ 2*(n_obj-1) = 4, but n_var=24 with k=4 gives only 20 distance params)
- **Fix**: Use standard research parameters: n_var=24, n_obj=3, k=4 (this is actually correct per WFG spec where k must be divisible by n_obj-1)

### 4. **Model 5 (TVSR) Has Incomplete Code**
- The actual `run_sgmgpso_tvsr()` function is missing — replaced with a comment `# ... [Previous Stability and run_sgmgpso_tvsr functions remain the same] ...`
- **Fix**: Reconstruct the full function

### 5. **Algorithmic Improvements Needed for Paper Quality**

| Improvement | Impact | All Models |
|---|---|---|
| Velocity clamping (Vmax = 0.2 * range) | Prevents explosion | ✅ |
| Polynomial mutation (pm=20, prob=1/n_var) | Escapes local optima | ✅ |
| Larger archive (200→300) | Better Pareto front coverage | ✅ |
| More generations (200→300 for ZDT, 500 for WFG) | Better convergence | ✅ |
| Epsilon-dominance pbest update | More robust personal best | ✅ |
| Proper velocity initialization (small random) | Better exploration start | ✅ |
| Non-dominated personal best with tie-breaking | Handles equal fronts | ✅ |
| Larger neighborhood (radius 2→3) for hard problems | Better information flow | ✅ |

### 6. **Pareto Front Comparison Plots — Major Improvements Needed**

Current plots are minimal. For paper quality:
- Publication-quality matplotlib styling (serif fonts, proper sizing)
- High DPI (300+) PNG/PDF output
- Proper axis labels with LaTeX formatting
- Grid, tight layout, proper legend placement
- For 3D WFG: multiple viewing angles, better alpha/color
- Save the **best run** (lowest IGD) not the last run

## Optimization Strategy Per Model

| Model | File | Key Change |
|---|---|---|
| 1. SGPSO | sgpso-model-1.ipynb | Add Vmax clamping, mutation, vectorize velocity update |
| 2. MGPSO | mgpso-model-2.ipynb | Increase generations, add mutation, fix leader selection |
| 3. SGMGPSO | sgmgpso-model-3.ipynb | Add mutation, better stability sampling, increase archive |
| 4. SGMGPSO-PP-PD | sgmgpso-pp-pd-model-4.ipynb | Add mutation, increase rejection attempts, vectorize |
| 5. SGMGPSO-TVSR | sgmgpso-tvsr-model-5.ipynb | Reconstruct full code, add mutation, TVSR logic |
| 6. SGMGPSO-PD-TVSR | sgmgpso-pd-tvsr-model-6.ipynb | Add mutation, fix TVSR bounds, larger archive |

## Common Template Changes

All models will receive:
1. **Polynomial mutation** after position update
2. **Velocity clamping** at ±Vmax
3. **Archive size**: 300
4. **Generations**: ZDT=300, WFG=400
5. **Particles**: 150
6. **Publication-quality Pareto front comparison plots**
7. **Best-run selection** (lowest IGD) for plotting
8. **Proper HV reference point**: [1.1]*n_obj for ZDT (normalized), [3,5,7]*appropriate for WFG
