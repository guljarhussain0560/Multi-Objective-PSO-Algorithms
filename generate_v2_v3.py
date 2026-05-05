import json

def create_notebook(filename, source_code):
    lines = [line + '\n' for line in source_code.split('\n')]
    # Remove the last newline from the last line to be neat
    if lines:
        lines[-1] = lines[-1].rstrip('\n')
        
    nb = {
        "cells": [
            {
                "cell_type": "code",
                "execution_count": None,
                "metadata": {},
                "outputs": [],
                "source": lines
            }
        ],
        "metadata": {
            "kernelspec": {
                "display_name": "Python 3",
                "language": "python",
                "name": "python3"
            },
            "language_info": {
                "codemirror_mode": {"name": "ipython", "version": 3},
                "file_extension": ".py",
                "mimetype": "text/x-python",
                "name": "python",
                "nbconvert_exporter": "python",
                "pygments_lexer": "ipython3",
                "version": "3.10.11"
            }
        },
        "nbformat": 4,
        "nbformat_minor": 4
    }
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(nb, f, indent=1)

code_v2 = """!pip install pymoo numpy pandas matplotlib seaborn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

ALGO_NAME = "SGMGPSO-FULL-V2"
N_PARTICLES = 150
ARCHIVE_SIZE = 300
N_RUNS = 30

def polynomial_mutation(X, xl, xu, eta=20, prob_multiplier=1.0):
    n, d = X.shape
    prob = (1.0 / d) * prob_multiplier
    mut_mask = np.random.random((n, d)) < prob
    delta = xu - xl
    delta = np.where(delta < 1e-10, 1e-10, delta)
    u = np.random.random((n, d))
    delta1 = np.clip((X - xl) / delta, 0, 1)
    delta2 = np.clip((xu - X) / delta, 0, 1)
    pow_val = 1.0 / (eta + 1.0)
    mask_low = u <= 0.5
    xy = 1.0 - delta1
    val = 2.0 * u + (1.0 - 2.0 * u) * np.power(np.clip(xy, 0, None), eta + 1.0)
    deltaq_low = np.power(np.clip(val, 0, None), pow_val) - 1.0
    xy = 1.0 - delta2
    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * np.power(np.clip(xy, 0, None), eta + 1.0)
    deltaq_high = 1.0 - np.power(np.clip(val, 0, None), pow_val)
    deltaq = np.where(mask_low, deltaq_low, deltaq_high)
    X_mutated = X + deltaq * delta * mut_mask
    return np.clip(X_mutated, xl, xu)

def sample_full_stability_matrix(n_particles, n_dims, w_bounds):
    W = np.full((n_particles, n_dims), 0.4)
    L = np.full((n_particles, n_dims), 0.5)
    C1 = np.full((n_particles, n_dims), 1.2)
    C2 = np.full((n_particles, n_dims), 1.2)
    C3 = np.full((n_particles, n_dims), 1.8)
    
    filled = np.zeros((n_particles, n_dims), dtype=bool)
    
    for attempt in range(30):
        remaining = ~filled
        n_remaining = remaining.sum()
        if n_remaining == 0:
            break
        rows, cols = np.where(remaining)
        w_mins = np.array([max(0.05, w_bounds[c][0]) for c in cols])
        w_maxs = np.array([min(0.95, w_bounds[c][1]) for c in cols])
        w_maxs = np.maximum(w_maxs, w_mins + 0.05)
        
        w = np.random.uniform(w_mins, w_maxs)
        lambd = np.random.uniform(0.2, 0.8, n_remaining)
        c1 = np.random.uniform(0.5, 2.0, n_remaining)
        c2 = np.random.uniform(0.5, 2.0, n_remaining)
        c3 = np.random.uniform(0.5, 3.0, n_remaining)
        
        C = c1 + lambd * c2 + (1 - lambd) * c3
        C_var = (c1**2 + (lambd**2)*(c2**2) + ((1-lambd)**2)*(c3**2))
        num = 4 * (1 - w**2)
        den = (1 - w) + (C_var * (1 + w)) / (3 * (C**2))
        
        valid = (C > 0) & (den > 0) & (C < (num / den))
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            r, c_idx = rows[idx], cols[idx]
            if not filled[r, c_idx]:
                W[r, c_idx] = w[idx]
                L[r, c_idx] = lambd[idx]
                C1[r, c_idx] = c1[idx]
                C2[r, c_idx] = c2[idx]
                C3[r, c_idx] = c3[idx]
                filled[r, c_idx] = True
    return W, L, C1, C2, C3

def update_archive(problem, archive, new_pop, max_size=ARCHIVE_SIZE):
    if archive is None or len(archive) == 0:
        combined = new_pop
    else:
        combined = Population.merge(archive, new_pop)
    survival = RankAndCrowdingSurvival()
    return survival.do(problem, combined, n_survive=min(len(combined), max_size))

def tournament_selection_archive(archive, size=5):
    if archive is None or len(archive) == 0: 
        return None
    indices = np.random.choice(len(archive), min(size, len(archive)), replace=False)
    candidates = archive[indices]
    cds = candidates.get("crowding")
    if cds is not None:
        cds_safe = np.nan_to_num(cds, nan=0.0, posinf=1e6)
        return candidates[np.argmax(cds_safe)].x
    return candidates[np.random.randint(len(candidates))].x

def run_sgmgpso_full_v2(problem_obj, n_particles=N_PARTICLES, n_gen=300, seed=0):
    np.random.seed(seed)
    xl, xu = problem_obj.bounds()
    n_dims = problem_obj.n_var
    vmax_initial = 0.2 * (xu - xl)
    
    X = np.random.uniform(xl, xu, (n_particles, n_dims))
    V = np.random.uniform(-0.1 * (xu - xl), 0.1 * (xu - xl), (n_particles, n_dims))
    F = problem_obj.evaluate(X)
    pop = Population.new("X", X, "F", F)
    
    pbest_X, pbest_F = np.copy(X), np.copy(F)
    archive = update_archive(problem_obj, Population(), pop)
    
    w_history = np.zeros((n_gen, n_dims))
    conv_0 = np.std(V, axis=0) / (np.std(X, axis=0) + 1e-10)
    
    # Adaptive parameter schedules
    mut_prob_schedule = np.linspace(2.0, 0.5, n_gen)
    
    for gen in range(n_gen):
        # Dynamic Vmax
        vmax = vmax_initial * (1 - 0.5 * (gen / n_gen))
        
        # Non-linear TVSR
        conv_t = np.std(V, axis=0) / (np.std(X, axis=0) + 1e-10)
        # Using exponential decay for smooth transitions
        shift_d = np.exp(- (conv_t / (conv_0 + 1e-10))**2)
        w_bounds = list(zip(0.4 - 0.3 * shift_d, 0.9 - 0.5 * shift_d))
        
        W, L, C1, C2, C3 = sample_full_stability_matrix(n_particles, n_dims, w_bounds)
        w_history[gen, :] = np.mean(W, axis=0) 
        
        # Adaptive neighborhood based on generation
        nb_radius = 3 if gen < 0.5 * n_gen else 5
        
        for i in range(n_particles):
            nb_idx = np.arange(i - nb_radius, i + nb_radius + 1) % n_particles
            lbest = pbest_X[nb_idx[np.argmin(np.sum(pbest_F[nb_idx], axis=1))]]
            
            a_hat = tournament_selection_archive(archive, size=7)
            if a_hat is None: a_hat = lbest
            
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            r3 = np.random.random(n_dims)
            
            V[i] = (W[i] * V[i] + 
                    C1[i] * r1 * (pbest_X[i] - X[i]) + 
                    L[i] * C2[i] * r2 * (lbest - X[i]) + 
                    (1 - L[i]) * C3[i] * r3 * (a_hat - X[i]))
            
            V[i] = np.clip(V[i], -vmax, vmax)
            X[i] = np.clip(X[i] + V[i], xl, xu)
            
        # Apply adaptive polynomial mutation
        X = polynomial_mutation(X, xl, xu, eta=20, prob_multiplier=mut_prob_schedule[gen])
        
        F = problem_obj.evaluate(X)
        for i in range(n_particles):
            if np.all(F[i] <= pbest_F[i]) and np.any(F[i] < pbest_F[i]):
                pbest_X[i], pbest_F[i] = np.copy(X[i]), np.copy(F[i])
            elif not (np.all(pbest_F[i] <= F[i]) and np.any(pbest_F[i] < F[i])):
                if np.random.random() < 0.3: # slightly stricter update
                    pbest_X[i], pbest_F[i] = np.copy(X[i]), np.copy(F[i])
        
        archive = update_archive(problem_obj, archive, Population.new("X", X, "F", F))

    return archive, w_history

def plot_pareto_comparison_2d(pf, obtained_F, algo_name, func_name, output_path, color='#1976D2'):
    fig, ax = plt.subplots(figsize=(8, 6))
    if pf is not None:
        sorted_pf = pf[pf[:, 0].argsort()]
        ax.plot(sorted_pf[:, 0], sorted_pf[:, 1], color='black', lw=2.0, 
                label='True Pareto Front', alpha=0.85, zorder=5)
    ax.scatter(obtained_F[:, 0], obtained_F[:, 1], facecolor='none', 
              edgecolor=color, s=40, linewidths=1.2, 
              label=f'{algo_name} Obtained', alpha=0.85, zorder=10)
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_title(f'{algo_name} vs True Pareto Front — {func_name}')
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_comparison_3d(pf, obtained_F, algo_name, func_name, output_path, color='#1976D2'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if pf is not None:
        ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c='gray', s=8, 
                  label='True Pareto Front', alpha=0.15, zorder=1)
    ax.scatter(obtained_F[:, 0], obtained_F[:, 1], obtained_F[:, 2], 
             facecolor='none', edgecolor=color, s=30, linewidths=1.0,
             label=f'{algo_name} Obtained', alpha=0.8, zorder=10)
    ax.set_xlabel(r'$f_1$', labelpad=10)
    ax.set_ylabel(r'$f_2$', labelpad=10)
    ax.set_zlabel(r'$f_3$', labelpad=10)
    ax.set_title(f'{algo_name} vs True Pareto Front — {func_name}')
    ax.legend(loc='best', framealpha=0.9)
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_w_heatmap(w_history, func_name, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(w_history.T, cmap='mako', ax=ax,
                cbar_kws={'label': 'Inertia weight (w)'})
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Dimension Index')
    ax.set_title(f'{ALGO_NAME} Per-Dimension w Evolution — {func_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    benchmarks = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", 
                  "wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]
    
    all_results = []
    
    for b_name in benchmarks:
        print(f"\\n{'='*60}")
        print(f"Processing: {b_name.upper()} ({ALGO_NAME})")
        print(f"{'='*60}")
        output_dir = f"./results/{ALGO_NAME}/{b_name}/"
        os.makedirs(output_dir, exist_ok=True)
        
        is_wfg = b_name.lower().startswith("wfg")
        if is_wfg:
            problem = get_problem(b_name, n_var=24, n_obj=3, k=4)
        else:
            problem = get_problem(b_name)
        
        n_gen = 400 if is_wfg else 300
        if b_name.lower() == 'zdt4':
            n_gen = 500
            
        pf = problem.pareto_front()
        ref_point = np.array([3.0, 5.0, 7.0]) if is_wfg else np.ones(problem.n_obj) * 2.5
        
        run_results = []
        best_igd = np.inf
        best_F, best_w_hist = None, None
        
        for run in range(N_RUNS):
            print(f"  Trial {run + 1}/{N_RUNS} running...", end="\\r")
            res_archive, w_hist = run_sgmgpso_full_v2(problem, n_gen=n_gen, seed=run)
            F = res_archive.get("F")
            
            igd = IGD(pf).do(F) if pf is not None else np.nan
            hv = HV(ref_point=ref_point).do(F) if pf is not None else np.nan
            
            run_results.append({"run": run + 1, "IGD": igd, "HV": hv})
            
            if igd < best_igd:
                best_igd = igd
                best_F = F.copy()
                best_w_hist = w_hist.copy()

        print(f"\\n  [Done] {b_name.upper()} — {N_RUNS} trials completed.")
        df = pd.DataFrame(run_results)
        df.to_csv(f"{output_dir}metrics.csv", index=False)
        
        igd_mean, igd_std = df['IGD'].mean(), df['IGD'].std()
        hv_mean, hv_std = df['HV'].mean(), df['HV'].std()
        
        print(f"  Final Metrics: IGD {igd_mean:.6f} ± {igd_std:.6f} | HV {hv_mean:.6f} ± {hv_std:.6f}")
        
        all_results.append({
            'Function': b_name.upper(),
            'IGD_Mean': igd_mean, 'IGD_Std': igd_std,
            'HV_Mean': hv_mean, 'HV_Std': hv_std
        })

        if best_w_hist is not None:
            plot_w_heatmap(best_w_hist, b_name.upper(), f"{output_dir}w_heatmap.png")

        if problem.n_obj == 2:
            plot_pareto_comparison_2d(pf, best_F, ALGO_NAME, b_name.upper(),
                                     f"{output_dir}pareto_front_comparison.png")
        else:
            plot_pareto_comparison_3d(pf, best_F, ALGO_NAME, b_name.upper(),
                                     f"{output_dir}pareto_front_comparison.png")
    
    print(f"\\n{'='*80}")
    print(f"FINAL SUMMARY TABLE — {ALGO_NAME}")
    print(f"{'='*80}")
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(f"./results/{ALGO_NAME}/summary.csv", index=False)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()"""

code_v3 = """!pip install pymoo numpy pandas matplotlib seaborn

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from pymoo.core.population import Population
from pymoo.problems import get_problem
from pymoo.indicators.igd import IGD
from pymoo.indicators.hv import HV
from pymoo.algorithms.moo.nsga2 import RankAndCrowdingSurvival

plt.rcParams.update({
    'font.family': 'serif',
    'font.size': 12,
    'axes.labelsize': 14,
    'axes.titlesize': 15,
    'legend.fontsize': 11,
    'xtick.labelsize': 11,
    'ytick.labelsize': 11,
    'figure.dpi': 150,
    'savefig.dpi': 300,
    'savefig.bbox': 'tight',
    'axes.grid': True,
    'grid.alpha': 0.3
})

ALGO_NAME = "SGMGPSO-FULL-V3"
N_PARTICLES = 150
ARCHIVE_SIZE = 300
N_RUNS = 30

def polynomial_mutation(X, xl, xu, eta=20):
    n, d = X.shape
    prob = 1.0 / d
    mut_mask = np.random.random((n, d)) < prob
    delta = xu - xl
    delta = np.where(delta < 1e-10, 1e-10, delta)
    u = np.random.random((n, d))
    delta1 = np.clip((X - xl) / delta, 0, 1)
    delta2 = np.clip((xu - X) / delta, 0, 1)
    pow_val = 1.0 / (eta + 1.0)
    mask_low = u <= 0.5
    xy = 1.0 - delta1
    val = 2.0 * u + (1.0 - 2.0 * u) * np.power(np.clip(xy, 0, None), eta + 1.0)
    deltaq_low = np.power(np.clip(val, 0, None), pow_val) - 1.0
    xy = 1.0 - delta2
    val = 2.0 * (1.0 - u) + 2.0 * (u - 0.5) * np.power(np.clip(xy, 0, None), eta + 1.0)
    deltaq_high = 1.0 - np.power(np.clip(val, 0, None), pow_val)
    deltaq = np.where(mask_low, deltaq_low, deltaq_high)
    X_mutated = X + deltaq * delta * mut_mask
    return np.clip(X_mutated, xl, xu)

def sample_full_stability_matrix(n_particles, n_dims, w_bounds):
    W = np.full((n_particles, n_dims), 0.4)
    L = np.full((n_particles, n_dims), 0.5)
    C1 = np.full((n_particles, n_dims), 1.2)
    C2 = np.full((n_particles, n_dims), 1.2)
    C3 = np.full((n_particles, n_dims), 1.8)
    
    filled = np.zeros((n_particles, n_dims), dtype=bool)
    
    for attempt in range(30):
        remaining = ~filled
        n_remaining = remaining.sum()
        if n_remaining == 0:
            break
        
        rows, cols = np.where(remaining)
        w_mins = np.array([max(0.05, w_bounds[c][0]) for c in cols])
        w_maxs = np.array([min(0.95, w_bounds[c][1]) for c in cols])
        w_maxs = np.maximum(w_maxs, w_mins + 0.05)
        
        w = np.random.uniform(w_mins, w_maxs)
        lambd = np.random.uniform(0.2, 0.8, n_remaining)
        c1 = np.random.uniform(0.5, 2.0, n_remaining)
        c2 = np.random.uniform(0.5, 2.0, n_remaining)
        c3 = np.random.uniform(0.5, 3.0, n_remaining)
        
        C = c1 + lambd * c2 + (1 - lambd) * c3
        C_var = (c1**2 + (lambd**2)*(c2**2) + ((1-lambd)**2)*(c3**2))
        num = 4 * (1 - w**2)
        den = (1 - w) + (C_var * (1 + w)) / (3 * (C**2))
        
        valid = (C > 0) & (den > 0) & (C < (num / den))
        valid_indices = np.where(valid)[0]
        for idx in valid_indices:
            r, c_idx = rows[idx], cols[idx]
            if not filled[r, c_idx]:
                W[r, c_idx] = w[idx]
                L[r, c_idx] = lambd[idx]
                C1[r, c_idx] = c1[idx]
                C2[r, c_idx] = c2[idx]
                C3[r, c_idx] = c3[idx]
                filled[r, c_idx] = True
    return W, L, C1, C2, C3

def update_archive(problem, archive, new_pop, max_size=ARCHIVE_SIZE):
    if archive is None or len(archive) == 0:
        combined = new_pop
    else:
        combined = Population.merge(archive, new_pop)
    survival = RankAndCrowdingSurvival()
    return survival.do(problem, combined, n_survive=min(len(combined), max_size))

def tournament_selection_archive(archive, size=5):
    if archive is None or len(archive) == 0: 
        return None
    indices = np.random.choice(len(archive), min(size, len(archive)), replace=False)
    candidates = archive[indices]
    cds = candidates.get("crowding")
    if cds is not None:
        cds_safe = np.nan_to_num(cds, nan=0.0, posinf=1e6)
        return candidates[np.argmax(cds_safe)].x
    return candidates[np.random.randint(len(candidates))].x

def run_sgmgpso_full_v3(problem_obj, n_particles=N_PARTICLES, n_gen=300, seed=0):
    np.random.seed(seed)
    xl, xu = problem_obj.bounds()
    n_dims = problem_obj.n_var
    vmax = 0.2 * (xu - xl)
    
    X = np.random.uniform(xl, xu, (n_particles, n_dims))
    V = np.random.uniform(-0.1 * (xu - xl), 0.1 * (xu - xl), (n_particles, n_dims))
    F = problem_obj.evaluate(X)
    pop = Population.new("X", X, "F", F)
    
    pbest_X, pbest_F = np.copy(X), np.copy(F)
    archive = update_archive(problem_obj, Population(), pop)
    
    w_history = np.zeros((n_gen, n_dims))
    conv_0 = np.std(V, axis=0) / (np.std(X, axis=0) + 1e-10)
    
    for gen in range(n_gen):
        conv_t = np.std(V, axis=0) / (np.std(X, axis=0) + 1e-10)
        
        # V3 Optimization: Sigmoid-based shift calculation for more responsive adaptation
        beta = 10.0
        exp_val = np.clip(beta * (conv_t / (conv_0 + 1e-10) - 0.5), -100, 100)
        shift_d = 1.0 / (1.0 + np.exp(exp_val))
        w_bounds = list(zip(0.4 - 0.3 * shift_d, 0.9 - 0.5 * shift_d))
        
        W, L, C1, C2, C3 = sample_full_stability_matrix(n_particles, n_dims, w_bounds)
        w_history[gen, :] = np.mean(W, axis=0) 
        
        # OBL (Opposition Based Learning) chance to reset bad pbests
        if gen > 0.1 * n_gen and gen % 20 == 0:
            for i in range(n_particles):
                if np.random.random() < 0.1:
                    obl_X = xl + xu - pbest_X[i]
                    obl_F = problem_obj.evaluate(obl_X)
                    if np.all(obl_F <= pbest_F[i]) and np.any(obl_F < pbest_F[i]):
                        pbest_X[i], pbest_F[i] = obl_X.copy(), obl_F.copy()

        for i in range(n_particles):
            # Global-best topology with random mixing (star/global blend)
            nb_idx = np.random.choice(n_particles, min(10, n_particles), replace=False)
            lbest = pbest_X[nb_idx[np.argmin(np.sum(pbest_F[nb_idx], axis=1))]]
            
            a_hat = tournament_selection_archive(archive, size=3)
            if a_hat is None: a_hat = lbest
            
            r1 = np.random.random(n_dims)
            r2 = np.random.random(n_dims)
            r3 = np.random.random(n_dims)
            
            V[i] = (W[i] * V[i] + 
                    C1[i] * r1 * (pbest_X[i] - X[i]) + 
                    L[i] * C2[i] * r2 * (lbest - X[i]) + 
                    (1 - L[i]) * C3[i] * r3 * (a_hat - X[i]))
            
            V[i] = np.clip(V[i], -vmax, vmax)
            X[i] = np.clip(X[i] + V[i], xl, xu)
            
        # Standard polynomial mutation
        X = polynomial_mutation(X, xl, xu, eta=20)
        
        F = problem_obj.evaluate(X)
        for i in range(n_particles):
            # Strict pareto domination for updating pbest
            if np.all(F[i] <= pbest_F[i]) and np.any(F[i] < pbest_F[i]):
                pbest_X[i], pbest_F[i] = np.copy(X[i]), np.copy(F[i])
            # If mutually non-dominating, random selection
            elif not (np.all(pbest_F[i] <= F[i]) and np.any(pbest_F[i] < F[i])):
                if np.random.random() < 0.5:
                    pbest_X[i], pbest_F[i] = np.copy(X[i]), np.copy(F[i])
        
        archive = update_archive(problem_obj, archive, Population.new("X", X, "F", F))

    return archive, w_history

def plot_pareto_comparison_2d(pf, obtained_F, algo_name, func_name, output_path, color='#4CAF50'):
    fig, ax = plt.subplots(figsize=(8, 6))
    if pf is not None:
        sorted_pf = pf[pf[:, 0].argsort()]
        ax.plot(sorted_pf[:, 0], sorted_pf[:, 1], color='black', lw=2.0, 
                label='True Pareto Front', alpha=0.85, zorder=5)
    ax.scatter(obtained_F[:, 0], obtained_F[:, 1], facecolor='none', 
              edgecolor=color, s=40, linewidths=1.2, 
              label=f'{algo_name} Obtained', alpha=0.85, zorder=10)
    ax.set_xlabel(r'$f_1$')
    ax.set_ylabel(r'$f_2$')
    ax.set_title(f'{algo_name} vs True Pareto Front — {func_name}')
    ax.legend(loc='best', framealpha=0.9)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_pareto_comparison_3d(pf, obtained_F, algo_name, func_name, output_path, color='#4CAF50'):
    fig = plt.figure(figsize=(10, 8))
    ax = fig.add_subplot(111, projection='3d')
    if pf is not None:
        ax.scatter(pf[:, 0], pf[:, 1], pf[:, 2], c='gray', s=8, 
                  label='True Pareto Front', alpha=0.15, zorder=1)
    ax.scatter(obtained_F[:, 0], obtained_F[:, 1], obtained_F[:, 2], 
             facecolor='none', edgecolor=color, s=30, linewidths=1.0,
             label=f'{algo_name} Obtained', alpha=0.8, zorder=10)
    ax.set_xlabel(r'$f_1$', labelpad=10)
    ax.set_ylabel(r'$f_2$', labelpad=10)
    ax.set_zlabel(r'$f_3$', labelpad=10)
    ax.set_title(f'{algo_name} vs True Pareto Front — {func_name}')
    ax.legend(loc='best', framealpha=0.9)
    ax.view_init(elev=25, azim=45)
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def plot_w_heatmap(w_history, func_name, output_path):
    fig, ax = plt.subplots(figsize=(12, 6))
    sns.heatmap(w_history.T, cmap='viridis', ax=ax,
                cbar_kws={'label': 'Inertia weight (w)'})
    ax.set_xlabel('Iteration')
    ax.set_ylabel('Dimension Index')
    ax.set_title(f'{ALGO_NAME} Per-Dimension w Evolution — {func_name}')
    plt.tight_layout()
    plt.savefig(output_path, dpi=300, bbox_inches='tight')
    plt.close()

def main():
    benchmarks = ["zdt1", "zdt2", "zdt3", "zdt4", "zdt6", 
                  "wfg1", "wfg2", "wfg3", "wfg4", "wfg5", "wfg6", "wfg7", "wfg8", "wfg9"]
    
    all_results = []
    
    for b_name in benchmarks:
        print(f"\\n{'='*60}")
        print(f"Processing: {b_name.upper()} ({ALGO_NAME})")
        print(f"{'='*60}")
        output_dir = f"./results/{ALGO_NAME}/{b_name}/"
        os.makedirs(output_dir, exist_ok=True)
        
        is_wfg = b_name.lower().startswith("wfg")
        if is_wfg:
            problem = get_problem(b_name, n_var=24, n_obj=3, k=4)
        else:
            problem = get_problem(b_name)
        
        n_gen = 400 if is_wfg else 300
        if b_name.lower() == 'zdt4':
            n_gen = 500
            
        pf = problem.pareto_front()
        ref_point = np.array([3.0, 5.0, 7.0]) if is_wfg else np.ones(problem.n_obj) * 2.5
        
        run_results = []
        best_igd = np.inf
        best_F, best_w_hist = None, None
        
        for run in range(N_RUNS):
            print(f"  Trial {run + 1}/{N_RUNS} running...", end="\\r")
            res_archive, w_hist = run_sgmgpso_full_v3(problem, n_gen=n_gen, seed=run)
            F = res_archive.get("F")
            
            igd = IGD(pf).do(F) if pf is not None else np.nan
            hv = HV(ref_point=ref_point).do(F) if pf is not None else np.nan
            
            run_results.append({"run": run + 1, "IGD": igd, "HV": hv})
            
            if igd < best_igd:
                best_igd = igd
                best_F = F.copy()
                best_w_hist = w_hist.copy()

        print(f"\\n  [Done] {b_name.upper()} — {N_RUNS} trials completed.")
        df = pd.DataFrame(run_results)
        df.to_csv(f"{output_dir}metrics.csv", index=False)
        
        igd_mean, igd_std = df['IGD'].mean(), df['IGD'].std()
        hv_mean, hv_std = df['HV'].mean(), df['HV'].std()
        
        print(f"  Final Metrics: IGD {igd_mean:.6f} ± {igd_std:.6f} | HV {hv_mean:.6f} ± {hv_std:.6f}")
        
        all_results.append({
            'Function': b_name.upper(),
            'IGD_Mean': igd_mean, 'IGD_Std': igd_std,
            'HV_Mean': hv_mean, 'HV_Std': hv_std
        })

        if best_w_hist is not None:
            plot_w_heatmap(best_w_hist, b_name.upper(), f"{output_dir}w_heatmap.png")

        if problem.n_obj == 2:
            plot_pareto_comparison_2d(pf, best_F, ALGO_NAME, b_name.upper(),
                                     f"{output_dir}pareto_front_comparison.png")
        else:
            plot_pareto_comparison_3d(pf, best_F, ALGO_NAME, b_name.upper(),
                                     f"{output_dir}pareto_front_comparison.png")
    
    print(f"\\n{'='*80}")
    print(f"FINAL SUMMARY TABLE — {ALGO_NAME}")
    print(f"{'='*80}")
    summary_df = pd.DataFrame(all_results)
    summary_df.to_csv(f"./results/{ALGO_NAME}/summary.csv", index=False)
    print(summary_df.to_string(index=False))

if __name__ == "__main__":
    main()"""

# create_notebook(r"c:\Downloads\FinalModels\sgmgpso-pd-tvsr-model-6-v2.ipynb", code_v2)
create_notebook(r"c:\Downloads\FinalModels\sgmgpso-pd-tvsr-model-6-v3.ipynb", code_v3)
