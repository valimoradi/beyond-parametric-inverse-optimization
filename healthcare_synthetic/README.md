# Synthetic Radiotherapy Study (Section 4.2 and Appendix B.6, Figures 3, 4, 7)

A synthetic test bed. A ground-truth forward planner with quadratic, linear and exponential
organ-at-risk penalties produces optimal plans for randomly generated 2-D anatomies
(55 × 55 voxel grid, 100 beamlets). From the observed plans the method recovers the
objective under the 2 × 2 ablation (additivity × smoothness); the additive classes return the three
penalty functions. Each class is scored by
the held-out beamlet prediction error (Rel-L2).

## Contents

| Path | Contents |
| --- | --- |
| `paper_run/Cancer simulation code seed 142.ipynb` | generates the cohort from seed 142 and recovers the four classes |
| `paper_run/results_high_std/` | Additive + Smooth (`*_add_smooth.npy`) |
| `paper_run/results_high_std_add_nonsmooth_lp/` | Additive (`*_lp.npy`) |
| `paper_run/results_high_std_non_add_smooth/` | Smooth (`*_smooth.npy`) |
| `paper_run/results_high_std_non_add_nonsmooth/` | Convex only (`*_convex.npy`) |
| `paper_run/fig4_data/` | Figure 4 data, one CSV per class |
| `paper_run/figure_scripts/` | the Figure 3, 4 and 7 scripts and the PNGs they write |
| `predict.py`, `config.py` | held-out predictions (Figure 4 data) |

For each training size N = 20, 40, …, 200, a results folder holds the recovered values δ
(`final_delta_*`), gradients λ (`final_lambda_*`), the observed outcomes (`Z_hat_obs_*`),
`beta_star_*` (the β of the last stage; none for Additive) and `epsilon_star_*` (the Stage 1
minimum ε₀). Rows 0 and 1 of every training set are the best- and worst-case anchors, with f = 0
and f = U_max (1000 for the additive classes, 10000 for Smooth and Convex only); N counts both
anchors.

| Notebook cells | Step |
| --- | --- |
| 1–4 | imports, anatomy, dose-influence matrix, forward problem |
| 5, 6 | worst- and best-case anchor search |
| 7, 8 | cohort generation and aggregation |
| 9, 10 | validity filter, 200-patient test split, anchors prepended to the training set |
| 11, 12, 13, 14 | recovery: Additive + Smooth, Additive, Smooth, Convex only |

| Class | Stage 1 | β* | Solver |
| --- | --- | --- | --- |
| Additive + Smooth | min ε at β̄ = 2000 | bisection from [0.01, 10], upper end doubled until feasible; gap 0.11 | MOSEK |
| Additive | min ε (LP) | — | GUROBI |
| Smooth | min ε at β̄ = 2000 | bisection from [0.01, 20], upper end doubled until feasible; gap 0.2 | MOSEK |
| Convex only | min ε at fixed β = 10⁵ | — | MOSEK |

Later stages use ε₀ + 10⁻⁶. The last stage maximizes the sum of δ (the pessimistic representative).

## Reproducing the results

Figures from the shipped results (run from this folder):

```bash
python paper_run/figure_scripts/make_true_vs_retrieved.py   # Figure 3
python paper_run/figure_scripts/make_combined_mape.py       # Figure 4
python paper_run/figure_scripts/make_fig3_zoom.py           # Figure 7
```

Everything from the seed:

1. Run notebook cells 1–10 from `paper_run/`. They generate the cohort
   (`paper_run/ml_data_high_std/` and intermediate folders, about 12 GB in total).
2. Run cells 11–14 to recover the four classes into the `results_*` folders.
3. Compute the Figure 4 data:
   ```bash
   for m in add_smooth add_lp nonadd_smooth nonadd_nonsmooth; do
       SYNTH_DATA_ROOT=paper_run python predict.py $m
   done
   ```
   Three of the 200 test patients are borderline infeasible: the prediction solve finds no feasible
   plan for them at any N, so scores are means over the other 197.
4. Run the three figure scripts.

## Requirements

Python 3 with `numpy`, `scipy`, `cvxpy`, `pandas`, `matplotlib` and `scikit-learn`
(`pip install -r ../requirements.txt`). **MOSEK** solves the forward problem (cell 4 falls back to cvxpy's default solver and then
SCS if MOSEK raises), the smooth and Convex-only classes and the Figure 3 and 7 evaluations. **GUROBI** solves the Additive class
(cell 12 and `predict.py add_lp`).
