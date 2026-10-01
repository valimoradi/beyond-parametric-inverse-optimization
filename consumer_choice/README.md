# Consumer choice (Section 4.1, Figures 1–2)

A consumer chooses a bundle `x >= 0` by solving

```
max_x  U(x) - p^T x
```

for a price vector `p` and a concave utility `U`. From bundles observed under many price
vectors we impute `U` under four structural classes, then predict held-out bundles by
re-solving the forward problem with the imputed utility. The error is the test-set relative
L2 error of the predicted bundle, averaged over 200 held-out price vectors.

|               | no smoothness | β-smooth          |
| ------------- | ------------- | ----------------- |
| no additivity | Convex only   | Smooth            |
| additive      | Additive      | Additive + Smooth |

The additive classes are recovered and predicted good by good.

## Data

Two ground-truth utilities, each drawn once:

* `smooth` (Figure 1): `U(x) = 40 Σ_i log(a_i x_i + b_i)`, `a_i ~ U[1,1000]`, `b_i ~ U[2,60]`.
* `kicks3` (Figure 2): `U(x) = 40 min{U_1, U_2, U_3}`, three coupled-log branches over the
  same goods (nonadditive and nonsmooth). Branches 2–3 permute the slope pool of branch 1 and
  rescale goods 2 and 4 by 1.5 in opposite directions; the interaction matrix `W` couples
  eight pairs, seven symmetrically.

600 price vectors `p ~ U[8,20]^5`; 200 are held out for testing and 400 form the training
pool. Two synthetic anchors are added: `U(0) = 0` and `U(1000·1) = 1000`. A training set of
size N holds the two anchors and the first N − 2 bundles of an ordering of the pool, for
N = 20, 40, …, 200. Rep 0 uses the generated order and reps 1–19 use seed `1000 + r`. In the
`not-perturbed` regime the training bundles are exact optima; in the `perturbed` regime each
coordinate is scaled by `1 + U[-0.05, 0.05]`. The test set is exact in both.

`data/` holds the generated data used for the paper.

## Method

All four classes use the exact inverse-optimality certificate with a free witness per
observation: (C4) for the joint classes and (C8) for the additive classes. Model 1 minimizes
ε (the smooth classes at β̄ = 1000) and sets ε* = ε₀ + 10⁻⁶. For the smooth classes,
Model 2 bisects for the smallest feasible β. Model 3 is the conservative selection,
minimizing the sum of recovered utility values. All programs are solved with MOSEK at
tolerance 10⁻⁶, with the retry steps defined in `_solve`, `_min_epsilon` and
`_BETA_ESCALATION`; a cell whose solve still fails is left out of the results.

## Requirements

Python with `numpy`, `cvxpy`, `matplotlib` and `pandas` (`pip install -r ../requirements.txt`),
plus **MOSEK** with a license (free for academics at mosek.com). Every program here, including
data generation and prediction, is solved with MOSEK through CVXPY; GUROBI is not used.

## Reproduce

Run from this folder.

```
# Figures 1-2 from the shipped results (a few seconds)
python make_paper_panels.py results figures

# Full recovery run: 2 utilities x 2 regimes x 4 classes x 10 sizes x 20 replications
CONSUMER_WORKERS=10 python run_parallel.py --utilities smooth kicks3 --reps 20 \
    --out-name paper --cert c8 --cache-dir data
python make_paper_panels.py results_paper_c8 figures_paper
```

The full run takes about three days with 10 workers (one MOSEK thread each); a single Smooth
or Additive + Smooth cell at N = 200 takes about an hour. `run_parallel.py` writes
`results_paper_c8/` (per-cell checkpoint, CSVs, solver log) and `figures_paper_c8/` (±1 SE
bands), and resumes from the checkpoint if restarted. `make_paper_panels.py` draws the paper
panels with ±2 SE bands.

## Shipped results

`results/` holds the run behind the paper: 3,198 of the 3,200 cells. Two fits did not solve and
are omitted from their means: Additive + Smooth (`smooth`, not-perturbed, N = 200, rep 16) and
Additive (`kicks3`, not-perturbed, N = 120, rep 8).

* `_cells_checkpoint.csv`: one row per solved cell (test Rel-L2 and β*); the figures are built from it.
* `results_<utility>_<regime>_reps.csv`: the per-cell values.
* `results_<utility>_<regime>.csv`: the per-size mean and SE.

`figures/` holds the four paper panels and the legend.

Mean test Rel-L2 over 20 replications (not-perturbed / perturbed):

Figure 1, additive smooth truth:

| N   | Convex only | Additive | Smooth | Additive + Smooth |
| --- | ----------- | -------- | ------ | ----------------- |
| 20  | 0.275 / 0.275 | 0.114 / 0.117 | 0.120 / 0.129 | 0.093 / 0.095 |
| 100 | 0.179 / 0.178 | 0.028 / 0.036 | 0.081 / 0.107 | 0.024 / 0.034 |
| 200 | 0.153 / 0.151 | 0.013 / 0.023 | 0.072 / 0.109 | 0.008 / 0.020 |

Figure 2, nonadditive nonsmooth truth:

| N   | Convex only | Additive | Smooth | Additive + Smooth |
| --- | ----------- | -------- | ------ | ----------------- |
| 20  | 0.326 / 0.324 | 0.159 / 0.160 | 0.190 / 0.195 | 0.148 / 0.147 |
| 100 | 0.197 / 0.197 | 0.122 / 0.120 | 0.123 / 0.133 | 0.107 / 0.108 |
| 200 | 0.160 / 0.159 | 0.120 / 0.121 | 0.102 / 0.113 | 0.107 / 0.107 |

Additive + Smooth for `smooth`, not-perturbed, N = 200 averages 19 replications.

## Layout

```
consumer_inverse_optimization.py   models, certificates, predictors, data generation, plotting
run_parallel.py                    multiprocess driver with per-cell checkpointing
make_paper_panels.py               paper panels from a checkpoint (±2 SE bands)
data/                              generated data caches (smooth, kicks3)
results/                           the paper run: checkpoint and CSVs
figures/                           the four paper panels and the legend
```
