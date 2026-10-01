# Beyond Parametric Inverse Optimization

Code and results for *Beyond Parametric Inverse Optimization* (Valimoradi and Li).

Given observed optimal decisions, the method imputes a convex objective, or equivalently a
concave utility, without a parametric form. Additivity and smoothness (a β-Lipschitz gradient)
can be imposed optionally. Out-of-sample decisions are predicted by re-solving the forward
problem with the imputed objective.

The repository follows the paper's three experiments (Section 4).

| Directory | Paper | Contents |
| --- | --- | --- |
| [`consumer_choice/`](consumer_choice/) | Section 4.1, Figures 1–2 | code, generated data, the replicated run and its figures |
| [`healthcare_synthetic/`](healthcare_synthetic/) | Section 4.2, Figures 3, 4, 7 | code, recovered parameters and figure scripts |
| [`portpy_clinical/`](portpy_clinical/) | Section 4.3, Figures 5–6, Table 3 | code, recovered penalty functions (N = 3–6), predictions and benchmarks |

Each directory has its own README with the exact commands.

## Install

```
pip install -r requirements.txt
```

**MOSEK** is required for the conic programs; free academic licenses are available at
mosek.com. **GUROBI** is used by the synthetic radiotherapy study (see its README). The
clinical study also needs [PortPy](https://github.com/PortPy-Project/PortPy) and its prostate
dataset.

## Data

* `consumer_choice/` ships its generated data.
* `healthcare_synthetic/` regenerates its cohort from a fixed seed; the recovered parameters
  behind the figures are shipped.
* `portpy_clinical/` does not redistribute the PortPy dataset or the forward caches derived
  from it; its README shows how to rebuild them.

## Citation

See [`CITATION.cff`](CITATION.cff).

## License

MIT; see [`LICENSE`](LICENSE).
