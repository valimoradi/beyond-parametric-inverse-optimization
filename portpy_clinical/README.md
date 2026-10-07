# Clinical-Scale Radiotherapy Study (Section 4.3, Figures 5–6, Table 3)

Prostate cases from the public [PortPy](https://github.com/PortPy-Project/PortPy) dataset
(`Prostate_26Fx` protocol). A forward planner with per-voxel quadratic overdose penalties on
six structures produces the reference plans. The recovery imputes the six shared penalty
functions from nested training cohorts of N = 3, 4, 5, 6 patients. The recovered objective then
predicts the plans of 20 held-out patients, against KKT-residual parametric benchmarks with
linear, quadratic and cubic penalties.

## Contents

| Path | Contents |
| --- | --- |
| `config.py`, `data/`, `forward/`, `inverse/`, `recovery/` | forward planner and recovery model: per-voxel outcomes, C5 adjacent-pair smoothness, C6 monotonicity, C7 two-anchor normalization, the per-patient LP dual and the first-order certificate (Appendix B) |
| `prediction.py`, `prediction_solver.py`, `anchor_reduction.py` | the prediction SOCP of Appendix B.4 and its tangent-line reduction |
| `parametric_io.py`, `run_parametric_comparison.py` | parametric benchmarks |
| `results/recovery/N{3,4,5,6}/` | recovered `δ_k`, `λ_k`, outcomes `Z_k`, `ε*`, `β*`, `β_used` (Figure 5, Table 3) |
| `results/prediction/doses_N{3,4,5,6}/` | true and predicted doses and beamlet weights of the 20 test patients (Figure 6) |
| `results/parametric_comparison/` | benchmark metrics per N and ρ (Figure 6) |
| `test_patients_canonical.txt` | the 20 held-out patients |

## Cohorts

Each training cohort contains two boundary anchors and N − 2 unmodified patients. The anchors
are built from patients 61 (best) and 6 (worst) by `build_synthetic_anchors.py`.

| N | Unmodified patients |
| --- | --- |
| 3 | 35 |
| 4 | 44, 35 |
| 5 | 44, 35, 20 |
| 6 | 44, 35, 20, 37 |

## Reproducing the results

The PortPy patient data are not redistributed. Download them from PortPy and set
`PORTPY_DATA_DIR` in `config.py`. The forward caches (`results/all_patients/*.pkl`) are rebuilt
by steps 1–2.

```bash
# 1. Forward plans for all patients
python solve_all_forward.py

# 2. Boundary anchors and nested training caches
python build_synthetic_anchors.py
python build_nested_caches.py

# 3. Recovery, N = 3..6: Stage 1 minimum epsilon at beta_bar = 1000,
#    Stage 2 bisection on beta from 100, Stage 3 selection
bash run_recovery.sh

# 4. Predictions for the 20 test patients
bash run_predictions.sh

# 5. Parametric benchmarks (four metrics)
for m in w dose penalty mape; do
  t=$([ $m = w ] && echo kkt || echo ${m}_kkt)
  python run_parametric_comparison.py --method kkt --metric $m --tag $t
done

# 6. Figures 5 and 6 (results/figures/) and Table 3
python make_figure5.py
python make_figure6.py
python make_table3.py
```

Steps 5–6 run in minutes. Steps 3–4 are long and memory-intensive: the recovery takes about
2–10 h per cohort with MOSEK's default thread count. A prediction takes 0.3–4.0 h per patient at one
MOSEK thread. Figures 5–6 and Table 3 can be rebuilt
from the shipped results without steps 1–5. Figure 6 and Table 3 also need
`results/all_patients/forward_cache_all.pkl` (Figure 6, for the test patients' structures) and
`forward_cache_nested_N3.pkl` (Table 3, anchors column).

## Requirements

PortPy, `numpy`, `scipy`, `cvxpy`, `pandas`, `matplotlib`, and **MOSEK** (required for the conic
programs).
