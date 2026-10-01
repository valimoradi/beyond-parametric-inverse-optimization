"""Predict the plans of held-out test patients from the recovered objective.

usage: python predict_test_patients.py N IDX

Reads the recovery in results/recovery/N{N}, reduces its tangent lines (Appendix B.4) and solves the
prediction SOCP for test patient IDX (0-based position in test_patients_canonical.txt; a comma-separated
list of positions is also accepted).
Writes results/prediction/doses_N{N}/<patient>.npz with the true and predicted doses and beamlet weights.
"""
import os
import sys
import time

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
import config
from anchor_reduction import dedup_recovered
from prediction import load_recovered_beta, load_recovered_functions, load_test_patients
from prediction_solver import solve_prediction

ANCHOR_TOL = 1e-5   # tangent-line reduction tolerance, relative to the largest recovered value
config.MOSEK_TOLERANCES['MSK_IPAR_NUM_THREADS'] = int(os.environ.get('MOSEK_THREADS', '1'))

N = int(sys.argv[1])
indices = [int(x) for x in sys.argv[2].split(',')]
res_dir = f'results/recovery/N{N}'
dose_dir = f'results/prediction/doses_N{N}'

recovered = dedup_recovered(load_recovered_functions(res_dir, N), ANCHOR_TOL, verbose=True)
beta = load_recovered_beta(res_dir)
with open('test_patients_canonical.txt') as f:
    test_ids = [line.strip() for line in f if line.strip()]
patients, w_sols, _ = load_test_patients('results/all_patients/forward_cache_all.pkl', test_ids)
os.makedirs(dose_dir, exist_ok=True)

for i in indices:
    p, wt = patients[i], w_sols[i]
    out = f'{dose_dir}/{p["patient_id"]}.npz'
    t0 = time.time()
    wp = solve_prediction(p['D'], p['organ_indices'], recovered, beta_val=beta, mosek_tol=1e-5, timeout=21600.0)
    dt = time.time() - t0
    if wp is None:
        print(f'[{i}] {p["patient_id"]} FAILED ({dt:.0f}s)', flush=True)
        continue
    rel = np.linalg.norm(wp - wt) / max(np.linalg.norm(wt), 1e-10)
    np.savez(out, d_true=p['D'] @ wt, d_pred=p['D'] @ wp, w_true=wt, w_pred=wp,
             ptv_idx=p['organ_indices'].get('ptv', np.array([], dtype=int)))
    print(f'[{i}] {p["patient_id"]} OK ({dt:.0f}s, beamlet error {rel:.4f})', flush=True)
