"""
run_parametric_comparison.py
============================
Parametric IO benchmarks (Keshavarz et al. 2011) on the nested PortPy cohorts
(the parametric series of Figure 6).

The true forward objective is f_k(z_v) = alpha_k * z_v^2 (quadratic).
Three parametric forms are fitted:
  rho=1  (linear):    misspecified
  rho=2  (quadratic): correct form
  rho=3  (cubic):     misspecified

For each N in {3,4,5,6} and each rho:
  1. Load the training patients from the nested cache.
  2. Fit alpha_k with the estimator chosen by --method (kkt: the KKT residual
     estimator fit_keshavarz_kkt reported in the paper; oracle and nnls are
     the other estimators of parametric_io.py).
  3. Predict the plans of the 20 test patients with the fitted parametric
     forward model.
  4. Score them with --metric (w: beamlet Rel-L2; dose: dose Rel-L2;
     penalty: true-objective regret; mape: high-dose dose MAPE).

Outputs (suffixed by --tag, default = method)
----------------------------------------------
  results/parametric_comparison/parametric_mape_nested_<tag>.csv

Usage
-----
  python run_parametric_comparison.py --method kkt --metric w       --tag kkt
  python run_parametric_comparison.py --method kkt --metric dose    --tag dose_kkt
  python run_parametric_comparison.py --method kkt --metric penalty --tag penalty_kkt
  python run_parametric_comparison.py --method kkt --metric mape    --tag mape_kkt
"""

import os, sys, argparse, pickle, time
import numpy as np

os.environ.setdefault("PYTHONUNBUFFERED", "1")

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import (
    OAR_KEYS, OAR_GROUPS, N_FUNCTIONS, FUNC_LABELS, FORWARD_PARAMS,
    DOSE_THRESHOLDS,
)


def _true_penalty(dose, organ_indices):
    """True quadratic overdose penalty J = sum_k alpha_k sum_v max(0,d_v-theta_k)^2."""
    J = 0.0
    for k in OAR_KEYS:
        idx = organ_indices.get(k, np.array([], dtype=int))
        if len(idx) == 0:
            continue
        o = np.maximum(0.0, np.asarray(dose)[idx] - DOSE_THRESHOLDS[k])
        J += FORWARD_PARAMS[f'alpha_{k}'] * np.sum(o ** 2)
    return J


def penalty_regret_pct(w_pred, w_true, D, organ_indices):
    """Percentage true-objective regret 100*(J(w_pred)-J(w_true))/J(w_true)."""
    d_true = np.asarray(D @ np.asarray(w_true)).ravel()
    d_pred = np.asarray(D @ np.asarray(w_pred)).ravel()
    Jt = _true_penalty(d_true, organ_indices)
    Jp = _true_penalty(d_pred, organ_indices)
    if Jt <= 0:
        return np.nan
    return 100.0 * (Jp - Jt) / Jt


# Low-dose threshold for the dose-MAPE metric: the AAPM TG-218 low-dose threshold
# (Miften et al., Med. Phys. 2018), 10% of the maximum dose, applied per patient.
# Voxels below it form the low-dose bath, where a per-voxel percentage error
# divides by a near-zero dose. The mask is defined on the ground-truth dose
# d_true = D w_true, so every predictor is scored on the same voxels.
MAPE_LOW_DOSE_FRAC = 0.10  # TG-218 low-dose threshold: 10% of the maximum dose


def dose_mape_masked(w_pred, w_true, D, frac=MAPE_LOW_DOSE_FRAC):
    """Per-voxel MAPE over voxels above the TG-218 low-dose threshold
    (d_true >= frac * max(d_true)), evaluated on the ground-truth dose."""
    d_true = np.asarray(D @ np.asarray(w_true)).ravel()
    d_pred = np.asarray(D @ np.asarray(w_pred)).ravel()
    m = d_true >= frac * d_true.max()
    if not np.any(m):
        return np.nan
    return 100.0 * np.mean(np.abs(d_pred[m] - d_true[m]) / d_true[m])
from parametric_io import (
    fit_parametric_alpha, fit_keshavarz_kkt, fit_keshavarz_obj_value,
    solve_forward_parametric, rel_error_beamlets, rel_error_dose,
)
from prediction import load_test_patients


N_VALUES   = [3, 4, 5, 6]
RHO_VALUES = [1, 2, 3]

OUT_DIR = 'results/parametric_comparison'


def load_nested_training(N):
    """
    Load the N training patients (two anchors and N - 2 real patients) from
    the nested cache. Returns (patients, w_solutions, obj_values).
    """
    cache_file = f'results/all_patients/forward_cache_nested_N{N}.pkl'
    if not os.path.exists(cache_file):
        raise FileNotFoundError(f"Cache not found: {cache_file}")
    with open(cache_file, 'rb') as f:
        cache = pickle.load(f)
    patients    = cache['patients']
    w_solutions = cache['w_solutions']
    obj_values  = cache.get('obj_values', None)
    return patients, [np.asarray(w) for w in w_solutions], obj_values


def fit_alpha_dispatch(method, training_patients, training_w, rho,
                       training_obj, ptv_bind_tol, verbose):
    """Route to the requested estimator."""
    if method == 'oracle':
        return fit_parametric_alpha(
            training_patients, training_w, rho,
            training_obj_values=training_obj, verbose=verbose)
    elif method == 'kkt':
        return fit_keshavarz_kkt(
            training_patients, training_w, rho,
            ptv_bind_tol=ptv_bind_tol, verbose=verbose)
    elif method == 'nnls':
        return fit_keshavarz_obj_value(
            training_patients, training_w, rho,
            training_obj_values=training_obj, verbose=verbose)
    raise ValueError(f"unknown method: {method}")


def run_one_N_rho(N, rho, test_patients, test_w, method='kkt',
                  ptv_bind_tol=0.1, verbose=True, metric='w'):
    """Fit parametric IO and evaluate on test patients. Returns the per-patient values of `metric`."""
    print(f"\n  -- N={N}, rho={rho}, method={method} --", flush=True)
    t0 = time.time()

    # Load training patients
    training_patients, training_w, training_obj = load_nested_training(N)
    print(f"    Training: {len(training_patients)} patients", flush=True)

    # Fit alpha_k
    try:
        alpha_hat = fit_alpha_dispatch(
            method, training_patients, training_w, rho,
            training_obj, ptv_bind_tol, verbose)
    except Exception as e:
        print(f"    FIT FAILED: {e}", flush=True)
        return [np.nan] * len(test_patients)

    # Per-function true reference (fold OARs by func_idx; femur_l/femur_r share
    # alpha, so assignment, not summation).
    true_pf = np.zeros(N_FUNCTIONS)
    for k in OAR_KEYS:
        true_pf[OAR_GROUPS[k]['func_idx']] = FORWARD_PARAMS[f'alpha_{k}']
    print(f"    alpha_hat = {np.round(alpha_hat, 4)}", flush=True)
    print(f"    True alpha (per function) = {np.round(true_pf, 4)}", flush=True)

    # Predict on each test patient
    mape_list = []
    for i, (patient, w_true) in enumerate(zip(test_patients, test_w)):
        t1 = time.time()
        w_pred = solve_forward_parametric(
            patient['D'], patient['organ_indices'],
            alpha_hat, rho, verbose=False
        )
        elapsed = time.time() - t1
        if w_pred is not None:
            if metric == 'dose':
                m = rel_error_dose(w_pred, np.asarray(w_true), patient['D'])
            elif metric == 'penalty':
                m = penalty_regret_pct(w_pred, np.asarray(w_true),
                                       patient['D'], patient['organ_indices'])
            elif metric == 'mape':
                m = dose_mape_masked(w_pred, np.asarray(w_true), patient['D'])
            else:
                m = rel_error_beamlets(w_pred, np.asarray(w_true))
            print(f"    test[{i:2d}] {patient['patient_id']}: "
                  f"RelErr={m:.4f}  ({elapsed:.1f}s)", flush=True)
        else:
            m = np.nan
            print(f"    test[{i:2d}] {patient['patient_id']}: "
                  f"FAILED ({elapsed:.1f}s)", flush=True)
        mape_list.append(m)

    total = time.time() - t0
    valid = [m for m in mape_list if not np.isnan(m)]
    print(f"    Mean RelErr = {np.mean(valid):.4f}  "
          f"({len(valid)}/{len(mape_list)} valid, {total:.1f}s total)", flush=True)
    return mape_list


def write_csv(results, out_path, N_values=None, rho_values=None):
    lines = ['N,rho,mean_rel_error,std_rel_error,n_valid']
    for N in (N_values or N_VALUES):
        for rho in (rho_values or RHO_VALUES):
            vals = [v for v in results.get((N, rho), []) if not np.isnan(v)]
            mean = np.mean(vals) if vals else np.nan
            std  = np.std(vals)  if vals else np.nan
            lines.append(f"{N},{rho},{mean:.4f},{std:.4f},{len(vals)}")
    with open(out_path, 'w') as f:
        f.write('\n'.join(lines) + '\n')
    print(f"Saved CSV: {out_path}", flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-test', type=int, default=20)
    parser.add_argument('--seed',   type=int, default=123)
    parser.add_argument('--rho',    nargs='+', type=float, default=RHO_VALUES,
                        help='Which rho values to run (default: 1 2 3); '
                             'floats allowed (e.g. 1.5 2.5)')
    parser.add_argument('--N-list', nargs='+', type=int, default=N_VALUES,
                        help='Which N values to run (default: 3 4 5 6)')
    parser.add_argument('--method', choices=['oracle', 'kkt', 'nnls'],
                        default='kkt',
                        help='alpha estimator: kkt (non-oracle Keshavarz KKT '
                             'residual; the paper method, DEFAULT), '
                             'oracle (true alpha; upper-bound diagnostic), '
                             'nnls (non-oracle objective-value NNLS)')
    parser.add_argument('--ptv-bind-tol', type=float, default=0.1,
                        help='binding-PTV dose tolerance (Gy) for KKT method')
    parser.add_argument('--tag', type=str, default=None,
                        help='filename suffix for outputs (default: method)')
    parser.add_argument('--metric', choices=['w', 'dose', 'penalty', 'mape'],
                        default='w',
                        help='error space: w (beamlet Rel-L2, paper default), '
                             'dose (voxel Rel-L2 on d = D w), '
                             'penalty (%% true-objective regret 100*(J_pred-J_true)/J_true), '
                             'or mape (%% per-voxel dose MAPE over d_true >= 10%% of max d_true)')
    parser.add_argument('--verbose', action='store_true', default=True)
    args = parser.parse_args()

    os.makedirs(OUT_DIR, exist_ok=True)
    tag = args.tag or args.method
    if args.method == 'oracle':
        print("WARNING: --method oracle injects the TRUE alpha_k (FORWARD_PARAMS) and is an "
              "upper-bound diagnostic ONLY. The paper reports --method kkt (default).",
              file=sys.stderr, flush=True)

    with open('test_patients_canonical.txt') as f:
        test_ids = [line.strip() for line in f if line.strip()][:args.n_test]
    test_patients, test_w, test_obj = load_test_patients(
        'results/all_patients/forward_cache_all.pkl', test_ids)
    print(f"Test set: {len(test_patients)} patients", flush=True)

    results = {}
    for N in args.N_list:
        for rho in args.rho:
            mape_list = run_one_N_rho(
                N, rho, test_patients, test_w,
                method=args.method,
                ptv_bind_tol=args.ptv_bind_tol, verbose=args.verbose,
                metric=args.metric,
            )
            results[(N, rho)] = mape_list

    csv_name = f'parametric_mape_nested_{tag}.csv'
    write_csv(results, os.path.join(OUT_DIR, csv_name),
              N_values=args.N_list, rho_values=args.rho)

    # Print summary table
    print("\n===== SUMMARY =====")
    print(f"{'N':>4}  " + "  ".join(f"{'rho='+str(r):>12}" for r in args.rho))
    for N in args.N_list:
        row = f"{N:>4}  "
        for rho in args.rho:
            vals = [v for v in results.get((N, rho), []) if not np.isnan(v)]
            mean = np.nanmean(vals) if vals else np.nan
            row += f"  {mean:>10.3f}%"
        print(row)


if __name__ == '__main__':
    main()
