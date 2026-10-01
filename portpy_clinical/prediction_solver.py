"""
prediction_solver.py
====================
Prediction SOCP (Appendix B.4).

solve_prediction minimizes the recovered objective over the planning constraints
(PTV_PRESCRIBED <= dose <= PTV_MAX on the PTV, w >= 0). Each OAR voxel's penalty
f_k(z_v) is the Proposition 1 conjugate in perspective form over the retained
anchors j, with one rotated second-order cone per (voxel, anchor) pair; the cones
of an OAR are built in one vectorized cp.SOC call. Returns the beamlet weights w,
or None if the solve fails or exceeds the timeout.
"""
import threading
import numpy as np
import cvxpy as cp

from config import (OAR_KEYS, OAR_GROUPS, DOSE_THRESHOLDS,
                    PTV_PRESCRIBED, PTV_MAX, MOSEK_TOLERANCES)


def solve_prediction(D, organ_indices, recovered_funcs, beta_val,
                          verbose=False, mosek_tol=1e-5, timeout=3600.0):
    n_voxels, n_beamlets = D.shape
    w = cp.Variable(n_beamlets, nonneg=True)
    dose = D @ w

    constraints = []
    obj_terms = []

    idx_ptv = organ_indices['ptv']
    if len(idx_ptv) > 0:
        constraints.append(dose[idx_ptv] >= PTV_PRESCRIBED)
        constraints.append(dose[idx_ptv] <= PTV_MAX)

    BETA = float(beta_val)

    for oar_key in OAR_KEYS:
        idx = organ_indices.get(oar_key, np.array([], dtype=int))
        if len(idx) == 0:
            continue

        func_idx = OAR_GROUPS[oar_key]['func_idx']
        rf = recovered_funcs[func_idx]
        theta = DOSE_THRESHOLDS[oar_key]

        n_v = len(idx)
        M = len(rf['z_hat'])
        z_anchors = np.asarray(rf['z_hat'], dtype=float)
        delta_anchors = np.asarray(rf['delta'], dtype=float)
        lambda_anchors = np.asarray(rf['lam'], dtype=float)

        z_v = cp.Variable(n_v, nonneg=True)
        constraints.append(z_v >= dose[idx] - theta)

        alpha = cp.Variable((n_v, M), nonneg=True)
        v_persp = cp.Variable((n_v, M))
        aux = cp.Variable((n_v, M), nonneg=True)

        constraints.append(cp.sum(alpha, axis=1) == 1)
        constraints.append(cp.sum(v_persp, axis=1) >= z_v)

        # Rotated SOC over all (v, j): (v_p - alpha*z_j)^2 <= alpha*aux
        #   <=>  || 2(v_p - alpha z_j) , (alpha - aux) ||_2 <= alpha + aux
        # Flattened in C order so the (v, j) pairing of U, S, T is preserved.
        U = v_persp - cp.multiply(alpha, z_anchors[None, :])   # (n_v, M)
        Uf = cp.reshape(U, (n_v * M,), order='C')
        Sf = cp.reshape(alpha, (n_v * M,), order='C')
        Tf = cp.reshape(aux, (n_v * M,), order='C')
        constraints.append(cp.SOC(Sf + Tf, cp.vstack([2 * Uf, Sf - Tf]), axis=0))

        obj_terms.append(
            (BETA / 2.0) * cp.sum(aux)
            + cp.sum(cp.multiply(v_persp, lambda_anchors[None, :]))
            - cp.sum(cp.multiply(alpha, (lambda_anchors * z_anchors)[None, :]))
            + cp.sum(cp.multiply(alpha, delta_anchors[None, :]))
        )

    if not obj_terms:
        return None

    prob = cp.Problem(cp.Minimize(sum(obj_terms)), constraints)

    tol = {**MOSEK_TOLERANCES,
           'MSK_DPAR_OPTIMIZER_MAX_TIME': max(timeout * 0.8, 480.0),
           'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   mosek_tol,
           'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   mosek_tol,
           'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': mosek_tol}
    _exc = [None]

    def _solve():
        try:
            prob.solve(solver='MOSEK', verbose=verbose, mosek_params=tol)
        except Exception as e:
            _exc[0] = e

    t = threading.Thread(target=_solve, daemon=True)
    t.start()
    t.join(timeout=timeout)

    if t.is_alive():
        return None
    if _exc[0] is not None:
        if isinstance(_exc[0], (cp.SolverError, MemoryError)):
            return None
        raise _exc[0]

    if prob.status in ('optimal', 'optimal_inaccurate'):
        return w.value
    return None
