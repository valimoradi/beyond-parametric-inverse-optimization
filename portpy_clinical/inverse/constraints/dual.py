"""
inverse/constraints/dual.py
============================
First-order certificate C8' (Appendix B.3) for each training patient i, with
h^(i) written through the exact LP dual of the planning subproblem (Appendix B.5):
  reduced cost  D_P^T (mu_2 - mu_1) + sum_k D_k^T nu_k >= 0,
  mu_1, mu_2 >= 0,  0 <= nu_v <= lambda_k[phi(i,v)],
  sum_k sum_v lambda_k[phi(i,v)] z_{k,v} - h^(i) <= epsilon.
"""

import cvxpy as cp
import numpy as np
import scipy.sparse as sp

from config import (
    PTV_PRESCRIBED, PTV_MAX,
    DOSE_THRESHOLDS, OAR_KEYS, OAR_GROUPS, N_FUNCTIONS,
)


def build_c8_dual(
    variables: dict,
    outcome_data: dict,
    patients: list[dict],
    w_solutions: list[np.ndarray],
    epsilon_val=None,
) -> list:
    N = len(patients)
    constraints = []
    patient_voxel_map = outcome_data['patient_voxel_map']
    per_patient_maps = _build_per_patient_maps(patient_voxel_map, N)

    delta_funcs = variables['delta_funcs']
    lambda_funcs = variables['lambda_funcs']
    Z_per_func = outcome_data['Z_per_func']

    for i in range(N):
        pdata = patients[i]
        organ_indices = pdata['organ_indices']
        w_i = w_solutions[i]
        D_i = pdata['D']  # sparse

        mu_1 = variables['mu_1'][i]
        mu_2 = variables['mu_2'][i]
        sigma = variables['sigma'][i]

        idx_ptv = organ_indices.get('ptv', np.array([], dtype=int))
        dose_i = pdata.get('dose')
        if dose_i is None:
            dose_i = np.asarray(D_i @ w_i).ravel()

        # =============================================================
        # Dual feasibility in w (reduced cost):
        # D_P^T(mu2 - mu1) + sum_k D_k^T nu_k - sigma >= 0, sigma >= 0
        # mu_1, mu_2 are duals of the PTV hard constraints
        # =============================================================
        stationarity = -sigma

        if mu_1 is not None and mu_2 is not None and len(idx_ptv) > 0:
            D_ptv = D_i[idx_ptv, :]
            if not sp.issparse(D_ptv):
                D_ptv = sp.csr_matrix(D_ptv)
            stationarity = stationarity + (mu_2 - mu_1) @ D_ptv

        for oar_key in OAR_KEYS:
            idx = organ_indices.get(oar_key, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            nu_k = variables[f'nu_{oar_key}'][i]
            if nu_k is None:
                continue
            D_oar = D_i[idx, :]
            if not sp.issparse(D_oar):
                D_oar = sp.csr_matrix(D_oar)
            stationarity = stationarity + nu_k @ D_oar

        constraints.append(stationarity >= 0)

        # =============================================================
        # Dual feasibility of the overdose multipliers: nu_v <= lambda_k[phi(i,v)]
        # (nu_v >= 0 by declaration; lambda_v = nu_v + kappa_v with kappa_v >= 0).
        # =============================================================
        for oar_key in OAR_KEYS:
            idx = organ_indices.get(oar_key, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            func_idx = OAR_GROUPS[oar_key]['func_idx']
            nu_k = variables[f'nu_{oar_key}'][i]
            if nu_k is None:
                continue
            mapping = per_patient_maps[i].get(oar_key)
            if mapping is None or len(mapping['local']) == 0:
                continue
            constraints.append(nu_k[mapping['local']] <= lambda_funcs[func_idx][mapping['global']])

        # =============================================================
        # C8': per-patient first-order certificate
        #
        #   sum_k sum_v lambda_k[phi(i,v)] * z_{k,v} - h^(i) <= epsilon
        #
        # where h^(i) = min_{w in W(D_i)} sum_k sum_v lambda_k[phi(i,v)]
        # max(0, [D_i w]_v - theta_k) enters through its LP dual objective
        # sum(mu_1) PTV_PRESCRIBED - sum(mu_2) PTV_MAX - sum_k theta_k sum(nu_k).
        # =============================================================
        primal_expr = 0
        has_primal = False
        for oar_key in OAR_KEYS:
            idx = organ_indices.get(oar_key, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            func_idx = OAR_GROUPS[oar_key]['func_idx']

            mapping = per_patient_maps[i].get(oar_key)
            if mapping is None or len(mapping['local']) == 0:
                continue

            global_idx = mapping['global']
            z_vals = Z_per_func[func_idx][global_idx]

            primal_expr = primal_expr + z_vals @ lambda_funcs[func_idx][global_idx]
            has_primal = True

        # Dual objective h^(i)
        dual_expr = 0
        has_dual = False
        if mu_1 is not None and mu_2 is not None and len(idx_ptv) > 0:
            dual_expr = dual_expr + cp.sum(mu_1) * PTV_PRESCRIBED - cp.sum(mu_2) * PTV_MAX
            has_dual = True

        for oar_key in OAR_KEYS:
            idx = organ_indices.get(oar_key, np.array([], dtype=int))
            if len(idx) == 0:
                continue
            nu_k = variables[f'nu_{oar_key}'][i]
            if nu_k is None:
                continue
            theta = DOSE_THRESHOLDS[oar_key]
            dual_expr = dual_expr - cp.sum(nu_k) * theta
            has_dual = True

        if has_primal and has_dual:
            constraints.append(primal_expr - dual_expr <= epsilon_val)

    return constraints


def _build_per_patient_maps(patient_voxel_map: list, N: int) -> list[dict]:
    """
    Build vectorized index maps per patient.
    Returns list of N dicts, each mapping oar_key -> {
        'local': array of local voxel indices,
        'global': array of global observation indices in S_k
    }
    """
    collector = [{} for _ in range(N)]

    for k, entries in enumerate(patient_voxel_map):
        for global_idx, (pat_idx, oar_key, v_local) in enumerate(entries):
            if oar_key not in collector[pat_idx]:
                collector[pat_idx][oar_key] = {'local': [], 'global': []}
            collector[pat_idx][oar_key]['local'].append(v_local)
            collector[pat_idx][oar_key]['global'].append(global_idx)

    for i in range(N):
        for oar_key in collector[i]:
            collector[i][oar_key]['local'] = np.array(collector[i][oar_key]['local'], dtype=int)
            collector[i][oar_key]['global'] = np.array(collector[i][oar_key]['global'], dtype=int)

    return collector
