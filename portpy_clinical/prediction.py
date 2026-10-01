"""Load a recovered objective and the held-out test patients for prediction."""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import N_FUNCTIONS, FUNC_LABELS


def compute_upper_envelope(z, delta, lam):
    """
    Upper envelope of the recovered tangent lines.

    Each tangent line is y = lam_j * x + (delta_j - lam_j * z_j); the envelope
    max_j { delta_j + lam_j * (x - z_j) } is piecewise linear and convex.
    Lines that never attain the envelope are dropped (first step of the anchor
    reduction, Appendix B.4).

    Returns indices of the lines on the envelope in the input arrays.
    """
    # Each line: y = a*x + b  where a=lam_j, b=delta_j - lam_j*z_j
    a = lam.copy()
    b = delta - lam * z

    n = len(a)
    if n <= 2:
        return np.arange(n)

    # Sort by slope (ascending)
    order = np.argsort(a)
    a = a[order]
    b = b[order]

    # Graham scan for upper envelope of lines
    stack = [0]
    for i in range(1, n):
        while len(stack) >= 2:
            j = stack[-1]
            k = stack[-2]
            # Check if line j is dominated at the intersection of k and i
            denom = a[i] - a[k]
            if abs(denom) > 1e-15:
                x_int = (b[k] - b[i]) / denom
                y_j = a[j] * x_int + b[j]
                y_env = a[i] * x_int + b[i]
                if y_j <= y_env + 1e-12:
                    stack.pop()
                else:
                    break
            else:
                if b[i] >= b[k]:
                    stack.pop()
                else:
                    break
        stack.append(i)

    return order[np.array(stack)]


def load_recovered_functions(res_dir, N):
    """
    Load recovered delta, lambda, z_hat for each function and keep the
    tangent lines on the upper envelope.

    Returns dict with keys 0..K-1, each containing:
      'z_hat': z values (envelope tangent points)
      'delta': delta values (recovered function values)
      'lam':   lambda values (recovered subgradients)
    """
    funcs = {}
    for k in range(N_FUNCTIONS):
        z = np.load(f'{res_dir}/Z_func{k}_N{N}.npy')
        delta = np.load(f'{res_dir}/delta_func{k}_N{N}.npy')
        lam = np.load(f'{res_dir}/lambda_func{k}_N{N}.npy')

        n_total = len(z)

        # Keep the lines on the upper envelope
        env_idx = compute_upper_envelope(z, delta, lam)
        z = z[env_idx]
        delta = delta[env_idx]
        lam = lam[env_idx]

        # Sort by z for clean ordering
        order = np.argsort(z)
        z = z[order]
        delta = delta[order]
        lam = lam[order]

        funcs[k] = {'z_hat': z, 'delta': delta, 'lam': lam}
        print(f"    func {k} ({FUNC_LABELS[k]}): {n_total} obs -> "
              f"{len(z)} envelope lines")

    return funcs


def load_recovered_beta(res_dir):
    """
    Return the beta of the recovered objective.

    The recovered f_k is the Proposition 1 conjugate at this beta, and the
    prediction SOCP (Appendix B.4) is evaluated at the same beta. Reads
    beta_used.npy (written by Stage 3), else beta_star.npy (Stage 2); raises
    if neither exists.
    """
    used = os.path.join(res_dir, 'beta_used.npy')
    star = os.path.join(res_dir, 'beta_star.npy')
    if os.path.exists(used):
        beta = float(np.load(used)[0])
        print(f"    beta_used (from Stage 3) = {beta:.4f}")
        return beta
    if os.path.exists(star):
        beta = float(np.load(star)[0])
        print(f"    beta_star (from Stage 2) = {beta:.4f}  "
              f"(beta_used.npy not present)")
        return beta
    raise FileNotFoundError(
        f"Neither beta_used.npy nor beta_star.npy in {res_dir}. "
        f"Cannot run prediction without the beta from Stage 2/3 because "
        f"the recovered f_k is the beta*-smooth conjugate; using any other "
        f"beta evaluates a different function (see Proposition 1).")


def load_test_patients(all_cache_file, patient_ids):
    """Return (patients, w_solutions, obj_values) of the listed patients, in the listed order."""
    with open(all_cache_file, 'rb') as f:
        cache = pickle.load(f)
    id_to_idx = {p['patient_id']: i for i, p in enumerate(cache['patients'])}
    missing = [pid for pid in patient_ids if pid not in id_to_idx]
    if missing:
        raise RuntimeError(f"Test patients not in {all_cache_file}: {missing}")
    selected = [id_to_idx[pid] for pid in patient_ids]
    return ([cache['patients'][i] for i in selected],
            [cache['w_solutions'][i] for i in selected],
            [float(cache['obj_values'][i]) for i in selected])
