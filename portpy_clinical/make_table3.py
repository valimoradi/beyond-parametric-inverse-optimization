"""Table 3: per-component observation counts, for the two anchors alone and for the cohorts N = 3..6.

usage: python make_table3.py
"""
import os
import pickle
import sys

import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from config import FUNC_LABELS, N_FUNCTIONS
from inverse.outcomes import extract_per_voxel_outcomes

N_VALUES = [3, 4, 5, 6]

with open('results/all_patients/forward_cache_nested_N3.pkl', 'rb') as f:
    cache = pickle.load(f)
columns = {'Anchors': extract_per_voxel_outcomes(cache['patients'][:2], cache['w_solutions'][:2])['obs_per_func']}
for N in N_VALUES:
    columns[f'N={N}'] = [len(np.load(f'results/recovery/N{N}/Z_func{k}_N{N}.npy')) for k in range(N_FUNCTIONS)]

print(f"{'Component':24s}" + "".join(f"{c:>10s}" for c in columns))
for k in range(N_FUNCTIONS):
    print(f"{FUNC_LABELS[k]:24s}" + "".join(f"{columns[c][k]:>10,d}" for c in columns))
print(f"{'Total':24s}" + "".join(f"{sum(columns[c]):>10,d}" for c in columns))
