"""
build_nested_caches.py
======================
Build the nested training caches results/all_patients/forward_cache_nested_N{N}.pkl
for N = 3, 4, 5, 6. Each cache holds the two synthetic anchors (best, worst)
followed by the real patients of COHORTS[N]; each cohort adds one patient to
the next smaller one.

Usage:
  python build_nested_caches.py
"""

import os
import sys
import argparse
import pickle
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from config import OAR_KEYS

# Real patients of each training cohort; the two anchors are added to every cohort.
COHORTS = {
    3: ['Prostate_Patient_35'],
    4: ['Prostate_Patient_44', 'Prostate_Patient_35'],
    5: ['Prostate_Patient_44', 'Prostate_Patient_35', 'Prostate_Patient_20'],
    6: ['Prostate_Patient_44', 'Prostate_Patient_35', 'Prostate_Patient_20', 'Prostate_Patient_37'],
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--n-list', type=str, default='3,4,5,6',
                        help='comma-separated cohort sizes to build (default 3,4,5,6)')
    args = parser.parse_args()
    n_list = [int(x) for x in args.n_list.split(',') if x.strip()]

    # Load full forward cache
    all_cache_file = 'results/all_patients/forward_cache_all.pkl'
    print(f"Loading {all_cache_file} ...")
    with open(all_cache_file, 'rb') as f:
        cache = pickle.load(f)

    all_patients = cache['patients']
    all_w = cache['w_solutions']
    all_obj = np.array(cache['obj_values'])
    print(f"  {len(all_patients)} patients total")

    # Load the synthetic anchors written by build_synthetic_anchors.py. They are
    # contour-modified versions of the best/worst real patients and serve as the
    # lower and upper C7 anchors.
    synth_cache_file = 'results/all_patients/forward_cache_anchors.pkl'
    print(f"Loading synthetic anchors from {synth_cache_file} ...")
    with open(synth_cache_file, 'rb') as f:
        synth_cache = pickle.load(f)
    synth_best = synth_cache['patients'][0]
    synth_worst = synth_cache['patients'][1]
    synth_best_w = synth_cache['w_solutions'][0]
    synth_worst_w = synth_cache['w_solutions'][1]
    synth_best_obj = float(synth_cache['obj_values'][0])
    synth_worst_obj = float(synth_cache['obj_values'][1])
    real_best_real_id = synth_best['patient_id'].replace('_synthetic_best', '')
    real_worst_real_id = synth_worst['patient_id'].replace('_synthetic_worst', '')
    print(f"  BEST  anchor: {synth_best['patient_id']}  obj={synth_best_obj:.2f}")
    print(f"  WORST anchor: {synth_worst['patient_id']}  obj={synth_worst_obj:.2f}")

    # Filter to complete-data REAL patients (excluding the real patients that
    # the synthetic anchors were derived from, to avoid duplicating their D).
    excluded_ids = {real_best_real_id, real_worst_real_id}
    complete_indices = []
    for i, p in enumerate(all_patients):
        if p['patient_id'] in excluded_ids:
            continue
        has_all = all(
            len(p['organ_indices'].get(oar_key, np.array([], dtype=int))) > 0
            for oar_key in OAR_KEYS
        )
        if has_all:
            complete_indices.append(i)

    print(f"  {len(complete_indices)} complete-data real patients available "
          f"(excluding the source patients of the synthetic anchors)")

    id_to_idx = {all_patients[i]['patient_id']: i for i in complete_indices}
    missing = [pid for ids in COHORTS.values() for pid in ids if pid not in id_to_idx]
    if missing:
        raise RuntimeError(f"Cohort patients not found in complete-data pool: {missing}")

    # Build nested sets: [synth_best, synth_worst, real_3, ..., real_N]
    out_dir = 'results/all_patients'
    os.makedirs(out_dir, exist_ok=True)

    for N in n_list:
        extra_idx = [id_to_idx[pid] for pid in COHORTS[N]]

        train_patients = [synth_best, synth_worst] + \
                         [all_patients[i] for i in extra_idx]
        train_w = [synth_best_w, synth_worst_w] + \
                  [all_w[i] for i in extra_idx]
        train_obj = [synth_best_obj, synth_worst_obj] + \
                    [float(all_obj[i]) for i in extra_idx]

        cache_file = f'{out_dir}/forward_cache_nested_N{N}.pkl'
        with open(cache_file, 'wb') as f:
            pickle.dump({
                'patients': train_patients,
                'w_solutions': train_w,
                'obj_values': train_obj,
            }, f)

        print(f"\nN={N}: saved {cache_file}")
        for pos, (p, o) in enumerate(zip(train_patients, train_obj)):
            tag = (' (SYNTH BEST)' if pos == 0
                   else ' (SYNTH WORST)' if pos == 1 else '')
            print(f"  [{pos}] {p['patient_id']:40s}  obj={o:15.2f}{tag}")

    print("\nAll nested caches built with synthetic anchors.")


if __name__ == '__main__':
    main()
