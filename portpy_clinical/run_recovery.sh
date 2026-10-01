#!/usr/bin/env bash
# Recovery (Stages 1-3) for the nested cohorts N = 3, 4, 5, 6.
# Stage 1: minimum epsilon at beta_bar = 1000; Stage 2: bisection on beta from 100;
# Stage 3: conservative selection. The two anchor objectives U_min and U_max are
# read from the training cache.
cd "$(dirname "$0")"
export PYTHONUNBUFFERED=1
read U_MIN_ANCHOR U_MAX_ANCHOR < <(python -c "import pickle; c = pickle.load(open('results/all_patients/forward_cache_nested_N3.pkl', 'rb')); print(repr(float(c['obj_values'][0])), repr(float(c['obj_values'][1])))")
export U_MIN_ANCHOR U_MAX_ANCHOR
for N in 3 4 5 6; do
  D=results/recovery/N$N
  mkdir -p "$D"
  python run_stage1_nested.py "$N" "$D" || exit 1
  python run_stage2_nested.py "$N" 100.0 "$D" || exit 1
  python run_stage3_nested_auto.py "$N" "$D" || exit 1
done
