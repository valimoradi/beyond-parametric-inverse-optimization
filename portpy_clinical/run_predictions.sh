#!/usr/bin/env bash
# Predictions for the 20 test patients at N = 3, 4, 5, 6, one patient per process, one MOSEK thread.
cd "$(dirname "$0")"
export MOSEK_THREADS=1 MSK_NUM_THREADS=1 OMP_NUM_THREADS=1 MKL_NUM_THREADS=1 OPENBLAS_NUM_THREADS=1
export PYTHONUNBUFFERED=1
for N in 3 4 5 6; do
  for IDX in $(seq 0 19); do
    python predict_test_patients.py "$N" "$IDX"
  done
done
