"""
config.py
=========
Configuration for the per-voxel clinical-scale PortPy study (Section 4.3).

PortPy prostate patients; seven OAR structures grouped into six penalty
functions.
"""

import os

# ---------------------------------------------------------------------------
# PortPy data paths
# ---------------------------------------------------------------------------
PORTPY_DATA_DIR = "../portpy_prostate/prostate_data/data"
PROTOCOL_NAME   = "Prostate_26Fx"   # 26 fractions x 2.7 Gy = 70.2 Gy

# ---------------------------------------------------------------------------
# Structures (must match PortPy naming after create_opt_structures)
# ---------------------------------------------------------------------------
STRUCTURE_NAMES = {
    'ptv':     'PTV',
    'bladder': 'BLADDER',
    'rectum':  'RECTUM',
    'femur_l': 'FEMUR_L',
    'femur_r': 'FEMUR_R',
    'rind_0':  'RIND_0',
    'rind_1':  'RIND_1',
    'rind_2':  'RIND_2',
}

# OAR grouping: 6 penalty functions (the two femoral heads share one)
OAR_GROUPS = {
    'bladder': {'func_idx': 0, 'label': 'Bladder'},
    'rectum':  {'func_idx': 1, 'label': 'Rectum'},
    'femur_l': {'func_idx': 2, 'label': 'Femoral Heads'},
    'femur_r': {'func_idx': 2, 'label': 'Femoral Heads'},
    'rind_0':  {'func_idx': 3, 'label': 'RIND 0-2mm'},
    'rind_1':  {'func_idx': 4, 'label': 'RIND 2-20mm'},
    'rind_2':  {'func_idx': 5, 'label': 'RIND 20-40mm'},
}

N_FUNCTIONS = 6
OAR_KEYS = ['bladder', 'rectum', 'femur_l', 'femur_r',
            'rind_0', 'rind_1', 'rind_2']

FUNC_LABELS = ['Bladder', 'Rectum', 'Femoral Heads',
               'RIND 0-2mm', 'RIND 2-20mm', 'RIND 20-40mm']

# ---------------------------------------------------------------------------
# Forward objective parameters (per-voxel penalty)
# ---------------------------------------------------------------------------
# Weights from the PortPy Prostate_26Fx optimization configuration.
# g(z_v) = alpha_k * z_v^2  where z_v = max(0, d_v - theta_k)
#
FORWARD_PARAMS = {
    'alpha_bladder': 20.0,
    'alpha_rectum':  20.0,
    'alpha_femur_l': 10.0,
    'alpha_femur_r': 10.0,
    'alpha_rind_0':  5.0,
    'alpha_rind_1':  5.0,
    'alpha_rind_2':  3.0,
}

# Dose thresholds (Gy), Appendix B.3:
#   - Rectum, Bladder: 15 Gy  (penalty reference dose of Wahl et al. 2016)
#   - Femoral heads:    5 Gy  (no clearly safe dose for avascular necrosis)
#   - RIND_0:          30 Gy  (abuts the PTV; doses near the prescription)
#   - RIND_1, RIND_2:   5 Gy  (dose-shaping shells)
DOSE_THRESHOLDS = {
    'bladder':  15.0,
    'rectum':   15.0,
    'femur_l':   5.0,
    'femur_r':   5.0,
    'rind_0':   30.0,
    'rind_1':    5.0,
    'rind_2':    5.0,
}

# PTV parameters (Gy) — from PortPy Prostate_26Fx protocol
# PTV dose is enforced by hard inequality constraints in forward/solver.py:
#   PTV_PRESCRIBED <= dose[v] <= PTV_MAX  for v in PTV.
PTV_PRESCRIBED = 70.2    # 26 x 2.7 Gy
PTV_MAX        = 77.0    # ~110% of prescribed

# Beamlet uniformity bounds
BETA_1 = 0.0
BETA_2 = 1e6

# ---------------------------------------------------------------------------
# Spatial downsampling (PortPy native)
# ---------------------------------------------------------------------------
OPT_VOX_XYZ_RES_MM = [15, 15, 5]

# ---------------------------------------------------------------------------
# Inverse optimization settings
# ---------------------------------------------------------------------------
# C7 anchor values. The stage scripts read U_MIN / U_MAX from the environment
# variables U_MIN_ANCHOR / U_MAX_ANCHOR, which run_recovery.sh sets to the
# forward objectives of the two anchor patients.
NORMALIZATION_SETTINGS = {
    'U_MAX': 100000.0,   # delta_global[worst_patient] = U_MAX
    'U_MIN': 0.0,        # delta_global[best_patient] = U_MIN
}
BETA_0         = 40.0    # true Lipschitz constant: 2 * max(alpha) = 2 * 20
STAGE1_BETA_BAR = 1000.0  # Model 1 (minimum epsilon) runs at beta_bar = 1000; beta* is then found by Model 2.
EPSILON_TOL    = 1e-6
BISECT_GAP_TOL = 0.11
BISECT_BETA_LOW = 1.0

# ---------------------------------------------------------------------------
# Solver settings
# ---------------------------------------------------------------------------
PREFERRED_SOLVER = "MOSEK"

MOSEK_TOLERANCES = {
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   1e-5,
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   1e-5,
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-5,
    'MSK_DPAR_OPTIMIZER_MAX_TIME':     120.0,   # 2 min for forward solves
}

MOSEK_HIGH_PRECISION = {
    'MSK_DPAR_INTPNT_CO_TOL_PFEAS':   1e-11,
    'MSK_DPAR_INTPNT_CO_TOL_DFEAS':   1e-11,
    'MSK_DPAR_INTPNT_CO_TOL_REL_GAP': 1e-11,
}

# MOSEK thread budget. If MSK_NUM_THREADS is set (> 0), every solve whose options
# are built from MOSEK_TOLERANCES or MOSEK_HIGH_PRECISION uses that many threads.
# Unset or 0 leaves MOSEK on its default (all cores).
_MSK_NUM_THREADS = int(os.environ.get("MSK_NUM_THREADS", "0"))
if _MSK_NUM_THREADS > 0:
    MOSEK_TOLERANCES['MSK_IPAR_NUM_THREADS'] = _MSK_NUM_THREADS
    MOSEK_HIGH_PRECISION['MSK_IPAR_NUM_THREADS'] = _MSK_NUM_THREADS

# ---------------------------------------------------------------------------
# Output
# ---------------------------------------------------------------------------
RESULTS_DIR = "./results"
NUM_QUERY_POINTS = 200
