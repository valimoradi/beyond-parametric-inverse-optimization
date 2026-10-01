"""
Configuration for the Figure 4 predictions of the synthetic radiotherapy study (Section 4.2).

Paths are resolved relative to DATA_ROOT (environment variable SYNTH_DATA_ROOT; default: this
folder). Run with SYNTH_DATA_ROOT=paper_run to use the notebook's data and results.
"""
import os

DATA_ROOT = os.environ.get("SYNTH_DATA_ROOT", os.path.dirname(os.path.abspath(__file__)))

ML_DIR = os.path.join(DATA_ROOT, "ml_data_high_std")   # test split written by the notebook

# Recovered parameters, per class
RESULTS = {
    "add_smooth":       os.path.join(DATA_ROOT, "results_high_std"),
    "add_lp":           os.path.join(DATA_ROOT, "results_high_std_add_nonsmooth_lp"),
    "nonadd_smooth":    os.path.join(DATA_ROOT, "results_high_std_non_add_smooth"),
    "nonadd_nonsmooth": os.path.join(DATA_ROOT, "results_high_std_non_add_nonsmooth"),
}

# Prediction CSVs (Figure 4 data), per class
PRED_OUT = {model: os.path.join(DATA_ROOT, "fig4_data") for model in RESULTS}
PRED_CSV = {
    "add_smooth":       "add_smooth__add_smooth.csv",
    "add_lp":           "additive__add_pure_lp.csv",
    "nonadd_smooth":    "smooth__nonadd_smooth.csv",
    "nonadd_nonsmooth": "convex_only__nonadd_nonsmooth.csv",
}

# Ground-truth clinical parameters of the forward problem
PARAMS = {
    "theta_prescribed": 78.0,   # PTV min dose
    "theta_max_PTV": 81.9,      # PTV max dose
    "theta_thresh_1": 59.15,    # OAR1 (quadratic) threshold
    "theta_thresh_2": 59.15,    # OAR2 (linear) threshold
    "beta_1": 0.5,              # beamlet lower-bound fraction
    "beta_2": 1.5,              # beamlet upper-bound fraction
    "exp_scale_k": 15.0,        # OAR3 exponential scale (kappa)
    "alpha_exp": 1e-3,          # OAR3 exponential multiplier
}

TRAINING_SIZES = [20, 40, 60, 80, 100, 120, 140, 160, 180, 200]
