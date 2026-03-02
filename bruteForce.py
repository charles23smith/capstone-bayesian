"""
bruteForce.py (fast deterministic fit)

Model:
- NEW t0 = 100ns before PCD avg (excluding PCD3_B) crosses 0.5V
- Auto NEGATE detection
- Onset = first drop-off point (derivative threshold)
- Peak = extremum within 1000ns after onset
- Stop = next oscillation start (turning point, with fallback)
- Split = real slope-change point (smoothed + sustained derivative threshold)
- Piecewise model:
    exp1 from onset->split, exp2 from split->stop
- Keeps full + zoom overlays and current pulse TXT export
"""

# -----------------------------
# MUST BE FIRST
# -----------------------------
import os
import re
import argparse
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

# =============================
# USER SETTINGS
# =============================
CSV_FILE = "data/27271_data.csv"
OUT_DIR  = "diode_fit_outputs"
SEED = 42

# Optional shot-family mapping (from test log) for model selection.
SHOT_FAMILY_MAP = {
    **{k: "SMAJ400A" for k in range(27270, 27290)},
    **{k: "MMSZ5226BT1G" for k in range(27290, 27298)},
    **{k: "CSD01060E" for k in range(27298, 27303)},
    **{k: "1N6517US" for k in range(27303, 27306)},
    **{k: "SMAJ400A" for k in range(27306, 27308)},
}
# Only this diode family uses the advanced recovery (sigmoid+tail).
# All other families use the earlier 2-exp implementation (27271 logic).
THREE_EXP_FAMILIES = {"1N6517US"}

# --- t0 definition ---
PCD_EXCLUDE = {"PCD3_B"}
PCD_TARGET_V = 0.5
T0_SHIFT_NS  = 100.0

# --- onset detection ---
ONSET_MIN_NS_AFTER_T0 = 0.0
ONSET_BASELINE_NS     = 80.0
ONSET_SMOOTH_NS       = 2.0
ONSET_SIGMA_MULT      = 10.0
ONSET_SUSTAIN_N       = 8
ONSET_ZERO_DROP_SHOTS = {27283, 27286}
ONSET_ZERO_DROP_SMOOTH_NS = 4.5
ONSET_ZERO_DROP_V_THRESH = 0.02
ONSET_ZERO_DROP_SIGMA_MULT = 3.0
ONSET_ZERO_DROP_SUSTAIN_N = 6
ONSET_ZERO_DROP_DELTA_V = 0.10
ONSET_ZERO_DROP_V_THRESH_BY_SHOT = {27283: 0.12}
ONSET_ZERO_DROP_DELTA_V_BY_SHOT = {27283: 0.22}
ONSET_ZERO_DROP_V_THRESH_BY_SHOT.update({27286: 0.08})
ONSET_ZERO_DROP_DELTA_V_BY_SHOT.update({27286: 0.18})

# --- peak selection ---
PEAK_SEARCH_NS = 1000.0
PEAK_SEARCH_NS_BY_SHOT = {27289: 450.0}
PEAK_SMOOTH_NS_BY_SHOT = {27289: 8.0}
ANALYSIS_ABS_WINDOW_NS_BY_SHOT = {27289: (0.0, 300.0)}
PEAK_ABS_WINDOW_NS_BY_SHOT = {27297: (140.0, 150.0)}
PEAK_SMOOTH_SIGMOID_NS = 0.0
FOUR_EXP_SHOTS = {27298}
START_AT_PCD_CROSS_SHOTS = {27298}

# --- stop selection (next oscillation start) ---
STOP_MIN_NS_AFTER_PEAK = 25.0
STOP_MAX_NS_AFTER_PEAK = 5000.0
STOP_SMOOTH_NS         = 3.0
STOP_SUSTAIN_N         = 8          # consecutive samples required for sign flip confirmation

# NEW: require substantial recovery before allowing "turning point" stop
STOP_RECOVERY_FRAC = 0.60           # raise to 0.7-0.8 if it still stops early
STOP_DVDT_MED_WIN  = 9              # median window (samples) for dv/dt sign

# strict stop at start of next period (strong slope reversal)
STOP_PERIOD_MIN_NS_AFTER_PEAK = 120.0
STOP_PERIOD_RECOVERY_FRAC = 0.95
STOP_PERIOD_SUSTAIN_N = 6
STOP_PERIOD_SIGMA_MULT = 2.5
STOP_PERIOD_FRAC_OF_P90 = 0.15
STOP_PERIOD_PRE_QUIET_N = 10
STOP_PERIOD_QUIET_FRAC_OF_P90ABS = 0.22

# explicit opposite-direction stop (2-exp families)
STOP_OPP_MIN_NS_AFTER_PEAK = 120.0
STOP_OPP_RECOVERY_FRAC = 0.35
STOP_OPP_SUSTAIN_N = 8
STOP_OPP_SIGMA_MULT = 3.0
STOP_OPP_FRAC_OF_REC_P95 = 1.00
STOP_OPP_CAND_REC_FRAC = 0.00
STOP_VALLEY_MIN_NS_AFTER_PEAK = 25.0
STOP_VALLEY_MAX_NS_AFTER_PEAK = 450.0
STOP_VALLEY_SMOOTH_NS = 4.0

# flat-tail cutoff (end exp2 when slope has flattened)
STOP_FLAT_MIN_NS_AFTER_PEAK = 35.0
STOP_FLAT_SUSTAIN_N = 14
STOP_FLAT_FRAC_OF_RECOVERY_P95 = 0.12
STOP_FLAT_MIN_DV_FROM_PEAK = 0.6

# fallback slope-threshold (if turning-point not found)
SLOPE_REF_START_NS = 3.0
SLOPE_REF_LEN_NS   = 12.0
SLOPE_TOL_FRAC     = 0.35
SLOPE_SUSTAIN_N    = 12

# --- split (true slope-change for exp handoff) ---
SPLIT_SMOOTH_NS = 4.0
SPLIT_MIN_NS_AFTER_PEAK = 6.0
SPLIT_PRE_SUSTAIN_N = 8
SPLIT_POST_SUSTAIN_N = 12
SPLIT_SIGMA_MULT = 4.0
SPLIT_PRE_THR_FRAC = 0.20
SPLIT_POST_THR_FRAC = 0.55
SPLIT_RISE_FRAC_OF_MAX = 0.45
SPLIT_PRE_MAX_FRAC = 0.45

# --- 3-exp segmentation ---
SPLIT1_MIN_NS_AFTER_PEAK = 2.0
SPLIT1_PRE_SUSTAIN_N = 6
SPLIT1_POST_SUSTAIN_N = 7
SPLIT1_PRE_THR_SIGMA = 0.8
SPLIT1_POST_FRAC_OF_P95 = 0.18
SPLIT1_MAX_FRAC_OF_PEAK_TO_STOP = 0.55
SPLIT1_MAX_NS_AFTER_PEAK = 80.0
SPLIT1_EARLY_EXTREMUM_MAX_NS = 70.0

# Family-1 (SMAJ400A) split rule: near-zero slope -> sustained large rise
SPLIT1_ZERO_RISE_MIN_NS_AFTER_PEAK = 6.0
SPLIT1_ZERO_RISE_PRE_N = 10
SPLIT1_ZERO_RISE_POST_N = 8
SPLIT1_ZERO_RISE_ZERO_FRAC = 0.28
SPLIT1_ZERO_RISE_POST_FRAC = 0.55
SPLIT1_ZERO_RISE_RECOVERY_FRAC = 0.50
SPLIT1_ZERO_RISE_RECOVERY_LOOKAHEAD_NS = 1800.0
SPLIT1_ZERO_RISE_POST_HIT_FRAC = 0.65
SPLIT1_ZERO_RISE_LOCAL_RISE_FRAC = 0.18
SPLIT1_ZERO_RISE_LOCAL_RISE_NS = 80.0
SPLIT1_ZERO_RISE_HOLD_NS = 45.0
SPLIT1_ZERO_RISE_HOLD_FRAC = 0.45
SPLIT1_ZERO_RISE_HOLD_HIT_FRAC = 0.55
SMAJ_FIT_SMOOTH_NOISE_THRESH_V = 0.020
SMAJ_FIT_SMOOTH_NS = 3.5
SHOT_FIT_SMOOTH_NS_OVERRIDE = {27282: 5.5, 27283: 6.5, 27286: 5.5, 27287: 8.5, 27289: 6.5, 27295: 5.5}
SHOT_FORCE_FIT_SMOOTH_NS = {27296: 10.0, 27297: 10.0, 27298: 10.0, 27299: 18.0}
ONSET_FORCE_T0_SHOTS = set()
SPLIT1_PEAK_ANCHOR_START_NS = 2.0
SPLIT1_PEAK_ANCHOR_WIN_NS = 26.0
SPLIT1_PEAK_ANCHOR_SUSTAIN_N = 7
SPLIT1_PEAK_ANCHOR_SIGMA_MULT = 2.0
SPLIT1_PEAK_ANCHOR_HIT_FRAC = 0.70
SPLIT1_PEAK_ANCHOR_MAX_SPAN_NS = 900.0
SPLIT1_STRICT_INFLECT_SMOOTH_NS = 5.0
SPLIT1_STRICT_INFLECT_MIN_NS_AFTER_PEAK = 8.0
SPLIT1_STRICT_INFLECT_POST_N = 8
SPLIT1_STRICT_INFLECT_PRE_N = 8
SPLIT1_STRICT_INFLECT_SLOPE_FRAC_P90 = 0.45
SPLIT1_STRICT_INFLECT_ACCEL_FRAC_P90 = 0.30
SPLIT1_STRICT_INFLECT_PRE_FRAC_POST = 0.35
SPLIT1_STRICTER_SHOTS = {27277}
SPLIT1_STRICTER_MULT = 1.00
SPLIT1_STRICTER_MIN_NS_AFTER_PEAK = 22.0
SPLIT1_STRICTER_MIN_NS_BY_SHOT = {27282: 34.0, 27283: 100.0}
SPLIT1_DISABLE_PEAK_ANCHOR_SHOTS = {27282, 27283}
SPLIT1_FORCE_AT_PEAK_SHOTS = {27287, 27289, 27290, 27291, 27295, 27296, 27297, 27298}
SPLIT1_SKIP_STRICT_FALLBACK_SHOTS = {27287}
SPLIT1_POS_TO_NEG_SLOPE_SHOTS = set()
SPLIT1_SHIFT_LEFT_NS_BY_SHOT = {27300: 12.0, 27307: -50.0}
SPLIT1_FORCE_ABS_MIN_SHOTS = {27296}
ONSET_FLUCT_START_WINDOWS_NS_BY_SHOT = {27297: (80.0, 130.0), 27299: (0.0, 180.0)}
ONSET_SHIFT_NS_BY_SHOT = {27299: 20.0}
ONSET_PREPEAK_WINDOW_NS_BY_SHOT = {27301: (0.0, 400.0)}
SPLIT1_TARGET_V_BY_SHOT = {27297: 2.0, 27306: -1.5, 27307: -1.45}
SPLIT1_TARGET_WINDOW_NS_BY_SHOT = {27297: (120.0, 155.0), 27307: (500.0, 38000.0)}
SPLIT1_FORCE_ABS_MAX_SHOTS = {27299}

# Shot-specific stop override: choose next-cycle local peak in a bounded window.
STOP_NEXT_PEAK_SHOT_WINDOWS_NS = {
    27283: (200.0, 260.0),  # peak is ~146 ns, this targets ~346-406 ns absolute
    27287: (120.0, 170.0),  # peak is ~157 ns, target first-cycle local max (~277-327 ns)
}
STOP_NEXT_PEAK_SMOOTH_NS = 6.0
SPLIT1_SHOT_MIN_WINDOW_NS = {
    27283: (55.0, 120.0),   # peak ~146 ns -> target split ~201-266 ns (deeper first valley)
}
EXP1_ENDPOINT_CONSTRAINED_SHOTS = {27282, 27283, 27286, 27287, 27289, 27296, 27299}
EXP1_ENDPOINT_TAU_MAX_MULT_BY_SHOT = {27286: 0.9, 27287: 1.0, 27289: 1.2, 27296: 1.0, 27299: 1.1}
EXP1_ENDPOINT_RAW_TARGET_SHOTS = {27287, 27289, 27296}
EXP2_ENDPOINT_TARGET_STOP_SHOTS = {27283, 27286}
EXP2_CLASSIC_ENDPOINT_SHOTS = {27287, 27297, 27299}
EXP2_FIT_RAW_SHOTS = {27297}
EXP2_ENDPOINT_Y_OFFSET_BY_SHOT = {27286: -0.25}
STRETCHED_ENDPOINT_SHOTS = {27286, 27287, 27289, 27295}
EXP1_STRETCHED_ONLY_SHOTS = {27296, 27298, 27299}
STOP_AFTER_SPLIT_LOCAL_MAX_WINDOWS_NS = {27283: (70.0, 95.0)}
STOP_AFTER_SPLIT_RAW_MAX_WINDOWS_NS = {27283: (60.0, 85.0), 27290: (70.0, 120.0)}
STOP_AFTER_SPLIT_LOCAL_MAX_WINDOWS_NS[27290] = (70.0, 120.0)
STOP_AFTER_SPLIT_GLOBAL_MIN_SHOTS = {27289}
STOP_AFTER_SPLIT_GLOBAL_MIN_WINDOWS_NS = {27289: (20.0, 240.0)}
STOP_AFTER_SPLIT_ABS_MAX_WINDOWS_NS = {27291: (175.0, 300.0), 27295: (200.0, 300.0), 27307: (50000.0, 90000.0)}
STOP_AFTER_SPLIT_ABS_MIN_WINDOWS_NS = {27297: (165.0, 190.0)}
STOP_ABS_MIN_BEFORE_NS_BY_SHOT = {27299: 240.0}
STOP_FIRST_ZERO_AFTER_NS_BY_SHOT = {27300: 240.0, 27301: 300.0, 27302: 300.0, 27306: 0.0}
STOP_FIRST_ZERO_SMOOTH_NS_BY_SHOT = {27300: 10.0, 27301: 10.0, 27302: 10.0, 27306: 10.0}
EXP2_TWO_STAGE_SHOTS = {27295}
EXP2_TWO_STAGE_SPLIT_ABS_NS_BY_SHOT = {27295: 200.0}
EXP2_SIGMOID_ONLY_SHOTS = set()
SIGMOID_ONLY_FORCE_SPLIT1_NEAR_ONSET_SHOTS = set()
EXP2_LINEAR_THEN_EXP_SHOTS = {27300, 27306, 27307}
EXP2_LINEAR_SPLIT_ABS_NS_BY_SHOT = {27300: 240.0, 27306: -25000.0, 27307: 36000.0}
EXP2_LINEAR_END_V_BY_SHOT = {27306: 1.275, 27307: 1.275}
EXP2_LINEAR_END_DELAY_NS_BY_SHOT = {27306: 3800.0, 27307: 3800.0}
EXP2_LINEAR_END_V_PICK_BY_SHOT = {27306: "first_cross", 27307: "first_cross"}
EXP2_LINEAR_END_V_TOL_BY_SHOT = {27306: 0.03, 27307: 0.03}
EXP2_LINEAR_END_V_NEAR_BACKOFF_BY_SHOT = {27306: 0, 27307: 0}
EXP2_LINEAR_EXP_FIT_RAW_SHOTS = {27306, 27307}
EXP2_LINEAR_EXP_TAU_MAX_MULT_BY_SHOT = {27306: 7.0, 27307: 2.5}
EXP2_LINEAR_EXP_USE_ALL_POINTS_SHOTS = {27306, 27307}
EXP2_LINEAR_END_USE_RAW_SHOTS = {27307}
EXP2_LINEAR_HANDOFF_WINDOW_NS_BY_SHOT = {27300: (273.0, 278.0)}
EXP2_LINEAR_FORCE_ABS_TARGET_SHOTS = {27307}
EXP2_LINEAR_MIN_NS_AFTER_SPLIT1_BY_SHOT = {27300: 60.0, 27306: 40.0, 27307: 40.0}
EXP2_LINEAR_STRICT_SLOPE_MULT_BY_SHOT = {27300: 3.2, 27306: 3.0, 27307: 3.0}
EXP2_LINEAR_STRICT_SUSTAIN_N_BY_SHOT = {27300: 12, 27306: 10, 27307: 10}
EXP2_LINEAR_DETECT_SMOOTH_NS_BY_SHOT = {27300: 14.0, 27306: 12.0, 27307: 12.0}
STOP_FLAT_2EXP_SHOTS = set()
PLOT_MODELED_ONLY_LABEL_SHOTS = set()
PLOT_HIDE_CONF_BAND_SHOTS = set()
MODEL_V_FLOOR_BY_SHOT = {}
EXP2_SIGMOID_FIT_RAW_SHOTS = set()
EXP2_SIGMOID_USE_ALL_POINTS_SHOTS = set()
EXP2_SIGMOID_TMID_SHIFT_NS_BY_SHOT = {}
EXP2_THEN_SIGMOID_SHOTS = {27301, 27302}
EXP2_SIG_START_WINDOW_NS_BY_SHOT = {27301: (460.0, 560.0), 27302: (300.0, 360.0)}
EXP2_SIG_SPLIT_SMOOTH_NS_BY_SHOT = {27301: 10.0, 27302: 10.0}

# Family-1 (SMAJ400A) stop rule: where slope settles to an approximately constant value
STOP_CONST_MIN_NS_AFTER_PEAK = 20.0
STOP_CONST_SUSTAIN_N = 14
STOP_CONST_CV_MAX = 0.22
STOP_CONST_MIN_SIGMA_MULT = 3.0

SPLIT2_MIN_NS_AFTER_SPLIT1 = 8.0
SPLIT2_POST_SUSTAIN_N = 8
SPLIT2_DROP_FRAC_FROM_MAX = 0.93
SPLIT2_ACCEL_MIN_FRAC_OF_RANGE = 0.12
SPLIT2_ACCEL_SUSTAIN_N = 10
SPLIT2_ACCEL_FRAC_OF_P95 = 0.30
SPLIT2_SLOPE_LEVEL_FRAC = 0.22
SPLIT2_STRICT_MIN_FRAC_OF_RANGE = 0.30
SPLIT2_STRICT_FRAC_OF_P95 = 0.60
SPLIT2_STRICT_SUSTAIN_N = 10
SPLIT2_STRICT_PRE_RATIO = 1.35

# --- anchored sigmoid shape tuning ---
SIGMOID_K_MAX_FRAC = 0.22
SIGMOID_TMID_MIN_FRAC = 0.38
SIGMOID_WEIGHT_MID_BOOST = 5.0
SIGMOID_WEIGHT_MID_FRAC = 0.16
SIGMOID_MISMATCH_MIN_FRAC = 0.62
SIGMOID_MISMATCH_SUSTAIN_N = 6
SIGMOID_MISMATCH_FRAC_OF_P90 = 0.12
SIGMOID_MISMATCH_REL_RATIO = 0.45

# Optional post-peak dip segment (for noisy/nonlinear immediate recovery)
DIP_ENABLE = True
DIP_SCAN_NS = 140.0
DIP_SMOOTH_NS = 2.5
DIP_SIGN_SUSTAIN_N = 6
DIP_MIN_DROP_V = 0.08
DIP_CONST_SUSTAIN_N = 10
DIP_MIN_FIT_AMP_V = 0.03
DIP_MAX_END_MISMATCH_V = 0.06

# CSD family recovery shape:
# - "sigmoid_exp_tail" keeps current sigmoid + exponential-tail handoff
# - "gompertz" uses one anchored Gompertz from peak -> stop
RECOVERY_SHAPE_CSD = "sigmoid_exp_tail"

# --- fit speed ---
SUBSAMPLE_RECOVERY_N = 380
RECOVERY_BAND_SIGMA = 1.96  # approx 95% band from fit residuals
SHOW_CONF_BAND = False

# --- current export (doc-style) ---
K_T = 0.65
K_V = 0.10
SKEW_FIRST_NS = 0.0
SKEW_MULT     = 1.0
TAIL_TAPER_NS = 50.0
TIME_SHIFT_S  = 1e-6

# plot zoom window padding (ns)
ZOOM_LEFT_PAD_NS  = 200.0
ZOOM_RIGHT_PAD_NS = 600.0
SHOW_EVENT_MARKERS = False


# =============================
# Helpers
# =============================
def _oddify(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)

def _ns_to_s(ns: float) -> float:
    return ns * 1e-9

def _s_to_ns(s: float) -> float:
    return s * 1e9

def _interp_to(x_src, y_src, x_new):
    return np.interp(x_new, x_src, y_src)

def exp_anchor_np(t, baseline, tau, y0):
    """Single exponential anchored at y0 (t=0), relaxing toward baseline."""
    t = np.asarray(t, dtype=float)
    tau = np.maximum(tau, 1e-12)
    return baseline + (y0 - baseline) * np.exp(-t / tau)

def anchored_sigmoid_np(t, amp, t_mid, k, m_tail, y0):
    """
    Anchored sigmoid + linear-tail model.
    This keeps tails straighter while preserving a sharp knee.
    """
    t = np.asarray(t, dtype=float)
    k = np.maximum(k, 1e-12)
    s = 1.0 / (1.0 + np.exp(-(t - t_mid) / k))
    s0 = 1.0 / (1.0 + np.exp(-(-t_mid) / k))
    denom = np.maximum(1.0 - s0, 1e-12)
    g = (s - s0) / denom
    return y0 + m_tail * t + amp * g

def exp_tail_to_zero_np(t, tau, y0):
    """Exponential tail from y0 toward 0V."""
    t = np.asarray(t, dtype=float)
    tau = np.maximum(float(tau), 1e-12)
    return y0 * np.exp(-t / tau)

def dip_biexp_np(t, A, tau_fast, tau_slow, y0):
    """
    Anchored dip/rebound transient:
      y(t)=y0 + A*(exp(-t/tau_fast)-exp(-t/tau_slow))
    with tau_fast < tau_slow, A>0 gives initial drop then recovery.
    """
    t = np.asarray(t, dtype=float)
    tau_fast = np.maximum(float(tau_fast), 1e-12)
    tau_slow = np.maximum(float(tau_slow), tau_fast + 1e-12)
    return y0 + A * (np.exp(-t/tau_fast) - np.exp(-t/tau_slow))

def fit_dip_biexp_fast(t_fit, y_fit):
    """
    Fit anchored dip/rebound biexponential on a short segment.
    Returns [A, tau_fast, tau_slow], y0_const, residual_sigma.
    """
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    y0_const = float(y_fit[0])
    dur = float(max(t_fit[-1], 1e-9))

    # amplitude guess from most negative excursion
    dmin = float(np.min(y_fit) - y0_const)  # usually <= 0
    A0 = max(1e-4, -2.0 * dmin)

    lb = [1e-6, 0.5e-9, 2e-9]
    ub = [max(10.0, 20.0*A0), max(0.20*dur, 20e-9), max(2.0*dur, 80e-9)]

    def df_fit(t, A, tau_fast, tau_slow):
        # enforce tau_slow > tau_fast by penalizing invalid region softly via clipping
        tau_slow = np.maximum(tau_slow, tau_fast + 1e-12)
        return dip_biexp_np(t, A, tau_fast, tau_slow, y0_const)

    candidates = [
        [A0, 0.03*dur, 0.18*dur],
        [1.5*A0, 0.05*dur, 0.30*dur],
        [0.8*A0, 0.08*dur, 0.45*dur],
    ]

    best = None
    best_sse = np.inf
    for p0 in candidates:
        p0 = [
            float(np.clip(p0[0], lb[0], ub[0])),
            float(np.clip(max(p0[1], 1e-12), lb[1], ub[1])),
            float(np.clip(max(p0[2], p0[1] + 1e-12), lb[2], ub[2])),
        ]
        try:
            popt, _ = curve_fit(df_fit, t_fit, y_fit, p0=p0, bounds=(lb, ub), maxfev=25000)
            sse = float(np.sum((y_fit - df_fit(t_fit, *popt))**2))
            if sse < best_sse:
                best_sse = sse
                best = popt
        except Exception:
            continue

    if best is None:
        best = np.array([A0, min(max(0.05*dur, lb[1]), ub[1]), min(max(0.30*dur, lb[2]), ub[2])], dtype=float)

    resid = y_fit - df_fit(t_fit, *best)
    fit_sigma = float(np.std(resid) + 1e-9)
    return np.asarray(best, dtype=float), y0_const, fit_sigma

def detect_post_peak_dip_end(t_rel_ns, t_abs_s, v, peak_idx, rec_sign):
    """
    Detect a short post-peak dip segment:
    requires sustained opposite-slope then sustained recovery slope.
    Returns (has_dip, dip_end_idx, mode).
    """
    if not DIP_ENABLE:
        return False, peak_idx, "dip_disabled"

    i_end = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + DIP_SCAN_NS))
    i_end = min(i_end, len(v)-1)
    if i_end <= peak_idx + 8:
        return False, peak_idx, "dip_short"

    vs, _ = smooth_by_ns(t_abs_s, v, DIP_SMOOTH_NS)
    dv = np.gradient(vs, t_abs_s)

    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dv))
        return float(np.median(dv[a:b]))

    pre_sign = -rec_sign
    N = max(4, int(DIP_SIGN_SUSTAIN_N))
    i0 = peak_idx + 1
    i1 = i_end - N - 1
    if i1 <= i0 + N:
        return False, peak_idx, "dip_short"

    # optional gate hint: visible drop from peak before recovery
    drop = float(np.min(v[i0:i_end+1]) - v[peak_idx]) if rec_sign > 0 else float(v[peak_idx] - np.max(v[i0:i_end+1]))
    has_visible_drop = (drop <= -abs(DIP_MIN_DROP_V))

    if has_visible_drop:
        for i in range(i0 + N, i1):
            pre_vals = [pre_sign * dmed(j) for j in range(i - N, i)]
            post_vals = [rec_sign * dmed(j) for j in range(i, i + N)]
            if (np.median(pre_vals) > 0.0) and (np.median(post_vals) > 0.0):
                return True, i, "dip_detected"

    # Fallback: if early post-peak segment is nonlinear, end dip when slope stabilizes.
    te = np.asarray(t_abs_s[i0:i_end+1], dtype=float)
    ye = np.asarray(vs[i0:i_end+1], dtype=float)
    if len(te) >= 10:
        p = np.polyfit(te - te[0], ye, 1)
        resid = ye - (p[0]*(te - te[0]) + p[1])
        if float(np.std(resid)) >= 0.35 * abs(DIP_MIN_DROP_V):
            Nc = max(6, int(DIP_CONST_SUSTAIN_N))
            for i in range(i0 + Nc, i1 - Nc):
                post = np.array([rec_sign * dmed(j) for j in range(i, i + Nc)], dtype=float)
                if np.median(post) <= 0:
                    continue
                cv = float(np.std(post) / max(abs(np.median(post)), 1e-12))
                if cv <= 0.45:
                    return True, i, "dip_nonlinear_fallback"

    return False, peak_idx, "dip_not_found"

def anchored_gompertz_np(t, baseline, b, k, y0):
    """
    Anchored Gompertz from y0 at t=0 toward baseline as t->inf.
    """
    t = np.asarray(t, dtype=float)
    b = np.maximum(float(b), 1e-9)
    k = np.maximum(float(k), 1e-12)
    g = np.exp(-b * np.exp(-t / k))
    g0 = np.exp(-b)
    denom = np.maximum(1.0 - g0, 1e-12)
    u = (g - g0) / denom
    return y0 + (baseline - y0) * u

def fit_anchored_exp_fast(t_fit, y_fit, seg_sign=None):
    """
    Fast, robust anchored single-exponential fit using bounded multi-start curve_fit.
    Returns: best params [baseline, tau], y0_const, residual_sigma
    """
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    y0_const = float(y_fit[0])
    dur = float(max(t_fit[-1], 1e-9))

    tail_n = max(10, len(y_fit)//10)
    tail = y_fit[-tail_n:]
    b0 = float(np.mean(tail))
    bstd = float(np.std(tail) + 1e-6)
    span = abs(y0_const - b0) + 1e-6

    # Keep baseline near the observed tail to avoid non-physical local minima.
    b_lb = b0 - (4.0*bstd + 0.20*span)
    b_ub = b0 + (4.0*bstd + 0.20*span)

    # Optional direction constraint: sign(dy/dt at start) must match seg_sign.
    # For anchored exponential, sign(dy/dt) = sign(baseline - y0).
    eps_dir = max(1e-4, 0.01*span)
    if seg_sign is not None:
        if seg_sign > 0:
            b_lb = max(b_lb, y0_const + eps_dir)
        elif seg_sign < 0:
            b_ub = min(b_ub, y0_const - eps_dir)
        if b_lb >= b_ub:
            # fallback to a minimal feasible directional interval around y0
            if seg_sign > 0:
                b_lb, b_ub = y0_const + eps_dir, y0_const + max(10*eps_dir, span)
            else:
                b_lb, b_ub = y0_const - max(10*eps_dir, span), y0_const - eps_dir
    lb = [b_lb, 0.5e-9]
    ub = [b_ub, max(2.0*dur, 40e-9)]

    def ef_fit(t, baseline, tau):
        return exp_anchor_np(t, baseline, tau, y0_const)

    candidates = [
        [b0, 0.05*dur],
        [b0, 0.12*dur],
        [b0, 0.30*dur],
        [b0, 0.70*dur],
    ]

    best = None
    best_sse = np.inf
    for p0 in candidates:
        p0 = [
            float(np.clip(p0[0], lb[0], ub[0])),
            float(np.clip(max(p0[1], 1e-12), lb[1], ub[1])),
        ]
        try:
            popt, _ = curve_fit(
                ef_fit, t_fit, y_fit,
                p0=p0, bounds=(lb, ub), maxfev=20000
            )
            sse = float(np.sum((y_fit - ef_fit(t_fit, *popt))**2))
            if sse < best_sse:
                best_sse = sse
                best = popt
        except Exception:
            continue

    if best is None:
        best = np.array([b0, min(max(0.20*dur, 1e-9), ub[1])], dtype=float)

    resid = y_fit - ef_fit(t_fit, *best)
    fit_sigma = float(np.std(resid) + 1e-9)
    return np.asarray(best, dtype=float), y0_const, fit_sigma

def fit_anchored_exp_through_endpoint_fast(t_fit, y_fit, seg_sign=None, tau_max_mult=6.0):
    """
    Anchored exponential fit constrained to hit both endpoints:
      y(0)=y0 and y(T)=yT exactly.
    Fits only tau; baseline is derived from tau and endpoint constraint.
    Returns [baseline, tau], y0_const, residual_sigma.
    """
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    y0_const = float(y_fit[0])
    yT = float(y_fit[-1])
    T = float(max(t_fit[-1], 1e-12))

    def baseline_from_tau(tau):
        tau = max(float(tau), 1e-12)
        eT = float(np.exp(-T / tau))
        den = max(1.0 - eT, 1e-12)
        return (yT - y0_const * eT) / den

    def model_tau(t, tau):
        b = baseline_from_tau(tau)
        return exp_anchor_np(t, b, tau, y0_const)

    dur = T
    tau_lb = 0.2e-9
    tau_ub = max(float(tau_max_mult) * dur, 20e-9)
    tau_guesses = [0.05*dur, 0.12*dur, 0.30*dur, 0.70*dur]

    best_tau = None
    best_sse = np.inf
    for g in tau_guesses:
        p0 = [float(np.clip(max(g, 1e-12), tau_lb, tau_ub))]
        try:
            popt, _ = curve_fit(
                lambda tt, tau: model_tau(tt, tau),
                t_fit, y_fit,
                p0=p0, bounds=([tau_lb], [tau_ub]), maxfev=25000
            )
            tau = float(popt[0])
            b = baseline_from_tau(tau)
            if seg_sign is not None:
                slope_sign = np.sign(b - y0_const)
                if (seg_sign > 0 and slope_sign < 0) or (seg_sign < 0 and slope_sign > 0):
                    continue
            sse = float(np.sum((y_fit - model_tau(t_fit, tau))**2))
            if sse < best_sse:
                best_sse = sse
                best_tau = tau
        except Exception:
            continue

    if best_tau is None:
        best_tau = float(np.clip(max(0.25*dur, 1e-12), tau_lb, tau_ub))
    b_best = float(baseline_from_tau(best_tau))
    resid = y_fit - model_tau(t_fit, best_tau)
    fit_sigma = float(np.std(resid) + 1e-9)
    return np.asarray([b_best, best_tau], dtype=float), y0_const, fit_sigma

def stretched_exp_endpoint_np(t, tau, k, y0, yT, T):
    """
    Endpoint-constrained stretched exponential:
      y(0)=y0, y(T)=yT
      y(t)=b + (y0-b)*exp(-((t/tau)^k)), with b solved from endpoint.
    """
    t = np.asarray(t, dtype=float)
    tau = max(float(tau), 1e-12)
    k = max(float(k), 1e-9)
    T = max(float(T), 1e-12)
    eT = float(np.exp(-((T / tau) ** k)))
    den = max(1.0 - eT, 1e-12)
    b = (float(yT) - float(y0) * eT) / den
    return b + (float(y0) - b) * np.exp(-((np.maximum(t, 0.0) / tau) ** k))

def fit_stretched_exp_endpoint_fast(t_fit, y_fit, seg_sign=None, tau_max_mult=3.0, y_end_target=None):
    """
    Fit endpoint-constrained stretched exponential (tau, k).
    Returns [tau, k], y0_const, yT_const, residual_sigma.
    """
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    y0_const = float(y_fit[0])
    yT_const = float(y_fit[-1] if y_end_target is None else y_end_target)
    T = float(max(t_fit[-1], 1e-12))

    def mf(t, tau, k):
        return stretched_exp_endpoint_np(t, tau, k, y0_const, yT_const, T)

    tau_lb = 0.2e-9
    tau_ub = max(float(tau_max_mult) * T, 20e-9)
    k_lb, k_ub = 0.4, 4.0

    guesses = [
        (0.12*T, 0.8),
        (0.20*T, 1.0),
        (0.35*T, 1.3),
        (0.55*T, 1.7),
    ]

    best = None
    best_sse = np.inf
    for tg, kg in guesses:
        p0 = [float(np.clip(max(tg, 1e-12), tau_lb, tau_ub)), float(np.clip(kg, k_lb, k_ub))]
        try:
            popt, _ = curve_fit(
                mf, t_fit, y_fit, p0=p0,
                bounds=([tau_lb, k_lb], [tau_ub, k_ub]), maxfev=30000
            )
            tau, kval = map(float, popt)
            # sign check via endpoint-derived baseline
            eT = float(np.exp(-((T / max(tau, 1e-12)) ** max(kval, 1e-9))))
            den = max(1.0 - eT, 1e-12)
            b = (yT_const - y0_const * eT) / den
            if seg_sign is not None:
                slope_sign = np.sign(b - y0_const)
                if (seg_sign > 0 and slope_sign < 0) or (seg_sign < 0 and slope_sign > 0):
                    continue
            sse = float(np.sum((y_fit - mf(t_fit, tau, kval))**2))
            if sse < best_sse:
                best_sse = sse
                best = (tau, kval)
        except Exception:
            continue

    if best is None:
        best = (float(np.clip(max(0.25*T, 1e-12), tau_lb, tau_ub)), 1.0)

    resid = y_fit - mf(t_fit, best[0], best[1])
    fit_sigma = float(np.std(resid) + 1e-9)
    return np.asarray(best, dtype=float), y0_const, yT_const, fit_sigma

def exp2_accel_endpoint_np(t, alpha, y0, yT, T):
    """
    Endpoint-anchored accelerating exponential profile:
      y(0)=y0, y(T)=yT
      y(t)=y0 + (yT-y0)*((exp(alpha*t/T)-1)/(exp(alpha)-1))
    alpha>0 increases curvature toward the endpoint (steeper near valley).
    """
    t = np.asarray(t, dtype=float)
    T = max(float(T), 1e-12)
    u = np.clip(t / T, 0.0, 1.0)
    a = max(float(alpha), 1e-9)
    den = max(np.exp(a) - 1.0, 1e-12)
    h = (np.exp(a * u) - 1.0) / den
    return y0 + (yT - y0) * h

def fit_exp2_accel_endpoint_fast(t_fit, y_fit, y_end_target=None):
    """
    Fit alpha for endpoint-anchored accelerating exponential profile.
    Returns [alpha], y0_const, yT_const, residual_sigma.
    """
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    y0_const = float(y_fit[0])
    yT_const = float(y_fit[-1] if y_end_target is None else y_end_target)
    T = float(max(t_fit[-1], 1e-12))

    def mf(t, alpha):
        return exp2_accel_endpoint_np(t, alpha, y0_const, yT_const, T)

    lb = [0.8]
    ub = [12.0]
    guesses = [0.6, 1.2, 2.0, 3.5, 5.0]
    best = None
    best_sse = np.inf
    for g in guesses:
        p0 = [float(np.clip(g, lb[0], ub[0]))]
        try:
            popt, _ = curve_fit(mf, t_fit, y_fit, p0=p0, bounds=(lb, ub), maxfev=25000)
            sse = float(np.sum((y_fit - mf(t_fit, *popt))**2))
            if sse < best_sse:
                best_sse = sse
                best = popt
        except Exception:
            continue
    if best is None:
        best = np.array([1.5], dtype=float)
    resid = y_fit - mf(t_fit, *best)
    fit_sigma = float(np.std(resid) + 1e-9)
    return np.asarray(best, dtype=float), y0_const, yT_const, fit_sigma

def fit_anchored_sigmoid_fast(t_fit, y_fit, seg_sign=None):
    """
    Fast anchored sigmoid+linear-tail fit.
    Returns [amp, t_mid, k, m_tail], y0_const, residual_sigma.
    """
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    y0_const = float(y_fit[0])
    dur = float(max(t_fit[-1], 1e-9))

    tail_n = max(10, len(y_fit)//10)
    tail = y_fit[-tail_n:]
    b0 = float(np.mean(tail))
    bstd = float(np.std(tail) + 1e-6)
    span = abs(y0_const - b0) + 1e-6

    dur_safe = max(dur, 1e-12)
    m0_est = float((y_fit[-1] - y_fit[0]) / dur_safe)
    eps_dir = max(1e-4, 0.01*span)
    if seg_sign is None:
        a_lb, a_ub = -3.0*span, 3.0*span
        m_lb, m_ub = -max(5.0*abs(m0_est), 5e6), max(5.0*abs(m0_est), 5e6)
    elif seg_sign > 0:
        a_lb, a_ub = eps_dir, 3.0*span
        m_lb, m_ub = 0.0, max(5.0*abs(m0_est), 5e6)
    else:
        a_lb, a_ub = -3.0*span, -eps_dir
        m_lb, m_ub = -max(5.0*abs(m0_est), 5e6), 0.0

    lb = [a_lb, SIGMOID_TMID_MIN_FRAC*dur, 0.5e-9, m_lb]
    ub = [a_ub, 0.98*dur, max(SIGMOID_K_MAX_FRAC*dur, 8e-9), m_ub]

    def sf_fit(t, amp, t_mid, k, m_tail):
        return anchored_sigmoid_np(t, amp, t_mid, k, m_tail, y0_const)

    candidates = [
        [0.55*span*np.sign(m0_est if abs(m0_est) > 0 else 1.0), 0.50*dur, 0.05*dur, 0.60*m0_est],
        [0.85*span*np.sign(m0_est if abs(m0_est) > 0 else 1.0), 0.62*dur, 0.07*dur, 0.50*m0_est],
        [1.10*span*np.sign(m0_est if abs(m0_est) > 0 else 1.0), 0.72*dur, 0.09*dur, 0.40*m0_est],
    ]

    # Emphasize the knee region so inflection is sharper and better aligned to data.
    dy = np.gradient(y_fit, t_fit)
    d_eff = dy if (seg_sign is None) else (seg_sign * dy)
    i_mid = int(np.argmax(d_eff))
    half_w = max(3, int(SIGMOID_WEIGHT_MID_FRAC * len(t_fit)))
    a = max(0, i_mid - half_w)
    b = min(len(t_fit), i_mid + half_w + 1)
    sigma_w = np.ones_like(t_fit, dtype=float)
    sigma_w[a:b] = 1.0 / max(SIGMOID_WEIGHT_MID_BOOST, 1.0)

    best = None
    best_sse = np.inf
    for p0 in candidates:
        p0 = [
            float(np.clip(p0[0], lb[0], ub[0])),
            float(np.clip(p0[1], lb[1], ub[1])),
            float(np.clip(max(p0[2], 1e-12), lb[2], ub[2])),
            float(np.clip(p0[3], lb[3], ub[3])),
        ]
        try:
            popt, _ = curve_fit(
                sf_fit, t_fit, y_fit,
                p0=p0, bounds=(lb, ub), sigma=sigma_w, absolute_sigma=False, maxfev=25000
            )
            sse = float(np.sum((y_fit - sf_fit(t_fit, *popt))**2))
            if sse < best_sse:
                best_sse = sse
                best = popt
        except Exception:
            continue

    if best is None:
        best = np.array([
            min(max(0.8*span*np.sign(m0_est if abs(m0_est) > 0 else 1.0), lb[0]), ub[0]),
            min(max(0.62*dur, lb[1]), ub[1]),
            min(max(0.08*dur, lb[2]), ub[2]),
            min(max(0.5*m0_est, lb[3]), ub[3]),
        ], dtype=float)

    resid = y_fit - sf_fit(t_fit, *best)
    fit_sigma = float(np.std(resid) + 1e-9)
    return np.asarray(best, dtype=float), y0_const, fit_sigma

def fit_anchored_gompertz_fast(t_fit, y_fit, seg_sign=None):
    """
    Fast anchored Gompertz fit.
    Returns [baseline, b, k], y0_const, residual_sigma.
    """
    t_fit = np.asarray(t_fit, dtype=float)
    y_fit = np.asarray(y_fit, dtype=float)
    y0_const = float(y_fit[0])
    dur = float(max(t_fit[-1], 1e-9))

    tail_n = max(10, len(y_fit)//10)
    tail = y_fit[-tail_n:]
    b0 = float(np.mean(tail))
    bstd = float(np.std(tail) + 1e-6)
    span = abs(y0_const - b0) + 1e-6

    blb = b0 - (4.0*bstd + 0.35*span)
    bub = b0 + (4.0*bstd + 0.35*span)
    eps_dir = max(1e-4, 0.01*span)
    if seg_sign is not None:
        if seg_sign > 0:
            blb = max(blb, y0_const + eps_dir)
        elif seg_sign < 0:
            bub = min(bub, y0_const - eps_dir)
        if blb >= bub:
            if seg_sign > 0:
                blb, bub = y0_const + eps_dir, y0_const + max(10*eps_dir, span)
            else:
                blb, bub = y0_const - max(10*eps_dir, span), y0_const - eps_dir

    lb = [blb, 0.2, 2e-9]
    ub = [bub, 40.0, max(0.9*dur, 30e-9)]

    def gf_fit(t, baseline, b, k):
        return anchored_gompertz_np(t, baseline, b, k, y0_const)

    candidates = [
        [b0, 1.2, 0.10*dur],
        [b0, 2.0, 0.18*dur],
        [b0, 3.0, 0.28*dur],
        [b0, 5.0, 0.40*dur],
    ]

    best = None
    best_sse = np.inf
    for p0 in candidates:
        p0 = [
            float(np.clip(p0[0], lb[0], ub[0])),
            float(np.clip(p0[1], lb[1], ub[1])),
            float(np.clip(max(p0[2], 1e-12), lb[2], ub[2])),
        ]
        try:
            popt, _ = curve_fit(gf_fit, t_fit, y_fit, p0=p0, bounds=(lb, ub), maxfev=25000)
            sse = float(np.sum((y_fit - gf_fit(t_fit, *popt))**2))
            if sse < best_sse:
                best_sse = sse
                best = popt
        except Exception:
            continue

    if best is None:
        best = np.array([b0, 2.0, min(max(0.2*dur, lb[2]), ub[2])], dtype=float)

    resid = y_fit - gf_fit(t_fit, *best)
    fit_sigma = float(np.std(resid) + 1e-9)
    return np.asarray(best, dtype=float), y0_const, fit_sigma

def smooth_by_ns(t_s: np.ndarray, y: np.ndarray, smooth_ns: float):
    dt = float(np.median(np.diff(t_s)))
    win = int(max(9, round(_ns_to_s(smooth_ns) / dt)))
    win = _oddify(win)
    win = min(win, max(9, (len(y)//2)*2 - 1))
    if win % 2 == 0:
        win -= 1
    return savgol_filter(y, window_length=win, polyorder=3), win

def rolling_mean(y: np.ndarray, win: int):
    y = np.asarray(y, dtype=float)
    win = max(3, int(win))
    if win % 2 == 0:
        win += 1
    if win >= len(y):
        return np.full_like(y, float(np.mean(y)))
    k = np.ones(win, dtype=float) / float(win)
    return np.convolve(y, k, mode="same")

def find_next_big_min_by_rolling(t_rel_ns, v, idx_start, min_delay_ns=12.0, max_delay_ns=320.0, smooth_ns=10.0, prom_frac=0.16):
    """
    Find the next prominent relative minimum after idx_start using rolling-average smoothing.
    Returns (idx or None, mode).
    """
    if idx_start is None or idx_start >= len(v) - 5:
        return None, "rolling_min_none"

    dt_ns = float(np.median(np.diff(t_rel_ns)))
    win = max(5, int(round(float(smooth_ns) / max(dt_ns, 1e-9))))
    vs = rolling_mean(np.asarray(v, dtype=float), win)

    t0 = float(t_rel_ns[idx_start]) + float(min_delay_ns)
    t1 = float(t_rel_ns[idx_start]) + float(max_delay_ns)
    i0 = int(np.searchsorted(t_rel_ns, t0))
    i1 = int(np.searchsorted(t_rel_ns, t1))
    i0 = max(i0, idx_start + 2)
    i1 = min(i1, len(vs) - 1)
    if i1 <= i0 + 6:
        return None, "rolling_min_none"

    seg = vs[i0:i1+1]
    scale = float(max(np.percentile(seg, 90) - np.percentile(seg, 10), np.std(seg) + 1e-12))
    prom = max(float(prom_frac) * scale, 1e-4)
    mins, _ = find_peaks(-seg, prominence=prom, distance=max(4, win // 2))
    if len(mins) > 0:
        return int(i0 + mins[0]), "rolling_min"

    j = int(np.argmin(seg))
    return int(i0 + j), "rolling_min_fallback"

def find_tailoff_zero_slope_idx(t_rel_ns, v, idx_start, min_delay_ns=40.0, max_delay_ns=500.0, smooth_ns=10.0, sustain_n=10, frac_p90=0.18):
    """
    Find where waveform starts tailing off to ~0 slope (sustained low |dv/dt|).
    """
    if idx_start is None or idx_start >= len(v) - sustain_n - 2:
        return None, "tailoff_none"
    dt_ns = float(np.median(np.diff(t_rel_ns)))
    win = max(5, int(round(float(smooth_ns) / max(dt_ns, 1e-9))))
    vs = rolling_mean(np.asarray(v, dtype=float), win)
    t_abs = _ns_to_s(t_rel_ns)
    dvdt = np.gradient(vs, t_abs)

    t0 = float(t_rel_ns[idx_start]) + float(min_delay_ns)
    t1 = float(t_rel_ns[idx_start]) + float(max_delay_ns)
    i0 = int(np.searchsorted(t_rel_ns, t0))
    i1 = int(np.searchsorted(t_rel_ns, t1))
    i0 = max(i0, idx_start + 2)
    i1 = min(i1, len(v) - sustain_n - 1)
    if i1 <= i0 + sustain_n:
        return None, "tailoff_none"

    ref = np.abs(dvdt[i0:i1+1])
    thr = max(float(frac_p90) * float(np.percentile(ref, 90)), 1e-4)
    for i in range(i0, i1):
        blk = np.abs(dvdt[i:i+sustain_n])
        if len(blk) == sustain_n and float(np.median(blk)) <= thr:
            return int(i), "tailoff_zero_slope"
    return None, "tailoff_none"

def find_next_rel_max_by_rolling(t_rel_ns, v, idx_start, min_delay_ns=8.0, max_delay_ns=220.0, smooth_ns=10.0, prom_frac=0.12):
    """
    Find next prominent relative maximum after idx_start on rolling-mean waveform.
    Returns (idx or None, mode).
    """
    if idx_start is None or idx_start >= len(v) - 5:
        return None, "rolling_max_none"
    dt_ns = float(np.median(np.diff(t_rel_ns)))
    win = max(5, int(round(float(smooth_ns) / max(dt_ns, 1e-9))))
    vs = rolling_mean(np.asarray(v, dtype=float), win)

    t0 = float(t_rel_ns[idx_start]) + float(min_delay_ns)
    t1 = float(t_rel_ns[idx_start]) + float(max_delay_ns)
    i0 = int(np.searchsorted(t_rel_ns, t0))
    i1 = int(np.searchsorted(t_rel_ns, t1))
    i0 = max(i0, idx_start + 2)
    i1 = min(i1, len(vs) - 1)
    if i1 <= i0 + 6:
        return None, "rolling_max_none"

    seg = vs[i0:i1+1]
    scale = float(max(np.percentile(seg, 90) - np.percentile(seg, 10), np.std(seg) + 1e-12))
    prom = max(float(prom_frac) * scale, 1e-4)
    pks, _ = find_peaks(seg, prominence=prom, distance=max(4, win // 2))
    if len(pks) > 0:
        return int(i0 + pks[0]), "rolling_max"
    j = int(np.argmax(seg))
    return int(i0 + j), "rolling_max_fallback"

def cubic_hermite_tail_np(t, y0, y1, s0, s1):
    """
    Cubic Hermite segment on t in [0, T]:
      y(0)=y0, y(T)=y1, y'(0)=s0, y'(T)=s1
    """
    t = np.asarray(t, dtype=float)
    T = max(float(t[-1]) if len(t) else 1e-12, 1e-12)
    u = np.clip(t / T, 0.0, 1.0)
    h00 = 2*u**3 - 3*u**2 + 1
    h10 = u**3 - 2*u**2 + u
    h01 = -2*u**3 + 3*u**2
    h11 = u**3 - u**2
    return h00*y0 + h10*(T*s0) + h01*y1 + h11*(T*s1)

def cubic_anchor_max_np(t, y0, y1, t_anchor, y_anchor, s1=0.0):
    """
    Cubic y=a t^3 + b t^2 + c t + d with:
      y(0)=y0, y(T)=y1, y(t_anchor)=y_anchor, y'(T)=s1
    """
    t = np.asarray(t, dtype=float)
    T = max(float(t[-1]) if len(t) else 1e-12, 1e-12)
    ta = float(np.clip(t_anchor, 1e-12, T - 1e-12))
    d = float(y0)
    A = np.array([
        [T**3,  T**2,  T],
        [ta**3, ta**2, ta],
        [3*T**2, 2*T, 1.0],
    ], dtype=float)
    bvec = np.array([
        float(y1) - d,
        float(y_anchor) - d,
        float(s1),
    ], dtype=float)
    try:
        a, b, c = np.linalg.solve(A, bvec)
    except Exception:
        # fallback: Hermite-like coefficients with zero end-slope
        a, b, c = 0.0, 0.0, (float(y1) - d) / T
    return ((a*t + b)*t + c)*t + d

def build_fit_waveform(t_rel_ns: np.ndarray, t_abs_s: np.ndarray, v_raw: np.ndarray, family: str, use_three_exp: bool, shot_id=None):
    """
    Build waveform used for fitting/split selection.
    Keep raw for event timing/stop and plotting; smooth only noisy SMAJ 2-exp shots.
    """
    if shot_id in SHOT_FORCE_FIT_SMOOTH_NS:
        smooth_ns = float(SHOT_FORCE_FIT_SMOOTH_NS[shot_id])
        v_sm, _ = smooth_by_ns(t_abs_s, v_raw, smooth_ns)
        return v_sm, 0.0, smooth_ns, "smooth"

    if use_three_exp or (family != "SMAJ400A"):
        return v_raw, 0.0, 0.0, "raw"

    pre = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(pre) < 20:
        n = max(30, int(0.08 * len(v_raw)))
        pre = np.zeros_like(v_raw, dtype=bool)
        pre[:n] = True
    noise_v = float(np.std(v_raw[pre]) + 1e-18)
    smooth_ns = float(SHOT_FIT_SMOOTH_NS_OVERRIDE.get(shot_id, SMAJ_FIT_SMOOTH_NS))
    force_smooth = shot_id in SHOT_FIT_SMOOTH_NS_OVERRIDE
    if (noise_v < SMAJ_FIT_SMOOTH_NOISE_THRESH_V) and (not force_smooth):
        return v_raw, noise_v, 0.0, "raw"

    v_sm, _ = smooth_by_ns(t_abs_s, v_raw, smooth_ns)
    return v_sm, noise_v, smooth_ns, "smooth"

def first_crossing_time(t: np.ndarray, y: np.ndarray, level: float):
    above = y >= level
    idx = np.where(above)[0]
    if len(idx) == 0:
        raise RuntimeError(f"Signal never crosses {level}V.")
    i = int(idx[0])
    if i == 0:
        return float(t[0])
    t0, t1 = t[i-1], t[i]
    y0, y1 = y[i-1], y[i]
    if y1 == y0:
        return float(t1)
    frac = (level - y0) / (y1 - y0)
    return float(t0 + frac*(t1 - t0))

def first_crossing_time_or_nearest(t: np.ndarray, y: np.ndarray, level: float):
    """
    Prefer true upward crossing time; fallback to nearest-to-level sample time.
    Returns (time_s, mode, y_at_pick).
    """
    try:
        tc = first_crossing_time(t, y, level)
        yi = float(np.interp(tc, t, y))
        return float(tc), "crossing", yi
    except RuntimeError:
        i = int(np.argmin(np.abs(y - level)))
        return float(t[i]), "nearest_level", float(y[i])

def auto_negate_decision(t_rel_ns: np.ndarray, v: np.ndarray):
    post = (t_rel_ns >= 0) & (t_rel_ns <= 1200.0)
    if np.sum(post) < 20:
        post = t_rel_ns >= 0
    vpost = v[post]
    vmax = float(np.max(vpost))
    vmin = float(np.min(vpost))
    return (abs(vmin) > 1.15*abs(vmax))

def infer_shot_and_family(csv_file: str):
    name = os.path.basename(csv_file)
    m = re.search(r"(\d+)", name)
    shot = int(m.group(1)) if m else None
    fam = SHOT_FAMILY_MAP.get(shot, "UNKNOWN") if shot is not None else "UNKNOWN"
    return shot, fam


# =============================
# CSV parsing (wide)
# =============================
def load_wide_csv(csv_file: str):
    df = pd.read_csv(csv_file)
    cols = list(df.columns)

    pairs = []
    i = 0
    while i < len(cols) - 1:
        pairs.append((cols[i], cols[i+1]))
        i += 2

    series = {}
    for tcol, scol in pairs:
        t = df[tcol].to_numpy(dtype=float)
        y = df[scol].to_numpy(dtype=float)
        m = np.isfinite(t) & np.isfinite(y)
        t = t[m]
        y = y[m]
        order = np.argsort(t)
        t = t[order]
        y = y[order]
        series[scol] = (t, y)

    if "Diode" not in series:
        raise RuntimeError(f"Missing 'Diode' column. Found: {list(series.keys())[:12]}...")
    return series

def build_pcd_avg(series: dict, t_ref: np.ndarray):
    pcd_names = [k for k in series.keys() if k.upper().startswith("PCD")]
    used = [n for n in pcd_names if n not in PCD_EXCLUDE]
    if len(used) == 0:
        raise RuntimeError(f"No PCD columns found (excluding {PCD_EXCLUDE}).")
    vals = []
    for name in used:
        t, v = series[name]
        vals.append(_interp_to(t, v, t_ref))
    return np.mean(np.vstack(vals), axis=0), used


# =============================
# Event picking
# =============================
def find_onset_first_drop(t_rel_ns, v, t_abs_s):
    t = t_abs_s
    vs, _ = smooth_by_ns(t, v, ONSET_SMOOTH_NS)
    dvdt = np.gradient(vs, t)

    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True

    sigma = float(np.std(dvdt[base]) + 1e-18)
    thr = ONSET_SIGMA_MULT * sigma
    start_idx = int(np.searchsorted(t_rel_ns, ONSET_MIN_NS_AFTER_T0))

    for i in range(start_idx, len(t_rel_ns) - ONSET_SUSTAIN_N):
        if np.all(np.abs(dvdt[i:i+ONSET_SUSTAIN_N]) >= thr):
            return i, thr, sigma

    # fallback: voltage departure
    vbase = float(np.median(v[base]))
    vr = float(np.std(v[base]) + 1e-12)
    idx = np.where(np.abs(v - vbase) > 6.0*vr)[0]
    if len(idx) == 0:
        return start_idx, thr, sigma
    return int(idx[0]), thr, sigma

def find_onset_zero_drop_strict(t_rel_ns, v, t_abs_s, shot_id=None):
    """
    Shot-specific onset: first strong drop below 0 V.
    """
    t = t_abs_s
    vs, _ = smooth_by_ns(t, v, ONSET_ZERO_DROP_SMOOTH_NS)
    dvdt = np.gradient(vs, t)

    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)
    thr = ONSET_ZERO_DROP_SIGMA_MULT * sigma
    v_thresh = float(ONSET_ZERO_DROP_V_THRESH_BY_SHOT.get(shot_id, ONSET_ZERO_DROP_V_THRESH))
    dv_thresh = float(ONSET_ZERO_DROP_DELTA_V_BY_SHOT.get(shot_id, ONSET_ZERO_DROP_DELTA_V))
    N = max(4, int(ONSET_ZERO_DROP_SUSTAIN_N))

    i0 = int(np.searchsorted(t_rel_ns, ONSET_MIN_NS_AFTER_T0))
    i1 = max(i0 + 1, len(vs) - N - 1)
    for i in range(i0, i1):
        if vs[i] > -v_thresh:
            continue
        dblk = dvdt[i:i+N]
        if float(np.median(dblk)) > -thr:
            continue
        vdrop = float(vs[i] - np.min(vs[i:i+N]))
        if vdrop < dv_thresh:
            continue
        return i, thr, sigma

    idx = np.where(vs[i0:] <= -v_thresh)[0]
    if len(idx) > 0:
        return int(i0 + idx[0]), thr, sigma
    return i0, thr, sigma

def pick_peak_idx(t_rel_ns, v, onset_idx, negate: bool, t_abs_s=None, smooth_ns: float = 0.0):
    t0 = t_rel_ns[onset_idx]
    mask = (t_rel_ns >= t0) & (t_rel_ns <= (t0 + PEAK_SEARCH_NS))
    idxs = np.where(mask)[0]
    if len(idxs) < 10:
        idxs = np.arange(onset_idx, min(len(v), onset_idx + 80))
    if (t_abs_s is not None) and (smooth_ns is not None) and (smooth_ns > 0):
        vs, _ = smooth_by_ns(np.asarray(t_abs_s), np.asarray(v), float(smooth_ns))
        seg = vs[idxs]
    else:
        seg = v[idxs]
    j = int(np.argmin(seg)) if negate else int(np.argmax(seg))
    return int(idxs[j])

def stop_by_next_turning_point(t_rel_ns, v, peak_idx, onset_idx=None):
    """
    Stop at the start of the NEXT oscillation (true turnaround), robustly:
    - compute baseline (pre-onset preferred)
    - require recovery to regain STOP_RECOVERY_FRAC of amplitude before allowing stop
    - then find first sustained dv/dt sign flip (median-window stabilized)

    This prevents stopping early on tiny inflections.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    t_peak = float(t_rel_ns[peak_idx])

    # baseline estimate (prefer pre-onset)
    if onset_idx is not None:
        iB1 = max(0, onset_idx - 200)
        iB2 = max(0, onset_idx - 20)
        if iB2 <= iB1:
            iB1 = max(0, peak_idx - 500)
            iB2 = max(0, peak_idx - 50)
    else:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)

    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    peak_v = float(v[peak_idx])

    amp = abs(baseline_v - peak_v) + 1e-12
    recovered_level = peak_v + np.sign(baseline_v - peak_v) * STOP_RECOVERY_FRAC * amp

    # bounds
    start_ns = t_peak + STOP_MIN_NS_AFTER_PEAK
    end_ns   = t_peak + STOP_MAX_NS_AFTER_PEAK
    i0 = int(np.searchsorted(t_rel_ns, start_ns))
    i1 = int(np.searchsorted(t_rel_ns, end_ns))
    i1 = min(i1, len(t_rel_ns) - 1)

    if i0 >= i1:
        return i1, float(dvdt[min(peak_idx, len(dvdt)-1)]), "turning_point_fallback_cap"

    # initial recovery slope sign (median after peak)
    local = dvdt[peak_idx+1:peak_idx+1+max(10, STOP_DVDT_MED_WIN*2)]
    if len(local) < 5:
        return i1, float(dvdt[min(peak_idx, len(dvdt)-1)]), "turning_point_fallback_cap"

    s0 = float(np.median(local))
    sgn0 = 1.0 if s0 >= 0 else -1.0

    # recovery gate
    if baseline_v >= peak_v:
        rec_ok_idxs = np.where(v[i0:i1] >= recovered_level)[0]
    else:
        rec_ok_idxs = np.where(v[i0:i1] <= recovered_level)[0]

    if len(rec_ok_idxs) == 0:
        return i1, s0, "turning_point_fallback_cap"
    i_rec_ok = i0 + int(rec_ok_idxs[0])

    # stabilized dv/dt sign via median window
    W = max(3, int(STOP_DVDT_MED_WIN))
    halfW = W // 2

    def dvdt_sign_med(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        m = float(np.median(dvdt[a:b]))
        return 1.0 if m >= 0 else -1.0

    for i in range(i_rec_ok, i1 - STOP_SUSTAIN_N):
        ok = True
        for j in range(i, i + STOP_SUSTAIN_N):
            if dvdt_sign_med(j) != -sgn0:
                ok = False
                break
        if ok:
            return i, s0, "turning_point"

    return i1, s0, "turning_point_fallback_cap"

def stop_by_slope_threshold(t_rel_ns, v, peak_idx):
    """
    Fallback stop:
    stop when dv/dt matches the early recovery slope again (within tolerance),
    sustained for SLOPE_SUSTAIN_N samples.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    t_peak = float(t_rel_ns[peak_idx])
    r0 = t_peak + SLOPE_REF_START_NS
    r1 = r0 + SLOPE_REF_LEN_NS
    ref_mask = (t_rel_ns >= r0) & (t_rel_ns <= r1)
    if np.sum(ref_mask) < 8:
        ref_mask = (t_rel_ns >= (t_peak + 2.0)) & (t_rel_ns <= (t_peak + 20.0))

    ref = float(np.median(dvdt[ref_mask])) if np.any(ref_mask) else float(np.median(dvdt[peak_idx+1:peak_idx+30]))
    ref_mag = max(abs(ref), 1e-18)
    tol = SLOPE_TOL_FRAC * ref_mag

    start_ns = t_peak + STOP_MIN_NS_AFTER_PEAK
    end_ns = t_peak + STOP_MAX_NS_AFTER_PEAK
    i0 = int(np.searchsorted(t_rel_ns, start_ns))
    i1 = int(np.searchsorted(t_rel_ns, end_ns))
    i1 = min(i1, len(t_rel_ns)-1)

    def matches(i):
        return abs(dvdt[i] - ref) <= tol

    for i in range(i0, i1 - SLOPE_SUSTAIN_N):
        if np.all([matches(j) for j in range(i, i+SLOPE_SUSTAIN_N)]):
            return i, ref, tol, "slope_threshold"

    return i1, ref, tol, "slope_threshold_fallback_cap"

def stop_by_new_period_start(t_rel_ns, v, peak_idx, onset_idx=None):
    """
    Strict stop selection:
    stop where the next period starts, detected as the first sustained strong
    reversal of slope after substantial recovery from the peak.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    t_peak = float(t_rel_ns[peak_idx])

    # baseline estimate (prefer pre-onset)
    if onset_idx is not None:
        iB1 = max(0, onset_idx - 200)
        iB2 = max(0, onset_idx - 20)
        if iB2 <= iB1:
            iB1 = max(0, peak_idx - 500)
            iB2 = max(0, peak_idx - 50)
    else:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)

    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    peak_v = float(v[peak_idx])
    amp = abs(baseline_v - peak_v) + 1e-12
    recovered_level = peak_v + np.sign(baseline_v - peak_v) * STOP_PERIOD_RECOVERY_FRAC * amp

    # search bounds
    start_ns = t_peak + STOP_PERIOD_MIN_NS_AFTER_PEAK
    end_ns = t_peak + STOP_MAX_NS_AFTER_PEAK
    i0 = int(np.searchsorted(t_rel_ns, start_ns))
    i1 = int(np.searchsorted(t_rel_ns, end_ns))
    i1 = min(i1, len(t_rel_ns) - 1)
    if i0 >= i1:
        return i1, 0.0, "period_change_fallback_cap"

    # gate: require substantial recovery before allowing next-period detection
    if baseline_v >= peak_v:
        rec_ok = np.where(v[i0:i1] >= recovered_level)[0]
    else:
        rec_ok = np.where(v[i0:i1] <= recovered_level)[0]
    if len(rec_ok) == 0:
        return i1, 0.0, "period_change_fallback_cap"
    i_start = i0 + int(rec_ok[0])

    # stabilized derivative via median window
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2

    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    # Recovery direction from geometry (toward baseline) is more robust than local dv/dt.
    geom = baseline_v - peak_v
    if abs(geom) < 1e-12:
        # fallback to delayed local derivative sign
        r0_ns = t_peak + 8.0
        r1_ns = t_peak + 45.0
        mrec = (t_rel_ns >= r0_ns) & (t_rel_ns <= r1_ns)
        local = dvdt[mrec]
        if len(local) < 5:
            local = dvdt[peak_idx+1:peak_idx+1+max(12, STOP_DVDT_MED_WIN*3)]
        if len(local) < 5:
            return i1, 0.0, "period_change_fallback_cap"
        s0 = float(np.median(local))
        rec_sign = 1.0 if s0 >= 0 else -1.0
    else:
        rec_sign = 1.0 if geom >= 0 else -1.0

    # threshold for "drastic" change
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    # Opposite-slope statistics and absolute-slope stats in the scan region.
    opp = np.array([(-rec_sign) * dmed(i) for i in range(i_start, i1)])
    abs_scan = np.array([abs(dmed(i)) for i in range(i_start, i1)])
    opp_pos = opp[opp > 0]
    p90 = float(np.percentile(opp_pos, 90)) if len(opp_pos) > 10 else 0.0
    p90_abs = float(np.percentile(abs_scan, 90)) if len(abs_scan) > 10 else 0.0
    strong_thr = max(STOP_PERIOD_SIGMA_MULT * sigma, STOP_PERIOD_FRAC_OF_P90 * p90)
    quiet_thr = max(2.0 * sigma, STOP_PERIOD_QUIET_FRAC_OF_P90ABS * p90_abs)

    # detect first plateau->drop transition:
    # quiet/flat pre-window, then sustained opposite strong slope.
    for i in range(max(i_start, peak_idx + STOP_PERIOD_PRE_QUIET_N), i1 - STOP_PERIOD_SUSTAIN_N):
        pre_vals = [dmed(j) for j in range(i - STOP_PERIOD_PRE_QUIET_N, i)]
        post_vals = [dmed(j) for j in range(i, i + STOP_PERIOD_SUSTAIN_N)]
        pre_quiet = float(np.median(np.abs(pre_vals))) <= quiet_thr
        post_drop = np.all([(-rec_sign * x) >= strong_thr for x in post_vals])
        if pre_quiet and post_drop:
            return i, strong_thr, "period_change"

    # Fallback: first sustained opposite strong slope.
    for i in range(i_start, i1 - STOP_PERIOD_SUSTAIN_N):
        vals = [dmed(j) for j in range(i, i + STOP_PERIOD_SUSTAIN_N)]
        if np.all([(-rec_sign * x) >= strong_thr for x in vals]):
            return i, strong_thr, "period_change_fallback_strong"

    return i1, strong_thr, "period_change_fallback_cap"

def stop_by_recovery_flattening(t_rel_ns, v, peak_idx, onset_idx=None, min_recovery_frac=0.0):
    """
    End the modeled segment where recovery slope flattens and stays near zero.
    Useful when there is a long quasi-flat tail before the next cycle.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    t_peak = float(t_rel_ns[peak_idx])
    peak_v = float(v[peak_idx])

    # baseline for expected recovery direction
    if onset_idx is not None:
        iB1 = max(0, onset_idx - 200)
        iB2 = max(0, onset_idx - 20)
        if iB2 <= iB1:
            iB1 = max(0, peak_idx - 500)
            iB2 = max(0, peak_idx - 50)
    else:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - peak_v) >= 0 else -1.0
    amp = abs(baseline_v - peak_v) + 1e-12

    start_ns = t_peak + STOP_FLAT_MIN_NS_AFTER_PEAK
    end_ns = t_peak + STOP_MAX_NS_AFTER_PEAK
    i0 = int(np.searchsorted(t_rel_ns, start_ns))
    i1 = int(np.searchsorted(t_rel_ns, end_ns))
    i1 = min(i1, len(t_rel_ns) - 1)
    if i0 >= i1:
        return None, 0.0, "flat_none"

    # stabilized derivative
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2

    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    rec_mag = np.array([rec_sign * dmed(i) for i in range(i0, i1)])
    rec_pos = rec_mag[rec_mag > 0]
    if len(rec_pos) < 12:
        return None, 0.0, "flat_none"

    p95 = float(np.percentile(rec_pos, 95))
    flat_thr = STOP_FLAT_FRAC_OF_RECOVERY_P95 * p95

    for i in range(i0, i1 - STOP_FLAT_SUSTAIN_N):
        dv = abs(float(v[i] - peak_v))
        if dv < STOP_FLAT_MIN_DV_FROM_PEAK:
            continue
        prog = rec_sign * (float(v[i]) - peak_v) / amp
        if prog < float(min_recovery_frac):
            continue
        vals = [rec_sign * dmed(j) for j in range(i, i + STOP_FLAT_SUSTAIN_N)]
        if np.all([x <= flat_thr for x in vals]):
            return i, flat_thr, "flat_slope"

    return None, flat_thr, "flat_none"

def stop_by_opposite_slope_change(t_rel_ns, v, peak_idx, onset_idx=None):
    """
    Stop at first sustained strong slope reversal (opposite of recovery direction),
    after substantial recovery has occurred.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    # baseline/recovery direction
    if onset_idx is not None:
        iB1 = max(0, onset_idx - 200)
        iB2 = max(0, onset_idx - 20)
        if iB2 <= iB1:
            iB1 = max(0, peak_idx - 500)
            iB2 = max(0, peak_idx - 50)
    else:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    peak_v = float(v[peak_idx])
    rec_sign = 1.0 if (baseline_v - peak_v) >= 0 else -1.0
    amp = abs(baseline_v - peak_v) + 1e-12
    rec_level = peak_v + rec_sign * STOP_OPP_RECOVERY_FRAC * amp

    # derivative noise floor
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    # scan bounds
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + STOP_OPP_MIN_NS_AFTER_PEAK))
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + STOP_MAX_NS_AFTER_PEAK))
    i1 = min(i1, len(t_rel_ns) - 1)
    if i0 >= i1:
        return None, 0.0, "opp_slope_none"

    # require substantial recovery before looking for reversal
    if rec_sign > 0:
        r = np.where(v[i0:i1] >= rec_level)[0]
    else:
        r = np.where(v[i0:i1] <= rec_level)[0]
    iscan = i0 + int(r[0]) if len(r) > 0 else i0

    # stabilized derivative
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    rec_vals = np.array([rec_sign * dmed(i) for i in range(iscan, i1)], dtype=float)
    rec_pos = rec_vals[rec_vals > 0]
    rec_p95 = float(np.percentile(rec_pos, 95)) if len(rec_pos) > 10 else 0.0
    thr = max(STOP_OPP_SIGMA_MULT * sigma, STOP_OPP_FRAC_OF_REC_P95 * rec_p95)
    N = max(4, int(STOP_OPP_SUSTAIN_N))

    for i in range(iscan, i1 - N):
        prog = rec_sign * (float(v[i]) - peak_v) / amp
        if prog < STOP_OPP_CAND_REC_FRAC:
            continue
        vals = [(-rec_sign) * dmed(j) for j in range(i, i + N)]
        if np.all([x >= thr for x in vals]):
            return i, thr, "opp_slope_change"

    return None, thr, "opp_slope_none"

def stop_by_constant_slope_onset(t_rel_ns, v, peak_idx, onset_idx=None):
    """
    Detect onset of sustained approximately-constant recovery slope.
    Used for family-1 behavior to end exp2 where linear-like ramp begins.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    t_peak = float(t_rel_ns[peak_idx])

    # baseline and recovery sign
    if onset_idx is not None:
        iB1 = max(0, onset_idx - 200)
        iB2 = max(0, onset_idx - 20)
        if iB2 <= iB1:
            iB1 = max(0, peak_idx - 500)
            iB2 = max(0, peak_idx - 50)
    else:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - float(v[peak_idx])) >= 0 else -1.0

    # derivative noise floor
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    start_ns = t_peak + STOP_CONST_MIN_NS_AFTER_PEAK
    end_ns = t_peak + STOP_MAX_NS_AFTER_PEAK
    i0 = int(np.searchsorted(t_rel_ns, start_ns))
    i1 = int(np.searchsorted(t_rel_ns, end_ns))
    i1 = min(i1, len(t_rel_ns)-1)
    if i0 >= i1:
        return None, "const_slope_none"

    # stabilized derivative
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    N = max(6, int(STOP_CONST_SUSTAIN_N))
    thr = STOP_CONST_MIN_SIGMA_MULT * sigma
    for i in range(i0, i1 - N):
        post = np.array([rec_sign * dmed(j) for j in range(i, i+N)], dtype=float)
        med = float(np.median(post))
        if med <= thr:
            continue
        cv = float(np.std(post) / max(abs(med), 1e-12))
        if cv <= STOP_CONST_CV_MAX:
            return i, "const_slope_onset"

    return None, "const_slope_none"

def stop_by_recovery_apex_reversal(t_rel_ns, v, peak_idx, onset_idx=None):
    """
    Fallback stop for 2-exp families:
    find recovery apex after peak, then first sustained opposite-direction slope.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    # baseline/recovery direction
    if onset_idx is not None:
        iB1 = max(0, onset_idx - 200)
        iB2 = max(0, onset_idx - 20)
        if iB2 <= iB1:
            iB1 = max(0, peak_idx - 500)
            iB2 = max(0, peak_idx - 50)
    else:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - float(v[peak_idx])) >= 0 else -1.0

    # derivative noise floor
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    # scan bounds
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + STOP_MIN_NS_AFTER_PEAK))
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + STOP_MAX_NS_AFTER_PEAK))
    i1 = min(i1, len(v) - 1)
    if i1 <= i0 + 8:
        return None, 0.0, "apex_reversal_none"

    # apex in recovery direction
    seg = vs[i0:i1+1]
    j = int(np.argmax(seg)) if rec_sign > 0 else int(np.argmin(seg))
    i_apex = i0 + j

    # stabilized derivative
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    thr = 2.0 * sigma
    N = max(4, int(STOP_OPP_SUSTAIN_N))
    i_start = max(i_apex + 1, i0 + 1)
    for i in range(i_start, i1 - N):
        vals = [(-rec_sign) * dmed(k) for k in range(i, i + N)]
        if np.all([x >= thr for x in vals]):
            return i, thr, "apex_reversal"

    return None, thr, "apex_reversal_none"

def stop_by_first_valley_minimum(t_rel_ns, v, peak_idx):
    """
    For positive-peak responses (273/274 style), stop at the first-cycle absolute
    valley after peak by taking the minimum on a short post-peak window.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_VALLEY_SMOOTH_NS)
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + STOP_VALLEY_MIN_NS_AFTER_PEAK))
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + STOP_VALLEY_MAX_NS_AFTER_PEAK))
    i0 = max(i0, peak_idx + 3)
    i1 = min(i1, len(vs) - 1)
    if i1 <= i0 + 6:
        return None, "valley_none"
    j = int(np.argmin(vs[i0:i1+1]))
    return int(i0 + j), "valley_minimum"

def stop_by_next_peak_window(t_rel_ns, v, peak_idx, w_from_ns, w_to_ns):
    """
    Shot-specific stop: pick local maximum in a post-peak window.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, STOP_NEXT_PEAK_SMOOTH_NS)
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + float(w_from_ns)))
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + float(w_to_ns)))
    i0 = max(i0, peak_idx + 3)
    i1 = min(i1, len(vs) - 1)
    if i1 <= i0 + 4:
        return None, "next_peak_window_none"
    j = int(np.argmax(vs[i0:i1+1]))
    return int(i0 + j), "next_peak_window"

def split1_by_min_window(t_rel_ns, v, peak_idx, w_from_ns, w_to_ns, smooth_ns=6.0):
    """
    Shot-specific split1: choose local minimum in a bounded window after peak.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, smooth_ns)
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + float(w_from_ns)))
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + float(w_to_ns)))
    i0 = max(i0, peak_idx + 2)
    i1 = min(i1, len(vs) - 1)
    if i1 <= i0:
        return None, "split1_window_none"
    j = int(np.argmin(vs[i0:i1+1]))
    return int(i0 + j), "split1_window_min"

def split1_by_pos_to_neg_slope(t_rel_ns, v, onset_idx, peak_idx, stop_idx, smooth_ns=8.0):
    """
    Find split at first robust +dV/dt to -dV/dt transition (local max turn).
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, smooth_ns)
    dvdt = np.gradient(vs, t_abs)

    dt_ns = float(np.median(np.diff(t_rel_ns)))
    back_n = max(8, int(round(80.0 / max(dt_ns, 1e-9))))
    fwd_n = max(6, int(round(60.0 / max(dt_ns, 1e-9))))
    i0 = max(onset_idx + 3, peak_idx - back_n)
    i1 = min(stop_idx - 3, peak_idx + fwd_n)
    if i1 <= i0 + 4:
        return int(peak_idx), 0.0, "split1_posneg_fallback_peak"

    ref = dvdt[i0:i1]
    thr = max(float(np.percentile(np.abs(ref), 60)) * 0.35, 1e-12)
    pre_n = 4
    post_n = 4
    for i in range(i0 + pre_n, i1 - post_n):
        pre = dvdt[i-pre_n:i]
        post = dvdt[i:i+post_n]
        if (np.median(pre) > thr) and (np.median(post) < -thr):
            return int(i), float(np.std(ref) + 1e-18), "split1_pos_to_neg_slope"

    return int(peak_idx), float(np.std(ref) + 1e-18), "split1_posneg_fallback_peak"

def stop_by_local_max_after_split_window(t_rel_ns, v, split_idx, w_from_ns, w_to_ns, smooth_ns=6.0):
    """
    Choose stop as first local maximum after split in a bounded window.
    Falls back to absolute maximum in the same window.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, smooth_ns)
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[split_idx]) + float(w_from_ns)))
    i0 = max(i0, split_idx + 2, 0)
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[split_idx]) + float(w_to_ns)))
    i1 = min(i1, len(vs) - 1)
    if i1 <= i0 + 3:
        return None, "stop_after_split_local_none"
    dv = np.gradient(vs, t_abs)
    N = 4
    for i in range(i0 + N, i1 - N):
        pre = dv[i-N:i]
        post = dv[i:i+N]
        if (float(np.median(pre)) > 0.0) and (float(np.median(post)) < 0.0):
            return int(i), "stop_after_split_local_max"
    seg = vs[i0:i1+1]
    vmax = float(np.max(seg))
    near = np.where(seg >= 0.96 * vmax)[0]
    if len(near) > 0:
        return int(i0 + int(near[0])), "stop_after_split_nearmax_fallback"
    j = int(np.argmax(seg))
    return int(i0 + j), "stop_after_split_max_fallback"

def stop_by_raw_max_after_split_window(t_rel_ns, v, split_idx, w_from_ns, w_to_ns):
    """
    Shot-specific stop on raw waveform: absolute max after split in bounded window.
    """
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[split_idx]) + float(w_from_ns)))
    i0 = max(i0, split_idx + 2, 0)
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[split_idx]) + float(w_to_ns)))
    i1 = min(i1, len(v) - 1)
    if i1 <= i0:
        return None, "stop_after_split_raw_none"
    j = int(np.argmax(v[i0:i1+1]))
    return int(i0 + j), "stop_after_split_raw_max"

def find_split_by_slope_change(t_rel_ns, v, onset_idx, peak_idx, stop_idx, negate: bool):
    """
    Find the real handoff point where slope changes direction robustly.
    Uses smoothed derivative and sustained threshold logic.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, SPLIT_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)
    thr = max(SPLIT_SIGMA_MULT * sigma, 1e6)

    # median-smoothed derivative for robust sign decision
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2

    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    pre_sign = -1.0 if negate else 1.0
    post_sign = -pre_sign

    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT_MIN_NS_AFTER_PEAK))
    i0 = max(i0, peak_idx + SPLIT_PRE_SUSTAIN_N + 1)
    i1 = max(i0 + 1, stop_idx - SPLIT_POST_SUSTAIN_N - 1)

    # Dynamic "real rise/drop" threshold from local max derivative magnitude.
    dseg = np.array([post_sign * dmed(j) for j in range(i0, i1 + SPLIT_POST_SUSTAIN_N)])
    if len(dseg) > 8:
        local_peak = float(np.percentile(dseg, 95))
    else:
        local_peak = float(np.max(dseg)) if len(dseg) else 0.0
    rise_thr = max(SPLIT_POST_THR_FRAC * thr, SPLIT_RISE_FRAC_OF_MAX * max(local_peak, 0.0))

    for i in range(i0, i1):
        pre_vals = [dmed(j) for j in range(i - SPLIT_PRE_SUSTAIN_N, i)]
        post_vals = [dmed(j) for j in range(i, i + SPLIT_POST_SUSTAIN_N)]

        pre_lvl = float(np.median([post_sign * x for x in pre_vals]))
        post_lvl = [post_sign * x for x in post_vals]
        pre_ok = pre_lvl <= (SPLIT_PRE_MAX_FRAC * rise_thr)
        post_ok = np.all([x >= rise_thr for x in post_lvl])
        if pre_ok and post_ok:
            return i, thr, sigma, "slope_change"

    # Fallback: first sustained crossing of half local-peak derivative magnitude.
    half_thr = 0.50 * max(local_peak, rise_thr)
    for i in range(i0, i1):
        post_vals = [post_sign * dmed(j) for j in range(i, i + SPLIT_POST_SUSTAIN_N)]
        if np.all([x >= half_thr for x in post_vals]):
            return i, thr, sigma, "slope_change_fallback_halfpeak"

    # Last fallback: middle between peak and stop to avoid collapsing at peak.
    i_fallback = int(round(0.65 * peak_idx + 0.35 * stop_idx))
    i_fallback = max(min(i_fallback, stop_idx - 2), peak_idx + 2)
    return i_fallback, thr, sigma, "slope_change_fallback_mid"

def find_split1_opposite_turn(t_rel_ns, v, onset_idx, peak_idx, stop_idx):
    """
    split1: first robust change to the opposite slope direction after the first peak.
    This is the exp1->exp2 handoff.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, SPLIT_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    # baseline sigma (noise floor for derivative thresholds)
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    # recovery sign toward baseline relative to peak
    iB1 = max(0, onset_idx - 200)
    iB2 = max(0, onset_idx - 20)
    if iB2 <= iB1:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - float(v[peak_idx])) >= 0 else -1.0
    pre_sign = -rec_sign

    # median derivative helper
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_MIN_NS_AFTER_PEAK))
    i0 = max(i0, peak_idx + SPLIT1_PRE_SUSTAIN_N + 1)
    i1 = max(i0 + 1, stop_idx - SPLIT1_POST_SUSTAIN_N - 1)
    i1_cap = peak_idx + int(max(8, SPLIT1_MAX_FRAC_OF_PEAK_TO_STOP * (stop_idx - peak_idx)))
    i1_cap_ns = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_MAX_NS_AFTER_PEAK))
    i1_cap = min(i1_cap, i1_cap_ns)
    i1 = min(i1, i1_cap)
    if i1 <= i0 + 3:
        i = max(min(peak_idx + 6, stop_idx - 2), peak_idx + 2)
        return i, sigma, "split1_fallback_short"

    # Early-turnaround candidate: first local extremum after peak in a short window.
    i_cap_early = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_EARLY_EXTREMUM_MAX_NS))
    i_cap_early = min(i_cap_early, i1)
    if i_cap_early > i0 + 4:
        seg = vs[peak_idx:i_cap_early+1]
        j = int(np.argmin(seg)) if rec_sign > 0 else int(np.argmax(seg))
        i_ext = peak_idx + j
        if (i0 <= i_ext < i1) and (i_ext > peak_idx + 1):
            return i_ext, sigma, "split1_early_extremum"

    rec_scan = np.array([rec_sign * dmed(i) for i in range(i0, i1 + SPLIT1_POST_SUSTAIN_N)])
    rec_pos = rec_scan[rec_scan > 0]
    p95_rec = float(np.percentile(rec_pos, 95)) if len(rec_pos) > 10 else 0.0
    post_thr = max(SPLIT1_POST_FRAC_OF_P95 * p95_rec, 0.25 * SPLIT1_PRE_THR_SIGMA * sigma)
    pre_thr = 0.25 * SPLIT1_PRE_THR_SIGMA * sigma

    for i in range(i0, i1):
        pre_vals = [pre_sign * dmed(j) for j in range(i - SPLIT1_PRE_SUSTAIN_N, i)]
        post_vals = [rec_sign * dmed(j) for j in range(i, i + SPLIT1_POST_SUSTAIN_N)]
        pre_ok = np.median(pre_vals) >= pre_thr
        post_ok = np.all([x >= post_thr for x in post_vals])
        if pre_ok and post_ok:
            return i, sigma, "split1_opposite_turn"

    # softer sign-based fallback: first sustained sign reversal
    for i in range(i0, i1):
        pre_vals = [pre_sign * dmed(j) for j in range(i - SPLIT1_PRE_SUSTAIN_N, i)]
        post_vals = [rec_sign * dmed(j) for j in range(i, i + SPLIT1_POST_SUSTAIN_N)]
        pre_ok = np.median(pre_vals) > 0.0
        post_ok = np.median(post_vals) > 0.0
        if pre_ok and post_ok:
            return i, sigma, "split1_sign_reversal"

    # fallback: first sustained recovery slope rise
    for i in range(i0, i1):
        post_vals = [rec_sign * dmed(j) for j in range(i, i + SPLIT1_POST_SUSTAIN_N)]
        if np.all([x >= post_thr for x in post_vals]):
            return i, sigma, "split1_fallback_recovery"

    i = max(min(peak_idx + 10, i1 - 1), peak_idx + 2)
    return i, sigma, "split1_fallback_near_peak"

def find_split1_zero_to_rise(t_rel_ns, v, onset_idx, peak_idx, stop_idx):
    """
    Family-1 split:
    end exp1 where slope leaves ~0 plateau and enters sustained strong rise.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, SPLIT_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    # baseline sigma for derivative scale
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    # recovery sign from geometry
    iB1 = max(0, onset_idx - 200)
    iB2 = max(0, onset_idx - 20)
    if iB2 <= iB1:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - float(v[peak_idx])) >= 0 else -1.0
    amp = abs(baseline_v - float(v[peak_idx])) + 1e-12
    rec_level = float(v[peak_idx]) + rec_sign * SPLIT1_ZERO_RISE_RECOVERY_FRAC * amp
    i_rec_end = int(np.searchsorted(
        t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_ZERO_RISE_RECOVERY_LOOKAHEAD_NS
    ))
    i_rec_end = min(max(i_rec_end, stop_idx), len(v) - 1)

    # median-derivative helper
    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_ZERO_RISE_MIN_NS_AFTER_PEAK))
    i0 = max(i0, peak_idx + SPLIT1_ZERO_RISE_PRE_N + 1)
    i1 = max(i0 + 1, stop_idx - SPLIT1_ZERO_RISE_POST_N - 1)
    if i1 <= i0 + 3:
        i = max(min(peak_idx + 8, stop_idx - 2), peak_idx + 2)
        return i, sigma, "split1_zero_rise_fallback_short"

    dscan = np.array([rec_sign * dmed(i) for i in range(i0, i1 + SPLIT1_ZERO_RISE_POST_N)])
    p95 = float(np.percentile(dscan[dscan > 0], 95)) if np.any(dscan > 0) else 0.0
    post_thr = max(SPLIT1_ZERO_RISE_POST_FRAC * p95, 2.0*sigma)
    zero_thr = max(SPLIT1_ZERO_RISE_ZERO_FRAC * post_thr, 1.0*sigma)

    Npre = max(4, int(SPLIT1_ZERO_RISE_PRE_N))
    Npost = max(4, int(SPLIT1_ZERO_RISE_POST_N))
    hold_thr = SPLIT1_ZERO_RISE_HOLD_FRAC * post_thr

    def hold_ok(i):
        i_hold = int(np.searchsorted(t_rel_ns, float(t_rel_ns[i]) + SPLIT1_ZERO_RISE_HOLD_NS))
        i_hold = min(i_hold, i1 + Npost - 1, len(v)-1)
        if i_hold <= i + Npost:
            return False
        vals = np.array([rec_sign * dmed(j) for j in range(i + Npost, i_hold + 1)], dtype=float)
        if len(vals) < 4:
            return False
        return (float(np.median(vals)) >= hold_thr) and (
            float(np.mean(vals >= hold_thr)) >= SPLIT1_ZERO_RISE_HOLD_HIT_FRAC
        )

    for i in range(i0 + Npre, i1):
        pre = np.array([rec_sign * dmed(j) for j in range(i - Npre, i)], dtype=float)
        post = np.array([rec_sign * dmed(j) for j in range(i, i + Npost)], dtype=float)
        pre_near_zero = float(np.median(np.abs(pre))) <= zero_thr
        post_strong = (float(np.median(post)) >= post_thr) and (
            float(np.mean(post >= post_thr)) >= SPLIT1_ZERO_RISE_POST_HIT_FRAC
        )
        if pre_near_zero and post_strong and hold_ok(i):
            # must be followed by substantial recovery (reject noisy local flips)
            if rec_sign > 0:
                rec_ok = np.any(v[i:i_rec_end+1] >= rec_level)
            else:
                rec_ok = np.any(v[i:i_rec_end+1] <= rec_level)
            if not rec_ok:
                continue
            return i, sigma, "split1_zero_to_rise"

    # fallback: first sustained strong rise
    for i in range(i0, i1):
        post = np.array([rec_sign * dmed(j) for j in range(i, i + Npost)], dtype=float)
        post_strong = (float(np.median(post)) >= post_thr) and (
            float(np.mean(post >= post_thr)) >= SPLIT1_ZERO_RISE_POST_HIT_FRAC
        )
        if post_strong and hold_ok(i):
            # local rise gate to avoid choosing tiny/noisy wiggles near the peak
            i_loc = int(np.searchsorted(t_rel_ns, float(t_rel_ns[i]) + SPLIT1_ZERO_RISE_LOCAL_RISE_NS))
            i_loc = min(max(i_loc, i + 1), min(stop_idx, len(v)-1))
            rise_loc = rec_sign * (float(v[i_loc]) - float(v[i]))
            if rise_loc < SPLIT1_ZERO_RISE_LOCAL_RISE_FRAC * amp:
                continue
            if rec_sign > 0:
                rec_ok = np.any(v[i:i_rec_end+1] >= rec_level)
            else:
                rec_ok = np.any(v[i:i_rec_end+1] <= rec_level)
            if not rec_ok:
                continue
            return i, sigma, "split1_zero_rise_fallback_strong"

    # fallback rescue for noisy traces: detect first sustained recovery knee
    # using a lower threshold on smoothed slope and a local rise requirement.
    p90 = float(np.percentile(dscan[dscan > 0], 90)) if np.any(dscan > 0) else 0.0
    knee_thr = max(1.2*sigma, 0.35*p90)
    for i in range(i0, i1):
        post = np.array([rec_sign * dmed(j) for j in range(i, i + Npost)], dtype=float)
        if (float(np.median(post)) < knee_thr) or (not hold_ok(i)):
            continue
        i_loc = int(np.searchsorted(t_rel_ns, float(t_rel_ns[i]) + SPLIT1_ZERO_RISE_LOCAL_RISE_NS))
        i_loc = min(max(i_loc, i + 1), min(stop_idx, len(v)-1))
        rise_loc = rec_sign * (float(v[i_loc]) - float(v[i]))
        if rise_loc < SPLIT1_ZERO_RISE_LOCAL_RISE_FRAC * amp:
            continue
        if rec_sign > 0:
            rec_ok = np.any(v[i:i_rec_end+1] >= rec_level)
        else:
            rec_ok = np.any(v[i:i_rec_end+1] <= rec_level)
        if not rec_ok:
            continue
        return i, sigma, "split1_zero_rise_fallback_knee"

    i = max(min(peak_idx + 10, stop_idx - 2), peak_idx + 2)
    return i, sigma, "split1_zero_rise_fallback_near_peak"

def should_anchor_split1_at_peak(t_rel_ns, v, onset_idx, peak_idx, stop_idx):
    """
    For noisy SMAJ shots, if slope immediately after peak strongly follows recovery
    direction, anchor split1 at peak (exp2 starts at peak).
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, SPLIT_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    # derivative noise floor
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    # recovery direction from baseline geometry
    iB1 = max(0, onset_idx - 200)
    iB2 = max(0, onset_idx - 20)
    if iB2 <= iB1:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - float(v[peak_idx])) >= 0 else -1.0

    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_PEAK_ANCHOR_START_NS))
    i1 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_PEAK_ANCHOR_WIN_NS))
    i1 = min(i1, stop_idx - 1, len(v) - 1)
    if i1 <= i0 + 3:
        return False, sigma

    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    vals = np.array([rec_sign * dmed(i) for i in range(i0, i1 + 1)], dtype=float)
    thr = SPLIT1_PEAK_ANCHOR_SIGMA_MULT * sigma
    if len(vals) < SPLIT1_PEAK_ANCHOR_SUSTAIN_N:
        return False, sigma

    N = int(SPLIT1_PEAK_ANCHOR_SUSTAIN_N)
    for k in range(0, len(vals) - N + 1):
        blk = vals[k:k+N]
        if (float(np.median(blk)) >= thr) and (float(np.mean(blk >= thr)) >= SPLIT1_PEAK_ANCHOR_HIT_FRAC):
            return True, sigma
    return False, sigma

def find_split1_strict_inflection(t_rel_ns, v, onset_idx, peak_idx, stop_idx, strict_mult=1.0, min_ns_after_peak=None):
    """
    Strict split1 rescue: first sustained recovery knee where slope and acceleration
    both rise after a low-slope pre-window.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, SPLIT1_STRICT_INFLECT_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    # derivative noise floor
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    # recovery direction from baseline geometry
    iB1 = max(0, onset_idx - 200)
    iB2 = max(0, onset_idx - 20)
    if iB2 <= iB1:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - float(v[peak_idx])) >= 0 else -1.0

    min_ns = float(SPLIT1_STRICT_INFLECT_MIN_NS_AFTER_PEAK if min_ns_after_peak is None else min_ns_after_peak)
    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + min_ns))
    i0 = max(i0, peak_idx + SPLIT1_STRICT_INFLECT_PRE_N + 1)
    i1 = min(stop_idx - SPLIT1_STRICT_INFLECT_POST_N - 1, len(v) - 2)
    if i1 <= i0 + 4:
        return None, sigma, "split1_strict_inflect_none"

    d_eff = rec_sign * dvdt
    dd_eff = np.gradient(d_eff, t_abs)
    pos = d_eff[i0:i1]
    pos = pos[pos > 0]
    if len(pos) < 8:
        return None, sigma, "split1_strict_inflect_none"
    p90_s = float(np.percentile(pos, 90))
    p90_a = float(np.percentile(dd_eff[i0:i1][dd_eff[i0:i1] > 0], 90)) if np.any(dd_eff[i0:i1] > 0) else 0.0
    slope_thr = max(1.5*sigma, strict_mult * SPLIT1_STRICT_INFLECT_SLOPE_FRAC_P90 * p90_s)
    accel_thr = strict_mult * SPLIT1_STRICT_INFLECT_ACCEL_FRAC_P90 * max(p90_a, 0.0)
    Npre = max(4, int(SPLIT1_STRICT_INFLECT_PRE_N))
    Npost = max(4, int(SPLIT1_STRICT_INFLECT_POST_N))

    for i in range(i0, i1):
        pre = np.array([d_eff[j] for j in range(i - Npre, i)], dtype=float)
        post = np.array([d_eff[j] for j in range(i, i + Npost)], dtype=float)
        acc = np.array([dd_eff[j] for j in range(i, i + Npost)], dtype=float)
        pre_ok = float(np.median(pre)) <= SPLIT1_STRICT_INFLECT_PRE_FRAC_POST * slope_thr
        post_ok = float(np.median(post)) >= slope_thr
        acc_ok = float(np.median(acc)) >= accel_thr
        if pre_ok and post_ok and acc_ok:
            return i, sigma, "split1_strict_inflection"

    return None, sigma, "split1_strict_inflect_none"

def find_split1_bottom_flip(t_rel_ns, v, onset_idx, peak_idx, stop_idx):
    """
    split1 at the bottom turn: where slope passes through ~0 and flips direction.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, SPLIT_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    # baseline / recovery direction
    iB1 = max(0, onset_idx - 200)
    iB2 = max(0, onset_idx - 20)
    if iB2 <= iB1:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - float(v[peak_idx])) >= 0 else -1.0
    pre_sign = -rec_sign

    # derivative noise floor
    base = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
    if np.sum(base) < 15:
        n = max(20, int(0.05*len(dvdt)))
        base = np.zeros_like(base, dtype=bool)
        base[:n] = True
    sigma = float(np.std(dvdt[base]) + 1e-18)

    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_MIN_NS_AFTER_PEAK))
    i1_ns = int(np.searchsorted(t_rel_ns, float(t_rel_ns[peak_idx]) + SPLIT1_MAX_NS_AFTER_PEAK))
    i1 = min(max(i0 + 2, i1_ns), stop_idx - 2)
    if i1 <= i0 + 3:
        i = max(min(peak_idx + 6, stop_idx - 2), peak_idx + 2)
        return i, sigma, "split1_bottom_fallback_short"

    pre_n = max(4, SPLIT1_PRE_SUSTAIN_N)
    post_n = max(4, SPLIT1_POST_SUSTAIN_N)
    pre_thr = 0.10 * sigma
    post_thr = 0.10 * sigma

    for i in range(i0 + pre_n, i1 - post_n):
        pre_vals = [pre_sign * dmed(j) for j in range(i - pre_n, i)]
        post_vals = [rec_sign * dmed(j) for j in range(i, i + post_n)]
        if (float(np.median(pre_vals)) >= pre_thr) and (float(np.median(post_vals)) >= post_thr):
            # lock to nearest zero-slope sample around candidate
            j0 = max(i - pre_n, i0)
            j1 = min(i + post_n, i1)
            j_star = min(range(j0, j1), key=lambda jj: abs(dmed(jj)))
            return int(j_star), sigma, "split1_bottom_flip"

    # fallback: local extremum near peak
    seg = vs[i0:i1+1]
    j = int(np.argmin(seg)) if rec_sign > 0 else int(np.argmax(seg))
    return int(i0 + j), sigma, "split1_bottom_fallback_extremum"

def find_split2_inflection(t_rel_ns, v, split1_idx, stop_idx, rec_sign):
    """
    split2: recovery inflection proxy at peak recovery slope (max dV/dt in recovery direction).
    This is the exp2->exp3 handoff.
    """
    t_abs = _ns_to_s(t_rel_ns)
    vs, _ = smooth_by_ns(t_abs, v, SPLIT_SMOOTH_NS)
    dvdt = np.gradient(vs, t_abs)

    W = max(5, int(STOP_DVDT_MED_WIN))
    if W % 2 == 0:
        W += 1
    halfW = W // 2
    def dmed(i):
        a = max(i - halfW, 0)
        b = min(i + halfW + 1, len(dvdt))
        return float(np.median(dvdt[a:b]))

    i0 = int(np.searchsorted(t_rel_ns, float(t_rel_ns[split1_idx]) + SPLIT2_MIN_NS_AFTER_SPLIT1))
    i0 = max(i0, split1_idx + 2)
    i1 = max(i0 + 1, stop_idx - SPLIT2_POST_SUSTAIN_N - 1)
    if i1 <= i0 + 3:
        i = max(min(split1_idx + 8, stop_idx - 2), split1_idx + 2)
        return i, "split2_fallback_short"

    d_eff = np.array([rec_sign * dmed(i) for i in range(i0, i1)])
    if len(d_eff) < 5:
        i = max(min(split1_idx + 8, stop_idx - 2), split1_idx + 2)
        return i, "split2_fallback_short"

    # 1) Acceleration-onset detector: first sustained increase in recovery slope.
    # This targets the beginning of the ramp (knee onset), not the late max-slope zone.
    t_slice = t_abs[i0:i1]
    dd_eff = np.gradient(d_eff, t_slice) if len(d_eff) >= 3 else np.zeros_like(d_eff)
    p95_slope = float(np.percentile(d_eff, 95))
    p95_acc = float(np.percentile(dd_eff[dd_eff > 0], 95)) if np.any(dd_eff > 0) else 0.0
    slope_base = float(np.median(d_eff[:max(5, int(0.30 * len(d_eff)))]))
    slope_thr = slope_base + SPLIT2_SLOPE_LEVEL_FRAC * max(p95_slope - slope_base, 0.0)
    acc_thr = SPLIT2_ACCEL_FRAC_OF_P95 * max(p95_acc, 0.0)

    n_acc = max(4, int(SPLIT2_ACCEL_SUSTAIN_N))
    i0_acc = i0 + int(SPLIT2_ACCEL_MIN_FRAC_OF_RANGE * max(1, (i1 - i0)))
    for i in range(i0_acc, i1 - n_acc):
        k = i - i0
        pre_s = d_eff[max(0, k-n_acc):k]
        post_s = d_eff[k:k+n_acc]
        post_a = dd_eff[k:k+n_acc]
        if len(pre_s) < max(3, n_acc//2):
            continue
        pre_med = float(np.median(pre_s))
        post_med = float(np.median(post_s))
        acc_med = float(np.median(post_a))
        if (post_med >= slope_thr) and (acc_med >= acc_thr) and (post_med >= pre_med):
            return i, "split2_accel_onset"

    # 2) Strict knee detector: late, sustained high-slope onset.
    i0_strict = i0 + int(SPLIT2_STRICT_MIN_FRAC_OF_RANGE * max(1, (i1 - i0)))
    p95 = float(np.percentile(d_eff, 95))
    rise_thr = SPLIT2_STRICT_FRAC_OF_P95 * p95
    n_str = max(4, int(SPLIT2_STRICT_SUSTAIN_N))
    for i in range(i0_strict, i1 - n_str):
        k = i - i0
        post = d_eff[k:k+n_str]
        pre = d_eff[max(0, k-n_str):k]
        if len(pre) < max(3, n_str//2):
            continue
        if np.all(post >= rise_thr) and (float(np.median(post)) >= SPLIT2_STRICT_PRE_RATIO * float(np.median(pre))):
            return i, "split2_strict_knee"

    jmax = int(np.argmax(d_eff))
    i_inflect = i0 + jmax
    dmax = float(d_eff[jmax])
    drop_thr = SPLIT2_DROP_FRAC_FROM_MAX * dmax

    # Require that slope begins sustained decline after the candidate.
    for i in range(i_inflect, i1 - SPLIT2_POST_SUSTAIN_N):
        post = [rec_sign * dmed(j) for j in range(i, i + SPLIT2_POST_SUSTAIN_N)]
        if np.all([x <= drop_thr for x in post]):
            return i_inflect, "split2_inflection"

    return i_inflect, "split2_fallback_maxslope"

def find_sigmoid_mismatch_idx(t_abs_seg, v_seg, v_sig_seg, rec_sign):
    """
    Find where sigmoid slope diverges from waveform slope (start of tail mismatch).
    Returns segment-local index.
    """
    if len(t_abs_seg) < 20:
        return max(2, len(t_abs_seg)//2), "sigmoid_mismatch_fallback_short"

    vs, _ = smooth_by_ns(t_abs_seg, v_seg, STOP_SMOOTH_NS)
    dv_data = np.gradient(vs, t_abs_seg)
    dv_sig = np.gradient(v_sig_seg, t_abs_seg)
    err = rec_sign * (dv_sig - dv_data)
    rel = err / np.maximum(np.abs(dv_data), 1e-9)

    i0 = int(SIGMOID_MISMATCH_MIN_FRAC * len(err))
    i0 = min(max(i0, 3), len(err)-3)
    p90 = float(np.percentile(np.abs(err[i0:]), 90)) if len(err[i0:]) > 10 else float(np.max(np.abs(err[i0:])))
    thr = SIGMOID_MISMATCH_FRAC_OF_P90 * max(p90, 1e-12)
    N = max(4, int(SIGMOID_MISMATCH_SUSTAIN_N))

    for i in range(i0, len(err) - N):
        block = err[i:i+N]
        block_rel = rel[i:i+N]
        if np.all(block >= thr) and np.median(block_rel) >= SIGMOID_MISMATCH_REL_RATIO:
            return i, "sigmoid_mismatch"

    # fallback: earliest time error reaches a fraction of its later maximum
    e_tail = err[i0:]
    if len(e_tail) > 5:
        emax = float(np.max(e_tail))
        if emax > 0:
            idx = np.where(e_tail >= 0.35 * emax)[0]
            if len(idx) > 0:
                return int(i0 + idx[0]), "sigmoid_mismatch_fallback_early"
    return max(i0, len(err)-N-1), "sigmoid_mismatch_fallback_end"


# =============================
# Main
# =============================
def main(csv_file=None, out_dir=None):
    csv_file = csv_file or CSV_FILE
    out_dir = out_dir or OUT_DIR
    os.makedirs(out_dir, exist_ok=True)

    # Load
    series = load_wide_csv(csv_file)
    t_diode, v_diode = series["Diode"]

    # PCD avg on diode time
    pcd_avg, pcd_used = build_pcd_avg(series, t_diode)
    pcd_s, _ = smooth_by_ns(t_diode, pcd_avg, 3.0)

    # NEW t0
    t_cross, t_cross_mode, pcd_pick_v = first_crossing_time_or_nearest(t_diode, pcd_s, PCD_TARGET_V)
    new_t0_abs = t_cross - _ns_to_s(T0_SHIFT_NS)
    t_rel_ns = _s_to_ns(t_diode - new_t0_abs)

    # NEGATE decision
    NEGATE = auto_negate_decision(t_rel_ns, v_diode)
    shot_id, family = infer_shot_and_family(csv_file)
    # Enforce family-gated behavior:
    # - 1N6517US: exp1 + sigmoid + exp tail
    # - everyone else: earlier 2-exp model
    use_four_exp = (shot_id in FOUR_EXP_SHOTS)
    use_three_exp = (family in THREE_EXP_FAMILIES) and (not use_four_exp)
    use_gompertz = bool(use_three_exp and (RECOVERY_SHAPE_CSD.lower() == "gompertz"))
    model_name = "4-exp-custom" if use_four_exp else ("exp1+gompertz" if use_gompertz else ("exp1+sigmoid+exp_tail" if use_three_exp else "2-exp"))
    v_fit, pre_noise_v, fit_smooth_ns, fit_mode = build_fit_waveform(
        t_rel_ns, t_diode, v_diode, family, use_three_exp, shot_id=shot_id
    )

    print("\nPCD avg columns used:", pcd_used)
    if t_cross_mode == "crossing":
        print(f"PCD crossing at {PCD_TARGET_V:.3f}V: t = {t_cross:.9e} s")
    else:
        print(f"PCD never crossed {PCD_TARGET_V:.3f}V; using nearest level time: t = {t_cross:.9e} s (PCD={pcd_pick_v:.6f} V)")
    print(f"NEW t0 = crossing - {T0_SHIFT_NS:.1f}ns -> {new_t0_abs:.9e} s")
    print(f"NEGATE auto = {NEGATE}")
    print(f"\nShot/family: shot={shot_id}, family={family}, model={model_name}")
    if fit_mode == "smooth":
        print(f"Fit waveform smoothing: mode=savgol, pre-noise={pre_noise_v:.4f} V, window={fit_smooth_ns:.2f} ns")

    # Onset
    onset_idx, thr, sigma = find_onset_first_drop(t_rel_ns, v_diode, t_diode)
    onset_mode = "first_drop"
    if shot_id in ONSET_ZERO_DROP_SHOTS:
        o_idx, o_thr, o_sig = find_onset_zero_drop_strict(t_rel_ns, v_diode, t_diode, shot_id=shot_id)
        onset_idx, thr, sigma = int(o_idx), float(o_thr), float(o_sig)
        onset_mode = "zero_drop_strict"
    if shot_id in ONSET_FLUCT_START_WINDOWS_NS_BY_SHOT:
        w0, w1 = ONSET_FLUCT_START_WINDOWS_NS_BY_SHOT[shot_id]
        m = (t_rel_ns >= float(w0)) & (t_rel_ns <= float(w1))
        idxs = np.where(m)[0]
        if len(idxs) > 8:
            vs_loc, _ = smooth_by_ns(t_diode, v_diode, 8.0)
            pre = (t_rel_ns >= -ONSET_BASELINE_NS) & (t_rel_ns <= 0.0)
            b0 = float(np.median(vs_loc[pre])) if np.any(pre) else float(np.median(vs_loc[:max(10, idxs[0])]))
            n0 = float(np.std(vs_loc[pre]) + 1e-12) if np.any(pre) else float(np.std(vs_loc[idxs[:max(5, len(idxs)//4)]]) + 1e-12)
            thr_v = max(4.0*n0, 0.03)
            pick = None
            for ii in idxs:
                jj = min(ii + 3, len(vs_loc) - 1)
                if np.max(np.abs(vs_loc[ii:jj+1] - b0)) >= thr_v:
                    pick = ii
                    break
            if pick is None:
                pick = int(idxs[0])
            onset_idx = int(pick)
            onset_mode = "fluct_start_window"
    if shot_id in START_AT_PCD_CROSS_SHOTS:
        onset_idx = int(np.argmin(np.abs(t_diode - t_cross)))
        onset_mode = "pcd_0p5_cross"
    if shot_id in ONSET_PREPEAK_WINDOW_NS_BY_SHOT:
        w0_pk, w1_pk = ONSET_PREPEAK_WINDOW_NS_BY_SHOT[shot_id]
        m_pk = (t_rel_ns >= float(w0_pk)) & (t_rel_ns <= float(w1_pk))
        idxs_pk = np.where(m_pk)[0]
        if len(idxs_pk) > 0:
            j_pk = int(np.argmax(v_diode[idxs_pk]))
            onset_idx = int(idxs_pk[j_pk])
            onset_mode = "prepeak_window_max"
    if shot_id in ONSET_FORCE_T0_SHOTS:
        onset_idx = int(np.argmin(np.abs(t_rel_ns - 0.0)))
        onset_mode = "forced_t0"
    if shot_id in ONSET_SHIFT_NS_BY_SHOT:
        t_target = float(t_rel_ns[onset_idx]) + float(ONSET_SHIFT_NS_BY_SHOT[shot_id])
        sidx = int(np.searchsorted(t_rel_ns, t_target))
        onset_idx = int(np.clip(sidx, 0, len(t_rel_ns) - 1))
        onset_mode = f"{onset_mode}_shifted"
    onset_t_ns = float(t_rel_ns[onset_idx])

    # Peak
    peak_smooth_ns = PEAK_SMOOTH_SIGMOID_NS if use_three_exp else 0.0
    peak_smooth_ns = float(PEAK_SMOOTH_NS_BY_SHOT.get(shot_id, peak_smooth_ns))
    peak_idx = pick_peak_idx(t_rel_ns, v_diode, onset_idx, negate=NEGATE, t_abs_s=t_diode, smooth_ns=peak_smooth_ns)
    if shot_id in ANALYSIS_ABS_WINDOW_NS_BY_SHOT:
        w0_abs, w1_abs = ANALYSIS_ABS_WINDOW_NS_BY_SHOT[shot_id]
        m_abs = (t_rel_ns >= float(w0_abs)) & (t_rel_ns <= float(w1_abs)) & (np.arange(len(v_diode)) >= onset_idx)
        idxs = np.where(m_abs)[0]
        if len(idxs) >= 5:
            vv = v_diode[idxs]
            j = int(np.argmin(vv)) if NEGATE else int(np.argmax(vv))
            peak_idx = int(idxs[j])
    if shot_id in PEAK_ABS_WINDOW_NS_BY_SHOT:
        w0, w1 = PEAK_ABS_WINDOW_NS_BY_SHOT[shot_id]
        m_pk = (t_rel_ns >= float(w0)) & (t_rel_ns <= float(w1)) & (np.arange(len(v_diode)) >= onset_idx)
        idxs = np.where(m_pk)[0]
        if len(idxs) >= 5:
            if shot_id == 27297:
                j = int(np.argmax(v_diode[idxs]))
            else:
                j = int(np.argmax(np.abs(v_diode[idxs])))
            peak_idx = int(idxs[j])
    peak_t_ns = float(t_rel_ns[peak_idx])
    peak_v = float(v_diode[peak_idx])

    print("\nOnset selection:")
    print(f"  onset_mode = {onset_mode}")
    print(f"  onset_t = {onset_t_ns:.3f} ns | V_onset = {v_diode[onset_idx]:.6f} V")
    print(f"  onset dV/dt baseline sigma={sigma:.3e}, thr={thr:.3e} (V/s)")

    print("\nPeak selection:")
    print(f"  peak_t  = {peak_t_ns:.3f} ns | V_peak = {peak_v:.6f} V")
    if use_three_exp and peak_smooth_ns > 0:
        print(f"  peak selected on smoothed waveform ({PEAK_SMOOTH_SIGMOID_NS:.1f} ns window)")

    if (not use_three_exp) and (shot_id in SPLIT1_FORCE_ABS_MAX_SHOTS):
        i0 = max(onset_idx + 2, 0)
        i1 = min(len(v_diode) - 3, int(np.searchsorted(t_rel_ns, float(STOP_ABS_MIN_BEFORE_NS_BY_SHOT.get(shot_id, 240.0)))))
        if i1 > i0 + 2:
            j = int(np.argmax(v_diode[i0:i1+1]))
            peak_idx = int(i0 + j)
            peak_t_ns = float(t_rel_ns[peak_idx])
            peak_v = float(v_diode[peak_idx])

    # Stop: strict next-period start first, then legacy fallbacks
    stop_idx_np, stop_thr, stop_mode = stop_by_new_period_start(
        t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx
    )
    if stop_mode == "period_change":
        stop_idx = stop_idx_np
        ref = None
        tol = None
    else:
        # For 2-exp families, prefer explicit opposite-direction reversal stop.
        if not use_three_exp:
            stop_idx_opp, opp_thr, opp_mode = stop_by_opposite_slope_change(
                t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx
            )
            if stop_idx_opp is not None:
                stop_idx = int(stop_idx_opp)
                stop_mode = opp_mode
                ref = None
                tol = None
            else:
                stop_idx_tp, s0, stop_mode_tp = stop_by_next_turning_point(
                    t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx
                )
                if stop_mode_tp == "turning_point":
                    stop_idx = stop_idx_tp
                    stop_mode = stop_mode_tp
                    ref = None
                    tol = None
                else:
                    stop_idx_ar, ar_thr, ar_mode = stop_by_recovery_apex_reversal(
                        t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx
                    )
                    if stop_idx_ar is not None:
                        stop_idx = int(stop_idx_ar)
                        stop_mode = ar_mode
                        ref = None
                        tol = None
                    else:
                        stop_idx, ref, tol, stop_mode2 = stop_by_slope_threshold(t_rel_ns, v_diode, peak_idx)
                        stop_mode = stop_mode2
        else:
            stop_idx_tp, s0, stop_mode_tp = stop_by_next_turning_point(
                t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx
            )
            if stop_mode_tp == "turning_point":
                stop_idx = stop_idx_tp
                stop_mode = stop_mode_tp
                ref = None
                tol = None
            else:
                stop_idx, ref, tol, stop_mode2 = stop_by_slope_threshold(t_rel_ns, v_diode, peak_idx)
                stop_mode = stop_mode2

    # For advanced (3-exp) family only, optionally prefer flattening cutoff.
    # For 2-exp families, keep stop tied to opposite-direction turning/new-period logic.
    if use_three_exp:
        stop_idx_flat, flat_thr, flat_mode = stop_by_recovery_flattening(
            t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx, min_recovery_frac=0.75
        )
        if stop_idx_flat is not None and (peak_idx + 5) < stop_idx_flat < stop_idx:
            stop_idx = stop_idx_flat
            stop_mode = flat_mode

    # Positive-peak SMAJ override: stop at first-cycle valley minimum.
    if (family == "SMAJ400A") and (not use_three_exp) and (peak_v > 0.0):
        stop_idx_val, stop_mode_val = stop_by_first_valley_minimum(t_rel_ns, v_diode, peak_idx)
        if (stop_idx_val is not None) and (peak_idx + 5 < stop_idx_val < len(v_diode)):
            stop_idx = int(stop_idx_val)
            stop_mode = stop_mode_val

    # Shot-specific stop override (e.g., 27283): next local peak window.
    if (not use_three_exp) and (shot_id in STOP_NEXT_PEAK_SHOT_WINDOWS_NS):
        w0, w1 = STOP_NEXT_PEAK_SHOT_WINDOWS_NS[shot_id]
        stop_idx_pk, stop_mode_pk = stop_by_next_peak_window(t_rel_ns, v_diode, peak_idx, w0, w1)
        if (stop_idx_pk is not None) and (peak_idx + 5 < stop_idx_pk < len(v_diode)):
            stop_idx = int(stop_idx_pk)
            stop_mode = stop_mode_pk

    stop_t_ns = float(t_rel_ns[stop_idx])
    print("\nStop selection:")
    print(f"  stop_mode = {stop_mode}")
    print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")

    # Ensure ordering
    if not (onset_idx < peak_idx < stop_idx):
        stop_idx = max(stop_idx, peak_idx + 10)
        stop_t_ns = float(t_rel_ns[stop_idx])

    # =============================
    # split1: start of second function
    # - CSD path: force at peak (user-requested)
    # - non-CSD path: keep robust turn detector
    # =============================
    if use_three_exp:
        split1_idx = int(peak_idx)
        split1_sigma = 0.0
        split1_mode = "peak_anchor"
    else:
        if shot_id in SPLIT1_SHOT_MIN_WINDOW_NS:
            w0, w1 = SPLIT1_SHOT_MIN_WINDOW_NS[shot_id]
            s_idx, s_mode = split1_by_min_window(t_rel_ns, v_fit, peak_idx, w0, w1, smooth_ns=max(6.0, fit_smooth_ns))
            if s_idx is not None:
                split1_idx = int(s_idx)
                split1_sigma = 0.0
                split1_mode = s_mode
            else:
                split1_idx = int(peak_idx)
                split1_sigma = 0.0
                split1_mode = "split1_window_fallback_peak"
        elif shot_id in SPLIT1_FORCE_ABS_MAX_SHOTS:
            i0 = max(onset_idx + 2, 0)
            i1 = min(len(v_diode) - 3, int(np.searchsorted(t_rel_ns, float(STOP_ABS_MIN_BEFORE_NS_BY_SHOT.get(shot_id, 240.0)))))
            if i1 > i0 + 2:
                j = int(np.argmax(v_diode[i0:i1+1]))
                split1_idx = int(i0 + j)
                split1_sigma = 0.0
                split1_mode = "split1_forced_abs_max"
            else:
                split1_idx = int(peak_idx)
                split1_sigma = 0.0
                split1_mode = "split1_absmax_fallback_peak"
        elif shot_id in SPLIT1_TARGET_V_BY_SHOT:
            w0, w1 = SPLIT1_TARGET_WINDOW_NS_BY_SHOT.get(shot_id, (float(t_rel_ns[onset_idx]), float(t_rel_ns[stop_idx])))
            m = (t_rel_ns >= float(w0)) & (t_rel_ns <= float(w1))
            idxs = np.where(m)[0]
            if len(idxs) > 0:
                vt = float(SPLIT1_TARGET_V_BY_SHOT[shot_id])
                j = int(np.argmin(np.abs(v_diode[idxs] - vt)))
                split1_idx = int(idxs[j])
                split1_sigma = 0.0
                split1_mode = "split1_target_v_window"
            else:
                split1_idx = int(peak_idx)
                split1_sigma = 0.0
                split1_mode = "split1_target_v_fallback_peak"
        elif shot_id in SPLIT1_POS_TO_NEG_SLOPE_SHOTS:
            split1_idx, split1_sigma, split1_mode = split1_by_pos_to_neg_slope(
                t_rel_ns, v_fit, onset_idx, peak_idx, stop_idx, smooth_ns=max(8.0, fit_smooth_ns)
            )
        elif shot_id in SPLIT1_FORCE_AT_PEAK_SHOTS:
            split1_idx = int(peak_idx)
            split1_sigma = 0.0
            split1_mode = "split1_forced_peak"
        elif family == "SMAJ400A":
            peak_anchor, peak_anchor_sigma = should_anchor_split1_at_peak(
                t_rel_ns, v_fit, onset_idx, peak_idx, stop_idx
            )
            stop_span_ns = float(t_rel_ns[stop_idx] - t_rel_ns[peak_idx])
            anchor_allowed = (not str(stop_mode).endswith("fallback_cap")) and (stop_span_ns <= SPLIT1_PEAK_ANCHOR_MAX_SPAN_NS)
            if shot_id in SPLIT1_DISABLE_PEAK_ANCHOR_SHOTS:
                anchor_allowed = False
            if peak_anchor and anchor_allowed:
                split1_idx = int(peak_idx)
                split1_sigma = float(peak_anchor_sigma)
                split1_mode = "split1_peak_reversal_anchor"
            else:
                split1_idx, split1_sigma, split1_mode = find_split1_zero_to_rise(
                    t_rel_ns, v_fit, onset_idx, peak_idx, stop_idx
                )
                if (split1_mode == "split1_zero_rise_fallback_near_peak") and (shot_id not in SPLIT1_SKIP_STRICT_FALLBACK_SHOTS):
                    strict_mult = SPLIT1_STRICTER_MULT if (shot_id in SPLIT1_STRICTER_SHOTS) else 1.0
                    strict_min_ns = SPLIT1_STRICTER_MIN_NS_BY_SHOT.get(
                        shot_id, SPLIT1_STRICTER_MIN_NS_AFTER_PEAK if (shot_id in SPLIT1_STRICTER_SHOTS) else None
                    )
                    s_idx, s_sig, s_mode = find_split1_strict_inflection(
                        t_rel_ns, v_fit, onset_idx, peak_idx, stop_idx,
                        strict_mult=strict_mult, min_ns_after_peak=strict_min_ns
                    )
                    if s_idx is not None:
                        split1_idx, split1_sigma, split1_mode = int(s_idx), float(s_sig), s_mode
        else:
            split1_idx, split1_sigma, split1_mode = find_split1_opposite_turn(
                t_rel_ns, v_fit, onset_idx, peak_idx, stop_idx
            )
    if not (onset_idx < split1_idx < stop_idx - 2):
        split1_idx = max(onset_idx + 8, min(stop_idx - 6, peak_idx + 8))
    if shot_id in SPLIT1_FORCE_ABS_MIN_SHOTS:
        i0 = max(onset_idx + 2, 0)
        i1 = min(stop_idx - 2, len(v_diode) - 1)
        if i1 > i0 + 2:
            j = int(np.argmin(v_diode[i0:i1+1]))
            split1_idx = int(i0 + j)
            split1_mode = "split1_forced_abs_min"
    if shot_id in SPLIT1_SHIFT_LEFT_NS_BY_SHOT:
        t_target = float(t_rel_ns[split1_idx]) - float(SPLIT1_SHIFT_LEFT_NS_BY_SHOT[shot_id])
        sidx = int(np.searchsorted(t_rel_ns, t_target))
        split1_idx = int(np.clip(sidx, onset_idx + 2, stop_idx - 3))
        split1_mode = f"{split1_mode}_shift_left"
    if shot_id in SIGMOID_ONLY_FORCE_SPLIT1_NEAR_ONSET_SHOTS:
        split1_idx = int(np.clip(onset_idx + 2, onset_idx + 2, stop_idx - 3))
        split1_mode = "split1_forced_near_onset_sigmoid_only"
    split1_t_ns = float(t_rel_ns[split1_idx])
    print("\nSplit1 selection (exp1->exp2):")
    print(f"  split1_mode = {split1_mode}")
    print(f"  split1_t    = {split1_t_ns:.3f} ns | V_split1 = {v_diode[split1_idx]:.6f} V")
    print(f"  split1 dV/dt baseline sigma={split1_sigma:.3e} (V/s)")

    # Shot-specific stop override after split is finalized:
    # stop at absolute max after split in a bounded window.
    if (not use_three_exp) and (shot_id in STOP_AFTER_SPLIT_LOCAL_MAX_WINDOWS_NS):
        if shot_id in STOP_AFTER_SPLIT_RAW_MAX_WINDOWS_NS:
            w0, w1 = STOP_AFTER_SPLIT_RAW_MAX_WINDOWS_NS[shot_id]
            sidx, smode = stop_by_raw_max_after_split_window(
                t_rel_ns, v_diode, split1_idx, w0, w1
            )
        else:
            w0, w1 = STOP_AFTER_SPLIT_LOCAL_MAX_WINDOWS_NS[shot_id]
            sidx, smode = stop_by_local_max_after_split_window(
                t_rel_ns, v_diode, split1_idx, w0, w1, smooth_ns=max(6.0, fit_smooth_ns)
            )
        if (sidx is not None) and (split1_idx + 4 < sidx < len(v_diode)):
            stop_idx = int(sidx)
            stop_mode = smode
            stop_t_ns = float(t_rel_ns[stop_idx])
            print("\nStop adjusted after split:")
            print(f"  stop_mode = {stop_mode}")
            print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")
    if (not use_three_exp) and (shot_id in STOP_AFTER_SPLIT_GLOBAL_MIN_SHOTS):
        w0, w1 = STOP_AFTER_SPLIT_GLOBAL_MIN_WINDOWS_NS.get(shot_id, (10.0, 300.0))
        t0 = float(t_rel_ns[split1_idx]) + float(w0)
        t1 = float(t_rel_ns[split1_idx]) + float(w1)
        if shot_id in ANALYSIS_ABS_WINDOW_NS_BY_SHOT:
            _, abs_hi = ANALYSIS_ABS_WINDOW_NS_BY_SHOT[shot_id]
            t1 = min(t1, float(abs_hi))
        i0 = int(np.searchsorted(t_rel_ns, t0))
        i1 = int(np.searchsorted(t_rel_ns, t1))
        i0 = max(i0, split1_idx + 2)
        i1 = min(i1, len(v_diode) - 1)
        if i1 > i0 + 2:
            j = int(np.argmin(v_diode[i0:i1+1]))
            sidx = i0 + j
            if sidx > split1_idx + 2:
                stop_idx = int(sidx)
                stop_mode = "stop_after_split_window_min"
                stop_t_ns = float(t_rel_ns[stop_idx])
                print("\nStop adjusted after split:")
                print(f"  stop_mode = {stop_mode}")
                print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")
    if (not use_three_exp) and (shot_id in STOP_AFTER_SPLIT_ABS_MAX_WINDOWS_NS):
        w0_abs, w1_abs = STOP_AFTER_SPLIT_ABS_MAX_WINDOWS_NS[shot_id]
        m = (t_rel_ns >= float(w0_abs)) & (t_rel_ns <= float(w1_abs))
        idxs = np.where(m)[0]
        if len(idxs) > 0:
            idxs = idxs[idxs > split1_idx]
        if len(idxs) > 0:
            j = int(np.argmax(v_diode[idxs]))
            sidx = int(idxs[j])
            stop_idx = sidx
            stop_mode = "stop_after_split_abs_window_max"
            stop_t_ns = float(t_rel_ns[stop_idx])
            print("\nStop adjusted after split:")
            print(f"  stop_mode = {stop_mode}")
            print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")
    if (not use_three_exp) and (shot_id in STOP_AFTER_SPLIT_ABS_MIN_WINDOWS_NS):
        w0_abs, w1_abs = STOP_AFTER_SPLIT_ABS_MIN_WINDOWS_NS[shot_id]
        m = (t_rel_ns >= float(w0_abs)) & (t_rel_ns <= float(w1_abs))
        idxs = np.where(m)[0]
        if len(idxs) > 0:
            idxs = idxs[idxs > split1_idx]
        if len(idxs) > 0:
            j = int(np.argmin(v_diode[idxs]))
            stop_idx = int(idxs[j])
            stop_mode = "stop_after_split_abs_window_min"
            stop_t_ns = float(t_rel_ns[stop_idx])
            print("\nStop adjusted after split:")
            print(f"  stop_mode = {stop_mode}")
            print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")
    if (not use_three_exp) and (shot_id in STOP_ABS_MIN_BEFORE_NS_BY_SHOT):
        t_cap = float(STOP_ABS_MIN_BEFORE_NS_BY_SHOT[shot_id])
        i0 = max(split1_idx + 2, 0)
        i1 = int(np.searchsorted(t_rel_ns, t_cap))
        i1 = min(i1, len(v_diode) - 1)
        if i1 > i0 + 2:
            j = int(np.argmin(v_diode[i0:i1+1]))
            stop_idx = int(i0 + j)
            stop_mode = "stop_abs_min_before_cap"
            stop_t_ns = float(t_rel_ns[stop_idx])
            print("\nStop adjusted after split:")
            print(f"  stop_mode = {stop_mode}")
            print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")
    if (not use_three_exp) and (shot_id in STOP_FIRST_ZERO_AFTER_NS_BY_SHOT):
        t0z = float(STOP_FIRST_ZERO_AFTER_NS_BY_SHOT[shot_id])
        i0 = max(split1_idx + 2, int(np.searchsorted(t_rel_ns, t0z)))
        if i0 < len(v_diode) - 2:
            z_smooth_ns = float(STOP_FIRST_ZERO_SMOOTH_NS_BY_SHOT.get(shot_id, 8.0))
            vz, _ = smooth_by_ns(t_diode, v_diode, z_smooth_ns)
            if vz[i0] <= 0.0:
                hits = np.where(vz[i0:] >= 0.0)[0]
            else:
                hits = np.where(vz[i0:] <= 0.0)[0]
            if len(hits) > 0:
                stop_idx = int(i0 + int(hits[0]))
                stop_mode = "stop_first_zero_after"
            else:
                j = int(np.argmin(np.abs(vz[i0:])))
                stop_idx = int(i0 + j)
                stop_mode = "stop_nearest_zero_after"
            stop_t_ns = float(t_rel_ns[stop_idx])
            print("\nStop adjusted after split:")
            print(f"  stop_mode = {stop_mode}")
            print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")
    if (not use_three_exp) and (shot_id in STOP_FLAT_2EXP_SHOTS):
        stop_idx_flat, flat_thr, flat_mode = stop_by_recovery_flattening(
            t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx, min_recovery_frac=0.75
        )
        if (stop_idx_flat is not None) and (stop_idx_flat > split1_idx + 6):
            stop_idx = int(stop_idx_flat)
            stop_mode = f"{flat_mode}_2exp"
            stop_t_ns = float(t_rel_ns[stop_idx])
            print("\nStop adjusted after split:")
            print(f"  stop_mode = {stop_mode}")
            print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")

    stop_mid_idx = None
    split3_idx = None
    split3_t_ns = None
    four_exp_mode = None
    if (not use_three_exp) and (shot_id in FOUR_EXP_SHOTS):
        # Segment 2 end stays at the current stop marker.
        stop_mid_idx = int(stop_idx)
        split2_idx = int(stop_mid_idx)
        split2_mode = "exp3_start_old_stop"
        split2_t_ns = float(t_rel_ns[split2_idx])
        i_tail, m_tail = find_tailoff_zero_slope_idx(
            t_rel_ns, v_fit, split2_idx, min_delay_ns=40.0, max_delay_ns=500.0,
            smooth_ns=max(10.0, fit_smooth_ns), sustain_n=10, frac_p90=0.18
        )
        if i_tail is None:
            i_tail = min(len(v_diode) - 1, split2_idx + max(24, int(round(140.0 / max(float(np.median(np.diff(t_rel_ns))), 1e-9)))))
            m_tail = "tailoff_none_fallback"
        stop_idx = int(max(split2_idx + 6, i_tail))
        stop_mode = f"four_exp_exp_{m_tail}"
        stop_t_ns = float(t_rel_ns[stop_idx])
        four_exp_mode = "exp12_plus_exp3"

        max_delay = max(20.0, float(t_rel_ns[stop_idx] - t_rel_ns[split2_idx] - 5.0))
        i_inflect, m_inflect = find_next_rel_max_by_rolling(
            t_rel_ns, v_fit, split2_idx, min_delay_ns=8.0, max_delay_ns=max_delay,
            smooth_ns=max(10.0, fit_smooth_ns), prom_frac=0.12
        )
        if i_inflect is None:
            i_inflect = min(stop_idx - 2, split2_idx + max(8, int(round(0.45 * (stop_idx - split2_idx)))))
            m_inflect = "rolling_max_none_fallback"
        split3_idx = int(np.clip(i_inflect, split2_idx + 2, stop_idx - 2))
        split3_t_ns = float(t_rel_ns[split3_idx])

        print("\nFour-exp boundaries:")
        print(f"  seg2 end / exp3 start = {split2_t_ns:.3f} ns | V = {v_diode[split2_idx]:.6f} V")
        print(f"  exp3 shape anchor (rolling max) = {split3_t_ns:.3f} ns | V = {v_diode[split3_idx]:.6f} V ({m_inflect})")
        print(f"  exp3 end (tail-off)   = {stop_t_ns:.3f} ns | V = {v_diode[stop_idx]:.6f} V ({m_tail})")

    # recovery sign for downstream segmentation
    iB1 = max(0, onset_idx - 200)
    iB2 = max(0, onset_idx - 20)
    if iB2 <= iB1:
        iB1 = max(0, peak_idx - 500)
        iB2 = max(0, peak_idx - 50)
    baseline_v = float(np.median(v_fit[iB1:iB2])) if (iB2 > iB1 + 10) else float(np.median(v_fit[:max(20, peak_idx)]))
    rec_sign = 1.0 if (baseline_v - peak_v) >= 0 else -1.0
    pre_sign = -rec_sign

    if 'split2_idx' not in locals():
        split2_idx = None
    if 'split2_mode' not in locals():
        split2_mode = "n/a"
    if 'split2_t_ns' not in locals():
        split2_t_ns = None
    if 'split3_idx' not in locals():
        split3_idx = None
    if 'split3_t_ns' not in locals():
        split3_t_ns = None
    dip_end_idx = None
    dip_mode = "n/a"
    has_dip = False

    # For CSD family, model recovery as sigmoid -> quadratic-tail.
    if use_three_exp:
        stop_idx_sig, sig_thr, sig_mode = stop_by_recovery_flattening(
            t_rel_ns, v_diode, peak_idx, onset_idx=onset_idx, min_recovery_frac=0.95
        )
        if stop_idx_sig is not None and stop_idx_sig > split1_idx + 12:
            stop_idx = int(stop_idx_sig)
            stop_mode = "gompertz_tail" if use_gompertz else "sigmoid_tail"
            stop_t_ns = float(t_rel_ns[stop_idx])
            print("\nStop adjusted for recovery tail-off:")
            print(f"  stop_mode = {stop_mode}")
            print(f"  stop_t    = {stop_t_ns:.3f} ns | V_stop = {v_diode[stop_idx]:.6f} V")

        has_dip, dip_end_idx, dip_mode = detect_post_peak_dip_end(
            t_rel_ns, t_diode, v_diode, split1_idx, rec_sign
        )
        if has_dip and not (split1_idx + 4 < dip_end_idx < stop_idx - 8):
            has_dip = False
            dip_end_idx = None
            dip_mode = "dip_rejected_order"
        if has_dip:
            print("\nDip segment detection (post-peak):")
            print(f"  dip_mode   = {dip_mode}")
            print(f"  dip_end_t  = {float(t_rel_ns[dip_end_idx]):.3f} ns | V_dip_end = {v_diode[dip_end_idx]:.6f} V")

    # =============================
    # Fit model segments
    # =============================
    t_seg_abs = t_diode[onset_idx:stop_idx+1]
    t_seg_ns  = _s_to_ns(t_seg_abs - new_t0_abs)

    # exp1: onset -> split1
    t1_abs = t_diode[onset_idx:split1_idx+1]
    v1 = v_fit[onset_idx:split1_idx+1]
    t1 = t1_abs - t1_abs[0]
    idx1 = np.linspace(0, len(t1)-1, min(SUBSAMPLE_RECOVERY_N, len(t1)), dtype=int)
    use_endpoint_constrained = (family == "SMAJ400A") and (split1_mode == "split1_peak_reversal_anchor") and (not use_three_exp)
    if use_endpoint_constrained or ((not use_three_exp) and (shot_id in EXP1_ENDPOINT_CONSTRAINED_SHOTS)):
        tau1_mult = float(EXP1_ENDPOINT_TAU_MAX_MULT_BY_SHOT.get(shot_id, 3.0))
        if (shot_id in STRETCHED_ENDPOINT_SHOTS) or (shot_id in EXP1_STRETCHED_ONLY_SHOTS):
            y1_end_target = float(v_diode[split1_idx]) if (shot_id in EXP1_ENDPOINT_RAW_TARGET_SHOTS) else None
            p1s, y1_0, y1_end, sig1 = fit_stretched_exp_endpoint_fast(
                t1[idx1], v1[idx1], seg_sign=pre_sign, tau_max_mult=tau1_mult, y_end_target=y1_end_target
            )
            tau1, k1 = map(float, p1s)
            print("\nExp1 fit (onset->split1, stretched-endpoint):")
            print(f"  tau1={tau1*1e9:.3f} ns, k1={k1:.3f}, y1_0={y1_0:.6f}, y1_end={y1_end:.6f}")
            V1_model = stretched_exp_endpoint_np(t1, tau1, k1, y1_0, y1_end, float(t1[-1]))
            b1 = np.nan
        else:
            p1, y1_0, sig1 = fit_anchored_exp_through_endpoint_fast(
                t1[idx1], v1[idx1], seg_sign=pre_sign, tau_max_mult=tau1_mult
            )
            b1, tau1 = map(float, p1)
            print("\nExp1 fit (onset->split1):")
            print(f"  baseline1={b1:.6f}, tau1={tau1*1e9:.3f} ns")
            V1_model = exp_anchor_np(t1, b1, tau1, y1_0)
    else:
        p1, y1_0, sig1 = fit_anchored_exp_fast(t1[idx1], v1[idx1], seg_sign=pre_sign)
        b1, tau1 = map(float, p1)
        print("\nExp1 fit (onset->split1):")
        print(f"  baseline1={b1:.6f}, tau1={tau1*1e9:.3f} ns")
        V1_model = exp_anchor_np(t1, b1, tau1, y1_0)

    V_model = np.empty_like(t_seg_abs, dtype=float)
    V_lo = np.full_like(V_model, np.nan, dtype=float)
    V_hi = np.full_like(V_model, np.nan, dtype=float)

    n1 = len(V1_model)
    V_model[:n1] = V1_model

    if use_three_exp:
        # second function starts at peak
        sig_start_idx = split1_idx
        if has_dip:
            td_abs = t_diode[split1_idx:dip_end_idx+1]
            vd = v_diode[split1_idx:dip_end_idx+1]
            td = td_abs - td_abs[0]
            idxd = np.linspace(0, len(td)-1, min(SUBSAMPLE_RECOVERY_N, len(td)), dtype=int)
            p_dip, y0d, sigd = fit_dip_biexp_fast(td[idxd], vd[idxd])
            Ad, tfd, tsd = map(float, p_dip)
            Vd_model = dip_biexp_np(td, Ad, tfd, tsd, y0d)

            # Enforce continuity at dip end (otherwise looks like a step at sigmoid handoff).
            if len(td) > 1 and td[-1] > 0:
                end_delta = float(vd[-1] - Vd_model[-1])
                Vd_model = Vd_model + (td / td[-1]) * end_delta
            else:
                end_delta = float(vd[-1] - Vd_model[-1])

            # Quality gates: reject degenerate dip fits that behave like a step/flat line.
            amp_eff = float(np.max(np.abs(Vd_model - y0d)))
            end_mis = float(abs(Vd_model[-1] - vd[-1]))
            lin_ref = np.linspace(vd[0], vd[-1], len(vd))
            err_dip = float(np.std(vd - Vd_model))
            err_lin = float(np.std(vd - lin_ref) + 1e-12)

            if (abs(Ad) < DIP_MIN_FIT_AMP_V) or (amp_eff < DIP_MIN_FIT_AMP_V) or (end_mis > DIP_MAX_END_MISMATCH_V) or (err_dip > 1.05 * err_lin):
                has_dip = False
                dip_mode = "dip_fit_rejected"
                dip_end_idx = None
            else:
                print("Dip fit (peak->dip_end):")
                print(f"  A_dip={Ad:.6f}, tau_fast={tfd*1e9:.3f} ns, tau_slow={tsd*1e9:.3f} ns")
                V_model[n1-1:n1-1+len(Vd_model)] = Vd_model
                sig_start_idx = dip_end_idx

        t2_abs = t_diode[sig_start_idx:stop_idx+1]
        v2 = v_diode[sig_start_idx:stop_idx+1]
        t2 = t2_abs - t2_abs[0]
        idx2 = np.linspace(0, len(t2)-1, min(SUBSAMPLE_RECOVERY_N, len(t2)), dtype=int)

        if use_gompertz:
            pg, y2_0, sigg = fit_anchored_gompertz_fast(t2[idx2], v2[idx2], seg_sign=rec_sign)
            bg, bgom, kg = map(float, pg)
            print("Gompertz fit (peak->stop):")
            print(f"  baseline={bg:.6f}, b={bgom:.4f}, k={kg*1e9:.3f} ns")
            V2_model = anchored_gompertz_np(t2, bg, bgom, kg, y2_0)
            i2 = (n1 - 1) + (sig_start_idx - split1_idx)
            V_model[i2:] = V2_model

            t2grid = np.linspace(0, float(t2[-1]), max(300, len(t2)))
            mean2 = anchored_gompertz_np(t2grid, bg, bgom, kg, y2_0)
            band2 = RECOVERY_BAND_SIGMA * sigg
            lo2 = mean2 - band2
            hi2 = mean2 + band2
            V_lo[i2:] = np.interp(t2, t2grid, lo2)
            V_hi[i2:] = np.interp(t2, t2grid, hi2)
        else:
            p2s, y2_0, sig2 = fit_anchored_sigmoid_fast(t2[idx2], v2[idx2], seg_sign=rec_sign)
            a2s, tm2s, k2s, m2s = map(float, p2s)
            print("Sigmoid fit (start->stop):")
            print(f"  amp={a2s:.6f}, t_mid={tm2s*1e9:.3f} ns, k={k2s*1e9:.3f} ns, m_tail={m2s:.3e} V/s")

            V2_full = anchored_sigmoid_np(t2, a2s, tm2s, k2s, m2s, y2_0)
            split2_local, split2_mode = find_sigmoid_mismatch_idx(t2_abs, v2, V2_full, rec_sign)
            split2_local = int(np.clip(split2_local, 3, len(t2)-3))
            split2_idx = sig_start_idx + split2_local
            split2_t_ns = float(t_rel_ns[split2_idx])
            print("\nSplit2 selection (sigmoid->exp-tail mismatch):")
            print(f"  split2_mode = {split2_mode}")
            print(f"  split2_t    = {split2_t_ns:.3f} ns | V_split2 = {v_diode[split2_idx]:.6f} V")

            V2_model = V2_full[:split2_local+1]
            i2 = (n1 - 1) + (sig_start_idx - split1_idx)
            V_model[i2:i2+len(V2_model)] = V2_model

            t3_abs = t_diode[split2_idx:stop_idx+1]
            t3 = t3_abs - t3_abs[0]
            v3_obs = v_diode[split2_idx:stop_idx+1]
            y0_t = float(V2_full[split2_local])
            dv2 = np.gradient(V2_full, t2_abs)
            s0_t = float(dv2[split2_local])
            tau_guess = max(5e-9, abs(y0_t) / max(abs(s0_t), 1e-9))
            tau_lb = 1e-9
            tau_ub = max(20e-9, 6.0*float(max(t3[-1], 1e-9)))
            try:
                popt_tau, _ = curve_fit(
                    lambda tt, tau: exp_tail_to_zero_np(tt, tau, y0_t),
                    t3, v3_obs,
                    p0=[min(max(tau_guess, tau_lb), tau_ub)],
                    bounds=([tau_lb], [tau_ub]),
                    maxfev=20000,
                )
                tau_tail = float(popt_tau[0])
            except Exception:
                tau_tail = float(min(max(tau_guess, tau_lb), tau_ub))
            print(f"Exp tail fit (split2->stop): tau_tail={tau_tail*1e9:.3f} ns")
            V3_model = exp_tail_to_zero_np(t3, tau_tail, y0_t)
            i_start3 = (n1 - 1) + (sig_start_idx - split1_idx) + split2_local
            V_model[i_start3:] = V3_model

            t2grid = np.linspace(0, float(t2[-1]), max(300, len(t2)))
            mean2 = anchored_sigmoid_np(t2grid, a2s, tm2s, k2s, m2s, y2_0)
            band2 = RECOVERY_BAND_SIGMA * sig2
            lo2 = mean2 - band2
            hi2 = mean2 + band2
            V_lo[i2:i2+len(V2_model)] = np.interp(t2[:split2_local+1], t2grid, lo2)
            V_hi[i2:i2+len(V2_model)] = np.interp(t2[:split2_local+1], t2grid, hi2)
    else:
        if use_four_exp:
            # 4-exp custom mode (shot-specific): keep exp1+exp2, then single exponential tail from old stop to tail-off.
            iA0, iA1 = onset_idx, split1_idx
            iB0, iB1 = split1_idx, split2_idx

            def _fit_stretched_seg(i0, i1, name, y_end_target=None, tau_mult=2.5):
                ta_abs = t_diode[i0:i1+1]
                ya = v_fit[i0:i1+1]
                ta = ta_abs - ta_abs[0]
                ia = np.linspace(0, len(ta)-1, min(SUBSAMPLE_RECOVERY_N, len(ta)), dtype=int)
                y_end = float(v_diode[i1]) if (y_end_target is None) else float(y_end_target)
                sgn = float(np.sign(y_end - float(ya[0])))
                sgn = None if sgn == 0 else sgn
                p, y0, yT, sig = fit_stretched_exp_endpoint_fast(
                    ta[ia], ya[ia], seg_sign=sgn, tau_max_mult=tau_mult, y_end_target=y_end
                )
                tau, kval = map(float, p)
                ym = stretched_exp_endpoint_np(ta, tau, kval, y0, yT, float(ta[-1]))
                print(f"{name}: tau={tau*1e9:.3f} ns, k={kval:.3f}, y0={y0:.6f}, y_end={yT:.6f}")
                return ym, tau, kval, y0, yT, sig

            print("Exp2 fit (4-exp custom / exponential tail):")
            V1_model, tau1, k1, y1_0, y1_end, sig1 = _fit_stretched_seg(iA0, iA1, "  seg1 onset->split1", y_end_target=float(v_diode[iA1]), tau_mult=2.5)
            V2a_model, tau2a, k2a, y2a_0, y2a_end, sig2a = _fit_stretched_seg(iB0, iB1, "  seg2 split1->split2", y_end_target=float(v_diode[iB1]), tau_mult=2.5)

            # Single exponential tail: start at old stop marker and end where slope tails off to ~0.
            tC_abs = t_diode[split2_idx:stop_idx+1]
            yC_obs = v_fit[split2_idx:stop_idx+1]
            tC = tC_abs - tC_abs[0]
            yC0 = float(v_diode[split2_idx])
            yC1 = float(v_diode[stop_idx])
            idxC = np.linspace(0, len(tC)-1, min(SUBSAMPLE_RECOVERY_N, len(tC)), dtype=int)
            pC, yC_0, yC_end, sigC = fit_stretched_exp_endpoint_fast(
                tC[idxC], yC_obs[idxC], seg_sign=np.sign(yC1 - yC0), tau_max_mult=2.5, y_end_target=yC1
            )
            tauC, kC = map(float, pC)
            VC_model = stretched_exp_endpoint_np(tC, tauC, kC, yC_0, yC_end, float(tC[-1]))
            # Blend in rolling-max anchor via weighted average to better follow smoothed shape.
            if split3_idx is not None and (split3_idx > split2_idx) and (split3_idx < stop_idx):
                t_anchor = float(t_diode[split3_idx] - t_diode[split2_idx])
                y_anchor = float(v_fit[split3_idx])
                i_anchor = int(np.argmin(np.abs(tC - t_anchor)))
                delta = y_anchor - float(VC_model[i_anchor])
                T = max(float(tC[-1]), 1e-12)
                w = (tC / T) * (1.0 - tC / T)
                w = w / max(np.max(w), 1e-12)
                VC_model = VC_model + 0.55 * delta * w
                VC_model[-1] = yC1
                VC_model[0] = yC0
            else:
                t_anchor = 0.5 * float(tC[-1])
                y_anchor = float(np.interp(t_anchor, tC, yC_obs))
            sigC = float(np.std(yC_obs - VC_model) + 1e-9)
            print(f"  seg3 exp split2->stop: tau={tauC*1e9:.3f} ns, k={kC:.3f}, y0={yC0:.6f}, y1={yC1:.6f}, anchor={t_anchor*1e9:.3f} ns")

            t_seg_abs = t_diode[onset_idx:stop_idx+1]
            t_seg_ns  = _s_to_ns(t_seg_abs - new_t0_abs)
            V_model = np.empty_like(t_seg_abs, dtype=float)
            V_lo = np.full_like(V_model, np.nan, dtype=float)
            V_hi = np.full_like(V_model, np.nan, dtype=float)

            oA = iA0 - onset_idx
            oB = iB0 - onset_idx
            oC = split2_idx - onset_idx
            V_model[oA:oA+len(V1_model)] = V1_model
            V_model[oB:oB+len(V2a_model)] = V2a_model
            V_model[oC:oC+len(VC_model)] = VC_model

            # Pin boundaries to observed points for clean visual continuity.
            for ib in (iA1, iB1, stop_idx):
                V_model[ib - onset_idx] = float(v_diode[ib])

            # Approximate fit bands by per-segment residual sigma.
            bA = RECOVERY_BAND_SIGMA * sig1
            bB = RECOVERY_BAND_SIGMA * sig2a
            bC = RECOVERY_BAND_SIGMA * sigC
            V_lo[oA:oA+len(V1_model)] = V1_model - bA
            V_hi[oA:oA+len(V1_model)] = V1_model + bA
            V_lo[oB:oB+len(V2a_model)] = V2a_model - bB
            V_hi[oB:oB+len(V2a_model)] = V2a_model + bB
            V_lo[oC:oC+len(VC_model)] = VC_model - bC
            V_hi[oC:oC+len(VC_model)] = VC_model + bC

            exp2_mode = "four_stage_exp_tail"
        elif shot_id in EXP2_SIGMOID_ONLY_SHOTS:
            # Shot-specific 2nd segment: anchored sigmoid from split1 directly to stop.
            t2_abs = t_diode[split1_idx:stop_idx+1]
            if shot_id in EXP2_SIGMOID_FIT_RAW_SHOTS:
                v2 = v_diode[split1_idx:stop_idx+1]
            else:
                v2 = v_fit[split1_idx:stop_idx+1]
            t2 = t2_abs - t2_abs[0]
            if shot_id in EXP2_SIGMOID_USE_ALL_POINTS_SHOTS:
                idx2 = np.arange(len(t2), dtype=int)
            else:
                idx2 = np.linspace(0, len(t2)-1, min(SUBSAMPLE_RECOVERY_N, len(t2)), dtype=int)
            p2s, y2_0, sig2 = fit_anchored_sigmoid_fast(t2[idx2], v2[idx2], seg_sign=rec_sign)
            a2s, tm2s, k2s, m2s = map(float, p2s)
            if shot_id in EXP2_SIGMOID_TMID_SHIFT_NS_BY_SHOT:
                tm_shift = _ns_to_s(float(EXP2_SIGMOID_TMID_SHIFT_NS_BY_SHOT[shot_id]))
                tm2s = float(np.clip(tm2s + tm_shift, 0.0, max(float(t2[-1]) * 0.98, 1e-12)))
            print("Exp2 fit (split1->stop, anchored-sigmoid):")
            print(f"  amp2={a2s:.6f}, tmid2={tm2s*1e9:.3f} ns, k2={k2s*1e9:.3f} ns, m2={m2s:.6e}")
            V2_model = anchored_sigmoid_np(t2, a2s, tm2s, k2s, m2s, y2_0)
            V_model[n1-1:] = V2_model
            exp2_mode = "sigmoid_only"

            t2grid = np.linspace(0, float(t2[-1]), max(300, len(t2)))
            mean2 = anchored_sigmoid_np(t2grid, a2s, tm2s, k2s, m2s, y2_0)
            band2 = RECOVERY_BAND_SIGMA * sig2
            lo2 = mean2 - band2
            hi2 = mean2 + band2
            V_lo[n1-1:] = np.interp(t2, t2grid, lo2)
            V_hi[n1-1:] = np.interp(t2, t2grid, hi2)
        # exp2: split1 -> stop
        elif shot_id in EXP2_LINEAR_THEN_EXP_SHOTS:
            split2_target_ns = float(EXP2_LINEAR_SPLIT_ABS_NS_BY_SHOT.get(shot_id, 240.0))
            split2_target_v = EXP2_LINEAR_END_V_BY_SHOT.get(shot_id, None)
            split2_target_delay_ns = float(EXP2_LINEAR_END_DELAY_NS_BY_SHOT.get(shot_id, 0.0))
            split2_target_pick = str(EXP2_LINEAR_END_V_PICK_BY_SHOT.get(shot_id, "first_cross"))
            split2_target_tol = float(EXP2_LINEAR_END_V_TOL_BY_SHOT.get(shot_id, 0.02))
            split2_target_near_backoff = int(EXP2_LINEAR_END_V_NEAR_BACKOFF_BY_SHOT.get(shot_id, 0))
            exp2b_tau_max_mult = float(EXP2_LINEAR_EXP_TAU_MAX_MULT_BY_SHOT.get(shot_id, 3.0))
            min_after_ns = float(EXP2_LINEAR_MIN_NS_AFTER_SPLIT1_BY_SHOT.get(shot_id, 40.0))
            slope_mult = float(EXP2_LINEAR_STRICT_SLOPE_MULT_BY_SHOT.get(shot_id, 3.0))
            sustain_n = int(EXP2_LINEAR_STRICT_SUSTAIN_N_BY_SHOT.get(shot_id, 10))
            det_smooth_ns = float(EXP2_LINEAR_DETECT_SMOOTH_NS_BY_SHOT.get(shot_id, max(10.0, fit_smooth_ns)))

            # Detect end-of-linear region: first sustained departure from initial linear trend
            # plus sustained slope increase.
            vs_det, _ = smooth_by_ns(t_diode, v_fit, det_smooth_ns)
            dv = np.gradient(vs_det, t_diode)
            i_start = int(np.searchsorted(t_rel_ns, float(t_rel_ns[split1_idx]) + min_after_ns))
            i_target = int(np.argmin(np.abs(t_rel_ns - split2_target_ns)))
            i_start = max(i_start, i_target)
            i_start = max(i_start, split1_idx + 4)
            i_end = max(i_start + sustain_n + 2, stop_idx - 4)
            i_end = min(i_end, stop_idx - 4)

            # Reference slope from early linear section after split1.
            ref_a = split1_idx + 2
            ref_b = min(i_start, split1_idx + max(10, sustain_n * 2))
            if ref_b <= ref_a + 3:
                ref_b = min(stop_idx - 6, split1_idx + 12)
            ref_mag = float(np.median(np.abs(dv[ref_a:ref_b])) + 1e-18) if ref_b > ref_a else float(np.median(np.abs(dv[split1_idx+1:split1_idx+8])) + 1e-18)
            thr = slope_mult * ref_mag
            t0_ref = float(t_diode[split1_idx])
            y0_ref = float(vs_det[split1_idx])
            if ref_b > ref_a:
                m0 = float(np.median(dv[ref_a:ref_b]))
            else:
                m0 = float(np.median(dv[split1_idx+1:split1_idx+8]))
            y_line = y0_ref + m0 * (t_diode - t0_ref)
            res = np.abs(vs_det - y_line)
            ref_res = float(np.std(res[ref_a:ref_b]) + 1e-18) if ref_b > ref_a else float(np.std(res[split1_idx+1:split1_idx+8]) + 1e-18)
            amp_local = float(np.std(vs_det[ref_a:i_end]) + 1e-18) if i_end > ref_a + 5 else ref_res
            res_thr = max(4.0 * ref_res, 0.18 * amp_local, 0.015)
            d2 = np.gradient(dv, t_diode)
            d2_scan = d2[i_start:i_end] if i_end > i_start + 5 else d2[i_start:]
            d2_pos = d2_scan[d2_scan > 0]
            d2_thr = float(0.55 * np.percentile(d2_pos, 95)) if len(d2_pos) > 8 else float(np.max(d2_scan) * 0.65 if len(d2_scan) > 0 else 0.0)
            dv_rise_thr = max(1.6 * ref_mag, 1e-18)

            split2_idx = None
            if shot_id in EXP2_LINEAR_FORCE_ABS_TARGET_SHOTS:
                split2_idx = int(np.argmin(np.abs(t_rel_ns - split2_target_ns)))
                split2_idx = int(np.clip(split2_idx, split1_idx + 4, stop_idx - 4))
                split2_mode = "exp2_linear_handoff_forced_abs_target"
            if (split2_idx is None) and (split2_target_v is not None):
                # Primary shot-specific handoff: end linear segment at target voltage
                # crossing on the RAW diode waveform.
                i_v0 = max(split1_idx + 2, 1)
                i_v1 = max(i_v0 + 1, stop_idx - 3)
                y_raw = v_diode[i_v0:i_v1+1]
                # Use sign consistent with the branch start (e.g., -1.25 V for negative waveform).
                target_signed = float(np.copysign(abs(float(split2_target_v)), float(v_diode[split1_idx])))
                d = y_raw - target_signed
                if len(d) >= 2:
                    if split2_target_pick == "last_near":
                        near = np.where(np.abs(d) <= split2_target_tol)[0]
                        if len(near) > 0:
                            k = max(0, len(near) - 1 - max(0, split2_target_near_backoff))
                            split2_idx = int(i_v0 + near[k])
                            split2_mode = "exp2_linear_handoff_target_v_raw_last_near"
                        else:
                            zc = np.where((d[:-1] == 0.0) | (d[:-1] * d[1:] <= 0.0))[0]
                            if len(zc) > 0:
                                split2_idx = int(i_v0 + zc[-1] + 1)
                                split2_mode = "exp2_linear_handoff_target_v_raw_last_cross"
                    else:
                        zc = np.where((d[:-1] == 0.0) | (d[:-1] * d[1:] <= 0.0))[0]
                        if len(zc) > 0:
                            split2_idx = int(i_v0 + zc[0] + 1)
                            split2_mode = "exp2_linear_handoff_target_v_raw_first_cross"
                    if split2_idx is None:
                        # Fallback in the same raw segment: pick nearest target point.
                        jn = int(np.argmin(np.abs(d)))
                        split2_idx = int(i_v0 + jn)
                        split2_mode = "exp2_linear_handoff_target_v_raw_nearest"
                if (split2_idx is not None) and (split2_target_delay_ns != 0.0):
                    t_late = float(t_rel_ns[split2_idx]) + split2_target_delay_ns
                    split2_idx = int(np.searchsorted(t_rel_ns, t_late))
                    split2_idx = int(np.clip(split2_idx, split1_idx + 4, stop_idx - 4))
                    split2_mode = f"{split2_mode}_delay"

            # Optional strict handoff window (shot-specific): choose inflection (max d2) inside window.
            if (split2_idx is None) and (shot_id in EXP2_LINEAR_HANDOFF_WINDOW_NS_BY_SHOT):
                w0_ns, w1_ns = EXP2_LINEAR_HANDOFF_WINDOW_NS_BY_SHOT[shot_id]
                m_hw = (t_rel_ns >= float(w0_ns)) & (t_rel_ns <= float(w1_ns))
                idx_hw = np.where(m_hw)[0]
                idx_hw = idx_hw[(idx_hw > split1_idx + 3) & (idx_hw < stop_idx - 3)]
                if len(idx_hw) > 0:
                    j_hw = int(np.argmax(d2[idx_hw]))
                    split2_idx = int(idx_hw[j_hw])
                    split2_mode = "exp2_linear_handoff_window_inflection"

            if split2_idx is None:
                for i in range(i_start, i_end - sustain_n):
                    seg = dv[i:i+sustain_n]
                    rseg = res[i:i+sustain_n]
                    # strict: sustained higher slope + sustained deviation from initial linear trend.
                    cond_slope = np.all(np.abs(seg) >= thr) and np.all(rseg >= res_thr)
                    d2seg = d2[i:i+sustain_n]
                    cond_knee = (np.median(d2seg) >= d2_thr) and (np.median(np.abs(seg)) >= dv_rise_thr)
                    if cond_slope or cond_knee:
                        split2_idx = int(i)
                        split2_mode = "exp2_linear_handoff_strict_slope_rise"
                        break

            if split2_idx is None:
                i_fb0 = max(i_start, split1_idx + 6)
                i_fb1 = max(i_fb0 + 3, stop_idx - 6)
                i_fb1 = min(i_fb1, len(d2) - 2)
                if i_fb1 > i_fb0 + 2:
                    j_acc = int(np.argmax(d2[i_fb0:i_fb1+1]))
                    split2_idx = int(i_fb0 + j_acc)
                    split2_mode = "exp2_linear_handoff_fallback_accel_knee"
                else:
                    split2_idx = int(np.argmin(np.abs(t_rel_ns - split2_target_ns)))
                    split2_mode = "exp2_linear_handoff_fallback_abs_target"
                split2_idx = int(np.clip(split2_idx, split1_idx + 4, stop_idx - 4))
            else:
                split2_idx = int(np.clip(split2_idx, split1_idx + 4, stop_idx - 4))
            split2_t_ns = float(t_rel_ns[split2_idx])
            print("\nSplit2 selection (linear->exp):")
            print(f"  split2_mode = {split2_mode}")
            print(f"  split2_t    = {split2_t_ns:.5f} ns | V_split2 = {v_diode[split2_idx]:.6f} V")
            if split2_target_v is not None:
                print(f"  split2_target_v = {float(split2_target_v):.6f} V")
                print(f"  split2_target_pick = {split2_target_pick}, tol={split2_target_tol:.4f} V")
                if split2_target_pick == "last_near":
                    print(f"  split2_target_near_backoff = {split2_target_near_backoff}")
            if split2_target_delay_ns != 0.0:
                print(f"  split2_target_delay = {split2_target_delay_ns:.3f} ns")

            # Segment 2a: linear from split1 to fixed handoff (~240 ns).
            t2a_abs = t_diode[split1_idx:split2_idx+1]
            t2a = t2a_abs - t2a_abs[0]
            y2a0 = float(v_diode[split1_idx])
            if shot_id in EXP2_LINEAR_END_USE_RAW_SHOTS:
                y2a1 = float(v_diode[split2_idx])
            else:
                y2a1 = float(v_fit[split2_idx])
            T2a = max(float(t2a[-1]), 1e-12)
            V2a_model = y2a0 + (y2a1 - y2a0) * (t2a / T2a)
            sig2a = float(np.std(v_diode[split1_idx:split2_idx+1] - V2a_model) + 1e-9)
            print("Exp2a fit (split1->split2, linear):")
            print(f"  y2a_0={y2a0:.6f}, y2a_1={y2a1:.6f}")

            # Segment 2b: final exponential from handoff to zero-reaching stop.
            t2b_abs = t_diode[split2_idx:stop_idx+1]
            if shot_id in EXP2_LINEAR_EXP_FIT_RAW_SHOTS:
                v2b = v_diode[split2_idx:stop_idx+1]
            else:
                v2b = v_fit[split2_idx:stop_idx+1]
            t2b = t2b_abs - t2b_abs[0]
            if shot_id in EXP2_LINEAR_EXP_USE_ALL_POINTS_SHOTS:
                idx2b = np.arange(len(t2b), dtype=int)
            else:
                idx2b = np.linspace(0, len(t2b)-1, min(SUBSAMPLE_RECOVERY_N, len(t2b)), dtype=int)
            p2b, y2_0, sig2b = fit_anchored_exp_through_endpoint_fast(
                t2b[idx2b], v2b[idx2b], seg_sign=rec_sign, tau_max_mult=exp2b_tau_max_mult
            )
            b2, tau2 = map(float, p2b)
            print("Exp2b fit (split2->stop, anchored-exp-endpoint):")
            print(f"  baseline2={b2:.6f}, tau2={tau2*1e9:.3f} ns")
            V2b_model = exp_anchor_np(t2b, b2, tau2, y2_0)

            i2a = n1 - 1
            V_model[i2a:i2a+len(V2a_model)] = V2a_model
            i2b = i2a + len(V2a_model) - 1
            V_model[i2b:] = V2b_model
            exp2_mode = "linear_then_exp"

            # Confidence bands per sub-segment.
            band2a = RECOVERY_BAND_SIGMA * sig2a
            V_lo[i2a:i2a+len(V2a_model)] = V2a_model - band2a
            V_hi[i2a:i2a+len(V2a_model)] = V2a_model + band2a

            t2bgrid = np.linspace(0, float(t2b[-1]), max(200, len(t2b)))
            mean2b = exp_anchor_np(t2bgrid, b2, tau2, y2_0)
            band2b = RECOVERY_BAND_SIGMA * sig2b
            lo2b = mean2b - band2b
            hi2b = mean2b + band2b
            V_lo[i2b:] = np.interp(t2b, t2bgrid, lo2b)
            V_hi[i2b:] = np.interp(t2b, t2bgrid, hi2b)
        elif shot_id in EXP2_THEN_SIGMOID_SHOTS:
            # exp2 ends where recovery starts fluctuating strongly (post-300ns window),
            # then sigmoid models the tail until stop.
            w0_ns, w1_ns = EXP2_SIG_START_WINDOW_NS_BY_SHOT.get(shot_id, (300.0, 360.0))
            sm_ns = float(EXP2_SIG_SPLIT_SMOOTH_NS_BY_SHOT.get(shot_id, max(8.0, fit_smooth_ns)))
            vs_det, _ = smooth_by_ns(t_diode, v_fit, sm_ns)
            dv_det = np.gradient(vs_det, t_diode)
            d2_det = np.gradient(dv_det, t_diode)

            idxs = np.where((t_rel_ns >= float(w0_ns)) & (t_rel_ns <= float(w1_ns)))[0]
            idxs = idxs[(idxs > split1_idx + 4) & (idxs < stop_idx - 8)]
            if len(idxs) > 0:
                # inflection onset proxy in this window: strongest positive acceleration.
                j = int(np.argmax(d2_det[idxs]))
                split2_idx = int(idxs[j])
                split2_mode = "exp2_end_fluct_window_inflection"
            else:
                split2_idx = int(np.argmin(np.abs(t_rel_ns - 320.0)))
                split2_idx = int(np.clip(split2_idx, split1_idx + 4, stop_idx - 8))
                split2_mode = "exp2_end_fluct_fallback_320ns"
            split2_t_ns = float(t_rel_ns[split2_idx])
            print("\nSplit2 selection (exp2->sigmoid):")
            print(f"  split2_mode = {split2_mode}")
            print(f"  split2_t    = {split2_t_ns:.3f} ns | V_split2 = {v_diode[split2_idx]:.6f} V")

            # Segment 2: anchored exponential from split1 -> split2 (hits endpoint exactly).
            t2a_abs = t_diode[split1_idx:split2_idx+1]
            v2a = v_fit[split1_idx:split2_idx+1]
            t2a = t2a_abs - t2a_abs[0]
            idx2a = np.linspace(0, len(t2a)-1, min(SUBSAMPLE_RECOVERY_N, len(t2a)), dtype=int)
            p2a, y2a_0, sig2a = fit_anchored_exp_through_endpoint_fast(
                t2a[idx2a], v2a[idx2a], seg_sign=rec_sign, tau_max_mult=3.0
            )
            b2a, tau2a = map(float, p2a)
            print("Exp2 fit (split1->split2, anchored-exp-endpoint):")
            print(f"  baseline2a={b2a:.6f}, tau2a={tau2a*1e9:.3f} ns")
            V2a_model = exp_anchor_np(t2a, b2a, tau2a, y2a_0)

            # Segment 3: anchored sigmoid from split2 -> stop.
            t2b_abs = t_diode[split2_idx:stop_idx+1]
            v2b = v_diode[split2_idx:stop_idx+1]
            t2b = t2b_abs - t2b_abs[0]
            idx2b = np.linspace(0, len(t2b)-1, min(SUBSAMPLE_RECOVERY_N, len(t2b)), dtype=int)
            p2b_s, y2b_0, sig2b = fit_anchored_sigmoid_fast(t2b[idx2b], v2b[idx2b], seg_sign=rec_sign)
            a2s, tm2s, k2s, m2s = map(float, p2b_s)
            print("Sigmoid fit (split2->stop):")
            print(f"  amp={a2s:.6f}, t_mid={tm2s*1e9:.3f} ns, k={k2s*1e9:.3f} ns, m_tail={m2s:.3e} V/s")
            V2b_model = anchored_sigmoid_np(t2b, a2s, tm2s, k2s, m2s, y2b_0)

            i2a = n1 - 1
            V_model[i2a:i2a+len(V2a_model)] = V2a_model
            i2b = i2a + len(V2a_model) - 1
            V_model[i2b:] = V2b_model
            exp2_mode = "exp_then_sigmoid"

            # Bands
            t2agrid = np.linspace(0, float(t2a[-1]), max(200, len(t2a)))
            mean2a = exp_anchor_np(t2agrid, b2a, tau2a, y2a_0)
            band2a = RECOVERY_BAND_SIGMA * sig2a
            V_lo[i2a:i2a+len(t2a)] = np.interp(t2a, t2agrid, mean2a - band2a)
            V_hi[i2a:i2a+len(t2a)] = np.interp(t2a, t2agrid, mean2a + band2a)

            t2bgrid = np.linspace(0, float(t2b[-1]), max(300, len(t2b)))
            mean2b = anchored_sigmoid_np(t2bgrid, a2s, tm2s, k2s, m2s, y2b_0)
            band2b = RECOVERY_BAND_SIGMA * sig2b
            V_lo[i2b:] = np.interp(t2b, t2bgrid, mean2b - band2b)
            V_hi[i2b:] = np.interp(t2b, t2bgrid, mean2b + band2b)
        elif shot_id in EXP2_TWO_STAGE_SHOTS:
            split2_target_ns = float(EXP2_TWO_STAGE_SPLIT_ABS_NS_BY_SHOT.get(shot_id, 200.0))
            split2_idx = int(np.argmin(np.abs(t_rel_ns - split2_target_ns)))
            split2_idx = int(np.clip(split2_idx, split1_idx + 4, stop_idx - 4))
            split2_mode = "exp2_twostage_abs_target"
            split2_t_ns = float(t_rel_ns[split2_idx])
            print("\nSplit2 selection (exp2_a->exp2_b):")
            print(f"  split2_mode = {split2_mode}")
            print(f"  split2_t    = {split2_t_ns:.3f} ns | V_split2 = {v_diode[split2_idx]:.6f} V")

            t2a_abs = t_diode[split1_idx:split2_idx+1]
            v2a = v_fit[split1_idx:split2_idx+1]
            t2a = t2a_abs - t2a_abs[0]
            idx2a = np.linspace(0, len(t2a)-1, min(SUBSAMPLE_RECOVERY_N, len(t2a)), dtype=int)
            y2a_end_target = float(v_fit[split2_idx])
            p2a_s, y2a_0, y2a_end, sig2a = fit_stretched_exp_endpoint_fast(
                t2a[idx2a], v2a[idx2a], seg_sign=rec_sign, tau_max_mult=2.0, y_end_target=y2a_end_target
            )
            tau2a, k2a = map(float, p2a_s)
            print("Exp2a fit (split1->split2, stretched-endpoint-smoothed):")
            print(f"  tau2a={tau2a*1e9:.3f} ns, k2a={k2a:.3f}, y2a_0={y2a_0:.6f}, y2a_end(target)={y2a_end:.6f}")
            V2a_model = stretched_exp_endpoint_np(t2a, tau2a, k2a, y2a_0, y2a_end, float(t2a[-1]))

            t2b_abs = t_diode[split2_idx:stop_idx+1]
            v2b = v_fit[split2_idx:stop_idx+1]
            t2b = t2b_abs - t2b_abs[0]
            idx2b = np.linspace(0, len(t2b)-1, min(SUBSAMPLE_RECOVERY_N, len(t2b)), dtype=int)
            y2b_end_target = float(v_diode[stop_idx]) + float(EXP2_ENDPOINT_Y_OFFSET_BY_SHOT.get(shot_id, 0.0))
            p2b_s, y2b_0, y2b_end, sig2b = fit_stretched_exp_endpoint_fast(
                t2b[idx2b], v2b[idx2b], seg_sign=rec_sign, tau_max_mult=2.5, y_end_target=y2b_end_target
            )
            tau2b, k2b = map(float, p2b_s)
            print("Exp2b fit (split2->stop, stretched-endpoint-target-stop):")
            print(f"  tau2b={tau2b*1e9:.3f} ns, k2b={k2b:.3f}, y2b_0={y2b_0:.6f}, y2b_end(target)={y2b_end:.6f}")
            V2b_model = stretched_exp_endpoint_np(t2b, tau2b, k2b, y2b_0, y2b_end, float(t2b[-1]))

            i2a = n1 - 1
            V_model[i2a:i2a+len(V2a_model)] = V2a_model
            i2b = i2a + len(V2a_model) - 1
            V_model[i2b:] = V2b_model
            exp2_mode = "two_stage_stretched"

            t2agrid = np.linspace(0, float(t2a[-1]), max(200, len(t2a)))
            mean2a = stretched_exp_endpoint_np(t2agrid, tau2a, k2a, y2a_0, y2a_end, float(t2a[-1]))
            band2a = RECOVERY_BAND_SIGMA * sig2a
            lo2a = mean2a - band2a
            hi2a = mean2a + band2a
            V_lo[i2a:i2a+len(t2a)] = np.interp(t2a, t2agrid, lo2a)
            V_hi[i2a:i2a+len(t2a)] = np.interp(t2a, t2agrid, hi2a)

            t2bgrid = np.linspace(0, float(t2b[-1]), max(200, len(t2b)))
            mean2b = stretched_exp_endpoint_np(t2bgrid, tau2b, k2b, y2b_0, y2b_end, float(t2b[-1]))
            band2b = RECOVERY_BAND_SIGMA * sig2b
            lo2b = mean2b - band2b
            hi2b = mean2b + band2b
            V_lo[i2b:] = np.interp(t2b, t2bgrid, lo2b)
            V_hi[i2b:] = np.interp(t2b, t2bgrid, hi2b)
        else:
            t2_abs = t_diode[split1_idx:stop_idx+1]
            if shot_id in EXP2_FIT_RAW_SHOTS:
                v2 = v_diode[split1_idx:stop_idx+1]
            else:
                v2 = v_fit[split1_idx:stop_idx+1]
            t2 = t2_abs - t2_abs[0]
            idx2 = np.linspace(0, len(t2)-1, min(SUBSAMPLE_RECOVERY_N, len(t2)), dtype=int)
            exp2_mode = "anchored_exp"
            if (shot_id in STRETCHED_ENDPOINT_SHOTS):
                y_end_target = float(v_diode[stop_idx]) + float(EXP2_ENDPOINT_Y_OFFSET_BY_SHOT.get(shot_id, 0.0))
                p2s, y2_0, y2_end, sig2 = fit_stretched_exp_endpoint_fast(
                    t2[idx2], v2[idx2], seg_sign=rec_sign, tau_max_mult=2.5, y_end_target=y_end_target
                )
                tau2, k2 = map(float, p2s)
                print("Exp2 fit (split1->stop, stretched-endpoint-target-stop):")
                print(f"  tau2={tau2*1e9:.3f} ns, k2={k2:.3f}, y2_0={y2_0:.6f}, y2_end(target)={y2_end:.6f}")
                V2_model = stretched_exp_endpoint_np(t2, tau2, k2, y2_0, y2_end, float(t2[-1]))
                exp2_mode = "stretched_endpoint"
            elif (shot_id in EXP2_CLASSIC_ENDPOINT_SHOTS):
                p2, y2_0, sig2 = fit_anchored_exp_through_endpoint_fast(
                    t2[idx2], v2[idx2], seg_sign=rec_sign, tau_max_mult=2.5
                )
                b2, tau2 = map(float, p2)
                print("Exp2 fit (split1->stop, anchored-exp-endpoint):")
                print(f"  baseline2={b2:.6f}, tau2={tau2*1e9:.3f} ns")
                V2_model = exp_anchor_np(t2, b2, tau2, y2_0)
                exp2_mode = "anchored_exp"
            elif (shot_id in EXP2_ENDPOINT_TARGET_STOP_SHOTS):
                y_end_target = float(v_diode[stop_idx]) + float(EXP2_ENDPOINT_Y_OFFSET_BY_SHOT.get(shot_id, 0.0))
                p2a, y2_0, y2_end, sig2 = fit_exp2_accel_endpoint_fast(
                    t2[idx2], v2[idx2], y_end_target=y_end_target
                )
                alpha2 = float(p2a[0])
                print("Exp2 fit (split1->stop, accel-endpoint-target-stop):")
                print(f"  alpha2={alpha2:.6f}, y2_0={y2_0:.6f}, y2_end(target)={y2_end:.6f}")
                V2_model = exp2_accel_endpoint_np(t2, alpha2, y2_0, y2_end, float(t2[-1]))
                exp2_mode = "accel_endpoint"
            elif use_endpoint_constrained:
                p2a, y2_0, y2_end, sig2 = fit_exp2_accel_endpoint_fast(t2[idx2], v2[idx2])
                alpha2 = float(p2a[0])
                print("Exp2 fit (split1->stop, accel-endpoint):")
                print(f"  alpha2={alpha2:.6f}, y2_0={y2_0:.6f}, y2_end={y2_end:.6f}")
                V2_model = exp2_accel_endpoint_np(t2, alpha2, y2_0, y2_end, float(t2[-1]))
                exp2_mode = "accel_endpoint"
            else:
                p2, y2_0, sig2 = fit_anchored_exp_fast(t2[idx2], v2[idx2], seg_sign=rec_sign)
                b2, tau2 = map(float, p2)
                print("Exp2 fit (split1->stop):")
                print(f"  baseline2={b2:.6f}, tau2={tau2*1e9:.3f} ns")
                V2_model = exp_anchor_np(t2, b2, tau2, y2_0)
            V_model[n1-1:] = V2_model
            if shot_id in EXP1_ENDPOINT_RAW_TARGET_SHOTS:
                # Keep current curve shapes, but guarantee the stitched handoff hits the true valley sample.
                V_model[n1-1] = float(v_diode[split1_idx])

            t2grid = np.linspace(0, float(t2[-1]), max(300, len(t2)))
            if exp2_mode == "accel_endpoint":
                mean2 = exp2_accel_endpoint_np(t2grid, alpha2, y2_0, y2_end, float(t2[-1]))
            elif exp2_mode == "stretched_endpoint":
                mean2 = stretched_exp_endpoint_np(t2grid, tau2, k2, y2_0, y2_end, float(t2[-1]))
            else:
                mean2 = exp_anchor_np(t2grid, b2, tau2, y2_0)
            band2 = RECOVERY_BAND_SIGMA * sig2
            lo2 = mean2 - band2
            hi2 = mean2 + band2
            V_lo[n1-1:] = np.interp(t2, t2grid, lo2)
            V_hi[n1-1:] = np.interp(t2, t2grid, hi2)

    # Optional per-shot floor to suppress non-physical undershoot.
    if shot_id in MODEL_V_FLOOR_BY_SHOT:
        v_floor = float(MODEL_V_FLOOR_BY_SHOT[shot_id])
        V_model = np.maximum(V_model, v_floor)
        V_lo = np.maximum(V_lo, v_floor)
        V_hi = np.maximum(V_hi, v_floor)

    mci = np.isfinite(V_lo) & np.isfinite(V_hi)
    plot_modeled_only_labels = (shot_id in PLOT_MODELED_ONLY_LABEL_SHOTS)
    hide_conf_band = (not SHOW_CONF_BAND) or (shot_id in PLOT_HIDE_CONF_BAND_SHOTS)

    # =============================
    # PLOTS
    # =============================

    # 1) Full trace (for context)
    plt.figure(figsize=(14, 6))
    diode_lbl = "_nolegend_" if plot_modeled_only_labels else "Diode (actual)"
    plt.plot(t_rel_ns, v_diode, "k-", lw=1.0, alpha=0.8, label=diode_lbl)
    if use_four_exp:
        model_lbl = "modeled (exp1 + exp2 + exp tail)"
    elif use_gompertz:
        model_lbl = "modeled (exp1 + gompertz)"
    elif use_three_exp:
        model_lbl = "modeled (exp1 + sigmoid + exp tail)"
    else:
        model_lbl = "modeled (exp1 + exp2)"
    plt.plot(t_seg_ns, V_model, "r-", lw=2.6, label=model_lbl)
    band_lbl = "approx 95% fit band (4-exp/exp-tail)" if use_four_exp else ("approx 95% fit band (gompertz)" if use_gompertz else ("approx 95% fit band (sigmoid segment)" if use_three_exp else "approx 95% fit band (exp2)"))
    if np.any(mci) and (not hide_conf_band):
        band_plot_lbl = "_nolegend_" if plot_modeled_only_labels else band_lbl
        plt.fill_between(t_seg_ns[mci], V_lo[mci], V_hi[mci], color="red", alpha=0.20, label=band_plot_lbl)

    t_cross_rel_ns = _s_to_ns(t_cross - new_t0_abs)
    if SHOW_EVENT_MARKERS and (not plot_modeled_only_labels):
        plt.axvline(0.0, color="red", ls="--", lw=2, label="NEW t0")
        plt.axvline(t_cross_rel_ns, color="orange", ls="--", lw=2, label="PCD=0.5V (+100ns)")
        plt.axvline(onset_t_ns, color="green", ls="--", lw=2, label="onset (model start)")
        plt.axvline(peak_t_ns, color="blue", ls="--", lw=2, label="peak")
    sig_start_t_ns = float(t_rel_ns[dip_end_idx]) if (use_three_exp and has_dip and dip_end_idx is not None) else split1_t_ns
    if SHOW_EVENT_MARKERS and (not plot_modeled_only_labels):
        plt.axvline(split1_t_ns, color="cyan", ls="--", lw=2, label=("peak / seg2 start" if use_three_exp else "exp2 start (turn)"))
        if (not use_three_exp) and ('exp2_mode' in locals()) and (exp2_mode in {"two_stage_stretched", "linear_then_exp", "exp_then_sigmoid"}) and (split2_t_ns is not None):
            if exp2_mode == "linear_then_exp":
                lbl = "linear->exp handoff"
            elif exp2_mode == "exp_then_sigmoid":
                lbl = "sigmoid start"
            else:
                lbl = "exp3 start (~200ns)"
            plt.axvline(split2_t_ns, color="magenta", ls="--", lw=2, label=lbl)
        if use_four_exp and (split2_t_ns is not None):
            plt.axvline(split2_t_ns, color="magenta", ls="--", lw=2, label="exp tail start (old stop)")
        if use_four_exp and (split3_t_ns is not None):
            plt.axvline(split3_t_ns, color="teal", ls="--", lw=2, label="shape anchor (rel max)")
        if use_three_exp and has_dip and dip_end_idx is not None:
            plt.axvline(sig_start_t_ns, color="teal", ls="--", lw=2, label="sigmoid start (after dip)")
        if (not use_gompertz) and use_three_exp and split2_t_ns is not None:
            plt.axvline(split2_t_ns, color="magenta", ls="--", lw=2, label="exp tail start (slope mismatch)")
        plt.axvline(stop_t_ns, color="purple", ls="--", lw=2, label="stop")

    seg_count_lbl = "4" if use_four_exp else ("3" if use_three_exp else "2")
    plt.title(f"Full overlay (NEW t0 axis): actual diode + piecewise {seg_count_lbl}-exp model", fontweight="bold")
    plt.xlabel("Time after NEW t0 (ns)")
    plt.ylabel("Volts")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    p_full = os.path.join(out_dir, "04_overlay_full_after_NEWt0.png")
    plt.savefig(p_full, dpi=170)
    plt.close()
    print(f"Saved: {p_full}")

    # 2) Zoomed overlay (judge fit here)
    x0 = onset_t_ns - ZOOM_LEFT_PAD_NS
    x1 = stop_t_ns + ZOOM_RIGHT_PAD_NS
    plt.figure(figsize=(14, 6))
    plt.plot(t_rel_ns, v_diode, "k-", lw=1.2, alpha=0.85, label=diode_lbl)
    plt.plot(t_seg_ns, V_model, "r-", lw=2.8, label=model_lbl)
    if np.any(mci) and (not hide_conf_band):
        plt.fill_between(t_seg_ns[mci], V_lo[mci], V_hi[mci], color="red", alpha=0.20, label=("_nolegend_" if plot_modeled_only_labels else band_lbl))

    if SHOW_EVENT_MARKERS and (not plot_modeled_only_labels):
        plt.axvline(0.0, color="red", ls="--", lw=2, label="NEW t0")
        plt.axvline(t_cross_rel_ns, color="orange", ls="--", lw=2, label="PCD=0.5V (+100ns)")
        plt.axvline(onset_t_ns, color="green", ls="--", lw=2, label="onset")
        plt.axvline(peak_t_ns, color="blue", ls="--", lw=2, label="peak")
        plt.axvline(split1_t_ns, color="cyan", ls="--", lw=2, label=("peak / seg2 start" if use_three_exp else "exp2 start"))
        if (not use_three_exp) and ('exp2_mode' in locals()) and (exp2_mode in {"two_stage_stretched", "linear_then_exp", "exp_then_sigmoid"}) and (split2_t_ns is not None):
            if exp2_mode == "linear_then_exp":
                lbl = "linear->exp handoff"
            elif exp2_mode == "exp_then_sigmoid":
                lbl = "sigmoid start"
            else:
                lbl = "exp3 start"
            plt.axvline(split2_t_ns, color="magenta", ls="--", lw=2, label=lbl)
        if use_four_exp and (split2_t_ns is not None):
            plt.axvline(split2_t_ns, color="magenta", ls="--", lw=2, label="exp tail start")
        if use_four_exp and (split3_t_ns is not None):
            plt.axvline(split3_t_ns, color="teal", ls="--", lw=2, label="shape anchor")
        if use_three_exp and has_dip and dip_end_idx is not None:
            plt.axvline(sig_start_t_ns, color="teal", ls="--", lw=2, label="sigmoid start")
        if (not use_gompertz) and use_three_exp and split2_t_ns is not None:
            plt.axvline(split2_t_ns, color="magenta", ls="--", lw=2, label="exp tail start")
        plt.axvline(stop_t_ns, color="purple", ls="--", lw=2, label="stop")
    plt.xlim(x0, x1)

    split2_tag = str(split2_mode) if (split2_mode is not None) else "n/a"
    plt.title(
        f"ZOOM overlay: piecewise {seg_count_lbl}-exp "
        f"(stop_mode={stop_mode}, split1_mode={split1_mode}, split2_mode={split2_tag})",
        fontweight="bold"
    )
    plt.xlabel("Time after NEW t0 (ns)")
    plt.ylabel("Volts")
    plt.grid(True, alpha=0.3)
    plt.legend(loc="best")
    plt.tight_layout()
    p_zoom = os.path.join(out_dir, "04b_overlay_ZOOM_after_NEWt0.png")
    plt.savefig(p_zoom, dpi=170)
    plt.close()
    print(f"Saved: {p_zoom}")

    # =============================
    # EXPORT waveform CSV
    # =============================
    export_wave_csv = os.path.join(out_dir, "export_modeled_segment_waveform.csv")
    dfw = pd.DataFrame({
        "time_s_abs": t_seg_abs,
        "time_ns_after_NEWt0": t_seg_ns,
        "V_model": V_model
    })
    dfw.to_csv(export_wave_csv, index=False)
    print(f"Saved: {export_wave_csv}")

    # =============================
    # EXPORT current pulse TXT (2 cols): time(s) current(A)
    # I(t)=K_T*t + K_V*V(t)  (+ optional skew + tail taper)
    # =============================
    t_out = (t_seg_abs - new_t0_abs) + TIME_SHIFT_S
    V_out = V_model
    I_out = K_T * t_out + K_V * V_out

    if SKEW_FIRST_NS > 0 and SKEW_MULT != 1.0:
        msk = (t_out - TIME_SHIFT_S) <= _ns_to_s(SKEW_FIRST_NS)
        I_out = I_out.copy()
        I_out[msk] *= SKEW_MULT

    if TAIL_TAPER_NS > 0:
        taper_s = _ns_to_s(TAIL_TAPER_NS)
        t_end = t_out[-1]
        t_start = max(t_out[0], t_end - taper_s)
        wgt = np.ones_like(t_out)
        m = t_out >= t_start
        wgt[m] = np.clip((t_end - t_out[m]) / max(taper_s, 1e-18), 0.0, 1.0)
        I_out = I_out * wgt

    export_i_txt = os.path.join(out_dir, "export_current_pulse.txt")
    with open(export_i_txt, "w") as f:
        for ti, ii in zip(t_out, I_out):
            f.write(f"{ti:.9e} {ii:.9e}\n")
    print(f"Saved: {export_i_txt}")

    # =============================
    # Print piecewise formulas
    # =============================
    t0_model_start = float(t_seg_abs[0])
    t1_boundary = float(t_diode[split1_idx])
    t2_boundary = float(t_diode[split2_idx]) if (use_three_exp and (not use_gompertz) and (split2_idx is not None)) else None
    t_model_end = float(t_seg_abs[-1])

    print("\nPiecewise V(t) model (t in absolute seconds):")
    print(f"  valid on [{t0_model_start:.9e}, {t_model_end:.9e}]")
    print(f"  Segment 1: t in [{t0_model_start:.9e}, {t1_boundary:.9e}]")
    if ('k1' in locals()) and ((shot_id in STRETCHED_ENDPOINT_SHOTS) or (shot_id in EXP1_STRETCHED_ONLY_SHOTS)):
        print("    V1(t) = b1 + (y1_0-b1)*exp(-((t-ts)/tau1)^k1), with b1 set by V1(te)=y1_end")
        print(f"    ts={t0_model_start:.9e}, te={t1_boundary:.9e}, y1_0={y1_0:.9e}, y1_end={y1_end:.9e}, tau1={tau1:.9e}, k1={k1:.9e}")
    else:
        print(f"    V1(t) = baseline1 + (y1_0 - baseline1) * exp(-(t - {t0_model_start:.9e})/tau1)")
        print(f"    baseline1={b1:.9e}, y1_0={y1_0:.9e}, tau1={tau1:.9e}")

    if use_four_exp and (split2_idx is not None):
        t2_boundary = float(t_diode[split2_idx])
        t3_boundary = float(t_diode[split3_idx]) if (split3_idx is not None) else (t2_boundary + 0.5*(t_model_end - t2_boundary))
        print(f"  Segment 2: t in [{t1_boundary:.9e}, {t2_boundary:.9e}]")
        print("    V2(t) = b2a + (y2a_0-b2a)*exp(-((t-ts)/tau2a)^k2a), with b2a set by V2(te)=y2a_end")
        print(f"    ts={t1_boundary:.9e}, te={t2_boundary:.9e}, y2a_0={y2a_0:.9e}, y2a_end={y2a_end:.9e}, tau2a={tau2a:.9e}, k2a={k2a:.9e}")
        print(f"  Segment 3: t in [{t2_boundary:.9e}, {t_model_end:.9e}]")
        print("    V3(t) = b3 + (y3_0-b3)*exp(-((t-ts)/tau3)^k3), with b3 set by V3(te)=y3_end, shape anchored near rolling-max")
        print(f"    ts={t2_boundary:.9e}, t_anchor={t3_boundary:.9e}, te={t_model_end:.9e}, y3_0={yC0:.9e}, y_anchor={y_anchor:.9e}, y3_end={yC1:.9e}, tau3={tauC:.9e}, k3={kC:.9e}")
    elif use_three_exp and use_gompertz:
        print(f"  Segment 2: t in [{t1_boundary:.9e}, {t_model_end:.9e}]")
        print("    V2(t) = y2_0 + (baseline_g - y2_0) * ((exp(-b_g*exp(-(t-ts)/k_g)) - exp(-b_g)) / (1-exp(-b_g)))")
        print(f"    ts={t1_boundary:.9e}, y2_0={y2_0:.9e}, baseline_g={bg:.9e}, b_g={bgom:.9e}, k_g={kg:.9e}")
    elif use_three_exp:
        sig_start_abs = float(t_diode[dip_end_idx]) if (has_dip and dip_end_idx is not None) else t1_boundary
        if has_dip and dip_end_idx is not None:
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {sig_start_abs:.9e}]")
            print("    V2(t) = y0_dip + A_dip*(exp(-(t-ts)/tau_fast)-exp(-(t-ts)/tau_slow))")
            print(f"    ts={t1_boundary:.9e}, y0_dip={y0d:.9e}, A_dip={Ad:.9e}, tau_fast={tfd:.9e}, tau_slow={tsd:.9e}")
            print(f"  Segment 3: t in [{sig_start_abs:.9e}, {t2_boundary:.9e}]")
            print("    V3(t) = y2_0 + m_tail*(t-ts_sig) + amp_s * ((sigmoid((t-ts_sig)-t_mid_s,k_s)-sigmoid(-t_mid_s,k_s))/(1-sigmoid(-t_mid_s,k_s)))")
            print(f"    ts_sig={sig_start_abs:.9e}, y2_0={y2_0:.9e}, amp_s={a2s:.9e}, t_mid_s={tm2s:.9e}, k_s={k2s:.9e}, m_tail={m2s:.9e}")
            print(f"  Segment 4: t in [{t2_boundary:.9e}, {t_model_end:.9e}]")
            print("    V4(t) = y3_0 * exp(-(t-ts2)/tau_tail)")
            print(f"    ts2={t2_boundary:.9e}, y3_0={y0_t:.9e}, tau_tail={tau_tail:.9e}")
        else:
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t2_boundary:.9e}]")
            print("    V2(t) = y2_0 + m_tail*(t-ts) + amp_s * ((sigmoid((t-ts)-t_mid_s,k_s)-sigmoid(-t_mid_s,k_s))/(1-sigmoid(-t_mid_s,k_s)))")
            print(f"    ts={t1_boundary:.9e}, y2_0={y2_0:.9e}, amp_s={a2s:.9e}, t_mid_s={tm2s:.9e}, k_s={k2s:.9e}, m_tail={m2s:.9e}")
            print(f"  Segment 3: t in [{t2_boundary:.9e}, {t_model_end:.9e}]")
            print("    V3(t) = y3_0 * exp(-(t-ts2)/tau_tail)")
            print(f"    ts2={t2_boundary:.9e}, y3_0={y0_t:.9e}, tau_tail={tau_tail:.9e}")
    else:
        if 'exp2_mode' in locals() and exp2_mode == "exp_then_sigmoid" and (split2_idx is not None):
            t2b_boundary = float(t_diode[split2_idx])
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t2b_boundary:.9e}]")
            print("    V2(t) = baseline2a + (y2a_0 - baseline2a) * exp(-(t - ts)/tau2a)")
            print(f"    ts={t1_boundary:.9e}, baseline2a={b2a:.9e}, y2a_0={y2a_0:.9e}, tau2a={tau2a:.9e}")
            print(f"  Segment 3: t in [{t2b_boundary:.9e}, {t_model_end:.9e}]")
            print("    V3(t) = y2b_0 + m_tail*(t-ts2) + amp_s * ((sigmoid((t-ts2)-t_mid_s,k_s)-sigmoid(-t_mid_s,k_s))/(1-sigmoid(-t_mid_s,k_s)))")
            print(f"    ts2={t2b_boundary:.9e}, y2b_0={y2b_0:.9e}, amp_s={a2s:.9e}, t_mid_s={tm2s:.9e}, k_s={k2s:.9e}, m_tail={m2s:.9e}")
        elif 'exp2_mode' in locals() and exp2_mode == "sigmoid_only":
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t_model_end:.9e}]")
            print("    V2(t) = y2_0 + m_tail*(t-ts) + amp_s * ((sigmoid((t-ts)-t_mid_s,k_s)-sigmoid(-t_mid_s,k_s))/(1-sigmoid(-t_mid_s,k_s)))")
            print(f"    ts={t1_boundary:.9e}, y2_0={y2_0:.9e}, amp_s={a2s:.9e}, t_mid_s={tm2s:.9e}, k_s={k2s:.9e}, m_tail={m2s:.9e}")
        elif 'exp2_mode' in locals() and exp2_mode == "linear_then_exp" and (split2_idx is not None):
            t2b_boundary = float(t_diode[split2_idx])
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t2b_boundary:.9e}]")
            print("    V2(t) = y2a_0 + (y2a_1-y2a_0) * ((t-ts)/(te-ts))")
            print(f"    ts={t1_boundary:.9e}, te={t2b_boundary:.9e}, y2a_0={y2a0:.9e}, y2a_1={y2a1:.9e}")
            print(f"  Segment 3: t in [{t2b_boundary:.9e}, {t_model_end:.9e}]")
            print("    V3(t) = baseline2 + (y2_0 - baseline2) * exp(-(t - ts2)/tau2)")
            print(f"    ts2={t2b_boundary:.9e}, baseline2={b2:.9e}, y2_0={y2_0:.9e}, tau2={tau2:.9e}")
        elif 'exp2_mode' in locals() and exp2_mode == "two_stage_stretched" and (split2_idx is not None):
            t2b_boundary = float(t_diode[split2_idx])
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t2b_boundary:.9e}]")
            print("    V2(t) = b2a + (y2a_0-b2a)*exp(-((t-ts)/tau2a)^k2a), with b2a set by V2(te)=y2a_end")
            print(f"    ts={t1_boundary:.9e}, te={t2b_boundary:.9e}, y2a_0={y2a_0:.9e}, y2a_end={y2a_end:.9e}, tau2a={tau2a:.9e}, k2a={k2a:.9e}")
            print(f"  Segment 3: t in [{t2b_boundary:.9e}, {t_model_end:.9e}]")
            print("    V3(t) = b2b + (y2b_0-b2b)*exp(-((t-ts2)/tau2b)^k2b), with b2b set by V3(te)=y2b_end")
            print(f"    ts2={t2b_boundary:.9e}, te={t_model_end:.9e}, y2b_0={y2b_0:.9e}, y2b_end={y2b_end:.9e}, tau2b={tau2b:.9e}, k2b={k2b:.9e}")
        elif 'exp2_mode' in locals() and exp2_mode == "accel_endpoint":
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t_model_end:.9e}]")
            print("    V2(t) = y2_0 + (y2_end-y2_0) * ((exp(alpha2*(t-ts)/(te-ts))-1)/(exp(alpha2)-1))")
            print(f"    ts={t1_boundary:.9e}, te={t_model_end:.9e}, y2_0={y2_0:.9e}, y2_end={y2_end:.9e}, alpha2={alpha2:.9e}")
        elif 'exp2_mode' in locals() and exp2_mode == "stretched_endpoint":
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t_model_end:.9e}]")
            print("    V2(t) = b2 + (y2_0-b2)*exp(-((t-ts)/tau2)^k2), with b2 set by V2(te)=y2_end")
            print(f"    ts={t1_boundary:.9e}, te={t_model_end:.9e}, y2_0={y2_0:.9e}, y2_end={y2_end:.9e}, tau2={tau2:.9e}, k2={k2:.9e}")
        else:
            print(f"  Segment 2: t in [{t1_boundary:.9e}, {t_model_end:.9e}]")
            print(f"    V2(t) = baseline2 + (y2_0 - baseline2) * exp(-(t - {t1_boundary:.9e})/tau2)")
            print(f"    baseline2={b2:.9e}, y2_0={y2_0:.9e}, tau2={tau2:.9e}")

    # Numeric-only formulas (constants substituted) for direct reuse.
    print("\nNumeric Piecewise V(t) (constants substituted; t in absolute seconds):")
    print(f"  valid on [{t0_model_start:.9e}, {t_model_end:.9e}]")
    if ('k1' in locals()) and ((shot_id in STRETCHED_ENDPOINT_SHOTS) or (shot_id in EXP1_STRETCHED_ONLY_SHOTS)):
        print(f"  if {t0_model_start:.9e} <= t <= {t1_boundary:.9e}:")
        print(f"    V(t) = {b1:.9e} + ({y1_0:.9e} - {b1:.9e}) * exp(-((t - {t0_model_start:.9e}) / {tau1:.9e})^{k1:.9e})")
    else:
        print(f"  if {t0_model_start:.9e} <= t <= {t1_boundary:.9e}:")
        print(f"    V(t) = {b1:.9e} + ({y1_0:.9e} - {b1:.9e}) * exp(-(t - {t0_model_start:.9e}) / {tau1:.9e})")

    if 'exp2_mode' in locals() and exp2_mode == "linear_then_exp" and ('t2b_boundary' in locals()):
        slope2 = (y2a1 - y2a0) / max((t2b_boundary - t1_boundary), 1e-18)
        print(f"  if {t1_boundary:.9e} < t <= {t2b_boundary:.9e}:")
        print(f"    V(t) = {y2a0:.9e} + ({slope2:.9e}) * (t - {t1_boundary:.9e})")
        print(f"  if {t2b_boundary:.9e} < t <= {t_model_end:.9e}:")
        print(f"    V(t) = {b2:.9e} + ({y2_0:.9e} - {b2:.9e}) * exp(-(t - {t2b_boundary:.9e}) / {tau2:.9e})")
    elif 'exp2_mode' in locals() and exp2_mode == "exp_then_sigmoid" and ('t2b_boundary' in locals()):
        print(f"  if {t1_boundary:.9e} < t <= {t2b_boundary:.9e}:")
        print(f"    V(t) = {b2a:.9e} + ({y2a_0:.9e} - {b2a:.9e}) * exp(-(t - {t1_boundary:.9e}) / {tau2a:.9e})")
        print(f"  if {t2b_boundary:.9e} < t <= {t_model_end:.9e}:")
        print(f"    V(t) = {y2b_0:.9e} + ({m2s:.9e})*(t - {t2b_boundary:.9e}) + ({a2s:.9e}) * "
              f"((1/(1+exp(-((t - {t2b_boundary:.9e}) - {tm2s:.9e})/{k2s:.9e})) - 1/(1+exp(-(-{tm2s:.9e})/{k2s:.9e})))"
              f" / (1 - 1/(1+exp(-(-{tm2s:.9e})/{k2s:.9e}))))")
    elif 'exp2_mode' in locals() and exp2_mode == "sigmoid_only":
        print(f"  if {t1_boundary:.9e} < t <= {t_model_end:.9e}:")
        print(f"    V(t) = {y2_0:.9e} + ({m2s:.9e})*(t - {t1_boundary:.9e}) + ({a2s:.9e}) * "
              f"((1/(1+exp(-((t - {t1_boundary:.9e}) - {tm2s:.9e})/{k2s:.9e})) - 1/(1+exp(-(-{tm2s:.9e})/{k2s:.9e})))"
              f" / (1 - 1/(1+exp(-(-{tm2s:.9e})/{k2s:.9e}))))")
    else:
        print(f"  if {t1_boundary:.9e} < t <= {t_model_end:.9e}:")
        if ('alpha2' in locals()) and ('y2_end' in locals()) and ('exp2_mode' in locals()) and exp2_mode == "accel_endpoint":
            print(f"    V(t) = {y2_0:.9e} + ({y2_end:.9e} - {y2_0:.9e}) * "
                  f"((exp({alpha2:.9e} * (t - {t1_boundary:.9e}) / ({t_model_end:.9e} - {t1_boundary:.9e})) - 1) / (exp({alpha2:.9e}) - 1))")
        else:
            print(f"    V(t) = {b2:.9e} + ({y2_0:.9e} - {b2:.9e}) * exp(-(t - {t1_boundary:.9e}) / {tau2:.9e})")

    print("\nNumeric I(t) model:")
    print(f"  I(t) = {K_T:.9e} * ((t - {new_t0_abs:.9e}) + {TIME_SHIFT_S:.9e}) + {K_V:.9e} * V(t)")

    print("\nDONE.")
    print("Check:")
    print(" - 04b_overlay_ZOOM_after_NEWt0.png")
    print(" - export_current_pulse.txt")


if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    parser = argparse.ArgumentParser(description="Single-file diode waveform model fit.")
    parser.add_argument("--csv", default=CSV_FILE, help="Input CSV path (single file).")
    parser.add_argument("--out", default=OUT_DIR, help="Output directory.")
    args = parser.parse_args()
    main(csv_file=args.csv, out_dir=args.out)
