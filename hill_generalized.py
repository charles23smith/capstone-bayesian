import argparse
import random
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import curve_fit

from bruteForce import (
    PCD_TARGET_V,
    T0_SHIFT_NS,
    T0_SHIFT_NS_BY_SHOT,
    build_pcd_avg,
    first_crossing_time_or_nearest,
    load_wide_csv,
    smooth_by_ns,
)

DATA_DIR = Path("data")
RANDOM_SEED = 42
OUTPUT_DIR = Path("loo_png_outputs") / "hill_generalized"

# This staged-family script keeps the core 300-series shots and adds the
# shared exponential-fit shots from the comparison sheet. Mid fits are kept
# intentionally and documented inline; shots marked bad are excluded.
ACTIVE_SHOT_IDS = [
    27271,
    27273,  # mid sigmoid/staged fit
    27280,
    27282,
    27283,
    27285,  # mid sigmoid/staged fit
    27288,  # mid sigmoid/staged fit
    27290,
    27291,
    27294,
    27295,
    27298,
    27296,
    27299,
    27300,
    27301,
    27302,
    27303,
    27304,
    27305,
    27306,
    27307,
]
SHARED_EXP_SHOT_IDS = {
    27271,
    27280,
    27282,
    27283,
    27290,
    27291,
    27294,
}
SPECIAL_NEIGHBOR_SHOT_IDS = {
}
SHOT_NOTES = {
    27271: "shared_exp",
    27273: "mid",
    27280: "shared_exp",
    27282: "shared_exp",
    27283: "shared_exp",
    27285: "mid",
    27288: "mid",
    27290: "shared_exp",
    27291: "shared_exp",
    27294: "shared_exp",
    27295: "sigmoid_family",
    27296: "sigmoid_family",
    27298: "sigmoid_family",
    27299: "sigmoid_family",
}
SHOT_DOSES = {
    27271: 1.28e10,
    27273: 1.25e10,
    27280: 5.63e10,
    27282: 6.41e10,
    27283: 8.11e10,
    27285: 4.86e10,
    27288: 6.02e11,
    27290: 1.22e10,
    27291: 9.31e9,
    27294: 5.43e10,
    27295: 5.44e10,
    27298: 1.21e10,
    27296: 4.81e11,
    27299: 1.05e10,
    27300: 5.22e10,
    27301: 5.46e11,
    27302: 8.88e11,
    27303: 1.01e10,
    27304: 6.23e10,
    27305: 5.56e11,
    27306: 1.15e10,
    27307: 5.09e11,
}
SHOT_WINDOWS_NS = {
    27271: (200.0, 314.0),
    27273: (152.0, 215.0),  # mid sigmoid/staged fit
    27280: (200.0, 323.0),
    27282: (200.0, 323.0),
    27283: (200.0, 323.0),
    27285: (119.0, 216.0),  # mid sigmoid/staged fit
    27288: (130.0, 268.0),  # mid sigmoid/staged fit
    27290: (137.0, 233.0),
    27291: (139.0, 235.0),
    27294: (140.0, 600.0),
    27295: (146.0, 266.0),
    27298: (131.0, 325.0),
    27296: (165.0, 521.0),
    27299: (150.0, 274.0),
    27300: (127.0, 450.0),
    27301: (421.0, 800.0),
    27302: (261.0, 600.0),
    27303: (150.0, 1330.0),
    27304: (160.0, 1500.0),
    27305: (250.0, 2000.0),
    27306: (-39000.0, 40000.0),
    27307: (500.0, 70000.0),
}
FULL_SHOT_WINDOWS_NS = {
    shot_id: ((start_ns if start_ns < 75.0 else 75.0), end_ns)
    for shot_id, (start_ns, end_ns) in SHOT_WINDOWS_NS.items()
}
DEFAULT_FIT_SMOOTH_NS = 6.0
FIT_SMOOTH_NS_BY_SHOT = {shot_id: DEFAULT_FIT_SMOOTH_NS for shot_id in ACTIVE_SHOT_IDS}
FIT_SMOOTH_NS_BY_SHOT.update({
    27301: 12.0,
    27302: 12.0,
})
EXCLUDED_TRAIN_SHOT_IDS = set()
WAVEFORM_FAMILY_BY_SHOT = {
    27271: "staged",
    27273: "staged",
    27280: "staged",
    27282: "staged",
    27283: "staged",
    27285: "staged",
    27288: "staged",
    27290: "staged",
    27291: "staged",
    27294: "staged",
    27295: "staged",
    27298: "staged",
    27296: "staged",
    27299: "staged",
    27300: "staged",
    27301: "staged",
    27302: "staged",
    27303: "staged",
    27304: "staged",
    27305: "staged",
    27306: "staged",
    27307: "staged",
}
MACRO_FAMILY_BY_SHOT = {
    27271: "shared_exp",
    27273: "shared_mid",
    27280: "shared_exp",
    27282: "shared_exp",
    27283: "shared_exp",
    27285: "shared_mid",
    27288: "shared_mid",
    27290: "shared_exp",
    27291: "shared_exp",
    27294: "shared_exp",
    27295: "shared_sigmoid",
    27298: "shared_sigmoid",
    27296: "shared_sigmoid",
    27299: "shared_sigmoid",
    27300: "csd_short",
    27301: "csd_short",
    27302: "csd_short",
    27303: "csd_short",
    27304: "csd_short",
    27305: "csd_short",
    27306: "long",
    27307: "long",
}
NEIGHBOR_FAMILY_BY_SHOT = {
    27271: "group_shared_exp_core",
    27273: "group_shared_exp_core",
    27280: "group_shared_exp_core",
    27282: "group_shared_exp_core",
    27283: "group_shared_exp_core",
    27290: "group_shared_exp_core",
    27291: "group_shared_exp_core",
    27294: "group_shared_exp_core",
    27296: "group_301_302_296",
    27301: "group_301_302_296",
    27302: "group_301_302_296",
    27298: "group_298_300_303_304_305",
    27300: "group_298_300_303_304_305",
    27303: "group_298_300_303_304_305",
    27304: "group_298_300_303_304_305",
    27305: "group_298_300_303_304_305",
    27285: "group_285_288_299_295",
    27288: "group_285_288_299_295",
    27299: "group_285_288_299_295",
    27295: "group_285_288_299_295",
    27306: "group_306_only",
    27307: "group_307_only",
}


def parse_holdout():
    parser = argparse.ArgumentParser(
        description="Generalized Hill-model leave-one-out fit for shots 27300-27307."
    )
    parser.add_argument("holdout_shot_id", nargs="?", type=int, default=None)
    parser.add_argument("--full", action="store_true", help="Use the broader 75 ns to current endpoint windows.")
    args, unknown = parser.parse_known_args()

    holdout_shot_id = args.holdout_shot_id
    if holdout_shot_id is None:
        for token in unknown:
            if token.startswith("--") and token[2:].isdigit():
                holdout_shot_id = int(token[2:])
                break
    if holdout_shot_id is None:
        rng = random.Random(RANDOM_SEED)
        holdout_shot_id = rng.choice(list(ACTIVE_SHOT_IDS))
        print(f"Random holdout selected: {holdout_shot_id}")
    holdout_shot_id = int(holdout_shot_id)
    if holdout_shot_id not in ACTIVE_SHOT_IDS:
        raise ValueError(f"holdout shot must be one of {ACTIVE_SHOT_IDS}")
    return holdout_shot_id, args.full


def ns_to_s(ns: float) -> float:
    return float(ns) * 1e-9


def s_to_ns(s) -> np.ndarray:
    return np.asarray(s, dtype=float) * 1e9


def compute_new_t0_info_local(csv_file: Path, shot_id: int):
    series = load_wide_csv(str(csv_file))
    t_diode_abs_s, v_diode = series["Diode"]
    pcd_avg, _ = build_pcd_avg(series, t_diode_abs_s)
    pcd_s, _ = smooth_by_ns(t_diode_abs_s, pcd_avg, 3.0)
    t_cross_abs_s, _, _ = first_crossing_time_or_nearest(t_diode_abs_s, pcd_s, PCD_TARGET_V)
    t0_shift_ns = float(T0_SHIFT_NS_BY_SHOT.get(shot_id, T0_SHIFT_NS))
    new_t0_abs_s = float(t_cross_abs_s - ns_to_s(t0_shift_ns))
    t_rel_ns = s_to_ns(np.asarray(t_diode_abs_s, dtype=float) - new_t0_abs_s)
    return {
        "t_diode_abs_s": np.asarray(t_diode_abs_s, dtype=float),
        "v_diode": np.asarray(v_diode, dtype=float),
        "new_t0_abs_s": new_t0_abs_s,
        "t_rel_ns": t_rel_ns,
    }


def generalized_hill_recovery_np(t_ns, c, m, amp_hill, t_half_ns, n, amp_fast, tau_fast_ns):
    t_ns = np.maximum(np.asarray(t_ns, dtype=float), 0.0)
    t_half_ns = max(float(t_half_ns), 1e-12)
    n = max(float(n), 1e-12)
    tau_fast_ns = max(float(tau_fast_ns), 1e-12)
    hill = float(amp_hill) * (
        np.power(t_ns, n) / (np.power(t_half_ns, n) + np.power(t_ns, n))
    )
    fast = float(amp_fast) * np.exp(-t_ns / tau_fast_ns)
    return float(c) + float(m) * t_ns + hill + fast


def _logistic_np(t_ns, t_mid_ns, k_ns):
    t_ns = np.asarray(t_ns, dtype=float)
    k_ns = max(float(k_ns), 1e-12)
    return 1.0 / (1.0 + np.exp(-(t_ns - float(t_mid_ns)) / k_ns))


def staged_recovery_np(t_ns, c, m, amp_fast, tau_fast_ns, amp_sigmoid, t_mid_ns, k_ns):
    t_ns = np.asarray(t_ns, dtype=float)
    tau_fast_ns = max(float(tau_fast_ns), 1e-12)
    return (
        float(c)
        + float(m) * t_ns
        + float(amp_fast) * np.exp(-t_ns / tau_fast_ns)
        + float(amp_sigmoid) * _logistic_np(t_ns, t_mid_ns, k_ns)
    )


def _median_abs_dev(values):
    values = np.asarray(values, dtype=float)
    med = float(np.median(values))
    mad = float(np.median(np.abs(values - med)))
    return med, max(mad, 1e-6)


def regress_parameter(train_rows, param_name, target_log_dose):
    x = np.asarray([row["log_dose"] for row in train_rows], dtype=float)
    y = np.asarray([row[param_name] for row in train_rows], dtype=float)
    if len(train_rows) == 1 or np.allclose(x, x[0]):
        return float(y[0]), 0.0, float(y[0])
    slope, intercept = np.polyfit(x, y, 1)
    pred = float(intercept + slope * target_log_dose)
    return float(intercept), float(slope), pred


def weighted_average_parameter(train_rows, param_name, weight_key="_bayes_weight"):
    y = np.asarray([row[param_name] for row in train_rows], dtype=float)
    w = np.asarray([max(float(row.get(weight_key, 1.0)), 1e-6) for row in train_rows], dtype=float)
    return float(np.average(y, weights=w))


def fit_metrics(y_true, y_hat):
    y_true = np.asarray(y_true, dtype=float)
    y_hat = np.asarray(y_hat, dtype=float)
    rmse = float(np.sqrt(np.mean((y_true - y_hat) ** 2)))
    ss_tot = float(np.sum((y_true - np.mean(y_true)) ** 2))
    r_squared = float(1.0 - np.sum((y_true - y_hat) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return rmse, r_squared


def _clip_p0(p0, bounds):
    lb, ub = bounds
    clipped = []
    for val, lo, hi in zip(p0, lb, ub):
        if hi < lo:
            lo, hi = hi, lo
        clipped.append(float(np.clip(val, lo, hi)))
    return clipped


def _window_scales(t_fit_ns, v_obs_fit):
    t_end = float(max(t_fit_ns[-1], 1e-9))
    v_obs_fit = np.asarray(v_obs_fit, dtype=float)
    v_span = float(max(np.max(v_obs_fit) - np.min(v_obs_fit), abs(v_obs_fit[-1] - v_obs_fit[0]), 1e-3))
    return t_end, v_span


def _family_fit_model(row):
    return "staged" if row["waveform_family"] == "staged" else "generalized"


def load_windowed_shot_local(shot_id: int, windows_ns):
    start_ns, end_ns = windows_ns[shot_id]
    info = compute_new_t0_info_local(DATA_DIR / f"{shot_id}_data.csv", shot_id)
    t_diode_abs_s = info["t_diode_abs_s"]
    v_diode = info["v_diode"]
    new_t0_abs_s = float(info["new_t0_abs_s"])

    window_start_abs_s = float(new_t0_abs_s + ns_to_s(start_ns))
    window_end_abs_s = float(new_t0_abs_s + ns_to_s(end_ns))
    mask = (t_diode_abs_s >= window_start_abs_s) & (t_diode_abs_s <= window_end_abs_s)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"window contains too few points for shot {shot_id}")

    t_win_abs_s = np.asarray(t_diode_abs_s[mask], dtype=float)
    t_win_rel_ns = s_to_ns(t_win_abs_s - new_t0_abs_s)
    v_win_raw = np.asarray(v_diode[mask], dtype=float)

    smooth_ns = float(FIT_SMOOTH_NS_BY_SHOT.get(shot_id, 0.0))
    if smooth_ns > 0.0:
        v_win_fit, smooth_win = smooth_by_ns(t_win_abs_s, v_win_raw, smooth_ns)
        v_win_fit = np.asarray(v_win_fit, dtype=float)
    else:
        v_win_fit = v_win_raw.copy()
        smooth_win = 0

    t_fit_ns = t_win_rel_ns - float(t_win_rel_ns[0])
    keep = np.isfinite(t_fit_ns) & np.isfinite(v_win_raw) & (t_fit_ns > 0.0)
    if np.count_nonzero(keep) < 8:
        raise ValueError(f"window has too few positive-time fit points for shot {shot_id}")

    return {
        "t_ns": np.asarray(t_fit_ns[keep], dtype=float),
        "t_window_rel_ns": np.asarray(t_win_rel_ns[keep], dtype=float),
        "v_obs_raw": np.asarray(v_win_raw[keep], dtype=float),
        "v_obs_fit": np.asarray(v_win_fit[keep], dtype=float),
        "smooth_ns": smooth_ns,
        "smooth_win": int(smooth_win),
        "new_t0_abs_s": new_t0_abs_s,
        "window_start_abs_s": window_start_abs_s,
        "window_end_abs_s": window_end_abs_s,
        "waveform_family": WAVEFORM_FAMILY_BY_SHOT[shot_id],
        "macro_family": MACRO_FAMILY_BY_SHOT[shot_id],
    }


def direct_fit_windowed_shot_local(shot_id: int, shot_data):
    t_fit_ns = np.asarray(shot_data[shot_id]["t_ns"], dtype=float)
    v_obs_fit = np.asarray(shot_data[shot_id]["v_obs_fit"], dtype=float)
    waveform_family = str(shot_data[shot_id]["waveform_family"])
    t_end, v_scale = _window_scales(t_fit_ns, v_obs_fit)
    v0 = float(v_obs_fit[0])
    vend = float(v_obs_fit[-1])
    span = float(vend - v0)
    slope0 = float((vend - v0) / max(t_end, 1e-9))

    if waveform_family == "staged":
        p0 = [
            float(vend - 0.1),
            float(0.2 * slope0),
            float(v0 - (vend - 0.1)),
            float(max(0.04 * t_end, 8.0)),
            float(max(0.7 * span, 1e-3)),
            float(0.60 * t_end),
            float(max(0.06 * t_end, 8.0)),
        ]
        bounds = (
            [
                float(np.min(v_obs_fit) - 1.0),
                -0.02,
                -6.0,
                0.5,
                0.0,
                0.20 * t_end,
                1.0,
            ],
            [
                float(np.max(v_obs_fit) + 1.0),
                0.02,
                0.0,
                max(120.0, 0.35 * t_end),
                max(6.0, 3.0 * abs(span) + 1.0),
                0.95 * t_end,
                max(0.30 * t_end, 8.0),
            ],
        )
        p0 = _clip_p0(p0, bounds)
        popt, _ = curve_fit(
            staged_recovery_np,
            t_fit_ns,
            v_obs_fit,
            p0=p0,
            bounds=bounds,
            maxfev=300000,
        )
        y_hat = staged_recovery_np(t_fit_ns, *popt)
        t_mid_rel = float(popt[5] / t_end)
        k_rel = float(popt[6] / t_end)
        fit_row = {
            "c": float(popt[0]),
            "m": float(popt[1]),
            "amp_fast": float(popt[2]),
            "tau_fast": float(popt[3]),
            "amp_sigmoid": float(popt[4]),
            "t_mid": float(popt[5]),
            "k": float(popt[6]),
            "m_norm": float(popt[1] * t_end / v_scale),
            "amp_fast_norm": float(popt[2] / v_scale),
            "tau_fast_rel": float(popt[3] / t_end),
            "amp_sigmoid_norm": float(popt[4] / v_scale),
            "t_mid_rel": t_mid_rel,
            "k_rel": k_rel,
            "v_start": float(v_obs_fit[0]),
        }
    else:
        amp_fast0 = float(min(0.0, v0 - np.median(v_obs_fit[: min(len(v_obs_fit), 20)])))
        p0 = [
            float(v0 - amp_fast0),
            float(0.25 * slope0),
            float(max(span, 1e-4)),
            float(max(0.45 * t_end, 1.0)),
            2.0,
            amp_fast0,
            float(max(0.08 * t_end, 5.0)),
        ]
        bounds = (
            [
                min(v0 - 1.0, float(np.min(v_obs_fit)) - 0.5),
                -0.02,
                0.0,
                1e-3,
                0.2,
                -6.0,
                0.5,
            ],
            [
                max(v0 + 1.0, float(np.max(v_obs_fit)) + 0.5),
                0.02,
                max(6.0, 3.0 * abs(span) + 1.0),
                max(t_end, 1e-3),
                20.0,
                6.0,
                max(150.0, 0.6 * t_end),
            ],
        )
        p0 = _clip_p0(p0, bounds)
        popt, _ = curve_fit(
            generalized_hill_recovery_np,
            t_fit_ns,
            v_obs_fit,
            p0=p0,
            bounds=bounds,
            maxfev=250000,
        )
        y_hat = generalized_hill_recovery_np(t_fit_ns, *popt)
        fit_row = {
            "c": float(popt[0]),
            "m": float(popt[1]),
            "amp_hill": float(popt[2]),
            "t_half": float(popt[3]),
            "n": float(popt[4]),
            "amp_fast": float(popt[5]),
            "tau_fast": float(popt[6]),
            "m_norm": float(popt[1] * t_end / v_scale),
            "amp_hill_norm": float(popt[2] / v_scale),
            "t_half_rel": float(popt[3] / t_end),
            "amp_fast_norm": float(popt[5] / v_scale),
            "tau_fast_rel": float(popt[6] / t_end),
            "v_start": float(v_obs_fit[0]),
        }

    rmse, r_squared = fit_metrics(v_obs_fit, y_hat)
    return {
        "shot_id": shot_id,
        "log_dose": float(shot_data[shot_id]["log_dose"]),
        "waveform_family": waveform_family,
        "macro_family": str(shot_data[shot_id]["macro_family"]),
        "neighbor_family": NEIGHBOR_FAMILY_BY_SHOT[shot_id],
        "fit_model": _family_fit_model({"waveform_family": waveform_family}),
        "window_len_ns": float(t_end),
        "v_scale": float(v_scale),
        "rmse_direct": rmse,
        "r_squared_direct": r_squared,
        "smooth_ns": float(shot_data[shot_id]["smooth_ns"]),
        **fit_row,
    }


def select_shape_neighbors(train_fit_rows, holdout_v_start_anchor: float):
    if len(train_fit_rows) <= 2:
        return list(train_fit_rows)
    _, scale_v_start = _median_abs_dev([row["v_start"] for row in train_fit_rows])
    ranked = sorted(train_fit_rows, key=lambda row: abs(row["v_start"] - holdout_v_start_anchor))
    selected = ranked[:2]
    second_gap = abs(selected[-1]["v_start"] - holdout_v_start_anchor)
    extra_gap_cutoff = max(1.5 * second_gap, second_gap + 0.5 * scale_v_start)
    for row in ranked[2:]:
        gap = abs(row["v_start"] - holdout_v_start_anchor)
        if gap <= extra_gap_cutoff:
            selected.append(row)
    return selected


def build_bayesian_family_rows(train_fit_rows, selected_rows, holdout_v_start_anchor: float):
    local_rows = list(selected_rows) if selected_rows else list(train_fit_rows)
    if not local_rows:
        return []

    family = str(local_rows[0]["waveform_family"])

    if family == "staged":
        pred_t_mid_rel = float(np.median([row["t_mid_rel"] for row in local_rows]))
        pred_k_rel = float(np.median([row["k_rel"] for row in local_rows]))
        pred_amp_fast_norm = float(np.median([row["amp_fast_norm"] for row in local_rows]))
        pred_amp_sigmoid_norm = float(np.median([row["amp_sigmoid_norm"] for row in local_rows]))

        _, scale_v_start = _median_abs_dev([row["v_start"] for row in local_rows])
        _, scale_t_mid_rel = _median_abs_dev([row["t_mid_rel"] for row in local_rows])
        _, scale_k_rel = _median_abs_dev([row["k_rel"] for row in local_rows])
        _, scale_amp_fast_norm = _median_abs_dev([row["amp_fast_norm"] for row in local_rows])
        _, scale_amp_sigmoid_norm = _median_abs_dev([row["amp_sigmoid_norm"] for row in local_rows])

        def score(row):
            return (
                0.9 * abs(row["v_start"] - holdout_v_start_anchor) / scale_v_start
                + abs(row["t_mid_rel"] - pred_t_mid_rel) / scale_t_mid_rel
                + abs(row["k_rel"] - pred_k_rel) / scale_k_rel
                + 0.8 * abs(row["amp_fast_norm"] - pred_amp_fast_norm) / scale_amp_fast_norm
                + 0.8 * abs(row["amp_sigmoid_norm"] - pred_amp_sigmoid_norm) / scale_amp_sigmoid_norm
            )
    else:
        pred_t_half_rel = float(np.median([row["t_half_rel"] for row in local_rows]))
        pred_n = float(np.median([row["n"] for row in local_rows]))
        pred_amp_fast_norm = float(np.median([row["amp_fast_norm"] for row in local_rows]))
        pred_tau_fast_rel = float(np.median([row["tau_fast_rel"] for row in local_rows]))

        _, scale_v_start = _median_abs_dev([row["v_start"] for row in local_rows])
        _, scale_t_half_rel = _median_abs_dev([row["t_half_rel"] for row in local_rows])
        _, scale_n = _median_abs_dev([row["n"] for row in local_rows])
        _, scale_amp_fast_norm = _median_abs_dev([row["amp_fast_norm"] for row in local_rows])
        _, scale_tau_fast_rel = _median_abs_dev([row["tau_fast_rel"] for row in local_rows])

        def score(row):
            return (
                0.9 * abs(row["v_start"] - holdout_v_start_anchor) / scale_v_start
                + abs(row["t_half_rel"] - pred_t_half_rel) / scale_t_half_rel
                + abs(row["n"] - pred_n) / scale_n
                + 0.8 * abs(row["amp_fast_norm"] - pred_amp_fast_norm) / scale_amp_fast_norm
                + 0.8 * abs(row["tau_fast_rel"] - pred_tau_fast_rel) / scale_tau_fast_rel
            )

    weighted_rows = []
    for row in local_rows:
        weight = float(np.exp(-0.5 * score(row)))
        weighted_row = dict(row)
        weighted_row["_bayes_weight"] = max(weight, 1e-6)
        weighted_rows.append(weighted_row)
    return weighted_rows


def scale_shape_to_zero_endpoint(y_hat: np.ndarray, v_start_anchor: float):
    y_hat = np.asarray(y_hat, dtype=float)
    if len(y_hat) == 0:
        return y_hat, 1.0
    y_end = float(y_hat[-1])
    if y_end <= 0.0 or v_start_anchor >= 0.0:
        return y_hat, 1.0
    denom = float(y_end - v_start_anchor)
    if denom <= 1e-9:
        return y_hat, 1.0
    scale = float(np.clip((-v_start_anchor) / denom, 0.0, 1.0))
    y_scaled = float(v_start_anchor) + scale * (y_hat - float(v_start_anchor))
    return np.asarray(y_scaled, dtype=float), scale


def main():
    holdout_shot_id, use_full_windows = parse_holdout()
    shot_ids = list(ACTIVE_SHOT_IDS)
    windows_ns = FULL_SHOT_WINDOWS_NS if use_full_windows else SHOT_WINDOWS_NS
    shot_data = {}

    for shot_id in shot_ids:
        loaded = load_windowed_shot_local(shot_id, windows_ns)
        loaded["log_dose"] = float(np.log(SHOT_DOSES[shot_id]))
        shot_data[shot_id] = loaded

    print("Dose values:")
    for shot_id in shot_ids:
        print(f"{shot_id}: {SHOT_DOSES[shot_id]:.6e}")
    print(f"Window mode: {'full' if use_full_windows else 'default'}")

    train_shot_ids = [
        shot_id
        for shot_id in shot_ids
        if shot_id != holdout_shot_id and shot_id not in EXCLUDED_TRAIN_SHOT_IDS
    ]
    print(f"\nLOO holdout: {holdout_shot_id} | train on {train_shot_ids}")
    print("Method: generalized Hill + line + fast-exp LOO")

    train_fit_rows = [direct_fit_windowed_shot_local(shot_id, shot_data) for shot_id in train_shot_ids]
    holdout_family = str(shot_data[holdout_shot_id]["waveform_family"])
    holdout_macro_family = str(shot_data[holdout_shot_id]["macro_family"])
    holdout_neighbor_family = NEIGHBOR_FAMILY_BY_SHOT[holdout_shot_id]
    holdout_v_start_anchor = float(shot_data[holdout_shot_id]["v_obs_fit"][0])
    if holdout_shot_id in SPECIAL_NEIGHBOR_SHOT_IDS:
        special_neighbor_ids = set(SPECIAL_NEIGHBOR_SHOT_IDS[holdout_shot_id])
        selected_rows = [row for row in train_fit_rows if row["shot_id"] in special_neighbor_ids]
        if not selected_rows:
            raise ValueError(f"special neighbor set missing for holdout {holdout_shot_id}")
        bayes_rows = list(selected_rows)
    else:
        family_train_rows = [
            row for row in train_fit_rows if row["neighbor_family"] == holdout_neighbor_family
        ]
        if not family_train_rows:
            family_train_rows = train_fit_rows
        selected_rows = select_shape_neighbors(
            family_train_rows,
            holdout_v_start_anchor,
        )
        if (
            holdout_neighbor_family == "shared_exp"
            and abs(holdout_v_start_anchor) < 0.8
            and len(family_train_rows) >= 3
        ):
            ranked_family_rows = sorted(
                family_train_rows,
                key=lambda row: abs(row["v_start"] - holdout_v_start_anchor),
            )
            selected_rows = ranked_family_rows[:3]
        bayes_rows = build_bayesian_family_rows(
            family_train_rows,
            selected_rows,
            holdout_v_start_anchor,
        )
    holdout_t_fit_ns = np.asarray(shot_data[holdout_shot_id]["t_ns"], dtype=float)
    holdout_v_obs_fit = np.asarray(shot_data[holdout_shot_id]["v_obs_fit"], dtype=float)
    holdout_t_end, holdout_v_scale = _window_scales(holdout_t_fit_ns, holdout_v_obs_fit)
    holdout_trend_sign = np.sign(float(holdout_v_obs_fit[-1] - holdout_v_obs_fit[0]))
    if holdout_family == "staged":
        m_norm_hat = weighted_average_parameter(bayes_rows, "m_norm")
        amp_fast_norm_hat = weighted_average_parameter(bayes_rows, "amp_fast_norm")
        tau_fast_rel_hat = weighted_average_parameter(bayes_rows, "tau_fast_rel")
        amp_sigmoid_norm_hat = weighted_average_parameter(bayes_rows, "amp_sigmoid_norm")
        t_mid_rel_hat = weighted_average_parameter(bayes_rows, "t_mid_rel")
        k_rel_hat = weighted_average_parameter(bayes_rows, "k_rel")

        m_hat = float(m_norm_hat * holdout_v_scale / holdout_t_end)
        if holdout_trend_sign > 0:
            m_hat = max(0.0, m_hat)
        elif holdout_trend_sign < 0:
            m_hat = min(0.0, m_hat)
        amp_fast_hat = float(amp_fast_norm_hat * holdout_v_scale)
        tau_fast_hat = float(np.clip(tau_fast_rel_hat, 0.01, 0.35) * holdout_t_end)
        amp_sigmoid_hat = float(max(1e-4, amp_sigmoid_norm_hat * holdout_v_scale))
        t_mid_hat = float(np.clip(t_mid_rel_hat, 0.20, 0.92) * holdout_t_end)
        k_hat = float(np.clip(k_rel_hat, 0.01, 0.30) * holdout_t_end)
        s0 = float(_logistic_np(np.array([0.0]), t_mid_hat, k_hat)[0])
        c_hat = float(holdout_v_start_anchor - amp_fast_hat - amp_sigmoid_hat * s0)
    else:
        m_norm_hat = weighted_average_parameter(bayes_rows, "m_norm")
        amp_hill_norm_hat = weighted_average_parameter(bayes_rows, "amp_hill_norm")
        t_half_rel_hat = weighted_average_parameter(bayes_rows, "t_half_rel")
        n_hat = weighted_average_parameter(bayes_rows, "n")
        amp_fast_norm_hat = weighted_average_parameter(bayes_rows, "amp_fast_norm")
        tau_fast_rel_hat = weighted_average_parameter(bayes_rows, "tau_fast_rel")

        m_hat = float(m_norm_hat * holdout_v_scale / holdout_t_end)
        amp_hill_hat = float(max(1e-4, amp_hill_norm_hat * holdout_v_scale))
        t_half_hat = float(np.clip(t_half_rel_hat, 0.05, 0.95) * holdout_t_end)
        n_hat = float(max(0.05, min(20.0, n_hat)))
        amp_fast_hat = float(amp_fast_norm_hat * holdout_v_scale)
        tau_fast_hat = float(np.clip(tau_fast_rel_hat, 0.01, 0.60) * holdout_t_end)
        c_hat = float(holdout_v_start_anchor - amp_fast_hat)

    t_fit_ns = holdout_t_fit_ns
    t_plot_ns = np.asarray(shot_data[holdout_shot_id]["t_window_rel_ns"], dtype=float)
    v_obs_raw = np.asarray(shot_data[holdout_shot_id]["v_obs_raw"], dtype=float)
    v_obs_fit = holdout_v_obs_fit
    if holdout_family == "staged":
        y_hat = staged_recovery_np(
            t_fit_ns,
            c_hat,
            m_hat,
            amp_fast_hat,
            tau_fast_hat,
            amp_sigmoid_hat,
            t_mid_hat,
            k_hat,
        )
    else:
        y_hat = generalized_hill_recovery_np(
            t_fit_ns,
            c_hat,
            m_hat,
            amp_hill_hat,
            t_half_hat,
            n_hat,
            amp_fast_hat,
            tau_fast_hat,
        )

    y_hat, endpoint_scale = scale_shape_to_zero_endpoint(y_hat, holdout_v_start_anchor)
    if endpoint_scale < 1.0:
        if holdout_family == "staged":
            amp_fast_hat *= endpoint_scale
            amp_sigmoid_hat *= endpoint_scale
            c_hat = float(holdout_v_start_anchor - amp_fast_hat - amp_sigmoid_hat * s0)
        else:
            amp_hill_hat *= endpoint_scale
            amp_fast_hat *= endpoint_scale
            c_hat = float(holdout_v_start_anchor - amp_fast_hat)

    rmse_fit, r_squared_fit = fit_metrics(v_obs_fit, y_hat)
    rmse_raw, r_squared_raw = fit_metrics(v_obs_raw, y_hat)
    sigma_hat = float(np.std(v_obs_fit - y_hat, ddof=1)) if len(v_obs_fit) > 1 else float("nan")

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    out_png_linear = OUTPUT_DIR / f"hill_generalized_{holdout_shot_id}_fit_linear.png"

    plt.figure(figsize=(8, 5))
    plt.plot(t_plot_ns, v_obs_raw, color="0.7", linewidth=1.0, label="Observed raw")
    if float(shot_data[holdout_shot_id]["smooth_ns"]) > 0.0:
        plt.plot(t_plot_ns, v_obs_fit, color="black", linewidth=1.2, label="Observed smoothed")
    else:
        plt.plot(t_plot_ns, v_obs_fit, color="black", linewidth=1.2, label="Observed")
    plt.plot(t_plot_ns, y_hat, color="red", linewidth=2.0, label="LOO posterior mean fit")
    plt.xlabel("Time after NEW t0 (ns)")
    plt.ylabel("Diode voltage (V)")
    plt.title(f"Generalized LOO Fit: Shot {holdout_shot_id} (Linear Time)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(out_png_linear, dpi=150)
    plt.close()

    print(
        f"Predicted holdout {holdout_shot_id}: using shots {[row['shot_id'] for row in selected_rows]} | "
        f"family={holdout_family} c={c_hat:.4f} m={m_hat:.6f} "
        + (
            f"amp_sigmoid={amp_sigmoid_hat:.4f} t_mid={t_mid_hat:.4f} k={k_hat:.4f} "
            if holdout_family == "staged"
            else f"amp_hill={amp_hill_hat:.4f} t_half={t_half_hat:.4f} n={n_hat:.4f} "
        )
        + f"amp_fast={amp_fast_hat:.4f} tau_fast={tau_fast_hat:.4f} "
        f"RMSE_fit={rmse_fit:.6f} R^2_fit={r_squared_fit:.6f} "
        f"RMSE_raw={rmse_raw:.6f} R^2_raw={r_squared_raw:.6f}"
    )
    print(f"Saved: {out_png_linear}")


if __name__ == "__main__":
    main()
