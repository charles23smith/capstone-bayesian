<<<<<<< HEAD
from pathlib import Path
=======
import os
from pathlib import Path
import random
>>>>>>> 2d51a42 (Bayesian updates and local files)

import pytensor
pytensor.config.cxx = ""
pytensor.config.mode = "FAST_COMPILE"
pytensor.config.linker = "py"
<<<<<<< HEAD
=======
import pytensor.tensor as pt
>>>>>>> 2d51a42 (Bayesian updates and local files)

import arviz as az
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pymc as pm
<<<<<<< HEAD

from rsquaredfit import compute_new_t0_abs, prepare_window_data


SHOT_ID = 27277
DATA_DIR = Path("data")
OUT_CSV = Path("bayesian_recovery_27277_summary.csv")
OUT_PNG = Path("bayesian_recovery_27277_fit.png")
START_TIME_NS = 200.0
END_TIME_NS = 400.0
SUBSAMPLE = 4

SHOT_DOSES = {
    27277: 1.06e10,
    27278: 1.16e10,
    27279: 6.22e10,
    27286: 4.74e11,
}


def hill_recovery(t_ns, v0, t_half_ns, n):
    return v0 / (1.0 + t_ns / t_half_ns) ** n


def load_windowed_shot(shot_id: int):
    t_rel_ns, v_diode = compute_new_t0_abs(DATA_DIR / f"{shot_id}_data.csv", shot_id)
    t_ns, v_obs = prepare_window_data(
        t_rel_ns,
        v_diode,
        START_TIME_NS,
        END_TIME_NS,
    )[:2]
    return t_ns[::SUBSAMPLE], v_obs[::SUBSAMPLE]


def main():
    dose = SHOT_DOSES[SHOT_ID]
    log_dose = float(np.log(dose))
    t_ns, v_obs = load_windowed_shot(SHOT_ID)
=======
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
OUT_CSV = Path("bayesian_recovery_summary.csv")
OUT_LOO_CSV = Path("bayesian_recovery_loo_results.csv")
OUT_WINDOW_TIMES_CSV = Path("bayesian_recovery_window_times.csv")
RANDOM_SEED = 42
DRAWS = 400
TUNE = 500
TARGET_ACCEPT = 0.98
MAX_POINTS_PER_SHOT = 500
NUMPYRO_AVAILABLE = True

os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")

try:
    import numpyro  # noqa: F401
except ImportError:
    NUMPYRO_AVAILABLE = False

# Keep the broader candidate pool in the file for later reuse.
ALL_SHOT_DOSES = {
    27271: 1.28e10,
    27272: 1.28e10,
    27276: 1.40e10,
    27277: 1.06e10,
    27278: 1.16e10,
    27279: 6.22e10,
    27280: 5.63e10,
    27282: 6.41e10,
    27290: 1.22e10,
    27291: 9.31e9,
    27294: 5.43e10,
    27295: 5.44e10,
    27297: 6.10e11,
    27298: 1.21e10,
    27300: 5.22e10,
    27286: 4.74e11,
    27306: 1.15e10,
    27307: 5.09e11,
}

ALL_SHOT_WINDOWS_NS = {
    27271: (200.0, 314.0),
    27272: (200.0, 400.0),
    27276: (205.0, 500.0),
    27277: (200.0, 400.0),
    27278: (200.0, 400.0),
    27279: (210.0, 400.0),
    27280: (200.0, 323.0),
    27282: (200.0, 323.0),
    27286: (152.0, 300.0),
    27290: (137.0, 233.0),
    27291: (139.0, 235.0),
    27294: (140.0, 600.0),
    27295: (146.0, 266.0),
    27297: (140.0, 325.0),
    27298: (131.0, 325.0),
    27300: (127.0, 450.0),
    # Direct comparison fits suggest these are strong plain-Hill-rise candidates.
    27306: (-39000.0, 40000.0),
    27307: (150.0, 70000.0),
}

ACTIVE_SHOT_IDS = [27272, 27277, 27278, 27279]
LOO_HOLDOUT_SHOT_IDS = [27272]
SHOT_DOSES = {shot_id: ALL_SHOT_DOSES[shot_id] for shot_id in ACTIVE_SHOT_IDS}
SHOT_WINDOWS_NS = {shot_id: ALL_SHOT_WINDOWS_NS[shot_id] for shot_id in ACTIVE_SHOT_IDS}


def ns_to_s(ns: float) -> float:
    return float(ns) * 1e-9


def s_to_ns(s) -> np.ndarray:
    return np.asarray(s, dtype=float) * 1e9


def rise_then_drop_recovery(t_ns, v_start, amp_rise, t_half_ns, n, amp_drop, t_drop_ns, k_drop_ns):
    t_ns = pt.maximum(t_ns, 0.0)
    rise_ratio = pt.power(t_ns, n) / (pt.power(t_half_ns, n) + pt.power(t_ns, n))
    rise_term = amp_rise * rise_ratio

    sigmoid = 1.0 / (1.0 + pt.exp(-(t_ns - t_drop_ns) / k_drop_ns))
    sigmoid0 = 1.0 / (1.0 + pt.exp(-(-t_drop_ns) / k_drop_ns))
    drop_term = amp_drop * (sigmoid - sigmoid0) / pt.maximum(1.0 - sigmoid0, 1e-9)
    return v_start + rise_term - drop_term


def rise_then_drop_recovery_np(t_ns, v_start, amp_rise, t_half_ns, n, amp_drop, t_drop_ns, k_drop_ns):
    t_ns = np.asarray(t_ns, dtype=float)
    t_ns = np.maximum(t_ns, 0.0)
    rise_ratio = np.power(t_ns, float(n)) / (
        np.power(float(t_half_ns), float(n)) + np.power(t_ns, float(n))
    )
    rise_term = float(amp_rise) * rise_ratio

    sigmoid = 1.0 / (1.0 + np.exp(-(t_ns - float(t_drop_ns)) / float(k_drop_ns)))
    sigmoid0 = 1.0 / (1.0 + np.exp(-(-float(t_drop_ns)) / float(k_drop_ns)))
    drop_term = float(amp_drop) * (sigmoid - sigmoid0) / max(1.0 - sigmoid0, 1e-9)
    return float(v_start) + rise_term - drop_term


def rise_then_drop_recovery_logtime(log10_t_ns, v_start, amp_rise, t_half_ns, n, amp_drop, t_drop_ns, k_drop_ns):
    t_ns = pt.power(10.0, log10_t_ns)
    return rise_then_drop_recovery(t_ns, v_start, amp_rise, t_half_ns, n, amp_drop, t_drop_ns, k_drop_ns)


def rise_then_drop_recovery_logtime_np(log10_t_ns, v_start, amp_rise, t_half_ns, n, amp_drop, t_drop_ns, k_drop_ns):
    t_ns = np.power(10.0, np.asarray(log10_t_ns, dtype=float))
    return rise_then_drop_recovery_np(t_ns, v_start, amp_rise, t_half_ns, n, amp_drop, t_drop_ns, k_drop_ns)


def positive_from_dose(intercept, slope, log_dose, floor=1e-6):
    return pt.softplus(intercept + slope * log_dose) + float(floor)


def negative_from_dose(intercept, slope, log_dose, floor=1e-6):
    return -(pt.softplus(intercept + slope * log_dose) + float(floor))


def choose_log_spaced_indices(t_ns: np.ndarray, n_keep: int) -> np.ndarray:
    if len(t_ns) <= n_keep:
        return np.arange(len(t_ns), dtype=int)

    # Keep denser coverage near t=0 while still covering the long tail.
    grid = np.geomspace(1.0, float(t_ns[-1]) + 1.0, n_keep)
    idx = np.searchsorted(t_ns + 1.0, grid, side="left")
    idx = np.clip(idx, 0, len(t_ns) - 1)
    idx = np.unique(np.concatenate(([0], idx, [len(t_ns) - 1])))
    return idx.astype(int)


def compute_new_t0_info(csv_file: Path, shot_id: int):
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


def load_windowed_shot(shot_id: int):
    start_ns, end_ns = SHOT_WINDOWS_NS[shot_id]
    info = compute_new_t0_info(DATA_DIR / f"{shot_id}_data.csv", shot_id)
    t_diode_abs_s = info["t_diode_abs_s"]
    v_diode = info["v_diode"]
    new_t0_abs_s = float(info["new_t0_abs_s"])

    window_start_abs_s = float(new_t0_abs_s + ns_to_s(start_ns))
    window_end_abs_s = float(new_t0_abs_s + ns_to_s(end_ns))
    mask = (t_diode_abs_s >= window_start_abs_s) & (t_diode_abs_s <= window_end_abs_s)
    if np.count_nonzero(mask) < 8:
        raise ValueError(f"window contains too few points for shot {shot_id}")

    t_win_abs_s = np.asarray(t_diode_abs_s[mask], dtype=float)
    t_win = s_to_ns(t_win_abs_s - new_t0_abs_s)
    v_win = np.asarray(v_diode[mask], dtype=float)
    sign_flipped = abs(float(np.min(v_win))) > abs(float(np.max(v_win)))

    # Put the fit on a common hill-recovery scale:
    # 1) flip so the dominant lobe is positive
    # 2) remove tail baseline so the recovery asymptotes to 0
    if sign_flipped:
        v_win = -v_win

    tail_n = max(8, min(64, len(v_win) // 5))
    baseline = float(np.median(v_win[-tail_n:]))
    y_win = v_win - baseline

    # Shift the fit window to start at 0 ns and keep all finite positive-time
    # samples from the selected diode-voltage window.
    t_fit_ns = t_win - float(t_win[0])
    keep = np.isfinite(t_fit_ns) & np.isfinite(v_diode[mask]) & (t_fit_ns > 0.0)
    if np.count_nonzero(keep) < 8:
        raise ValueError(f"window has too few positive-time fit points for shot {shot_id}")

    t_fit_ns = np.asarray(t_fit_ns[keep], dtype=float)
    y_fit = np.asarray(y_win[keep], dtype=float)
    t_window_rel_ns = np.asarray(t_win[keep], dtype=float)
    v_obs_raw = np.asarray(v_diode[mask][keep], dtype=float)
    sign_multiplier = -1.0 if sign_flipped else 1.0

    idx = choose_log_spaced_indices(t_fit_ns, MAX_POINTS_PER_SHOT)
    return {
        "t_ns": t_fit_ns[idx],
        "t_window_rel_ns": t_window_rel_ns[idx],
        "log10_t_ns": np.log10(t_fit_ns[idx]),
        "y_obs": y_fit[idx],
        "v_obs_raw": v_obs_raw[idx],
        "baseline": baseline,
        "new_t0_abs_s": new_t0_abs_s,
        "window_start_abs_s": window_start_abs_s,
        "window_end_abs_s": window_end_abs_s,
        "sign_multiplier": sign_multiplier,
        "sign_flipped": sign_flipped,
    }


def direct_fit_windowed_shot(shot_id: int, shot_data):
    t_fit_ns = np.asarray(shot_data[shot_id]["t_ns"], dtype=float)
    v_obs_raw = np.asarray(shot_data[shot_id]["v_obs_raw"], dtype=float)

    v_start0 = float(np.clip(v_obs_raw[0], -5.0, 0.5))
    amp_rise0 = float(np.clip(max(float(np.max(v_obs_raw) - v_obs_raw[0]), 0.05), 0.0, 5.0))
    tail_n = max(5, min(25, len(v_obs_raw) // 10))
    amp_drop0 = float(np.clip(np.max(v_obs_raw) - np.median(v_obs_raw[-tail_n:]), 0.0, 3.0))
    t_end = float(t_fit_ns[-1])
    t_drop0 = float(min(max(0.65 * t_end, 20.0), t_end))
    k_drop0 = float(min(max(0.06 * t_end, 1.0), max(80.0, 0.5 * t_end)))
    p0 = [v_start0, amp_rise0, min(25.0, max(5.0, 0.15 * t_end)), 1.5, amp_drop0, t_drop0, k_drop0]
    bounds = (
        [-5.0, 0.0, 1e-3, 0.1, 0.0, 5.0, 0.5],
        [0.5, 5.0, max(400.0, t_end), 20.0, 3.0, t_end, max(80.0, 0.5 * t_end)],
    )
    popt, _ = curve_fit(
        rise_then_drop_recovery_np,
        t_fit_ns,
        v_obs_raw,
        p0=p0,
        bounds=bounds,
        maxfev=200000,
    )
    y_hat = rise_then_drop_recovery_np(t_fit_ns, *popt)
    rmse = float(np.sqrt(np.mean((y_hat - v_obs_raw) ** 2)))
    ss_tot = float(np.sum((v_obs_raw - np.mean(v_obs_raw)) ** 2))
    r_squared = float(1.0 - np.sum((v_obs_raw - y_hat) ** 2) / ss_tot) if ss_tot > 0 else np.nan
    return {
        "shot_id": shot_id,
        "log_dose": float(shot_data[shot_id]["log_dose"]),
        "v_start": float(popt[0]),
        "amp_rise": float(popt[1]),
        "t_half": float(popt[2]),
        "n": float(popt[3]),
        "amp_drop": float(popt[4]),
        "t_drop": float(popt[5]),
        "k_drop": float(popt[6]),
        "rmse_direct": rmse,
        "r_squared_direct": r_squared,
    }


def select_consistent_training_shots(train_fit_rows, holdout_v_start_anchor):
    if len(train_fit_rows) <= 2:
        return train_fit_rows
    order = sorted(train_fit_rows, key=lambda row: abs(row["v_start"] - holdout_v_start_anchor))
    return order[:2]


def regress_parameter(train_rows, param_name, target_log_dose):
    x = np.asarray([row["log_dose"] for row in train_rows], dtype=float)
    y = np.asarray([row[param_name] for row in train_rows], dtype=float)
    if len(train_rows) == 1 or np.allclose(x, x[0]):
        return float(y[0]), 0.0, float(y[0])
    slope, intercept = np.polyfit(x, y, 1)
    pred = float(intercept + slope * target_log_dose)
    return float(intercept), float(slope), pred


def fit_trace(train_shot_ids, shot_data):
    with pm.Model() as model:
        a_start = pm.Normal("a_start", 0.3, 0.8)
        a_rise = pm.Normal("a_rise", -1.0, 1.0)
        a_half = pm.Normal("a_half", 2.5, 1.5)
        a_n = pm.Normal("a_n", 0.0, 0.8)
        a_drop = pm.Normal("a_drop", -2.5, 1.0)
        a_tdrop = pm.Normal("a_tdrop", 4.8, 0.8)
        a_kdrop = pm.Normal("a_kdrop", 1.5, 0.7)

        b_start = pm.Normal("b_start", 0.0, 0.10)
        b_rise = pm.Normal("b_rise", 0.0, 0.10)
        b_half = pm.Normal("b_half", 0.0, 0.20)
        b_n = pm.Normal("b_n", 0.0, 0.10)
        b_drop = pm.Normal("b_drop", 0.0, 0.10)
        b_tdrop = pm.Normal("b_tdrop", 0.0, 0.15)
        b_kdrop = pm.Normal("b_kdrop", 0.0, 0.12)
        sigma = pm.HalfNormal("sigma", 0.08)

        shot_param_names = []
        for shot_id in train_shot_ids:
            log10_t_ns = shot_data[shot_id]["log10_t_ns"]
            v_obs_raw = shot_data[shot_id]["v_obs_raw"]
            log_dose = shot_data[shot_id]["log_dose"]

            v_start_i = pm.Deterministic(f"v_start_{shot_id}", negative_from_dose(a_start, b_start, log_dose))
            amp_rise_i = pm.Deterministic(f"amp_rise_{shot_id}", positive_from_dose(a_rise, b_rise, log_dose, floor=1e-4))
            t_half_i = pm.Deterministic(f"t_half_{shot_id}", positive_from_dose(a_half, b_half, log_dose, floor=1.0))
            n_i = pm.Deterministic(f"n_{shot_id}", positive_from_dose(a_n, b_n, log_dose, floor=0.05))
            amp_drop_i = pm.Deterministic(f"amp_drop_{shot_id}", positive_from_dose(a_drop, b_drop, log_dose, floor=1e-4))
            t_drop_i = pm.Deterministic(f"t_drop_{shot_id}", positive_from_dose(a_tdrop, b_tdrop, log_dose, floor=5.0))
            k_drop_i = pm.Deterministic(f"k_drop_{shot_id}", positive_from_dose(a_kdrop, b_kdrop, log_dose, floor=0.5))

            mu_voltage = pm.Deterministic(
                f"mu_voltage_{shot_id}",
                rise_then_drop_recovery_logtime(
                    log10_t_ns, v_start_i, amp_rise_i, t_half_i, n_i, amp_drop_i, t_drop_i, k_drop_i
                ),
            )
            pm.Normal(f"obs_{shot_id}", mu=mu_voltage, sigma=sigma, observed=v_obs_raw)
            shot_param_names.extend(
                [
                    f"v_start_{shot_id}",
                    f"amp_rise_{shot_id}",
                    f"t_half_{shot_id}",
                    f"n_{shot_id}",
                    f"amp_drop_{shot_id}",
                    f"t_drop_{shot_id}",
                    f"k_drop_{shot_id}",
                ]
            )

        sample_kwargs = dict(
            draws=DRAWS,
            tune=TUNE,
            chains=1,
            cores=1,
            random_seed=RANDOM_SEED,
            target_accept=TARGET_ACCEPT,
            progressbar=True,
            return_inferencedata=True,
        )
        if NUMPYRO_AVAILABLE:
            sample_kwargs["nuts_sampler"] = "numpyro"
        trace = pm.sample(**sample_kwargs)

    summary = az.summary(
        trace,
        var_names=[
            "a_start",
            "a_rise",
            "a_half",
            "a_n",
            "a_drop",
            "a_tdrop",
            "a_kdrop",
            "b_start",
            "b_rise",
            "b_half",
            "b_n",
            "b_drop",
            "b_tdrop",
            "b_kdrop",
            "sigma",
            *shot_param_names,
        ],
    )
    return trace, summary


def main():
    rng = random.Random(RANDOM_SEED)
    shot_ids = list(SHOT_DOSES.keys())
    holdout_shot_ids = [shot_id for shot_id in LOO_HOLDOUT_SHOT_IDS if shot_id in shot_ids]
    if not holdout_shot_ids:
        raise ValueError("LOO_HOLDOUT_SHOT_IDS does not overlap ACTIVE_SHOT_IDS")
    plot_shot_id = rng.choice(holdout_shot_ids)

    shot_data = {}
    for shot_id in shot_ids:
        loaded = load_windowed_shot(shot_id)
        loaded["log_dose"] = float(np.log(SHOT_DOSES[shot_id]))
        shot_data[shot_id] = loaded
>>>>>>> 2d51a42 (Bayesian updates and local files)

    print("Dose values:")
    for shot, shot_dose in SHOT_DOSES.items():
        print(f"{shot}: {shot_dose:.6e}")
<<<<<<< HEAD

    with pm.Model() as model:
        a1 = pm.Normal("a1", 0.0, 1.0)
        a2 = pm.Normal("a2", 100.0, 200.0)
        a3 = pm.Normal("a3", 1.0, 2.0)
        b1 = pm.Normal("b1", 0.0, 0.1)
        b2 = pm.Normal("b2", 0.0, 20.0)
        b3 = pm.Normal("b3", 0.0, 0.2)
        sigma = pm.HalfNormal("sigma", 0.2)

        v0_i = pm.Deterministic("V0_i", a1 + b1 * log_dose)
        t_half_i = pm.Deterministic("t_half_i", a2 + b2 * log_dose)
        n_i = pm.Deterministic("n_i", a3 + b3 * log_dose)

        pm.Potential("v0_positive", pm.math.switch(pm.math.gt(v0_i, 0.0), 0.0, -np.inf))
        pm.Potential("t_half_positive", pm.math.switch(pm.math.gt(t_half_i, 0.0), 0.0, -np.inf))
        pm.Potential("n_positive", pm.math.switch(pm.math.gt(n_i, 0.0), 0.0, -np.inf))

        mu = hill_recovery(t_ns, v0_i, t_half_i, n_i)
        pm.Normal("obs", mu=mu, sigma=sigma, observed=v_obs)

        trace = pm.sample(
            draws=250,
            tune=250,
            chains=1,
            cores=1,
            random_seed=42,
            target_accept=0.9,
            progressbar=True,
            return_inferencedata=True,
        )

    summary = az.summary(trace, var_names=["a1", "a2", "a3", "b1", "b2", "b3", "sigma", "V0_i", "t_half_i", "n_i"])
    summary.to_csv(OUT_CSV)
    print(f"\nSaved: {OUT_CSV}")
    print(summary.to_string())

    post = trace.posterior
    v0_hat = float(post["V0_i"].mean())
    t_half_hat = float(post["t_half_i"].mean())
    n_hat = float(post["n_i"].mean())
    v_hat = hill_recovery(t_ns, v0_hat, t_half_hat, n_hat)

    plt.figure(figsize=(8, 5))
    plt.plot(t_ns, v_obs, color="black", linewidth=1.2, label="Observed")
    plt.plot(t_ns, v_hat, color="red", linewidth=2.0, label="Posterior mean fit")
    plt.xlabel("Time since window start (ns)")
    plt.ylabel("Voltage (V)")
    plt.title(f"Bayesian Recovery Fit: Shot {SHOT_ID}")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    plt.savefig(OUT_PNG, dpi=150)
    plt.close()
    print(f"Saved: {OUT_PNG}")
=======
    print(f"\nPlotting random shot: {plot_shot_id}")
    print("Method: robust two-stage direct-fit parameter regression")
    loo_rows = []
    plot_artifacts = None
    last_summary_df = None

    for holdout_shot_id in holdout_shot_ids:
        train_shot_ids = [shot_id for shot_id in shot_ids if shot_id != holdout_shot_id]
        print(f"\nLOO holdout: {holdout_shot_id} | train on {train_shot_ids}")
        train_fit_rows = [direct_fit_windowed_shot(shot_id, shot_data) for shot_id in train_shot_ids]
        holdout_v_start_anchor = float(shot_data[holdout_shot_id]["v_obs_raw"][0])
        selected_rows = select_consistent_training_shots(train_fit_rows, holdout_v_start_anchor)
        last_summary_df = pd.DataFrame(train_fit_rows)
        last_summary_df["used_for_regression"] = last_summary_df["shot_id"].isin([row["shot_id"] for row in selected_rows])

        log_dose = shot_data[holdout_shot_id]["log_dose"]
        v_start_hat = holdout_v_start_anchor
        _, _, amp_rise_hat = regress_parameter(selected_rows, "amp_rise", log_dose)
        _, _, t_half_hat = regress_parameter(selected_rows, "t_half", log_dose)
        _, _, n_hat = regress_parameter(selected_rows, "n", log_dose)
        _, _, amp_drop_hat = regress_parameter(selected_rows, "amp_drop", log_dose)
        _, _, t_drop_hat = regress_parameter(selected_rows, "t_drop", log_dose)
        _, _, k_drop_hat = regress_parameter(selected_rows, "k_drop", log_dose)

        log10_t_ns = shot_data[holdout_shot_id]["log10_t_ns"]
        v_obs_raw = shot_data[holdout_shot_id]["v_obs_raw"]
        y_hat = rise_then_drop_recovery_logtime_np(
            log10_t_ns,
            v_start_hat,
            amp_rise_hat,
            t_half_hat,
            n_hat,
            amp_drop_hat,
            t_drop_hat,
            k_drop_hat,
        )

        rmse = float(np.sqrt(np.mean((y_hat - v_obs_raw) ** 2)))
        ss_tot = float(np.sum((v_obs_raw - np.mean(v_obs_raw)) ** 2))
        r_squared = float(1.0 - np.sum((v_obs_raw - y_hat) ** 2) / ss_tot) if ss_tot > 0 else np.nan
        sigma_hat = float(np.std(v_obs_raw - y_hat, ddof=1)) if len(v_obs_raw) > 1 else float("nan")

        loo_rows.append({
            "holdout_shot": holdout_shot_id,
            "train_shots": " ".join(str(x) for x in train_shot_ids),
            "used_train_shots": " ".join(str(row["shot_id"]) for row in selected_rows),
            "dose_rate": SHOT_DOSES[holdout_shot_id],
            "v_start_pred": v_start_hat,
            "amp_rise_pred": amp_rise_hat,
            "t_half_pred": t_half_hat,
            "n_pred": n_hat,
            "amp_drop_pred": amp_drop_hat,
            "t_drop_pred": t_drop_hat,
            "k_drop_pred": k_drop_hat,
            "sigma_fit": sigma_hat,
            "rmse": rmse,
            "r_squared": r_squared,
            "n_points": int(len(v_obs_raw)),
        })
        print(
            f"Predicted holdout {holdout_shot_id}: "
            f"using shots {[row['shot_id'] for row in selected_rows]} | "
            f"v_start={v_start_hat:.4f} amp_rise={amp_rise_hat:.4f} "
            f"t_half={t_half_hat:.4f} n={n_hat:.4f} "
            f"amp_drop={amp_drop_hat:.4f} t_drop={t_drop_hat:.4f} k_drop={k_drop_hat:.4f} "
            f"RMSE={rmse:.6f} R^2={r_squared:.6f}"
        )

        if holdout_shot_id == plot_shot_id:
            plot_artifacts = (
                holdout_shot_id,
                shot_data[holdout_shot_id]["t_window_rel_ns"],
                v_obs_raw,
                y_hat,
            )

    if last_summary_df is not None:
        last_summary_df.to_csv(OUT_CSV, index=False)
        print(f"\nSaved: {OUT_CSV}")

    window_rows = []
    for shot_id, (start_ns, end_ns) in ALL_SHOT_WINDOWS_NS.items():
        info = compute_new_t0_info(DATA_DIR / f"{shot_id}_data.csv", shot_id)
        new_t0_abs_s = float(info["new_t0_abs_s"])
        window_rows.append({
            "shot_id": shot_id,
            "window_start_rel_ns": float(start_ns),
            "window_end_rel_ns": float(end_ns),
            "new_t0_abs_s": new_t0_abs_s,
            "window_start_abs_s": float(new_t0_abs_s + ns_to_s(start_ns)),
            "window_end_abs_s": float(new_t0_abs_s + ns_to_s(end_ns)),
        })
    pd.DataFrame(window_rows).sort_values("shot_id").to_csv(OUT_WINDOW_TIMES_CSV, index=False)
    print(f"Saved: {OUT_WINDOW_TIMES_CSV}")

    loo_df = pd.DataFrame(loo_rows)
    loo_df.to_csv(OUT_LOO_CSV, index=False)
    print(f"Saved: {OUT_LOO_CSV}")
    print(loo_df.to_string(index=False))

    if plot_artifacts is not None:
        holdout_shot_id, t_plot_ns, y_obs, y_hat = plot_artifacts
        out_png = Path(f"bayesian_recovery_{holdout_shot_id}_fit.png")
        out_png_linear = Path(f"bayesian_recovery_{holdout_shot_id}_fit_linear.png")

        plt.figure(figsize=(8, 5))
        plt.plot(t_plot_ns, y_obs, color="black", linewidth=1.2, label="Observed")
        plt.plot(t_plot_ns, y_hat, color="red", linewidth=2.0, label="LOO posterior mean fit")
        plt.xscale("log")
        plt.xlabel("Time after NEW t0 (ns)")
        plt.ylabel("Diode voltage (V)")
        plt.title(f"Bayesian Recovery LOO Fit: Shot {holdout_shot_id}")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png, dpi=150)
        plt.close()
        print(f"Saved: {out_png}")

        plt.figure(figsize=(8, 5))
        plt.plot(t_plot_ns, y_obs, color="black", linewidth=1.2, label="Observed")
        plt.plot(t_plot_ns, y_hat, color="red", linewidth=2.0, label="LOO posterior mean fit")
        plt.xlabel("Time after NEW t0 (ns)")
        plt.ylabel("Diode voltage (V)")
        plt.title(f"Bayesian Recovery LOO Fit: Shot {holdout_shot_id} (Linear Time)")
        plt.grid(True, alpha=0.3)
        plt.legend()
        plt.tight_layout()
        plt.savefig(out_png_linear, dpi=150)
        plt.close()
        print(f"Saved: {out_png_linear}")
>>>>>>> 2d51a42 (Bayesian updates and local files)


if __name__ == "__main__":
    main()
