# peak_voltage.py
#
# Within-diode model with IMPEDANCE AS GROUP (imp=0 vs imp=1)
# and prints BOTH:
#   - 95% CI of mean (mu)  [tight, what you probably want]
#   - 95% PI of observation [wide, includes noise]
#
# Still uses ONLY: dose, bias, imp.
# Still streaming extraction + uniform window.

import os
os.environ["PYTENSOR_FLAGS"] = "cxx=,linker=py,mode=FAST_COMPILE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pymc as pm

DATA_DIR = "data"
REFERENCE_WINDOW_FILE_ID = 27271
PREDICT_FILE_ID = 27271
EXCLUDE_FILE_IDS = {27270, 27281}
CHUNKSIZE = 250_000

PCD3_THRESHOLD_V = 0.5
T0_OFFSET_NS = 100.0
BASELINE_PRE_NS = 50.0

PEAK_SEARCH_MAX_NS = 300.0
TROUGH_SEARCH_MAX_NS = 600.0
WINDOW_GUARD_NS = 10.0

SEED = 42
CHAINS = 2
CORES = 1
DRAWS = 1600
TUNE = 1600
TARGET_ACCEPT = 0.99

SHOT_META = {
    27271: {"diode": "SMAJ400A", "dose_rate": 1.28e10, "bias_v": 0.0,  "imp_50ohm": 0},
    27272: {"diode": "SMAJ400A", "dose_rate": 1.28e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27273: {"diode": "SMAJ400A", "dose_rate": 1.25e10, "bias_v": 5.25, "imp_50ohm": 1},
    27274: {"diode": "SMAJ400A", "dose_rate": 1.19e10, "bias_v": 5.25, "imp_50ohm": 0},
    27275: {"diode": "SMAJ400A", "dose_rate": 1.10e10, "bias_v": 5.25, "imp_50ohm": 1},
    27276: {"diode": "SMAJ400A", "dose_rate": 1.40e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27277: {"diode": "SMAJ400A", "dose_rate": 1.06e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27278: {"diode": "SMAJ400A", "dose_rate": 1.16e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27279: {"diode": "SMAJ400A", "dose_rate": 6.22e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27280: {"diode": "SMAJ400A", "dose_rate": 5.63e10, "bias_v": 0.0,  "imp_50ohm": 0},
    27282: {"diode": "SMAJ400A", "dose_rate": 6.41e10, "bias_v": 0.0,  "imp_50ohm": 0},
    27283: {"diode": "SMAJ400A", "dose_rate": 8.11e10, "bias_v": 0.0,  "imp_50ohm": 0},
    27284: {"diode": "SMAJ400A", "dose_rate": 5.11e10, "bias_v": 5.25, "imp_50ohm": 1},
    27285: {"diode": "SMAJ400A", "dose_rate": 4.86e10, "bias_v": 5.25, "imp_50ohm": 0},
    27286: {"diode": "SMAJ400A", "dose_rate": 4.74e11, "bias_v": 0.0,  "imp_50ohm": 1},
    27287: {"diode": "SMAJ400A", "dose_rate": 6.25e11, "bias_v": 0.0,  "imp_50ohm": 0},
    27288: {"diode": "SMAJ400A", "dose_rate": 6.02e11, "bias_v": 5.25, "imp_50ohm": 1},
    27289: {"diode": "SMAJ400A", "dose_rate": 5.59e11, "bias_v": 5.25, "imp_50ohm": 0},

    27290: {"diode": "MMSZ5226BT1G", "dose_rate": 1.22e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27291: {"diode": "MMSZ5226BT1G", "dose_rate": 9.31e9,  "bias_v": 0.0,  "imp_50ohm": 0},
    27292: {"diode": "MMSZ5226BT1G", "dose_rate": 8.82e9,  "bias_v": 2.5,  "imp_50ohm": 1},
    27293: {"diode": "MMSZ5226BT1G", "dose_rate": 9.38e9,  "bias_v": 2.5,  "imp_50ohm": 0},
    27294: {"diode": "MMSZ5226BT1G", "dose_rate": 5.43e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27295: {"diode": "MMSZ5226BT1G", "dose_rate": 5.44e10, "bias_v": 0.0,  "imp_50ohm": 0},
    27296: {"diode": "MMSZ5226BT1G", "dose_rate": 4.81e11, "bias_v": 0.0,  "imp_50ohm": 1},
    27297: {"diode": "MMSZ5226BT1G", "dose_rate": 6.10e11, "bias_v": 0.0,  "imp_50ohm": 0},

    27298: {"diode": "CSD01060E", "dose_rate": 1.21e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27299: {"diode": "CSD01060E", "dose_rate": 1.05e10, "bias_v": 5.25, "imp_50ohm": 1},
    27300: {"diode": "CSD01060E", "dose_rate": 5.22e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27301: {"diode": "CSD01060E", "dose_rate": 5.46e11, "bias_v": 0.0,  "imp_50ohm": 1},
    27302: {"diode": "CSD01060E", "dose_rate": 8.88e11, "bias_v": 0.0,  "imp_50ohm": 1},
    27303: {"diode": "CSD01060E", "dose_rate": 1.01e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27304: {"diode": "CSD01060E", "dose_rate": 6.23e10, "bias_v": 0.0,  "imp_50ohm": 1},
    27305: {"diode": "CSD01060E", "dose_rate": 5.56e11, "bias_v": 0.0,  "imp_50ohm": 1},

    27306: {"diode": "SMAJ400A", "dose_rate": None, "bias_v": 0.0, "imp_50ohm": 1},
    27307: {"diode": "SMAJ400A", "dose_rate": None, "bias_v": 0.0, "imp_50ohm": 1},
}

def file_path(file_id: int) -> str:
    return os.path.join(DATA_DIR, f"{file_id}_data.csv")

def pct_interval(x: np.ndarray, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))

def standardize(train_vals: np.ndarray, all_vals: np.ndarray):
    m = float(np.mean(train_vals))
    s = float(np.std(train_vals, ddof=1))
    if (not np.isfinite(s)) or (s < 1e-12):
        s = 1.0
    return (all_vals - m) / s, m, s

def _needed_columns_from_header(csv_file: str):
    header = pd.read_csv(csv_file, nrows=0).columns.tolist()

    def time_before(col):
        j = header.index(col)
        if j <= 0:
            return None
        tcol = header[j - 1]
        if str(tcol).lower().startswith("time"):
            return tcol
        return None

    if "Diode" not in header:
        raise KeyError("Missing Diode column.")
    diode_tcol = time_before("Diode")
    if diode_tcol is None:
        raise KeyError("Missing time column before Diode.")

    pcd3_cols = [c for c in header if str(c).startswith("PCD3") and str(c) != "PCD3_B"]
    if not pcd3_cols:
        raise KeyError("No PCD3* columns found excluding PCD3_B.")
    pcd3_tcol = time_before(pcd3_cols[0])
    if pcd3_tcol is None:
        raise KeyError(f"Missing time column before {pcd3_cols[0]}.")

    return {"diode_tcol": diode_tcol, "pcd3_tcol": pcd3_tcol, "pcd3_cols": pcd3_cols}

def _stream_find_t_cross(csv_file: str, pcd3_tcol: str, pcd3_cols: list[str], threshold: float):
    last_t = None
    last_y = None

    for chunk in pd.read_csv(csv_file, usecols=[pcd3_tcol] + pcd3_cols, chunksize=CHUNKSIZE):
        t = chunk[pcd3_tcol].to_numpy(dtype=float)
        ys = [chunk[c].to_numpy(dtype=float) for c in pcd3_cols]
        y = np.mean(np.vstack(ys), axis=0)

        idx = np.where(y >= threshold)[0]
        if len(idx) == 0:
            last_t = float(t[-1]); last_y = float(y[-1])
            continue

        k = int(idx[0])
        tk = float(t[k]); yk = float(y[k])

        if k == 0:
            if last_t is not None and last_y is not None and last_y < threshold <= yk and abs(yk - last_y) > 1e-15:
                frac = (threshold - last_y) / (yk - last_y)
                return float(last_t + frac * (tk - last_t))
            return tk

        y0 = float(y[k - 1]); t0 = float(t[k - 1])
        if y0 < threshold <= yk and abs(yk - y0) > 1e-15:
            frac = (threshold - y0) / (yk - y0)
            return float(t0 + frac * (tk - t0))

        return tk

    raise RuntimeError(f"PCD3 avg never reached {threshold} V.")

def _baseline_median(v_pre: np.ndarray, v_fallback: np.ndarray) -> float:
    if v_pre.size >= 10:
        return float(np.median(v_pre))
    return float(np.median(v_fallback[: min(50, len(v_fallback))]))

def _polarity_flip_from_segment(seg: np.ndarray) -> float:
    seg = np.asarray(seg, float)
    return -1.0 if abs(np.min(seg)) > abs(np.max(seg)) else 1.0

def compute_window_len_from_reference(csv_file: str) -> float:
    cols = _needed_columns_from_header(csv_file)
    diode_tcol = cols["diode_tcol"]
    pcd3_tcol = cols["pcd3_tcol"]
    pcd3_cols = cols["pcd3_cols"]

    t_cross = _stream_find_t_cross(csv_file, pcd3_tcol, pcd3_cols, PCD3_THRESHOLD_V)
    t0 = float(t_cross - T0_OFFSET_NS * 1e-9)

    collect_lo = t0 - BASELINE_PRE_NS * 1e-9
    collect_hi = t_cross + max(PEAK_SEARCH_MAX_NS, TROUGH_SEARCH_MAX_NS) * 1e-9 + 200e-9

    t_all = []
    v_all = []
    for chunk in pd.read_csv(csv_file, usecols=[diode_tcol, "Diode"], chunksize=CHUNKSIZE):
        t = chunk[diode_tcol].to_numpy(float)
        v = chunk["Diode"].to_numpy(float)
        m = (t >= collect_lo) & (t <= collect_hi)
        if np.any(m):
            t_all.append(t[m]); v_all.append(v[m])
        if float(t[-1]) > (collect_hi + 50e-9):
            break
    if not t_all:
        raise RuntimeError("Could not collect reference diode samples for window sizing.")

    t_all = np.concatenate(t_all)
    v_all = np.concatenate(v_all)

    # baseline from pre region
    pre = (t_all >= (t0 - BASELINE_PRE_NS*1e-9)) & (t_all < t0)
    baseline = float(np.median(v_all[pre])) if np.sum(pre) >= 10 else float(np.median(v_all[:min(50,len(v_all))]))
    dv = v_all - baseline

    pol = (t_all >= t_cross) & (t_all <= t_cross + 120e-9)
    seg = dv[pol] if np.sum(pol) >= 3 else dv[: min(200, len(dv))]
    flip = _polarity_flip_from_segment(seg)
    x = flip * dv

    peak_mask = (t_all >= t_cross) & (t_all <= t_cross + PEAK_SEARCH_MAX_NS*1e-9)
    idx = np.where(peak_mask)[0]
    k_peak = int(idx[np.argmax(x[idx])])
    t_peak = float(t_all[k_peak])

    trough_mask = (t_all >= t_peak) & (t_all <= t_peak + TROUGH_SEARCH_MAX_NS*1e-9)
    idx2 = np.where(trough_mask)[0]
    idx2 = idx2[idx2 > k_peak] if np.any(idx2 > k_peak) else idx2
    k_trough = int(idx2[np.argmin(x[idx2])])
    t_trough = float(t_all[k_trough])

    window_end = float(t_trough + WINDOW_GUARD_NS*1e-9)
    window_len_ns = float((window_end - t0)*1e9)
    if window_len_ns <= 0:
        raise RuntimeError("Computed non-positive window length.")
    return window_len_ns

def stream_extract_peak_fixed_window(csv_file: str, window_len_ns: float):
    cols = _needed_columns_from_header(csv_file)
    diode_tcol = cols["diode_tcol"]
    pcd3_tcol = cols["pcd3_tcol"]
    pcd3_cols = cols["pcd3_cols"]

    t_cross = _stream_find_t_cross(csv_file, pcd3_tcol, pcd3_cols, PCD3_THRESHOLD_V)
    t0 = float(t_cross - T0_OFFSET_NS*1e-9)
    t1 = float(t0 + window_len_ns*1e-9)

    pre_lo = t0 - BASELINE_PRE_NS*1e-9
    pre_hi = t0
    pol_hi = t_cross + 120e-9

    v_pre, v_pol, v_win = [], [], []

    for chunk in pd.read_csv(csv_file, usecols=[diode_tcol, "Diode"], chunksize=CHUNKSIZE):
        t = chunk[diode_tcol].to_numpy(float)
        v = chunk["Diode"].to_numpy(float)

        m_pre = (t >= pre_lo) & (t < pre_hi)
        if np.any(m_pre): v_pre.append(v[m_pre])

        m_pol = (t >= t_cross) & (t <= pol_hi)
        if np.any(m_pol): v_pol.append(v[m_pol])

        m_win = (t >= t0) & (t <= t1)
        if np.any(m_win): v_win.append(v[m_win])

        if float(t[-1]) > (t1 + 50e-9):
            break

    if not v_win:
        raise RuntimeError("No diode samples found inside fixed window.")

    v_win = np.concatenate(v_win)
    v_pre_all = np.concatenate(v_pre) if v_pre else np.array([], float)
    baseline = _baseline_median(v_pre_all, v_win)

    dv_win = v_win - baseline
    seg = (np.concatenate(v_pol) - baseline) if v_pol else dv_win[: min(200, len(dv_win))]
    flip = _polarity_flip_from_segment(seg)

    x_win = flip * dv_win
    peak_mag = float(max(np.max(x_win), 0.0))
    return {"peak_mag": peak_mag}

def main():
    # uniform window
    ref_fp = file_path(REFERENCE_WINDOW_FILE_ID)
    window_len_ns = compute_window_len_from_reference(ref_fp)

    # extract
    rows = []
    for fid, meta in sorted(SHOT_META.items()):
        if fid in EXCLUDE_FILE_IDS: continue
        if meta.get("dose_rate") is None: continue
        fp = file_path(fid)
        if not os.path.exists(fp): continue

        try:
            peak = float(stream_extract_peak_fixed_window(fp, window_len_ns)["peak_mag"])
        except Exception as e:
            print(f"Skipping {fid}: {e}")
            continue

        if (not np.isfinite(peak)) or peak <= 0:
            print(f"Skipping {fid}: bad peak {peak}")
            continue

        rows.append({
            "file_id": fid,
            "diode": meta["diode"],
            "dose_rate": float(meta["dose_rate"]),
            "bias_v": float(meta["bias_v"]),
            "imp_50ohm": int(meta["imp_50ohm"]),
            "peak_mag": peak,
        })

    df_all = pd.DataFrame(rows).sort_values("file_id").reset_index(drop=True)
    print(f"\nAll shots extracted: {len(df_all)}")

    # within diode only
    target_diode = df_all.loc[df_all["file_id"] == PREDICT_FILE_ID, "diode"].iloc[0]
    df = df_all[df_all["diode"] == target_diode].copy().reset_index(drop=True)
    print(f"Shots used for diode {target_diode}: {len(df)}")

    test_mask = (df["file_id"].to_numpy(int) == PREDICT_FILE_ID)
    train_mask = ~test_mask
    print(f"Train shots: {int(train_mask.sum())}")
    print(f"Held-out: {PREDICT_FILE_ID}")

    # predictors
    dose = np.log10(df["dose_rate"].to_numpy(float))
    bias = df["bias_v"].to_numpy(float)
    imp = df["imp_50ohm"].to_numpy(int)          # 0/1
    imp_idx = imp.astype(int)                   # 0 or 1 (group)

    dose_z, _, _ = standardize(dose[train_mask], dose)
    bias_z, _, _ = standardize(bias[train_mask], bias)
    dose2 = dose_z**2

    # target in log space
    y = df["peak_mag"].to_numpy(float)
    y_log = np.log(np.maximum(y, 1e-12))
    y_z, y_m, y_s = standardize(y_log[train_mask], y_log)

    # train
    y_train = y_z[train_mask]
    dose_train = dose_z[train_mask]
    dose2_train = dose2[train_mask]
    bias_train = bias_z[train_mask]
    imp_train_idx = imp_idx[train_mask]

    # test
    dose_test = float(dose_z[test_mask][0])
    dose2_test = float(dose2[test_mask][0])
    bias_test = float(bias_z[test_mask][0])
    imp_test_idx = int(imp_idx[test_mask][0])
    y_actual = float(y[test_mask][0])

    # MODEL: impedance is a GROUP intercept + group noise
    with pm.Model() as model:
        alpha = pm.Normal("alpha", 0.0, 1.0)

        # group intercepts for imp=0 and imp=1
        sigma_imp = pm.HalfNormal("sigma_imp", 0.8)
        imp_raw = pm.Normal("imp_raw", 0.0, 1.0, shape=2)
        a_imp = pm.Deterministic("a_imp", imp_raw * sigma_imp)

        # allow different noise per impedance setup (helps intervals)
        sigma0 = pm.HalfNormal("sigma0", 0.6)
        sigma1 = pm.HalfNormal("sigma1", 0.6)
        sigma_by_imp = pm.Deterministic("sigma_by_imp", pm.math.stack([sigma0, sigma1]))

        # dose/bias model with mild flexibility
        b_dose = pm.Normal("b_dose", 0.0, 0.9)
        b_dose2 = pm.Normal("b_dose2", 0.0, 0.5)
        b_bias = pm.Normal("b_bias", 0.0, 0.9)

        # interactions
        b_dose_bias = pm.Normal("b_dose_bias", 0.0, 0.6)
        # let dose interact with impedance group too (critical)
        b_dose_imp = pm.Normal("b_dose_imp", 0.0, 0.6)

        nu = pm.Exponential("nu_minus_two", 1/10) + 2.0

        mu = (
            alpha
            + a_imp[imp_train_idx]
            + b_dose * dose_train
            + b_dose2 * dose2_train
            + b_bias * bias_train
            + b_dose_bias * (dose_train * bias_train)
            + b_dose_imp * (dose_train * (imp_train_idx.astype(float)))
        )

        pm.StudentT(
            "obs",
            nu=nu,
            mu=mu,
            sigma=sigma_by_imp[imp_train_idx],
            observed=y_train,
        )

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            progressbar=True,
        )

    post = trace.posterior
    rng = np.random.default_rng(SEED)

    def flat(name):
        return post[name].values.reshape(-1)

    alpha_s = flat("alpha")
    a_imp_s = post["a_imp"].values.reshape(-1, 2)
    b_dose_s = flat("b_dose")
    b_dose2_s = flat("b_dose2")
    b_bias_s = flat("b_bias")
    b_dose_bias_s = flat("b_dose_bias")
    b_dose_imp_s = flat("b_dose_imp")
    nu_s = flat("nu_minus_two") + 2.0
    sigma_by_imp_s = post["sigma_by_imp"].values.reshape(-1, 2)

    # mean (mu) draws for test
    mu_test = (
        alpha_s
        + a_imp_s[:, imp_test_idx]
        + b_dose_s * dose_test
        + b_dose2_s * dose2_test
        + b_bias_s * bias_test
        + b_dose_bias_s * (dose_test * bias_test)
        + b_dose_imp_s * (dose_test * float(imp_test_idx))
    )

    # 95% CI of mean response (tight)
    ylog_mean_draws = y_m + y_s * mu_test
    y_mean_draws = np.exp(ylog_mean_draws)
    ci_lo, ci_hi = pct_interval(y_mean_draws)

    # 95% PI of observation (includes noise)
    sigma_test = sigma_by_imp_s[:, imp_test_idx]
    z_obs = mu_test + sigma_test * rng.standard_t(df=nu_s)
    ylog_obs = y_m + y_s * z_obs
    y_obs = np.exp(ylog_obs)
    pi_lo, pi_hi = pct_interval(y_obs)

    pred_mean = float(np.mean(y_mean_draws))

    print("\n====================")
    print("WITHIN-DIODE + IMPEDANCE-GROUP MODEL")
    print("====================")
    print(f"Uniform window length (ns): {window_len_ns:.3f}")
    print(f"Actual extracted peak: {y_actual:.6f} V")
    print(f"\nPredicted mean (E[peak]): {pred_mean:.6f} V")
    print(f"95% CI of mean (model uncertainty): [{ci_lo:.6f}, {ci_hi:.6f}] V")
    print(f"95% PI of observation (includes noise): [{pi_lo:.6f}, {pi_hi:.6f}] V")

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()
