# peak_voltage_metadata_only.py
#
# Bayesian MCMC (PyMC NUTS) model for diode FIRST-LOBE PEAK VOLTAGE MAGNITUDE
# Metadata-only predictors:
# - dose rate (log10 + quadratic)
# - bias voltage
# - impedance flag
# - diode descriptors (v_nom, cj_pf)
#
# Target is still extracted from waveform (PCB-anchored first-lobe peak)
# but NO waveform-shape predictors are used.
#
# This makes the model a cleaner "predict from conditions + diode properties" model.

import os
os.environ["PYTENSOR_FLAGS"] = "cxx=,linker=py,mode=FAST_COMPILE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import pymc as pm

# -----------------------------
# SETTINGS
# -----------------------------
DATA_DIR = "data"

TEST_FILES = [27297, 27305, 27307]
EXCLUDE_FILE_IDS = {27270, 27281}

SEED = 42
CHAINS = 2
CORES = 1
DRAWS = 800
TUNE = 800
TARGET_ACCEPT = 0.95

MAD_Z_MAX = 4.0

# PCB / first-lobe extraction settings (used only to get target peak_mag)
PRE_BASELINE_NS = 20.0
SEARCH_END_NS = 120.0
POLARITY_DECIDE_NS = 60.0
RETURN_FRAC = 0.20
MIN_SUSTAIN = 3

# -----------------------------
# COMPACT DIODE DESCRIPTORS
# -----------------------------
DIODE_VPARAMS = {
    "SMAJ400A": {
        "v_nom": 447.0,
        "cj_pf": 12.0,
    },
    "MMSZ5226BT1G": {
        "v_nom": 3.3,
        "cj_pf": 280.0,
    },
    "CSD01060E": {
        "v_nom": 600.0,
        "cj_pf": 80.0,
    },
}

# -----------------------------
# SHOT METADATA
# -----------------------------
SHOT_META = {
    27271: {"diode": "SMAJ400A", "dose_rate": 1.28e10, "bias_v": 0.0, "imp_50ohm": 0},
    27272: {"diode": "SMAJ400A", "dose_rate": 1.28e10, "bias_v": 0.0, "imp_50ohm": 1},
    27273: {"diode": "SMAJ400A", "dose_rate": 1.25e10, "bias_v": 5.25, "imp_50ohm": 1},
    27274: {"diode": "SMAJ400A", "dose_rate": 1.19e10, "bias_v": 5.25, "imp_50ohm": 0},
    27275: {"diode": "SMAJ400A", "dose_rate": 1.10e10, "bias_v": 5.25, "imp_50ohm": 1},
    27276: {"diode": "SMAJ400A", "dose_rate": 1.40e10, "bias_v": 0.0, "imp_50ohm": 1},
    27277: {"diode": "SMAJ400A", "dose_rate": 1.06e10, "bias_v": 0.0, "imp_50ohm": 1},
    27278: {"diode": "SMAJ400A", "dose_rate": 1.16e10, "bias_v": 0.0, "imp_50ohm": 1},
    27279: {"diode": "SMAJ400A", "dose_rate": 6.22e10, "bias_v": 0.0, "imp_50ohm": 1},
    27280: {"diode": "SMAJ400A", "dose_rate": 5.63e10, "bias_v": 0.0, "imp_50ohm": 0},
    27282: {"diode": "SMAJ400A", "dose_rate": 6.41e10, "bias_v": 0.0, "imp_50ohm": 0},
    27283: {"diode": "SMAJ400A", "dose_rate": 8.11e10, "bias_v": 0.0, "imp_50ohm": 0},
    27284: {"diode": "SMAJ400A", "dose_rate": 5.11e10, "bias_v": 5.25, "imp_50ohm": 1},
    27285: {"diode": "SMAJ400A", "dose_rate": 4.86e10, "bias_v": 5.25, "imp_50ohm": 0},
    27286: {"diode": "SMAJ400A", "dose_rate": 4.74e11, "bias_v": 0.0, "imp_50ohm": 1},
    27287: {"diode": "SMAJ400A", "dose_rate": 6.25e11, "bias_v": 0.0, "imp_50ohm": 0},
    27288: {"diode": "SMAJ400A", "dose_rate": 6.02e11, "bias_v": 5.25, "imp_50ohm": 1},
    27289: {"diode": "SMAJ400A", "dose_rate": 5.59e11, "bias_v": 5.25, "imp_50ohm": 0},
    27290: {"diode": "MMSZ5226BT1G", "dose_rate": 1.22e10, "bias_v": 0.0, "imp_50ohm": 1},
    27291: {"diode": "MMSZ5226BT1G", "dose_rate": 9.31e9, "bias_v": 0.0, "imp_50ohm": 0},
    27292: {"diode": "MMSZ5226BT1G", "dose_rate": 8.82e9, "bias_v": 2.5, "imp_50ohm": 1},
    27293: {"diode": "MMSZ5226BT1G", "dose_rate": 9.38e9, "bias_v": 2.5, "imp_50ohm": 0},
    27294: {"diode": "MMSZ5226BT1G", "dose_rate": 5.43e10, "bias_v": 0.0, "imp_50ohm": 1},
    27295: {"diode": "MMSZ5226BT1G", "dose_rate": 5.44e10, "bias_v": 0.0, "imp_50ohm": 0},
    27296: {"diode": "MMSZ5226BT1G", "dose_rate": 4.81e11, "bias_v": 0.0, "imp_50ohm": 1},
    27297: {"diode": "MMSZ5226BT1G", "dose_rate": 6.10e11, "bias_v": 0.0, "imp_50ohm": 0},
    27298: {"diode": "CSD01060E", "dose_rate": 1.21e10, "bias_v": 0.0, "imp_50ohm": 1},
    27299: {"diode": "CSD01060E", "dose_rate": 1.05e10, "bias_v": 5.25, "imp_50ohm": 1},
    27300: {"diode": "CSD01060E", "dose_rate": 5.22e10, "bias_v": 0.0, "imp_50ohm": 1},
    27301: {"diode": "CSD01060E", "dose_rate": 5.46e11, "bias_v": 0.0, "imp_50ohm": 1},
    27302: {"diode": "CSD01060E", "dose_rate": 8.88e11, "bias_v": 0.0, "imp_50ohm": 1},
    27303: {"diode": "CSD01060E", "dose_rate": 1.01e10, "bias_v": 0.0, "imp_50ohm": 1},
    27304: {"diode": "CSD01060E", "dose_rate": 6.23e10, "bias_v": 0.0, "imp_50ohm": 1},
    27305: {"diode": "CSD01060E", "dose_rate": 5.56e11, "bias_v": 0.0, "imp_50ohm": 1},
    27306: {"diode": "SMAJ400A", "dose_rate": None, "bias_v": 0.0, "imp_50ohm": 1},
    27307: {"diode": "SMAJ400A", "dose_rate": None, "bias_v": 0.0, "imp_50ohm": 1},
}

def file_path(file_id: int) -> str:
    return os.path.join(DATA_DIR, f"{file_id}_data.csv")

def _get_time_col_before(df: pd.DataFrame, col_name: str):
    cols = list(df.columns)
    j = cols.index(col_name)
    if j <= 0:
        return None
    tcol = cols[j - 1]
    if str(tcol).lower().startswith("time"):
        return df[tcol].to_numpy(dtype=float)
    return None

def _load_pcb_avg(df: pd.DataFrame):
    pairs = []
    for name in ["PCD1", "PCD2", "PCD3"]:
        if name in df.columns:
            t = _get_time_col_before(df, name)
            if t is None:
                continue
            y = df[name].to_numpy(dtype=float)
            pairs.append((t, y))
    if not pairs:
        raise KeyError("No PCD1/PCD2/PCD3 columns found.")
    t_ref = pairs[0][0]
    ys = []
    for t, y in pairs:
        if len(t) == len(t_ref) and np.allclose(t, t_ref, rtol=0, atol=0):
            ys.append(y)
        else:
            ys.append(np.interp(t_ref, t, y))
    return t_ref, np.mean(np.vstack(ys), axis=0)

def extract_first_lobe_peak_mag(csv_file: str):
    """
    Returns only target-related quantities.
    Uses PCB-anchored first-lobe extraction and returns peak magnitude.
    """
    df = pd.read_csv(csv_file)
    if "Diode" not in df.columns:
        raise KeyError("Missing Diode column")

    t_d = _get_time_col_before(df, "Diode")
    if t_d is None:
        raise KeyError("No time column found before Diode")
    v_d = df["Diode"].to_numpy(dtype=float)

    # PCB event time = largest |avg(PCD1,PCD2,PCD3)|
    t_p, pcb = _load_pcb_avg(df)
    t0 = float(t_p[int(np.argmax(np.abs(pcb)))])

    # Baseline
    pre_mask = t_d < (t0 - PRE_BASELINE_NS * 1e-9)
    if np.sum(pre_mask) < 10:
        pre_mask = t_d < t0
    if np.sum(pre_mask) < 5:
        pre_mask = np.arange(len(t_d)) < min(50, len(t_d))

    baseline = float(np.median(v_d[pre_mask]))
    dv = v_d - baseline

    # Polarity orientation
    pol_mask = (t_d >= t0) & (t_d <= t0 + POLARITY_DECIDE_NS * 1e-9)
    if np.sum(pol_mask) < 3:
        raise RuntimeError("Insufficient samples in polarity window.")
    seg = dv[pol_mask]
    flip = -1.0 if abs(np.min(seg)) > abs(np.max(seg)) else 1.0
    x = flip * dv

    # First-lobe search
    search_mask = (t_d >= t0) & (t_d <= t0 + SEARCH_END_NS * 1e-9)
    idx_search = np.where(search_mask)[0]
    if len(idx_search) < 5:
        raise RuntimeError("Insufficient samples in first-lobe search window.")

    x_search = x[idx_search]
    k_peak = int(idx_search[np.argmax(x_search)])
    peak_mag = float(max(x[k_peak], 0.0))
    peak_delay_ns = float((t_d[k_peak] - t0) * 1e9)

    return {
        "peak_mag": peak_mag,
        "peak_delay_ns": peak_delay_ns,
    }

def mad_mask(y: np.ndarray, zmax: float) -> np.ndarray:
    y = np.asarray(y, dtype=float)
    med = float(np.median(y))
    mad = float(np.median(np.abs(y - med))) + 1e-12
    z = np.abs(y - med) / (1.4826 * mad)
    return z <= zmax

def pct_interval(x: np.ndarray, lo=2.5, hi=97.5):
    return float(np.percentile(x, lo)), float(np.percentile(x, hi))

def standardize(train_vals: np.ndarray, all_vals: np.ndarray):
    m = float(np.mean(train_vals))
    s = float(np.std(train_vals, ddof=1))
    if (not np.isfinite(s)) or (s < 1e-12):
        s = 1.0
    return (all_vals - m) / s, m, s

def main():
    rows = []

    for fid in sorted(SHOT_META.keys()):
        if fid in EXCLUDE_FILE_IDS:
            continue

        meta = SHOT_META[fid]
        if meta["dose_rate"] is None:
            continue  # drop missing-dose shots for metadata-only model

        diode = meta["diode"]
        if diode not in DIODE_VPARAMS:
            continue

        fp = file_path(fid)
        if not os.path.exists(fp):
            continue

        try:
            feats = extract_first_lobe_peak_mag(fp)
        except Exception as e:
            print(f"Skipping {fid}: {e}")
            continue

        rows.append({
            "file_id": fid,
            "diode": diode,
            "dose_rate": float(meta["dose_rate"]),
            "bias_v": float(meta["bias_v"]),
            "imp_50ohm": int(meta["imp_50ohm"]),
            "v_nom": float(DIODE_VPARAMS[diode]["v_nom"]),
            "cj_pf": float(DIODE_VPARAMS[diode]["cj_pf"]),
            "peak_mag": float(feats["peak_mag"]),
            "peak_delay_ns": float(feats["peak_delay_ns"]),  # just for printing diagnostics
            "is_test": fid in TEST_FILES,
        })

    if len(rows) < 12:
        raise RuntimeError(f"Not enough valid rows. Got {len(rows)}.")

    df = pd.DataFrame(rows)
    df["is_train"] = ~df["is_test"]

    present_tests = set(df.loc[df["is_test"], "file_id"].tolist())
    missing_tests = [t for t in TEST_FILES if t not in present_tests]
    if missing_tests:
        print(f"Dropped test files with missing dose_rate or missing file: {missing_tests}")

    print("\nTraining rows (metadata-only predictors, target from waveform):")
    for _, r in df[df["is_train"]].sort_values("file_id").iterrows():
        print(
            f"{int(r['file_id'])} | {r['diode']:<12} | "
            f"dose={r['dose_rate']:.3e} | bias={r['bias_v']:.2f} V | imp50={int(r['imp_50ohm'])} | "
            f"Vnom={r['v_nom']:.2f} V | Cj={r['cj_pf']:.2f} pF | "
            f"peak={r['peak_mag']:.6f} V | tpk={r['peak_delay_ns']:.3f} ns"
        )

    # Robust filter on training target only
    train_idx = df.index[df["is_train"]].to_numpy()
    keep_train = mad_mask(df.loc[train_idx, "peak_mag"].to_numpy(float), MAD_Z_MAX)
    keep_idx = set(train_idx[keep_train].tolist()) | set(df.index[df["is_test"]].tolist())
    df = df.loc[sorted(keep_idx)].reset_index(drop=True)

    train_mask = df["is_train"].to_numpy(bool)
    if train_mask.sum() < 10:
        raise RuntimeError("Too few training rows after MAD filtering.")

    # Diode indexing
    diode_names = sorted(df["diode"].unique().tolist())
    diode_to_idx = {d: i for i, d in enumerate(diode_names)}
    df["diode_idx"] = df["diode"].map(diode_to_idx).astype(int)
    n_diodes = len(diode_names)

    # -----------------------------
    # Metadata-only predictors
    # -----------------------------
    x_dose_all = np.log10(df["dose_rate"].to_numpy(float))
    x_dose_z, x_dose_m, x_dose_s = standardize(x_dose_all[train_mask], x_dose_all)
    x_dose2_all = x_dose_z ** 2

    x_bias_all = df["bias_v"].to_numpy(float)
    x_bias_z, x_bias_m, x_bias_s = standardize(x_bias_all[train_mask], x_bias_all)

    x_imp_all = df["imp_50ohm"].to_numpy(float)  # binary, leave unscaled

    x_vnom_all = np.log10(np.maximum(df["v_nom"].to_numpy(float), 1e-9))
    x_vnom_z, x_vnom_m, x_vnom_s = standardize(x_vnom_all[train_mask], x_vnom_all)

    x_cj_all = np.log10(np.maximum(df["cj_pf"].to_numpy(float), 1e-9))
    x_cj_z, x_cj_m, x_cj_s = standardize(x_cj_all[train_mask], x_cj_all)

    # Target = log peak magnitude
    y_all = df["peak_mag"].to_numpy(float)
    y_log_all = np.log(np.maximum(y_all, 1e-12))
    y_z_all, y_m, y_s = standardize(y_log_all[train_mask], y_log_all)

    # Train slices
    y_train = y_z_all[train_mask]
    dose_train = x_dose_z[train_mask]
    dose2_train = x_dose2_all[train_mask]
    bias_train = x_bias_z[train_mask]
    imp_train = x_imp_all[train_mask]
    vnom_train = x_vnom_z[train_mask]
    cj_train = x_cj_z[train_mask]

    diode_idx_all = df["diode_idx"].to_numpy(int)
    diode_idx_train = diode_idx_all[train_mask]

    # -----------------------------
    # Bayesian robust regression
    # -----------------------------
    with pm.Model() as model:
        # Global intercept
        alpha = pm.Normal("alpha", 0.0, 0.7)

        # Hierarchical diode intercepts
        sigma_diode = pm.HalfNormal("sigma_diode", 0.5)
        diode_offset_raw = pm.Normal("diode_offset_raw", 0.0, 1.0, shape=n_diodes)
        diode_offset = pm.Deterministic("diode_offset", diode_offset_raw * sigma_diode)

        # Hierarchical diode dose slopes
        b_dose_mu = pm.Normal("b_dose_mu", 0.0, 0.5)
        sigma_b_dose = pm.HalfNormal("sigma_b_dose", 0.35)
        b_dose_raw = pm.Normal("b_dose_raw", 0.0, 1.0, shape=n_diodes)
        b_dose_diode = pm.Deterministic("b_dose_diode", b_dose_mu + b_dose_raw * sigma_b_dose)

        # Hierarchical diode impedance slopes
        b_imp_mu = pm.Normal("b_imp_mu", 0.0, 0.35)
        sigma_b_imp = pm.HalfNormal("sigma_b_imp", 0.25)
        b_imp_raw = pm.Normal("b_imp_raw", 0.0, 1.0, shape=n_diodes)
        b_imp_diode = pm.Deterministic("b_imp_diode", b_imp_mu + b_imp_raw * sigma_b_imp)

        # Global fixed effects
        b_dose2 = pm.Normal("b_dose2", 0.0, 0.20)
        b_bias = pm.Normal("b_bias", 0.0, 0.35)
        b_vnom = pm.Normal("b_vnom", 0.0, 0.30)
        b_cj = pm.Normal("b_cj", 0.0, 0.30)

        sigma = pm.HalfNormal("sigma", 0.45)
        nu = pm.Exponential("nu_minus_two", 1 / 12) + 2.0

        mu = (
            alpha
            + diode_offset[diode_idx_train]
            + b_dose_diode[diode_idx_train] * dose_train
            + b_dose2 * dose2_train
            + b_bias * bias_train
            + b_imp_diode[diode_idx_train] * imp_train
            + b_vnom * vnom_train
            + b_cj * cj_train
        )

        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y_train)

        trace = pm.sample(
            draws=DRAWS,
            tune=TUNE,
            chains=CHAINS,
            cores=CORES,
            target_accept=TARGET_ACCEPT,
            random_seed=SEED,
            progressbar=True,
        )

    # -----------------------------
    # Flatten posterior
    # -----------------------------
    post = trace.posterior

    alpha_s = post["alpha"].values.reshape(-1)
    diode_offset_s = post["diode_offset"].values.reshape(-1, n_diodes)
    b_dose_diode_s = post["b_dose_diode"].values.reshape(-1, n_diodes)
    b_imp_diode_s = post["b_imp_diode"].values.reshape(-1, n_diodes)

    b_dose2_s = post["b_dose2"].values.reshape(-1)
    b_bias_s = post["b_bias"].values.reshape(-1)
    b_vnom_s = post["b_vnom"].values.reshape(-1)
    b_cj_s = post["b_cj"].values.reshape(-1)

    sigma_s = post["sigma"].values.reshape(-1)
    nu_s = post["nu_minus_two"].values.reshape(-1) + 2.0

    rng = np.random.default_rng(SEED)

    # -----------------------------
    # Predict test rows
    # -----------------------------
    test_rows = df[df["is_test"]].sort_values("file_id")
    if len(test_rows) == 0:
        print("\nNo test rows available after dropping missing-dose rows.")
        return

    print("\nPredicted vs actual (test files):")
    for _, r in test_rows.iterrows():
        j = int(r.name)
        d_idx = int(r["diode_idx"])

        dose_j = float(x_dose_z[j])
        dose2_j = float(x_dose2_all[j])
        bias_j = float(x_bias_z[j])
        imp_j = float(x_imp_all[j])
        vnom_j = float(x_vnom_z[j])
        cj_j = float(x_cj_z[j])

        mu_new = (
            alpha_s
            + diode_offset_s[:, d_idx]
            + b_dose_diode_s[:, d_idx] * dose_j
            + b_dose2_s * dose2_j
            + b_bias_s * bias_j
            + b_imp_diode_s[:, d_idx] * imp_j
            + b_vnom_s * vnom_j
            + b_cj_s * cj_j
        )

        # Posterior predictive in standardized log-space -> original volts
        z_new = mu_new + sigma_s * rng.standard_t(df=nu_s)
        y_pred_draws = np.exp(y_m + y_s * z_new)

        pred_mean = float(np.mean(y_pred_draws))
        pred_lo, pred_hi = pct_interval(y_pred_draws)

        actual = float(r["peak_mag"])
        err = y_pred_draws - actual
        err_mean = float(np.mean(err))
        err_lo, err_hi = pct_interval(err)

        print(
            f"{int(r['file_id'])} | actual_peak={actual:.6f} V | "
            f"pred_peak_mean={pred_mean:.6f} V | "
            f"pred_95%PI=[{pred_lo:.6f}, {pred_hi:.6f}] V | "
            f"err_mean(pred-actual)={err_mean:.6f} V | "
            f"err_95%=[{err_lo:.6f}, {err_hi:.6f}] V"
        )

    # Optional: print coefficient summaries (posterior means)
    print("\nPosterior mean coefficients (global):")
    print(f"alpha     = {alpha_s.mean(): .4f}")
    print(f"b_dose2   = {b_dose2_s.mean(): .4f}")
    print(f"b_bias    = {b_bias_s.mean(): .4f}")
    print(f"b_vnom    = {b_vnom_s.mean(): .4f}")
    print(f"b_cj      = {b_cj_s.mean(): .4f}")
    print(f"sigma     = {sigma_s.mean(): .4f}")
    print(f"nu        = {nu_s.mean(): .4f}")

    print("\nDiode-specific posterior mean slopes:")
    for d, i in diode_to_idx.items():
        print(
            f"{d:<12} | dose_slope={b_dose_diode_s[:, i].mean(): .4f} | "
            f"imp_slope={b_imp_diode_s[:, i].mean(): .4f} | "
            f"offset={diode_offset_s[:, i].mean(): .4f}"
        )

if __name__ == "__main__":
    from multiprocessing import freeze_support
    freeze_support()
    main()


    #needs to have random test left out instead of last 2
    #implement 2/24
    