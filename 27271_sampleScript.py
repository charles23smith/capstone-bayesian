"""
27271_autostop_bayes_final.py

Bayesian diode waveform fit around FIRST negative peak with AUTO-STOP on slope reversal.

What it does:
1) Load CSV (col0=time, col1=volts)
2) Auto-detect time units (s/us/ns) and convert to SECONDS
3) Find FIRST significant peak in y-domain (y = -V if NEGATE=True)  -> corresponds to FIRST negative peak in V
4) Window forward up to MAX_WINDOW_NS
5) Auto-stop when the recovery slope (in V-domain) turns negative for K consecutive points (next lobe/shot)
6) Fit delayed double-exponential recovery in y-domain:
      y(t) = b + A1 + A2                                      for t < d
           = b + A1*exp(-(t-d)/tau1) + A2*exp(-(t-d)/tau2)     for t >= d
   (Original voltage: V(t) = -y(t) if NEGATE=True)
7) Run Bayesian sampling:
   - Prefer NumPyro NUTS if JAX is installed
   - Otherwise fall back to ADVI (still gives uncertainty bands)
8) Compute:
   - Credible intervals on mean waveform (50% and 95%)
   - Predictive intervals including observation noise (50% and 95%)
   - Robust plotting that DOES NOT “flatten” the data
9) Save plots in ./diode_fit_outputs and print Xyce-ready expression to console.

Fixes your earlier issues:
- Windows multiprocessing safe entry point
- Avoids “plot looks like 0V” by:
   (a) computing intervals in V-domain directly
   (b) dropping non-finite posterior draws
   (c) y-limits locked to data range

Outputs:
  diode_fit_outputs/00_peak_window_full.png
  diode_fit_outputs/01_window_autostop.png
  diode_fit_outputs/01b_init_fit_curvefit.png
  diode_fit_outputs/02_fit_bayes_delayed_doubleexp.png
  diode_fit_outputs/03_residuals.png
  diode_fit_outputs/debug_intervals_lines.png  (optional extra sanity)
"""

# -----------------------------
# MUST BE FIRST (before importing pymc/pytensor)
# Prevent PyTensor from trying to compile lazylinker with a broken MinGW toolchain.
# NumPyro/JAX sampling does not rely on PyTensor C compilation in the same way.
# -----------------------------
import os
os.environ["PYTENSOR_FLAGS"] = "cxx=,linker=py,mode=FAST_COMPILE"
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"

import numpy as np
import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.signal import savgol_filter, find_peaks
from scipy.optimize import curve_fit

import pymc as pm
import arviz as az


# -----------------------------
# USER SETTINGS
# -----------------------------
CSV_FILE = "27271_data.csv"     # your csv (time, volts)
OUT_DIR  = "diode_fit_outputs"

NEGATE = True                   # fit y=-V if True (recommended for first negative peak)
SEED = 42

# Peak finding (first significant peak in y-domain)
SMOOTH_NS_PEAK = 2.0
PEAK_PROM_FRAC = 0.08           # try 0.03–0.15 if peak pick fails

# Auto-stop (stop when recovery slope flips negative)
MAX_WINDOW_NS   = 300.0
MIN_STOP_NS     = 25.0
SMOOTH_NS_SLOPE = 3.0
K_SUSTAIN       = 10

# Fit subsampling
SUBSAMPLE_N = 400               # points used in Bayesian fit (keep <= 600)

# Bayesian sampling defaults
CHAINS = 2
CORES  = 1                      # keep 1 on Windows
DRAWS  = 1200
TUNE   = 1200
TARGET_ACCEPT = 0.92

# If JAX not installed, fallback to ADVI
ADVI_ITERS = 60000
ADVI_DRAWS = 8000


# -----------------------------
# Numpy helper: delayed double-exp
# -----------------------------
def delayed_doubleexp_np(t, b, A1, tau1, A2, tau2, d):
    t = np.asarray(t, dtype=float)
    y = np.empty_like(t)
    pre = t < d
    y[pre] = b + A1 + A2
    td = t[~pre] - d
    y[~pre] = b + A1*np.exp(-td/tau1) + A2*np.exp(-td/tau2)
    return y


def _oddify(n: int) -> int:
    return n if (n % 2 == 1) else (n + 1)


def _rng(a):
    a = np.asarray(a)
    return float(np.nanmin(a)), float(np.nanmax(a))


def autodetect_time_to_seconds(time_raw: np.ndarray) -> np.ndarray:
    """
    Heuristic conversion to seconds if CSV time column is in us or ns.
    Prints what it decided.
    """
    time = time_raw.astype(float).copy()
    dtmed_raw = float(np.median(np.diff(time)))
    tspan_raw = float(time.max() - time.min())

    print("\nTIME SANITY (raw CSV):")
    print("  first 5 time values:", time[:5])
    print(f"  median dt raw:  {dtmed_raw:.6e}")
    print(f"  span raw:       {tspan_raw:.6e}")

    # Heuristics:
    # - microseconds: dt ~ 1e-5..1e-1, span ~ 1e-3..1e3  (numbers like -1..1)
    if (1e-5 < dtmed_raw < 1e-1) and (1e-3 < tspan_raw < 1e3):
        print("Detected time likely in microseconds. Converting us -> s.")
        time *= 1e-6

    # - nanoseconds as integers: dt ~ 0.5..5, span huge
    elif (0.5 < dtmed_raw < 5.0) and (tspan_raw > 1e3):
        print("Detected time likely in nanoseconds. Converting ns -> s.")
        time *= 1e-9
    else:
        print("Assuming time is already in seconds (no conversion).")

    dtmed = float(np.median(np.diff(time)))
    tspan = float(time.max() - time.min())
    print("\nTIME SANITY (seconds-assumed):")
    print(f"  median dt (s): {dtmed:.6e}")
    print(f"  span (s):      {tspan:.6e}")
    return time


def try_import_jax():
    try:
        import jax  # noqa: F401
        return True
    except Exception:
        return False


def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    # -----------------------------
    # Load data
    # -----------------------------
    data = pd.read_csv(CSV_FILE)
    time_raw = data.iloc[:, 0].to_numpy(dtype=float)
    vraw = data.iloc[:, 1].to_numpy(dtype=float)

    time = autodetect_time_to_seconds(time_raw)

    # y-domain for peak selection / fitting
    ysig = -vraw if NEGATE else vraw

    print(f"\nLoaded {len(time)} points from {CSV_FILE}")
    print(f"Voltage range (after NEGATE={NEGATE}): {ysig.min():.4f} to {ysig.max():.4f} V")

    # -----------------------------
    # STEP 1: Find FIRST significant peak in y-domain
    # -----------------------------
    dt = float(np.median(np.diff(time)))
    win_peak = int(max(9, round((SMOOTH_NS_PEAK * 1e-9) / dt)))
    win_peak = _oddify(win_peak)
    win_peak = min(win_peak, max(9, (len(ysig)//2)*2 - 1))
    if win_peak % 2 == 0:
        win_peak -= 1

    y_smooth = savgol_filter(ysig, window_length=win_peak, polyorder=3)

    yrange = float(y_smooth.max() - y_smooth.min())
    prom = max(1e-12, PEAK_PROM_FRAC * yrange)

    peaks, props = find_peaks(y_smooth, prominence=prom)
    if len(peaks) == 0:
        raise RuntimeError("No peaks found. Lower PEAK_PROM_FRAC or check NEGATE/time units.")

    peak_idx = int(peaks[0])
    peak_time = float(time[peak_idx])

    print("\nChosen FIRST significant peak:")
    print(f"  idx = {peak_idx}")
    print(f"  t0  = {peak_time*1e6:.5f} us")
    print(f"  V(t0) original = {vraw[peak_idx]:.4f} V")
    print(f"  y(t0) (negated) = {ysig[peak_idx]:.4f} V")

    # -----------------------------
    # STEP 2: Candidate window forward from peak
    # -----------------------------
    t_end = peak_time + MAX_WINDOW_NS * 1e-9
    mask = (time >= peak_time) & (time <= t_end)

    tw_full = time[mask] - peak_time
    y_full = ysig[mask]
    V_full = (-y_full) if NEGATE else y_full

    if len(tw_full) < 80:
        raise RuntimeError("Not enough points after the peak. Increase MAX_WINDOW_NS or check units.")

    # -----------------------------
    # STEP 3: AUTO-STOP when slope turns negative (recovery ends / next lobe begins)
    # Recovery in V-domain should have positive slope (V rising toward baseline).
    # -----------------------------
    dtw = float(np.median(np.diff(tw_full)))
    win_slope = int(max(9, round((SMOOTH_NS_SLOPE * 1e-9) / dtw)))
    win_slope = _oddify(win_slope)
    win_slope = min(win_slope, max(9, (len(V_full)//2)*2 - 1))
    if win_slope % 2 == 0:
        win_slope -= 1

    V_s = savgol_filter(V_full, window_length=win_slope, polyorder=3)
    dVdt = np.gradient(V_s, tw_full)

    min_i = int(np.searchsorted(tw_full, MIN_STOP_NS * 1e-9))
    neg = dVdt < 0

    stop_idx = len(tw_full) - 1
    for i in range(min_i, len(tw_full) - K_SUSTAIN):
        if np.all(neg[i:i + K_SUSTAIN]):
            stop_idx = i
            break

    t_cut = float(tw_full[stop_idx])
    print(f"\nAuto-stop cutoff at t = {t_cut*1e9:.2f} ns (slope turns negative → next lobe/shot)")

    tw_seg = tw_full[:stop_idx + 1]
    y_seg = y_full[:stop_idx + 1]
    V_seg = V_full[:stop_idx + 1]

    print(f"Final chosen segment: {len(tw_seg)} points, duration {tw_seg[-1]*1e9:.2f} ns")

    # -----------------------------
    # Sanity plots: full trace and window
    # -----------------------------
    plt.figure(figsize=(12, 5))
    plt.plot(time*1e6, vraw, "k-", lw=1, alpha=0.75, label="Original V")
    plt.plot(peak_time*1e6, vraw[peak_idx], "r*", ms=14, label="Chosen first negative peak")
    plt.axvspan(peak_time*1e6, (peak_time + MAX_WINDOW_NS*1e-9)*1e6, color="orange", alpha=0.15,
                label=f"Max window cap ({MAX_WINDOW_NS:.0f} ns)")
    plt.xlabel("Time (us)")
    plt.ylabel("Volts")
    plt.title("Chosen Peak + Max Window Cap (sanity)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p0 = os.path.join(OUT_DIR, "00_peak_window_full.png")
    plt.savefig(p0, dpi=170)
    plt.close()
    print(f"Saved: {p0}")

    plt.figure(figsize=(11, 5))
    plt.plot(tw_full*1e9, V_full, "0.75", lw=1, label="V within max cap")
    plt.plot(tw_seg*1e9, V_seg, "b.-", ms=3, label="V used for fit (auto-stop)")
    plt.axvline(t_cut*1e9, color="orange", ls="--", lw=2, label=f"cutoff {t_cut*1e9:.2f} ns")
    plt.xlabel("Time from peak (ns)")
    plt.ylabel("Volts (original sign)")
    plt.title("Auto-stop Segment (stops when slope turns negative)")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    p1 = os.path.join(OUT_DIR, "01_window_autostop.png")
    plt.savefig(p1, dpi=170)
    plt.close()
    print(f"Saved: {p1}")

    # -----------------------------
    # Subsample for fitting
    # -----------------------------
    N = min(SUBSAMPLE_N, len(tw_seg))
    idx = np.linspace(0, len(tw_seg) - 1, N, dtype=int)
    tw = tw_seg[idx]
    y = y_seg[idx]
    V_data = (-y) if NEGATE else y

    print(f"\nSubsampled for fit: {len(tw)} points, duration {tw[-1]*1e9:.2f} ns")
    print(f"V_data range: {V_data.min():.4f} to {V_data.max():.4f} V")

    # -----------------------------
    # curve_fit init (good starting values)
    # -----------------------------
    tail_n = max(10, len(y)//10)
    b0 = float(np.mean(y[-tail_n:]))
    y0 = float(y[0])
    A_tot = max(1e-6, y0 - b0)

    # delay guess: when y drops 5% toward baseline
    thresh = y0 - 0.05 * A_tot
    d0 = float(tw[np.argmax(y < thresh)]) if np.any(y < thresh) else 40e-9

    A1_0, A2_0 = 0.55*A_tot, 0.45*A_tot
    tau1_0, tau2_0 = 10e-9, 25e-9
    p_init = [b0, A1_0, tau1_0, A2_0, tau2_0, d0]

    lb = [b0 - 3*A_tot, 0.0, 1e-9, 0.0, 2e-9, 0.0]
    ub = [b0 + 3*A_tot, 5*A_tot, 250e-9, 5*A_tot, 600e-9, 200e-9]

    print("\nRunning curve_fit init...")
    try:
        popt, _ = curve_fit(delayed_doubleexp_np, tw, y, p0=p_init, bounds=(lb, ub), maxfev=50000)
    except Exception as e:
        print("curve_fit failed, using heuristic init:", e)
        popt = np.array(p_init, dtype=float)

    b_fit, A1_fit, tau1_fit, A2_fit, tau2_fit, d_fit = map(float, popt)
    print("Init (curve_fit) parameters:")
    print(f"  b    = {b_fit:.6f}")
    print(f"  A1   = {A1_fit:.6f}, tau1 = {tau1_fit*1e9:.3f} ns")
    print(f"  A2   = {A2_fit:.6f}, tau2 = {tau2_fit*1e9:.3f} ns")
    print(f"  d    = {d_fit*1e9:.3f} ns")

    # Init fit plot
    tplt = np.linspace(0, tw[-1], 1500)
    plt.figure(figsize=(12, 5))
    plt.plot(tw*1e9, y, "b.", ms=4, label="y data (negated domain)")
    plt.plot(tplt*1e9, delayed_doubleexp_np(tplt, *popt), "r-", lw=2.5, label="curve_fit init")
    plt.xlabel("Time from peak (ns)")
    plt.ylabel("y = -V (if NEGATE=True)")
    plt.title("Initialization Fit (curve_fit) in y-domain")
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()
    pinit = os.path.join(OUT_DIR, "01b_init_fit_curvefit.png")
    plt.savefig(pinit, dpi=170)
    plt.close()
    print(f"Saved: {pinit}")

    # -----------------------------
    # Bayesian model
    # Prefer NumPyro NUTS if JAX exists; else ADVI fallback
    # -----------------------------
    use_numpyro = try_import_jax()
    if use_numpyro:
        print("\nSampling with NumPyro NUTS (fast)...")
    else:
        print("\nJAX not found → using ADVI fallback (still produces credible bands).")
        print("To enable NumPyro NUTS: pip install jax jaxlib numpyro")

    # residual-based initial sigma guess
    res0 = y - delayed_doubleexp_np(tw, *popt)
    sig0 = float(np.std(res0) + 1e-6)

    with pm.Model() as model:
        # baseline around tail mean
        baseline = pm.Normal("baseline", mu=b_fit, sigma=max(0.05, 0.2*A_tot))

        # amplitudes (positive)
        A1 = pm.HalfNormal("A1", sigma=max(abs(A1_fit), 0.5))
        A2 = pm.HalfNormal("A2", sigma=max(abs(A2_fit), 0.5))

        # taus bounded to sane range (prevents crazy 1e45 blow-ups)
        tau1 = pm.TruncatedNormal("tau1", mu=tau1_fit, sigma=8e-9, lower=1e-9, upper=250e-9)
        tau2 = pm.TruncatedNormal("tau2", mu=tau2_fit, sigma=15e-9, lower=2e-9, upper=600e-9)

        # delay
        d = pm.TruncatedNormal("d", mu=d_fit, sigma=8e-9, lower=0.0, upper=250e-9)

        # noise model
        sigma = pm.HalfNormal("sigma", sigma=max(sig0*2.0, 1e-4))
        nu = pm.Exponential("nu", lam=1/10) + 2  # Student-t df > 2

        t = tw
        mu_pre = baseline + A1 + A2
        mu_post = baseline + A1*pm.math.exp(-(t - d)/tau1) + A2*pm.math.exp(-(t - d)/tau2)
        mu = pm.math.switch(t < d, mu_pre, mu_post)

        pm.StudentT("obs", nu=nu, mu=mu, sigma=sigma, observed=y)

        if use_numpyro:
            trace = pm.sample(
                draws=DRAWS, tune=TUNE,
                chains=CHAINS, cores=CORES,
                target_accept=TARGET_ACCEPT,
                random_seed=SEED,
                nuts_sampler="numpyro",
                progressbar=True,
            )
        else:
            approx = pm.fit(n=ADVI_ITERS, method="advi", random_seed=SEED, progressbar=True)
            trace = approx.sample(ADVI_DRAWS)

    # -----------------------------
    # Posterior curves + credible/predictive bands (ROBUST, V-domain)
    # -----------------------------
    post = trace.posterior

    def flat(name):
        return post[name].values.reshape(-1)

    b_s    = flat("baseline")
    A1_s   = flat("A1")
    A2_s   = flat("A2")
    tau1_s = flat("tau1")
    tau2_s = flat("tau2")
    d_s    = flat("d")
    sig_s  = flat("sigma")
    nu_s   = flat("nu")

    S = len(b_s)
    tgrid = np.linspace(0, tw[-1], 1800)

    curves_y = np.zeros((S, len(tgrid)), dtype=float)
    for i in range(S):
        curves_y[i] = delayed_doubleexp_np(
            tgrid, b_s[i], A1_s[i], tau1_s[i], A2_s[i], tau2_s[i], d_s[i]
        )

    curves_V = (-curves_y) if NEGATE else curves_y

    # Drop pathological draws (NaN/Inf)
    good = np.isfinite(curves_V).all(axis=1)
    bad_count = int((~good).sum())
    if bad_count > 0:
        print(f"WARNING: Dropping {bad_count}/{S} posterior draws with non-finite curves.")
    curves_V = curves_V[good]
    b_s = b_s[good]; A1_s = A1_s[good]; A2_s = A2_s[good]
    tau1_s = tau1_s[good]; tau2_s = tau2_s[good]; d_s = d_s[good]
    sig_s = sig_s[good]; nu_s = nu_s[good]

    V_mean = curves_V.mean(axis=0)
    V_ci95_lo, V_ci95_hi = np.percentile(curves_V, [2.5, 97.5], axis=0)
    V_ci50_lo, V_ci50_hi = np.percentile(curves_V, [25, 75], axis=0)

    # Predictive bands (approx): add Student-t noise per draw
    rng = np.random.default_rng(SEED)
    pred_V = curves_V.copy()
    for i in range(pred_V.shape[0]):
        noise = rng.standard_t(df=float(nu_s[i]), size=len(tgrid)) * float(sig_s[i])
        pred_V[i] = pred_V[i] + (-noise if NEGATE else noise)

    V_pi95_lo, V_pi95_hi = np.percentile(pred_V, [2.5, 97.5], axis=0)
    V_pi50_lo, V_pi50_hi = np.percentile(pred_V, [25, 75], axis=0)

    # Metrics in V-domain
    V_fit_w = np.interp(tw, tgrid, V_mean)
    resid_V = V_data - V_fit_w
    rmse_V = float(np.sqrt(np.mean(resid_V**2)))
    r2_V = float(1 - np.sum(resid_V**2) / np.sum((V_data - V_data.mean())**2))

    # Posterior summary
    summ = az.summary(trace, var_names=["baseline", "A1", "tau1", "A2", "tau2", "d", "sigma", "nu"])
    print("\n" + "="*80)
    print("POSTERIOR SUMMARY")
    print("="*80)
    print(summ)

    # Helpful means + 95% intervals
    def mean_ci(x):
        m = float(np.mean(x))
        lo, hi = np.percentile(x, [2.5, 97.5])
        return m, float(lo), float(hi)

    b_m, b_lo, b_hi = mean_ci(b_s)
    A1_m, A1_lo, A1_hi = mean_ci(A1_s)
    A2_m, A2_lo, A2_hi = mean_ci(A2_s)
    t1_m, t1_lo, t1_hi = mean_ci(tau1_s)
    t2_m, t2_lo, t2_hi = mean_ci(tau2_s)
    d_m, d_lo, d_hi = mean_ci(d_s)

    print("\n" + "="*80)
    print("FIT RESULTS (V-domain)")
    print("="*80)
    print(f"R²   = {r2_V:.6f}")
    print(f"RMSE = {rmse_V:.6f} V\n")
    print(f"baseline b  = {b_m:.6f}   [{b_lo:.6f}, {b_hi:.6f}] (y-domain)")
    print(f"A1          = {A1_m:.6f}   [{A1_lo:.6f}, {A1_hi:.6f}]")
    print(f"tau1        = {t1_m*1e9:.4f} ns   [{t1_lo*1e9:.4f}, {t1_hi*1e9:.4f}] ns")
    print(f"A2          = {A2_m:.6f}   [{A2_lo:.6f}, {A2_hi:.6f}]")
    print(f"tau2        = {t2_m*1e9:.4f} ns   [{t2_lo*1e9:.4f}, {t2_hi*1e9:.4f}] ns")
    print(f"d           = {d_m*1e9:.4f} ns   [{d_lo*1e9:.4f}, {d_hi*1e9:.4f}] ns")
    print("="*80)

    # -----------------------------
    # Main fit plot (V-domain) — robust y-limits based on data
    # -----------------------------
    fig, ax = plt.subplots(figsize=(12, 6))

    ax.fill_between(tgrid*1e9, V_pi95_lo, V_pi95_hi, alpha=0.12, label="95% predictive")
    ax.fill_between(tgrid*1e9, V_pi50_lo, V_pi50_hi, alpha=0.22, label="50% predictive")
    ax.fill_between(tgrid*1e9, V_ci95_lo, V_ci95_hi, alpha=0.20, label="95% credible")
    ax.fill_between(tgrid*1e9, V_ci50_lo, V_ci50_hi, alpha=0.35, label="50% credible")

    ax.plot(tgrid*1e9, V_mean, "r-", lw=2.5, label="posterior mean", zorder=5)
    ax.plot(tw*1e9, V_data, "bo", ms=3.5, alpha=0.85, label=f"data ({len(tw)} pts)", zorder=6)
    ax.axvline(d_m*1e9, color="orange", ls="--", lw=2, label=f"delay d ≈ {d_m*1e9:.2f} ns")

    ax.set_title(f"Delayed Double-Exponential Fit (auto-stop) | R²={r2_V:.5f} | RMSE={rmse_V:.5f} V",
                 fontweight="bold")
    ax.set_xlabel("Time from first negative peak (ns)")
    ax.set_ylabel("Volts (original sign)")
    ax.grid(True, alpha=0.3)
    ax.legend(fontsize=9)

    # Lock y-limits to data range so blown bands can’t flatten the plot
    pad = 0.15 * (V_data.max() - V_data.min() + 1e-9)
    ax.set_ylim(V_data.min() - pad, V_data.max() + pad)

    plt.tight_layout()
    p2 = os.path.join(OUT_DIR, "02_fit_bayes_delayed_doubleexp.png")
    plt.savefig(p2, dpi=170)
    plt.close()
    print(f"Saved: {p2}")

    # Optional extra debug: interval lines (no fill_between)
    plt.figure(figsize=(11, 4.5))
    plt.plot(tgrid*1e9, V_ci95_lo, "k--", lw=1, label="CI95 lo")
    plt.plot(tgrid*1e9, V_ci95_hi, "k--", lw=1, label="CI95 hi")
    plt.plot(tgrid*1e9, V_pi95_lo, "g--", lw=1, label="PI95 lo")
    plt.plot(tgrid*1e9, V_pi95_hi, "g--", lw=1, label="PI95 hi")
    plt.plot(tgrid*1e9, V_mean, "r-", lw=2, label="mean")
    plt.scatter(tw*1e9, V_data, s=10, alpha=0.7, label="data")
    plt.title("DEBUG: Interval line sanity (no fill_between)")
    plt.xlabel("Time from peak (ns)")
    plt.ylabel("Volts")
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()
    pdbg = os.path.join(OUT_DIR, "debug_intervals_lines.png")
    plt.savefig(pdbg, dpi=170)
    plt.close()
    print(f"Saved: {pdbg}")

    # -----------------------------
    # Residual plot (V-domain)
    # -----------------------------
    plt.figure(figsize=(11, 4.5))
    plt.scatter(tw*1e9, resid_V, s=18, alpha=0.75)
    plt.axhline(0, lw=2, color="red")
    plt.title("Residuals vs time (V-domain, posterior mean)")
    plt.xlabel("Time from peak (ns)")
    plt.ylabel("Residual (V)")
    plt.grid(True, alpha=0.3)
    plt.tight_layout()
    p3 = os.path.join(OUT_DIR, "03_residuals.png")
    plt.savefig(p3, dpi=170)
    plt.close()
    print(f"Saved: {p3}")

    # -----------------------------
    # Print Xyce-friendly equation
    # Note: b is in y-domain if NEGATE=True, so V(t) includes a leading minus.
    # -----------------------------
    print("\nXyce-style equation (t in seconds, t=0 at first negative peak):")
    if NEGATE:
        print("V(t) = - ( b + A1 + A2 )                                  for t < d")
        print("V(t) = - ( b + A1*exp(-(t-d)/tau1) + A2*exp(-(t-d)/tau2) )  for t >= d")
    else:
        print("V(t) = ( b + A1 + A2 )                                  for t < d")
        print("V(t) = ( b + A1*exp(-(t-d)/tau1) + A2*exp(-(t-d)/tau2) )  for t >= d")

    print("\nPosterior mean parameters (use these in Xyce):")
    print(f"b={b_m:.6f}")
    print(f"A1={A1_m:.6f}")
    print(f"tau1={t1_m:.6e}")
    print(f"A2={A2_m:.6f}")
    print(f"tau2={t2_m:.6e}")
    print(f"d={d_m:.6e}")

    print("\nDONE. Check these first:")
    print(f"  {os.path.join(OUT_DIR, '01_window_autostop.png')}")
    print(f"  {os.path.join(OUT_DIR, '02_fit_bayes_delayed_doubleexp.png')}")


if __name__ == "__main__":
    # Windows safe entry point
    from multiprocessing import freeze_support
    freeze_support()
    main()
