from pathlib import Path

import numpy as np
import pandas as pd


def read_metadata(path):
    """Validate SMAJ conditions and resolve measurement paths relative to the CSV."""
    path = Path(path).expanduser().resolve()
    frame = pd.read_csv(path)
    required = {'shot_id', 'diode_type', 'dose_rate', 'load_ohm', 'bias_v',
                'pcd_fwhm_ns', 'scope_scale_factor', 'cached_scope_scale_factor',
                'waveform_csv'}
    missing = required-set(frame.columns)
    if missing:
        raise ValueError(f'Metadata {path.name} is missing columns: {", ".join(sorted(missing))}')
    if frame.diode_type.isna().any():
        raise ValueError('Every metadata row needs a diode_type for input filtering.')
    frame['diode_type'] = frame.diode_type.astype(str).str.strip()
    frame = frame.loc[frame.diode_type.eq('SMAJ400A')].copy()
    if len(frame) < 4:
        raise ValueError('Metadata must contain at least four SMAJ400A tests for held-out evaluation.')
    numeric = ('shot_id', 'dose_rate', 'load_ohm', 'bias_v', 'pcd_fwhm_ns',
               'scope_scale_factor', 'cached_scope_scale_factor')
    if 'scope_attenuation_setting' in frame:
        numeric += ('scope_attenuation_setting',)
    for name in numeric:
        values = pd.to_numeric(frame[name], errors='coerce')
        invalid = ~np.isfinite(values)
        if invalid.any():
            raise ValueError(f'Metadata column {name} must contain finite numbers (CSV row {frame.index[invalid][0]+2}).')
        if name != 'bias_v' and (values <= 0).any():
            raise ValueError(f'Metadata column {name} must be positive.')
        frame[name] = values
    if (frame.shot_id != np.floor(frame.shot_id)).any():
        raise ValueError('Metadata shot_id values must be integers.')
    frame['shot_id'] = frame.shot_id.astype('int64')
    if frame.shot_id.duplicated().any():
        raise ValueError('Each SMAJ400A shot_id must appear exactly once in the metadata.')
    if 'raw_csv' not in frame:
        frame['raw_csv'] = ''  # Optional override for the legacy --raw-waveforms flag.
    for column in ('waveform_csv', 'raw_csv'):
        def resolve(value):
            if pd.isna(value) or not str(value).strip():
                return ''
            source = Path(str(value).strip()).expanduser()
            return str((path.parent/source).resolve())
        frame[column] = frame[column].map(resolve)
    if (frame.waveform_csv.eq('') & frame.raw_csv.eq('')).any():
        raise ValueError('Each metadata row needs a waveform_csv or raw_csv path.')
    return frame.reset_index(drop=True)


def _scope_channel(raw, name):
    """Use the channel's preceding time column and discard export zero padding."""
    index = list(raw.columns).index(name)
    if index == 0 or not raw.columns[index-1].startswith('time'):
        raise ValueError(f'No timebase before channel {name}')
    time = raw.iloc[:, index-1].to_numpy(float)
    values = raw[name].to_numpy(float)
    valid = np.isfinite(time) & np.isfinite(values)
    end = len(time)
    padded = (end > 1 and time[-1] == values[-1] == 0
              and ((time[-2] == values[-2] == 0) or time[-2] > 0))
    if padded:
        while end and time[end-1] == values[end-1] == 0:
            end -= 1
    valid[end:] = False
    time, values = time[valid], values[valid]
    if len(time) < 5 or np.any(np.diff(time) <= 0):
        raise ValueError(f'{name}: need at least five samples on a strictly increasing timebase')
    return time, values


def load_waveform(row, raw_waveforms=False):
    """Read either processed voltages or original scope channels from the CSV."""
    path = (getattr(row, 'raw_csv', '') if raw_waveforms else '') or row.waveform_csv
    if not path:
        raise ValueError(f'Test {row.shot_id} has no measurement path in the metadata.')
    if not Path(path).is_file():
        raise FileNotFoundError(f'Test {row.shot_id}: CSV not found: {path}\n'
                                'Update the measurement path in simease_ridge_metadata.csv.')
    header = list(pd.read_csv(path, nrows=0).columns)
    if {'time_ns', 'actual_v'}.issubset(header):
        wave = pd.read_csv(path, usecols=['time_ns', 'actual_v'])
        # Cached observations have already been divided by their original scale.
        # Undo that division before applying the editable calibration factor.
        wave['actual_v'] *= row.cached_scope_scale_factor/row.scope_scale_factor
    elif {'Diode', 'PCD3_B'}.issubset(header):
        needed = set()
        for channel in ('Diode', 'PCD3_B'):
            index = header.index(channel)
            if index == 0 or not header[index-1].startswith('time'):
                raise ValueError(f'Test {row.shot_id}: no timebase before {channel}.')
            needed.update((channel, header[index-1]))
        raw = pd.read_csv(path, usecols=[name for name in header if name in needed])
        time_s, voltage = _scope_channel(raw, 'Diode')
        pcd_time, pcd = _scope_channel(raw, 'PCD3_B')
        # Preserve the original v2 alignment, scaling, grid, and missing coverage.
        reference = float(pcd_time[np.argmax(pcd)])
        time_ns = (time_s-reference)*1e9
        grid = np.arange(-150.0, 5000.1, 2.0)
        actual = np.interp(grid, time_ns, voltage/row.scope_scale_factor,
                           left=np.nan, right=np.nan)
        wave = pd.DataFrame(dict(time_ns=grid, actual_v=actual))
    else:
        raise ValueError(f'Test {row.shot_id}: CSV needs time_ns/actual_v columns or '
                         'Diode/PCD3_B channels with preceding time columns.')
    time = wave.time_ns.to_numpy(float)
    voltage = wave.actual_v.to_numpy(float)
    if not np.isfinite(time).all() or (np.diff(time) <= 0).any():
        raise ValueError(f'Test {row.shot_id}: measurement times must be finite and increasing.')
    if np.isinf(voltage).any():
        raise ValueError(f'Test {row.shot_id}: measured voltage contains infinity.')
    return wave


from pathlib import Path
from html import escape
import sys
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator
# Metadata helpers are included above.

ROOT=Path(__file__).resolve().parent

TRAINING_DIODE = "SMAJ400A"
PREDICTOR_COLUMNS = ("dose_rate", "bias_v", "load_ohm", "pcd_fwhm_ns")


def require_smaj(conditions: pd.DataFrame) -> None:
    """Keep both training and evaluation within the supported diode family."""
    if conditions.empty or not conditions.diode_type.eq(TRAINING_DIODE).all():
        raise ValueError("simease_ridge supports SMAJ400A shots only.")

# Local v2 calculation routines preserve equations and candidate ordering.
GRID_NS = np.arange(-150.0, 5000.1, 2.0)


KNOTS_NS = np.array([-150, -100, -75, *range(-60, 81, 5),
                     100, 125, 150, 180, 220, 270, 330, 400, 500, 650,
                     800, 1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000], float)


TARGETS = {f"knot_{i:02d}_v": "identity" for i in range(len(KNOTS_NS))}
BASE_KNOTS_NS = KNOTS_NS.copy()


PROMPT = (GRID_NS >= -60) & (GRID_NS <= 1000)


def _crossing(time: np.ndarray, shape: np.ndarray, level: float,
              peak: int, rising: bool) -> float:
    if rising:
        hits = np.flatnonzero((shape[:-1] < level) & (shape[1:] >= level) & (np.arange(len(shape)-1) < peak))
        if not len(hits):
            return float("nan")
        left = int(hits[-1])
    else:
        hits = np.flatnonzero((shape[:-1] >= level) & (shape[1:] < level) & (np.arange(len(shape)-1) >= peak))
        if not len(hits):
            return float("nan")
        left = int(hits[0])
    fraction = (level-shape[left])/(shape[left+1]-shape[left])
    return float(time[left]+fraction*(time[left+1]-time[left]))


def waveform_parameters(time: np.ndarray, voltage: np.ndarray) -> dict:
    valid = np.isfinite(time) & np.isfinite(voltage)
    time, voltage = np.asarray(time)[valid], np.asarray(voltage)[valid]
    pre = (time >= -150) & (time < -60)
    prompt = (time >= -60) & (time <= 1000)
    if pre.sum() < 3 or prompt.sum() < 5:
        raise ValueError("Waveform must cover the pre-pulse baseline and prompt interval.")
    baseline = float(np.median(voltage[pre]))
    peak = np.flatnonzero(prompt)[np.argmax(np.abs(voltage[prompt]-baseline))]
    signed_peak = float(voltage[peak]-baseline)
    polarity = 1.0 if signed_peak >= 0 else -1.0
    amplitude = abs(signed_peak)
    shape = polarity*(voltage-baseline)/max(amplitude, 1e-12)
    t10, t50, t90 = [_crossing(time, shape, level, peak, True) for level in (.1, .5, .9)]
    f90, f50, f10 = [_crossing(time, shape, level, peak, False) for level in (.9, .5, .1)]
    after = np.flatnonzero((time > time[peak]) & (time <= 5000))
    minimum = after[np.argmin(shape[after])] if len(after) else peak
    late = (time >= 3000) & (time <= 5000)
    integral = (time >= -60) & (time <= 1000)
    return dict(peak_v=amplitude, baseline_v=baseline, polarity=polarity,
                peak_time_ns=float(time[peak]), rise_10_50_ns=t50-t10,
                rise_50_90_ns=t90-t50, rise_10_90_ns=t90-t10,
                peak_rounding_ns=float(time[peak])-t90,
                fall_90_ns=f90-float(time[peak]), recovery_50_ns=f50-float(time[peak]),
                recovery_10_ns=f10-float(time[peak]), fwhm_ns=f50-t50,
                undershoot_ratio=max(0.0, -float(shape[minimum])),
                undershoot_delay_ns=float(time[minimum]-time[peak]),
                tail_offset_v=float(np.median(voltage[late])-baseline) if late.any() else np.nan,
                area_v_ns=float(np.trapezoid(voltage[integral]-baseline, time[integral])),
                energy_v2_ns=float(np.trapezoid((voltage[integral]-baseline)**2, time[integral])))


def encode(frame: pd.DataFrame) -> np.ndarray:
    """Encode four numeric conditions and their interactions; no diode labels.

    Fixed unit scaling avoids fitting preprocessing on validation shots.
    """
    frame = frame.loc[:, list(PREDICTOR_COLUMNS)]
    dose = np.log10(frame.dose_rate.to_numpy(float)/1e10)
    bias = frame.bias_v.to_numpy(float)/5.25
    high = (frame.load_ohm.to_numpy(float) >= 1e5).astype(float)
    low = (1-high)*(frame.load_ohm.to_numpy(float)/81.3-1)
    width = frame.pcd_fwhm_ns.to_numpy(float)/40-1
    return np.column_stack([dose, bias, high, low, width,
                            dose*bias, dose*high, bias*high, dose*low])


def _kernel(left: pd.DataFrame, right: pd.DataFrame, kind: str) -> np.ndarray:
    """Condition-only covariance. Diode type is used solely at the input boundary."""
    x, y = encode(left), encode(right)
    if kind == "linear":
        # Retain the existing single-diode scale so alpha has the same meaning.
        return 1 + 2*(x@y.T)
    name, length_text = kind.split(":")
    if name not in ("rbf", "hybrid", "regime"):
        raise ValueError(f"Unsupported condition kernel: {kind}")
    length = float(length_text)
    if not np.isfinite(length) or length <= 0:
        raise ValueError("Kernel length must be finite and positive.")
    weights = np.array([1, 2, 1.5, 2, 2], float)
    distance = np.sum(((x[:, None, :5]-y[None, :, :5])*weights)**2, axis=2)
    covariance = np.exp(-distance/(2*length**2))
    if name in ("hybrid", "regime"):
        features = [0, 1, 2, 3, 5, 6, 7, 8] if name == "hybrid" else [1, 2, 3, 7]
        covariance += x[:, features]@y[:, features].T
    return covariance


def _solve(kernel: np.ndarray, targets: np.ndarray, alpha: float) -> tuple[list, np.ndarray]:
    """Fit observed knot groups; PRESS gives exact inner LOO residuals.

    Centered ridge equals ordered-pair regression with a scaled penalty,
    without materializing dependent pair rows. Missing tail samples are omitted.
    """
    valid = np.isfinite(targets)
    patterns: dict[bytes, list[int]] = {}
    for column in range(targets.shape[1]):
        patterns.setdefault(valid[:, column].tobytes(), []).append(column)
    models, errors = [], np.full_like(targets, np.nan)
    for columns in patterns.values():
        indices = np.flatnonzero(valid[:, columns[0]])
        if not len(indices):
            raise ValueError(f"No training coverage for knot {columns[0]}")
        values = targets[np.ix_(indices, columns)]
        k = kernel[np.ix_(indices, indices)]
        column_mean, grand_mean = k.mean(axis=0), float(k.mean())
        centered = k-column_mean[None, :]-column_mean[:, None]+grand_mean
        mean = values.mean(axis=0)
        inverse = np.linalg.inv(centered+alpha*np.eye(len(indices)))
        coefficient = inverse@(values-mean)
        hat = centered@inverse + 1/len(indices)
        if len(indices) > 1:
            errors[np.ix_(indices, columns)] = (values-(centered@coefficient+mean))/np.maximum(1-np.diag(hat)[:, None], 1e-10)
        models.append(dict(columns=columns, indices=indices, column_mean=column_mean,
                           grand_mean=grand_mean, mean=mean, coefficient=coefficient))
    return models, errors


def fit_model(conditions: pd.DataFrame, features: pd.DataFrame) -> dict:
    require_smaj(conditions)
    if len(conditions) < 3:
        raise ValueError("Need at least three training shots.")
    labels = features.set_index("shot_id").loc[conditions.shot_id]
    targets = labels[list(TARGETS)].to_numpy(float)
    peak = np.maximum(labels.peak_v.to_numpy(float), 1e-6)
    focus = (KNOTS_NS >= -60) & (KNOTS_NS <= 1000)
    times = KNOTS_NS[focus]
    weights = np.diff(np.r_[times[0], (times[:-1]+times[1:])/2, times[-1]])
    best, candidates = None, []
    # A single unpenalized intercept suffices for the SMAJ-only dataset.
    # Former task/sharing candidates reduce to these same RBF kernels.
    for kind in ("linear", "rbf:0.5", "rbf:1.0", "rbf:2.0"):
        kernel_matrix = kernel(conditions, conditions, kind)
        for alpha in (.01, .1, 1.0, 10.0):
            models, errors = _solve(kernel_matrix, targets, alpha)
            squared = (errors[:, focus]/peak[:, None])**2
            available = np.isfinite(squared)
            denom = np.sum(available*weights, axis=1)
            per_shot = np.nansum(squared*weights, axis=1)/np.maximum(denom, 1e-12)
            normalized_score = float(np.mean(per_shot[denom > 0]))
            # Align selection with the reported mean waveform RMSE in volts.
            score = float(np.mean(np.sqrt(per_shot[denom > 0])*peak[denom > 0]))
            candidates.append(dict(kernel=kind, alpha=alpha,
                                   inner_loo_rmse_v=score, inner_loo_normalized_mse=normalized_score))
            if best is None or score < best["inner_score"]:
                best = dict(kernel=kind, alpha=alpha,
                            inner_score=score, groups=models)
    best["conditions"] = conditions.reset_index(drop=True).copy()
    best["candidates"] = candidates
    return best


def predict(model: dict, conditions: pd.DataFrame) -> pd.DataFrame:
    require_smaj(model["conditions"])
    require_smaj(conditions)
    kernel_matrix = kernel(conditions, model["conditions"], model["kernel"])
    targets = np.empty((len(conditions), len(TARGETS)))
    for group in model["groups"]:
        k = kernel_matrix[:, group["indices"]]
        centered = k-group["column_mean"][None, :]-k.mean(axis=1)[:, None]+group["grand_mean"]
        values = centered@group["coefficient"]+group["mean"]
        targets[:, group["columns"]] = values
    result = conditions[["shot_id"]].reset_index(drop=True).copy()
    result = pd.concat([result, pd.DataFrame(targets, columns=list(TARGETS))], axis=1)
    landmarks = [waveform_parameters(GRID_NS, reconstruct(GRID_NS, row)) for row in result.to_dict("records")]
    return pd.concat([result, pd.DataFrame(landmarks)], axis=1)


def reconstruct(time: np.ndarray, parameters: dict) -> np.ndarray:
    values = np.array([parameters[name] for name in TARGETS], float)
    if not np.isfinite(values).all():
        raise ValueError("All predicted spline parameters must be finite.")
    interpolator = PchipInterpolator(KNOTS_NS, values, extrapolate=False)
    return interpolator(np.clip(time, KNOTS_NS[0], KNOTS_NS[-1]))


def _metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray | None = None) -> dict:
    """Score all observed waveform samples unless an explicit window is given."""
    valid = np.isfinite(actual) & np.isfinite(predicted)
    if mask is not None:
        valid &= mask
    if valid.sum() < 5:
        raise ValueError("Not enough observed samples to evaluate waveform")
    residual = actual[valid]-predicted[valid]
    sst = float(np.sum((actual[valid]-actual[valid].mean())**2))
    return dict(rmse_v=float(np.sqrt(np.mean(residual**2))),
                mae_v=float(np.mean(np.abs(residual))),
                r2=1-float(np.sum(residual**2))/sst if sst > 1e-15 else np.nan,
                n_samples=int(valid.sum()))


def draw_waveform(figure, wave, predicted, shot_id, diode_type, score, full):
    """Draw the same comparison in saved plots and the desktop GUI."""
    figure.clear()
    axes = figure.subplots(2, 1)
    for axis in axes:
        axis.plot(wave.time_ns, wave.actual_v, label='Actual diode output', color='#426b9a')
        axis.plot(wave.time_ns, predicted, label='Predicted output (held-out shot)',
                  color='#d26828', linestyle='--')
        axis.set_xlabel('Time (ns)')
        axis.set_ylabel('Diode output (V)')
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_title(f'Full waveform | RMSE: {full["rmse_v"]:.3f} V')
    axes[1].set_xlim(-60, 1000)
    prompt = wave.time_ns.between(-60, 1000).to_numpy()
    values = np.r_[wave.actual_v.to_numpy()[prompt], predicted[prompt]]
    values = values[np.isfinite(values)]
    if values.size:
        margin = max(float(np.ptp(values)) * 0.08, 0.01)
        axes[1].set_ylim(values.min() - margin, values.max() + margin)
    axes[1].set_title(f'Prompt window | RMSE: {score["rmse_v"]:.3f} V')
    figure.suptitle(f'Shot {shot_id} | {diode_type} | Leave-one-shot-out prediction')
    figure.tight_layout()


def plot_waveform(wave, predicted, shot_id, diode_type, score, full, output_dir):
    """Save the held-out prediction alongside the original measured voltage."""
    figure = plt.figure(figsize=(11, 8))
    draw_waveform(figure, wave, predicted, shot_id, diode_type, score, full)
    figure.savefig(output_dir / f'{shot_id}_predicted_vs_actual.png', dpi=160)
    plt.close(figure)


def fit_expanded(conditions, features, extra=False, balanced=False, gains=None):
    require_smaj(conditions)
    if len(conditions) < 3:
        raise ValueError("Need at least three training shots.")
    labels=features.set_index('shot_id').loc[conditions.shot_id]
    target=labels[list(TARGETS)].to_numpy(float)
    focus=(KNOTS_NS>=-60)&(KNOTS_NS<=1000)
    t=KNOTS_NS[focus]; weights=np.diff(np.r_[t[0],(t[:-1]+t[1:])/2,t[-1]])
    best=None
    gain_modes=(False,True) if gains is not None else (False,)
    for gain_mode in gain_modes:
      gain=gains if gain_mode else np.ones(len(conditions))
      y=target/gain[:,None]
      for kind in ['linear','rbf:0.5','rbf:1.0','rbf:2.0']+(['hybrid:0.5','hybrid:1.0','hybrid:2.0','regime:0.5','regime:1.0','regime:2.0'] if extra else []):
        k=kernel(conditions,conditions,kind)
        for alpha in (.01,.1,1.,10.):
            groups,errors=_solve(k,y,alpha)
            errors=errors*gain[:,None]
            available=np.isfinite(errors[:,focus]);denom=(available*weights).sum(axis=1)
            error=np.sqrt(np.nansum(errors[:,focus]**2*weights,axis=1)/np.maximum(denom,1e-12))
            score=error[denom>0].mean()
            if balanced:
                tail=KNOTS_NS>=0;times=KNOTS_NS[tail]
                tw=np.diff(np.r_[times[0],(times[:-1]+times[1:])/2,times[-1]])
                avail=np.isfinite(errors[:,tail]);td=(avail*tw).sum(axis=1)
                terr=np.sqrt(np.nansum(errors[:,tail]**2*tw,axis=1)/np.maximum(td,1e-12))
                score=.5*(score+terr[td>0].mean())
            if best is None or score<best['inner_score']:
                best=dict(kernel=kind,alpha=alpha,groups=groups,conditions=conditions.reset_index(drop=True).copy(),inner_score=score,gain_mode=gain_mode)
    return best


def kernel(left,right,kind):
    # Keep filtering/audit metadata outside the predictive kernel interface.
    predictors = list(PREDICTOR_COLUMNS)
    return _kernel(left.loc[:, predictors], right.loc[:, predictors], kind)


def configure_knots(spacing=10, coarse_tail=False):
    """Reset the experiment grid so successive runs do not accumulate knots."""
    global KNOTS_NS, TARGETS
    if not np.isfinite(spacing) or spacing <= 0:
        raise ValueError('Knot spacing must be positive.')
    KNOTS_NS = np.unique(np.r_[BASE_KNOTS_NS, np.arange(-60, 1001, spacing),
                              [] if coarse_tail else np.arange(1000, 5001, 25)])
    TARGETS = {f'dense_{i:03d}_v': 'identity' for i in range(len(KNOTS_NS))}


def load_smaj_conditions(metadata=None):
    """Reload the editable metadata; cached result metadata is not consulted."""
    return read_metadata(metadata if metadata is not None else ROOT/'simease_ridge_metadata.csv')


def read_waveform(row, raw_waveforms=False):
    return load_waveform(row, raw_waveforms)


def waveform_features(wave, shot_id, peak_v=None):
    good = wave.actual_v.notna()
    time, voltage = wave.time_ns.to_numpy()[good], wave.actual_v.to_numpy()[good]
    smooth = gaussian_filter1d(voltage, 1.5/2)
    if peak_v is None:
        peak_v = waveform_parameters(time, smooth)['peak_v']
    values = np.interp(KNOTS_NS, time, smooth, left=np.nan, right=np.nan)
    return dict(shot_id=shot_id, peak_v=peak_v, **dict(zip(TARGETS, values)))


def model_shot(shot_id, *, raw_waveforms=False, progress=None, metadata=None):
    """Fit one documented SMAJ holdout with the recommended hybrid/coarse setup.

    Like the batch script this configures module-level knots; callers must run
    fits sequentially. The GUI permits one worker at a time. Held-out waveform
    labels are read only after fitting and prediction, for display and scoring.
    """
    report = progress if progress is not None else lambda message: None
    conditions = load_smaj_conditions(metadata)
    test = conditions.loc[conditions.shot_id.eq(shot_id)]
    if len(test) != 1:
        raise ValueError(f'Test {shot_id} is not an available SMAJ400A test.')
    train = conditions.loc[conditions.shot_id.ne(shot_id)].reset_index(drop=True)
    configure_knots(10, coarse_tail=True)
    features = []
    for index, row in enumerate(train.itertuples(), 1):
        report(f'Loading training test {row.shot_id} ({index}/{len(train)})...')
        wave = read_waveform(row, raw_waveforms)
        features.append(waveform_features(wave, row.shot_id))
    report(f'Fitting on {len(train)} SMAJ400A tests; test {shot_id} is held out...')
    fitted = fit_expanded(train, pd.DataFrame(features), extra=True)
    parameters = predict(fitted, test).iloc[0].to_dict()
    report(f'Comparing the prediction with test {shot_id}...')
    row = next(test.itertuples())
    wave = read_waveform(row, raw_waveforms)
    time = wave.time_ns.to_numpy()
    predicted = reconstruct(time, parameters)
    score = _metrics(wave.actual_v.to_numpy(), predicted, (time >= -60) & (time <= 1000))
    full = _metrics(wave.actual_v.to_numpy(), predicted)
    return dict(shot_id=int(shot_id), diode_type=TRAINING_DIODE, wave=wave[['time_ns', 'actual_v']].copy(),
                predicted=predicted, score=score, full=full, train_shot_ids=train.shot_id.tolist(),
                kernel=fitted['kernel'], alpha=fitted['alpha'], conditions=conditions)


def run(spacing=10, extra=False, balanced=False, gain=False, coarse_tail=False, raw_waveforms=False, metadata=None):
    if spacing <= 0:
        raise ValueError('Knot spacing must be positive.')
    name='dense_'+str(spacing)+'ns'+('_hybrid' if extra else '')+('_balanced' if balanced else '')+('_gain' if gain else '')+('_coarse_tail' if coarse_tail else '')+('_documented' if '--documented' in sys.argv else '')
    name += '_smaj400a'
    output_dir=ROOT/'research'/f'{name}_waveforms'
    output_dir.mkdir(parents=True, exist_ok=True)
    c = load_smaj_conditions(metadata)
    print(f'Using {len(c)} SMAJ400A shots from {metadata or ROOT/"simease_ridge_metadata.csv"}.', flush=True)
    if gain and 'scope_attenuation_setting' not in c:
        raise ValueError('--gain requires scope_attenuation_setting in the metadata CSV.')
    c.to_csv(output_dir/'conditions_used.csv', index=False)
    configure_knots(spacing, coarse_tail)
    f=[]; waves={}
    for row in c.itertuples():
        w = read_waveform(row, raw_waveforms)
        waves[row.shot_id]=w
        f.append(waveform_features(w, row.shot_id))
        print(f'Loaded shot {row.shot_id} ({len(f)}/{len(c)})', flush=True)
    f=pd.DataFrame(f); rows=[]
    for row in c.itertuples():
        train=c[c.shot_id!=row.shot_id]; test=c[c.shot_id==row.shot_id]
        fitted=fit_expanded(train,f,extra,balanced,train.scope_attenuation_setting.to_numpy() if gain else None) if extra else fit_model(train,f)
        params=predict(fitted,test).iloc[0].to_dict()
        if fitted.get('gain_mode'): params.update({name:params[name]*float(test.scope_attenuation_setting.iloc[0]) for name in TARGETS})
        w=waves[row.shot_id]; pred=reconstruct(w.time_ns.to_numpy(),params)
        score=_metrics(w.actual_v.to_numpy(),pred,PROMPT)
        full=_metrics(w.actual_v.to_numpy(),pred)
        plot_waveform(w,pred,row.shot_id,row.diode_type,score,full,output_dir)
        pd.DataFrame(dict(time_ns=w.time_ns,actual_v=w.actual_v,predicted_v=pred)).to_csv(output_dir/f'{row.shot_id}_waveform.csv',index=False)
        rows.append(dict(shot_id=row.shot_id,diode_type=row.diode_type,**score,full_rmse_v=full['rmse_v'],kernel=fitted['kernel'],alpha=fitted['alpha'],gain_mode=fitted.get('gain_mode',False),n_train=len(train),train_shot_ids=';'.join(map(str, train.shot_id))))
        print(row.shot_id,score['rmse_v'],flush=True)
    result=pd.DataFrame(rows)
    result.to_csv(ROOT/'research'/f'{name}.csv',index=False)
    result.to_csv(output_dir/'metrics.csv',index=False)
    cards = []
    for diode_type, shots in result.groupby('diode_type', sort=True):
        cards.append(f'<h2>{escape(diode_type)}</h2><div class="grid">')
        for shot in shots.itertuples():
            filename = f'{shot.shot_id}_predicted_vs_actual.png'
            cards.append(
                f'<figure><a href="{filename}"><img src="{filename}" loading="lazy" '
                f'alt="Shot {shot.shot_id}: predicted and actual waveform"></a>'
                f'<figcaption>Shot {shot.shot_id} | Prompt RMSE: {shot.rmse_v:.3f} V'
                f'</figcaption></figure>')
        cards.append('</div>')
    (output_dir/'index.html').write_text(
        '<!doctype html><html lang="en"><meta charset="utf-8">'
        '<meta name="viewport" content="width=device-width, initial-scale=1">'
        '<title>Diode waveform predictions</title><style>'
        'body{font-family:system-ui;margin:2rem;background:#f4f6f8;color:#172330}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(min(100%,480px),1fr));gap:1rem}'
        'figure{margin:0;padding:1rem;background:white;border-radius:8px}'
        'img{width:100%;height:auto}figcaption{padding-top:.5rem}'
        '</style><h1>Diode waveform predictions</h1>'
        f'<p>{len(result)} SMAJ400A-only leave-one-shot-out predictions. Training and inner model selection use SMAJ400A shots only. Click an image to view full size.</p>'
        '<p><a href="metrics.csv">Download metrics</a></p>'
        + ''.join(cards) + '</html>', encoding='utf-8')
    print(f'Saved {len(result)} waveform PNGs and prediction CSVs to {output_dir}',flush=True)
    print(result.groupby('diode_type')[['rmse_v','full_rmse_v']].mean().to_string())
    print(result[['rmse_v','full_rmse_v']].mean().to_string())


if __name__=='__main__':
    parser = argparse.ArgumentParser(description='SMAJ400A ridge model driven by an editable metadata CSV.')
    parser.add_argument('spacing', type=int, nargs='?', default=10)
    parser.add_argument('--metadata', type=Path, default=ROOT/'simease_ridge_metadata.csv')
    parser.add_argument('--documented', action='store_true', help='Keep the historical documented output-folder suffix; input paths come from metadata.')
    for flag in ('extra', 'balanced', 'gain', 'coarse-tail', 'raw-waveforms'):
        parser.add_argument('--'+flag, action='store_true')
    args = parser.parse_args()
    run(args.spacing, args.extra, args.balanced, args.gain, args.coarse_tail, args.raw_waveforms, args.metadata)


