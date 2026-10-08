from pathlib import Path
from html import escape
import sys
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from scipy.ndimage import gaussian_filter1d
from scipy.interpolate import PchipInterpolator

ROOT=Path(__file__).resolve().parent

# Local v2 calculation routines preserve equations and candidate ordering.
GRID_NS = np.arange(-150.0, 5000.1, 2.0)


KNOTS_NS = np.array([-150, -100, -75, *range(-60, 81, 5),
                     100, 125, 150, 180, 220, 270, 330, 400, 500, 650,
                     800, 1000, 1250, 1600, 2000, 2500, 3200, 4000, 5000], float)


TARGETS = {f"knot_{i:02d}_v": "identity" for i in range(len(KNOTS_NS))}


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
    """Fixed unit scaling avoids fitting preprocessing on validation shots."""
    dose = np.log10(frame.dose_rate.to_numpy(float)/1e10)
    bias = frame.bias_v.to_numpy(float)/5.25
    high = (frame.load_ohm.to_numpy(float) >= 1e5).astype(float)
    low = (1-high)*(frame.load_ohm.to_numpy(float)/81.3-1)
    width = frame.pcd_fwhm_ns.to_numpy(float)/40-1
    return np.column_stack([dose, bias, high, low, width,
                            dose*bias, dose*high, bias*high, dose*low])


def _kernel(left: pd.DataFrame, right: pd.DataFrame, kind: str) -> np.ndarray:
    x, y = encode(left), encode(right)
    same = left.diode_type.to_numpy()[:, None] == right.diode_type.to_numpy()[None, :]
    if kind == "linear":
        return x@y.T + same*(1+x@y.T)
    if kind.startswith("task:"):
        _, length, sharing = kind.split(":")
        length, sharing = float(length), float(sharing)
        if length <= 0 or not 0 <= sharing <= 1:
            raise ValueError("Task kernel requires positive length and sharing in [0, 1].")
        weights = np.array([1, 2, 1.5, 2, 2], float)
        distance = np.sum(((x[:, None, :5]-y[None, :, :5])*weights)**2, axis=2)
        # PSD product of an RBF condition kernel and a task covariance matrix.
        return np.exp(-distance/(2*length**2))*(sharing+(1-sharing)*same)
    length = float(kind.split(":")[1])
    weights = np.array([1, 2, 1.5, 2, 2], float)
    distance = np.sum(((x[:, None, :5]-y[None, :, :5])*weights)**2, axis=2)
    return np.exp(-(distance+4*(~same))/(2*length**2))


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


def _solve_family_mean(kernel: np.ndarray, targets: np.ndarray, alpha: float,
                       families: np.ndarray) -> tuple[list, np.ndarray]:
    """Universal kernel ridge: unpenalized family mean plus kernel residual.

    The block normal equations jointly fit means and residuals. Centering by
    precomputed family averages would leak inner holdout labels into PRESS.
    See Rasmussen & Williams (2006), section 2.7, for explicit mean functions.
    """
    models, errors = [], np.full_like(targets, np.nan)
    patterns: dict[bytes, list[int]] = {}
    for column in range(targets.shape[1]):
        patterns.setdefault(np.isfinite(targets[:, column]).tobytes(), []).append(column)
    for columns in patterns.values():
        indices = np.flatnonzero(np.isfinite(targets[:, columns[0]]))
        if not len(indices):
            raise ValueError(f"No training coverage for knot {columns[0]}")
        values = targets[np.ix_(indices, columns)]
        names = sorted(set(families[indices]))
        trend = (families[indices, None] == np.array(names)[None, :]).astype(float)
        k = kernel[np.ix_(indices, indices)]
        n, p = trend.shape
        block = np.block([[k+alpha*np.eye(n), trend], [trend.T, np.zeros((p, p))]])
        inverse = np.linalg.inv(block)
        solution = inverse@np.vstack([values, np.zeros((p, values.shape[1]))])
        design = np.column_stack([k, trend])
        denominator = 1-np.diag(design@inverse[:, :n])
        # Removing a family's sole observation makes its free mean unidentified.
        # Such cases cannot contribute a PRESS score for this candidate.
        good = denominator > 1e-8
        errors[np.ix_(indices[good], columns)] = (values-design@solution)[good]/denominator[good, None]
        models.append(dict(columns=columns, indices=indices, families=names,
                           coefficient=solution[:n], trend=solution[n:]))
    return models, errors


def fit_model(conditions: pd.DataFrame, features: pd.DataFrame) -> dict:
    if len(conditions) < 3:
        raise ValueError("Need at least three training shots.")
    labels = features.set_index("shot_id").loc[conditions.shot_id]
    targets = labels[list(TARGETS)].to_numpy(float)
    peak = np.maximum(labels.peak_v.to_numpy(float), 1e-6)
    focus = (KNOTS_NS >= -60) & (KNOTS_NS <= 1000)
    times = KNOTS_NS[focus]
    weights = np.diff(np.r_[times[0], (times[:-1]+times[1:])/2, times[-1]])
    best, candidates = None, []
    candidates_spec = [(kind, False) for kind in ("linear", "rbf:0.5", "rbf:1.0", "rbf:2.0")]
    # Freeze a small research menu before evaluation; all choices use inner LOO.
    # Families with only one training shot cannot validate a free family mean.
    if conditions.groupby("diode_type").size().min() >= 2:
        candidates_spec += [(f"task:{length}:{sharing}", True)
                            for length in (.5, 1.0, 2.0) for sharing in (0.0, .25, 1.0)]
    for kind, family_mean in candidates_spec:
        kernel_matrix = kernel(conditions, conditions, kind)
        for alpha in (.01, .1, 1.0, 10.0):
            models, errors = (_solve_family_mean(kernel_matrix, targets, alpha, conditions.diode_type.to_numpy())
                              if family_mean else _solve(kernel_matrix, targets, alpha))
            squared = (errors[:, focus]/peak[:, None])**2
            available = np.isfinite(squared)
            denom = np.sum(available*weights, axis=1)
            per_shot = np.nansum(squared*weights, axis=1)/np.maximum(denom, 1e-12)
            normalized_score = float(np.mean(per_shot[denom > 0]))
            # Align selection with the reported mean waveform RMSE in volts.
            score = float(np.mean(np.sqrt(per_shot[denom > 0])*peak[denom > 0]))
            candidates.append(dict(kernel=kind, alpha=alpha, family_mean=family_mean,
                                   inner_loo_rmse_v=score, inner_loo_normalized_mse=normalized_score))
            if best is None or score < best["inner_score"]:
                best = dict(kernel=kind, alpha=alpha, family_mean=family_mean,
                            inner_score=score, groups=models)
    best["conditions"] = conditions.reset_index(drop=True).copy()
    best["candidates"] = candidates
    return best


def predict(model: dict, conditions: pd.DataFrame) -> pd.DataFrame:
    kernel_matrix = kernel(conditions, model["conditions"], model["kernel"])
    targets = np.empty((len(conditions), len(TARGETS)))
    for group in model["groups"]:
        k = kernel_matrix[:, group["indices"]]
        if model.get("family_mean", False):
            trend = (conditions.diode_type.to_numpy()[:, None] == np.array(group["families"])[None, :]).astype(float)
            # No family coverage at a late knot: use the available means equally.
            absent = trend.sum(axis=1) == 0
            trend[absent] = 1/len(group["families"])
            values = k@group["coefficient"]+trend@group["trend"]
        else:
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


def _metrics(actual: np.ndarray, predicted: np.ndarray, mask: np.ndarray) -> dict:
    valid = mask & np.isfinite(actual) & np.isfinite(predicted)
    if valid.sum() < 5:
        raise ValueError("Not enough observed samples to evaluate waveform")
    residual = actual[valid]-predicted[valid]
    sst = float(np.sum((actual[valid]-actual[valid].mean())**2))
    return dict(rmse_v=float(np.sqrt(np.mean(residual**2))),
                mae_v=float(np.mean(np.abs(residual))),
                r2=1-float(np.sum(residual**2))/sst if sst > 1e-15 else np.nan,
                n_samples=int(valid.sum()))


def plot_waveform(wave, predicted, shot_id, diode_type, score, full, output_dir):
    """Save the held-out prediction alongside the original measured voltage."""
    figure, axes = plt.subplots(2, 1, figsize=(11, 8))
    for axis in axes:
        axis.plot(wave.time_ns, wave.actual_v, label='Actual diode output', color='#426b9a')
        axis.plot(wave.time_ns, predicted, label='Predicted output (held-out shot)',
                  color='#d26828', linestyle='--')
        axis.set_xlabel('Time (ns)')
        axis.set_ylabel('Diode output (V)')
        axis.grid(alpha=0.25)
        axis.legend()
    axes[0].set_title(f'Full waveform | RMSE (time >= 0 ns): {full["rmse_v"]:.3f} V')
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
    figure.savefig(output_dir / f'{shot_id}_predicted_vs_actual.png', dpi=160)
    plt.close(figure)


def fit_expanded(conditions, features, extra=False, balanced=False, gains=None):
    labels=features.set_index('shot_id').loc[conditions.shot_id]
    target=labels[list(TARGETS)].to_numpy(float)
    focus=(KNOTS_NS>=-60)&(KNOTS_NS<=1000)
    t=KNOTS_NS[focus]; weights=np.diff(np.r_[t[0],(t[:-1]+t[1:])/2,t[-1]])
    best=None
    gain_modes=(False,True) if gains is not None else (False,)
    for gain_mode in gain_modes:
      gain=gains if gain_mode else np.ones(len(conditions))
      y=target/gain[:,None]
      for kind in ['linear','task:0.5:0.0','task:1.0:0.0','task:2.0:0.0']+(['hybrid:0.5','hybrid:1.0','hybrid:2.0','regime:0.5','regime:1.0','regime:2.0'] if extra else []):
        k=kernel(conditions,conditions,kind)
        for alpha in (.01,.1,1.,10.):
            groups,errors=_solve_family_mean(k,y,alpha,conditions.diode_type.to_numpy())
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
                best=dict(kernel=kind,alpha=alpha,family_mean=True,groups=groups,conditions=conditions,inner_score=score,gain_mode=gain_mode)
    return best


def kernel(left,right,kind):
    if kind.startswith(('hybrid:','regime:')):
        length=float(kind.split(':')[1]);x,y=encode(left),encode(right)
        same=left.diode_type.to_numpy()[:,None]==right.diode_type.to_numpy()[None,:]
        weights=np.array([1,2,1.5,2,2])
        d=np.sum(((x[:,None,:5]-y[None,:,:5])*weights)**2,axis=2)
        rb=np.exp(-d/(2*length**2))*same
        features=[0,1,2,3,5,6,7,8] if kind.startswith('hybrid') else [1,2,3,7]
        return rb+same*(x[:,features]@y[:,features].T)
    return ORIGINAL_KERNEL(left,right,kind)


ORIGINAL_KERNEL=_kernel


def run(spacing=10, extra=False, balanced=False, gain=False, coarse_tail=False, raw_waveforms=False):
    global KNOTS_NS, TARGETS
    if spacing <= 0:
        raise ValueError('Knot spacing must be positive.')
    name='dense_'+str(spacing)+'ns'+('_hybrid' if extra else '')+('_balanced' if balanced else '')+('_gain' if gain else '')+('_coarse_tail' if coarse_tail else '')+('_documented' if '--documented' in sys.argv else '')
    output_dir=ROOT/'research'/f'{name}_waveforms'
    output_dir.mkdir(parents=True, exist_ok=True)
    source=ROOT/('research/documented_v2_results' if '--documented' in sys.argv else 'simease_multiparams_v2_results')
    c=pd.read_csv(source/'conditions_used.csv')
    if gain: c=c.merge(pd.read_csv(ROOT/'research/scope_attenuation_settings.csv'),on='shot_id',validate='1:1')
    old=pd.read_csv(source/'predictions.csv').set_index('shot_id')
    KNOTS_NS=np.unique(np.r_[KNOTS_NS, np.arange(-60,1001,spacing), [] if coarse_tail else np.arange(1000,5001,25)])
    TARGETS={f'dense_{i:03d}_v':'identity' for i in range(len(KNOTS_NS))}
    f=[]; waves={}
    if raw_waveforms:
        from models.simease_spline_v2 import extract_shot
    for row in c.itertuples():
        if raw_waveforms:
            features, measured = extract_shot(ROOT/'data'/f'{row.shot_id}_data.csv', row.scope_scale_factor)
            if not np.isclose(features['peak_v'], old.loc[row.shot_id, 'actual_peak_v'], rtol=1e-7, atol=1e-10):
                raise ValueError(f'Shot {row.shot_id}: raw calibration does not match the selected dataset.')
            w = pd.DataFrame(dict(time_ns=measured['time_ns'], actual_v=measured['voltage_v']))
        else:
            w=pd.read_csv(source/f'{row.shot_id}_waveform.csv')
        waves[row.shot_id]=w
        good=w.actual_v.notna(); t=w.time_ns.to_numpy()[good]; v=w.actual_v.to_numpy()[good]
        smooth=gaussian_filter1d(v,1.5/2)
        values=np.interp(KNOTS_NS,t,smooth,left=np.nan,right=np.nan)
        f.append(dict(shot_id=row.shot_id,peak_v=old.loc[row.shot_id,'actual_peak_v'],**dict(zip(TARGETS,values))))
        print(f'Loaded shot {row.shot_id} ({len(f)}/{len(c)})', flush=True)
    f=pd.DataFrame(f); rows=[]
    for row in c.itertuples():
        train=c[c.shot_id!=row.shot_id]; test=c[c.shot_id==row.shot_id]
        fitted=fit_expanded(train,f,extra,balanced,train.scope_attenuation_setting.to_numpy() if gain else None) if extra else fit_model(train,f)
        params=predict(fitted,test).iloc[0].to_dict()
        if fitted.get('gain_mode'): params.update({name:params[name]*float(test.scope_attenuation_setting.iloc[0]) for name in TARGETS})
        w=waves[row.shot_id]; pred=reconstruct(w.time_ns.to_numpy(),params)
        score=_metrics(w.actual_v.to_numpy(),pred,PROMPT)
        full=_metrics(w.actual_v.to_numpy(),pred,GRID_NS>=0)
        plot_waveform(w,pred,row.shot_id,row.diode_type,score,full,output_dir)
        pd.DataFrame(dict(time_ns=w.time_ns,actual_v=w.actual_v,predicted_v=pred)).to_csv(output_dir/f'{row.shot_id}_waveform.csv',index=False)
        rows.append(dict(shot_id=row.shot_id,diode_type=row.diode_type,**score,full_rmse_v=full['rmse_v'],previous_rmse_v=old.loc[row.shot_id,'rmse_v'],kernel=fitted['kernel'],alpha=fitted['alpha'],gain_mode=fitted.get('gain_mode',False)))
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
        f'<p>{len(result)} leave-one-shot-out predictions. Click an image to view full size.</p>'
        '<p><a href="metrics.csv">Download metrics</a></p>'
        + ''.join(cards) + '</html>', encoding='utf-8')
    print(f'Saved {len(result)} waveform PNGs and prediction CSVs to {output_dir}',flush=True)
    print(result.groupby('diode_type')[['rmse_v','previous_rmse_v','full_rmse_v']].mean().to_string())
    print(result[['rmse_v','previous_rmse_v','full_rmse_v']].mean().to_string())


if __name__=='__main__':
    run(int(sys.argv[1]) if len(sys.argv)>1 else 10, '--extra' in sys.argv, '--balanced' in sys.argv, '--gain' in sys.argv, '--coarse-tail' in sys.argv, '--raw-waveforms' in sys.argv)


