""" Script Summary
Inputs: dose rate, diode type, load impedance, bias voltage, PCD pulse width,
and the scope calibration factor.

The model predicts peak voltage, baseline, polarity, peak time, rise time,
recovery time, and undershoot size.

For training, compare every shot with every other shot in both directions,
creating N * (N - 1) directed pairs. Fit ridge regression to the differences
between the encoded conditions and waveform parameters. For a new shot, the
model uses only its conditions to predict the waveform parameters, then rebuilds
an approximate waveform from those predictions.

Shots sit in data/ (for laptop environment integration)
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
RIDGE = 0.01
TARGETS = {"peak_v": "nonnegative", "baseline_v": "identity", "polarity": "sign",
           "peak_time_ns": "identity", "rise_10_90_ns": "log",
           "recovery_50_ns": "log", "undershoot_ratio": "log1p"}


def read_conditions(path: Path) -> pd.DataFrame:
    frame = pd.read_csv(path)
    required = {"shot_id", "dose_rate", "diode_type", "load_ohm", "bias_v",
                "pcd_fwhm_ns", "scope_scale_factor"}
    missing = required - set(frame.columns)
    if missing:
        raise ValueError(f"Missing condition columns: {sorted(missing)}")
    frame = frame.loc[frame.diode_type.eq("SMAJ400A")].copy()
    for column in ("dose_rate", "load_ohm", "bias_v", "pcd_fwhm_ns", "scope_scale_factor"):
        frame[column] = pd.to_numeric(frame[column], errors="raise")
    if (frame[["dose_rate", "load_ohm", "scope_scale_factor"]] <= 0).any().any():
        raise ValueError("Dose, load, and scope scale must be positive.")
    return frame.sort_values("shot_id").reset_index(drop=True)


def extract_shot(path: Path, scale: float) -> tuple[dict, dict]:
    raw = pd.read_csv(path, usecols=["time1", "Diode", "PCD3_B"])
    valid = raw.time1.notna() & raw.Diode.notna()
    time_s = raw.loc[valid, "time1"].to_numpy(float)
    voltage = raw.loc[valid, "Diode"].to_numpy(float) / scale
    pcd = raw.loc[valid, "PCD3_B"].to_numpy(float)
    if len(time_s) < 5 or not np.isfinite(pcd).any():
        raise ValueError(f"Insufficient waveform data in {path}")
    time = (time_s - time_s[np.nanargmax(pcd)]) * 1e9
    baseline_samples = voltage[(time >= -150) & (time < -60)]
    baseline = float(np.median(baseline_samples)) if len(baseline_samples) >= 3 else float(np.median(voltage[:10]))
    prompt = (time >= -60) & (time <= 1000)
    index = np.flatnonzero(prompt)[np.argmax(np.abs(voltage[prompt] - baseline))]
    signed_peak = float(voltage[index] - baseline)
    polarity = 1.0 if signed_peak >= 0 else -1.0
    peak_v = abs(signed_peak)
    shape = polarity * (voltage - baseline) / max(peak_v, 1e-9)

    def crossing(level: float, start: int, stop: int) -> float:
        hits = np.flatnonzero(shape[start:stop + 1] >= level)
        if not len(hits):
            return float("nan")
        right = start + int(hits[0])
        if right == 0 or shape[right] == shape[right - 1]:
            return float(time[right])
        fraction = (level - shape[right - 1]) / (shape[right] - shape[right - 1])
        return float(time[right - 1] + fraction * (time[right] - time[right - 1]))

    ten, ninety = crossing(.1, 0, index), crossing(.9, 0, index)
    fifty = crossing(.5, index, len(time) - 1)
    minimum = index + int(np.argmin(shape[index:]))
    features = dict(peak_v=peak_v, baseline_v=baseline, polarity=polarity,
                    peak_time_ns=float(time[index]),
                    rise_10_90_ns=max(ninety - ten, 1.0) if np.isfinite([ten, ninety]).all() else 20.0,
                    recovery_50_ns=max(fifty - time[index], 1.0) if np.isfinite(fifty) else 100.0,
                    undershoot_ratio=max(0.0, -float(shape[minimum])))
    return features, {"time_ns": time, "voltage_v": voltage}


def encode(frame: pd.DataFrame) -> np.ndarray:
    dose = np.sqrt(np.maximum(frame.dose_rate.to_numpy(float) / 1e10, 1e-12)) - 1.0
    bias = frame.bias_v.to_numpy(float) / 5.25
    load = frame.load_ohm.to_numpy(float)
    high_z = (load >= 1e5).astype(float)
    low_load = (1 - high_z) * (load / 81.3 - 1)
    fluence = dose * frame.pcd_fwhm_ns.to_numpy(float) / 40.0
    return np.column_stack([bias, high_z, bias * high_z, low_load, dose, fluence,
                            dose * bias, dose * high_z, dose * low_load])


def ridge_solution(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    center = x.mean(axis=0)
    centered = x - center
    system = centered.T @ centered + RIDGE * np.eye(x.shape[1])
    coefficient = np.linalg.solve(system, centered.T @ (y - y.mean()))
    return coefficient, float(y.mean() - center @ coefficient)


def pairwise_ridge_solution(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, float]:
    """Fit on all ordered pairs, then anchor the shared score absolutely."""
    count = len(y)
    if count < 3:
        raise ValueError("Need at least three shots for directed pair training.")
    differences_x = x[:, None, :] - x[None, :, :]
    differences_y = y[:, None] - y[None, :]
    mask = ~np.eye(count, dtype=bool)
    coefficient, _ = ridge_solution(differences_x[mask], differences_y[mask])
    intercept = float(y.mean() - x.mean(axis=0) @ coefficient)
    return coefficient, intercept


def transform(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        return np.log(np.maximum(values, 1e-9))
    if kind == "log1p":
        return np.log1p(np.maximum(values, 0.0))
    return np.asarray(values, float)


def inverse(values: np.ndarray, kind: str) -> np.ndarray:
    if kind == "log":
        return np.exp(np.clip(values, -30, 30))
    if kind == "log1p":
        return np.maximum(0, np.expm1(np.clip(values, 0, 30)))
    if kind == "nonnegative":
        return np.maximum(0, values)
    if kind == "sign":
        return np.where(values >= 0, 1.0, -1.0)
    return values


def fit_model(conditions: pd.DataFrame, features: pd.DataFrame) -> dict:
    labels = features.set_index("shot_id").loc[conditions.shot_id]
    model = {}
    for name, kind in TARGETS.items():
        values = labels[name].to_numpy(float)
        valid = np.isfinite(values)
        if kind == "log":
            valid &= values > 0
        if kind in {"nonnegative", "log1p"}:
            valid &= values >= 0
        if valid.sum() < 3:
            model[name] = (np.zeros(encode(conditions).shape[1]),
                           float(np.nanmedian(values)) if valid.any() else 0.0, kind)
            continue
        x = encode(conditions.iloc[np.flatnonzero(valid)])
        coefficient, intercept = pairwise_ridge_solution(x, transform(values[valid], kind))
        model[name] = (coefficient, intercept, kind)
    return model


def predict(model: dict, conditions: pd.DataFrame) -> pd.DataFrame:
    result = conditions[["shot_id"]].reset_index(drop=True).copy()
    x = encode(conditions)
    for name, (coefficient, intercept, kind) in model.items():
        result[name] = inverse(x @ coefficient + intercept, kind)
    return result


def reconstruct(time: np.ndarray, parameters: dict) -> np.ndarray:
    peak = float(parameters["peak_time_ns"])
    rise = max(float(parameters["rise_10_90_ns"]), 1.0)
    recovery = max(float(parameters["recovery_50_ns"]), 1.0)
    t10, t90 = peak - rise, peak - rise / 8
    end = peak + recovery * 4
    knots = np.array([t10 - rise / 8, t10, t90, peak, peak + recovery, end])
    values = np.array([0.0, .1, .9, 1.0, .5, .1])
    shape = np.interp(time, knots, values, left=0.0, right=np.nan)
    tail = time > end
    shape[tail] = .1 * np.exp(-(time[tail] - end) / recovery)
    amplitude = max(0.0, parameters["peak_v"] - parameters["polarity"] * parameters["baseline_v"])
    return parameters["baseline_v"] + parameters["polarity"] * amplitude * shape


def load_data(conditions: pd.DataFrame, data_dir: Path) -> tuple[pd.DataFrame, dict]:
    features, waves = [], {}
    for row in conditions.itertuples():
        feature, wave = extract_shot(data_dir / f"{row.shot_id}_data.csv", row.scope_scale_factor)
        feature["shot_id"] = int(row.shot_id)
        features.append(feature)
        waves[int(row.shot_id)] = wave
    return pd.DataFrame(features), waves


def evaluate(conditions: pd.DataFrame, features: pd.DataFrame, waves: dict,
             output: Path, holdouts: list[int] | None, plots: bool) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    selected = conditions if holdouts is None else conditions[conditions.shot_id.isin(holdouts)]
    rows = []
    for row in selected.itertuples():
        train = conditions[conditions.shot_id != row.shot_id]
        train_features = features[features.shot_id != row.shot_id]
        parameters = predict(fit_model(train, train_features),
                             pd.DataFrame([row._asdict()])).iloc[0].to_dict()
        wave = waves[row.shot_id]
        predicted = reconstruct(wave["time_ns"], parameters)
        mask = np.isfinite(predicted) & (wave["time_ns"] >= 0)
        actual_peak = float(features.loc[features.shot_id.eq(row.shot_id), "peak_v"].iloc[0])
        rows.append(dict(shot_id=row.shot_id, actual_peak_v=actual_peak,
                         predicted_peak_v=parameters["peak_v"],
                         rmse_v=float(np.sqrt(np.mean((wave["voltage_v"][mask] - predicted[mask]) ** 2)))))
        pd.DataFrame({"time_ns": wave["time_ns"], "actual_v": wave["voltage_v"],
                      "predicted_v": predicted}).to_csv(output / f"{row.shot_id}_waveform.csv", index=False)
        if plots:
            import matplotlib.pyplot as plt
            plt.plot(wave["time_ns"], wave["voltage_v"], label="Measured")
            plt.plot(wave["time_ns"], predicted, label="Predicted")
            plt.legend(); plt.xlabel("PCD-relative time (ns)"); plt.ylabel("Physical voltage (V)")
            plt.savefig(output / f"{row.shot_id}_waveform.png", dpi=120); plt.close()
    result = pd.DataFrame(rows)
    summary = dict(model="standalone_simple_landmarks", n_test_shots=len(result),
                   peak_rmse_v=float(np.sqrt(np.mean((result.predicted_peak_v - result.actual_peak_v) ** 2))))
    result.to_csv(output / "predictions.csv", index=False)
    (output / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    return summary


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--conditions", type=Path, default=ROOT / "simease_test_conditions.csv")
    parser.add_argument("--data-dir", type=Path, default=ROOT / "data")
    parser.add_argument("--output-dir", type=Path, default=ROOT / "simease_waveform_simple_outputs")
    parser.add_argument("--holdout", nargs="+", type=int)
    parser.add_argument("--no-plots", action="store_true")
    args = parser.parse_args()
    conditions = read_conditions(args.conditions)
    features, waves = load_data(conditions, args.data_dir)
    summary = evaluate(conditions, features, waves, args.output_dir, args.holdout, not args.no_plots)
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
