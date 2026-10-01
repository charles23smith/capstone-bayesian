# Siamese Ridge waveform prediction

## Overview

- Predicts diode waveforms from shot conditions using **kernel ridge regression** with an L2 penalty.
- Uses **leave-one-shot-out validation**: train on `N - 1` shots, predict the remaining shot, and repeat for every shot.
- Selects the kernel and ridge penalty using inner training-only leave-one-out validation.
- Uses the Siamese difference formulation `g(x_i) - g(x_j)` to model waveform differences through a compressed ridge solve, without creating pair rows.
- All distinct ordered pairs total `N * (N - 1)`; an outer training fold has `(N - 1) * (N - 2)` possible pairs. Family-mean fits use within-family pairs. Pairs reuse measurements and are not independent samples.

## Waveform targets

- Models **signed voltage at fixed time knots**, rather than separately fitting peak, rise time, or pulse width.
- Retains original v2 knots and adds **10 ns prompt knots** from -60 to 1000 ns and **25 ns tail knots** through 5000 ns.
- Lightly smooths training voltages; reconstructs predictions with **PCHIP interpolation**.
- Derives peak, rise/recovery times, FWHM, undershoot, tail offset, area, and energy from the predicted curve.

## Shot inputs

- **Diode type:** identifies the diode family.
- **Dose rate:** logarithmically scaled radiation dose rate.
- **Load resistance:** termination resistance in ohms.
- **Bias voltage:** applied bias in volts.
- **PCD FWHM:** reference pulse width in nanoseconds.
- Includes dose/bias and load interactions. Shot ID is only an identifier; scope scale is upstream calibration metadata.
- Exact inputs for each shot: [`documented_v2_results/conditions_used.csv`](documented_v2_results/conditions_used.csv).

## Run

From the repository root:

```powershell
.venv\Scripts\python.exe research\dense_waveform_experiment.py 10 --documented
```

- This single command generates a held-out prediction for **each shot**. The script has no single-shot option.
- Outputs: `research/dense_10ns_documented_waveforms/<shot_id>_predicted_vs_actual.png` and `<shot_id>_waveform.csv`.
- Combined errors: `research/dense_10ns_documented_waveforms/metrics.csv`.

## Research used

- [Bonilla et al., *Multi-task Gaussian Process Prediction* (2007)](https://proceedings.neurips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html): diode-task kernels and condition similarity.
- [Rasmussen and Williams, *GPML*, Section 2.7 (2006)](https://gaussianprocess.org/gpml/chapters/RW2.pdf): explicit family means and kernel residuals.
- [Cawley and Talbot, *Model Selection Bias* (2010)](https://www.jmlr.org/papers/v11/cawley10a.html): nested model selection and held-out evaluation.
- [`research_notes.md`](research_notes.md): Siamese pair equivalence and waveform experiments.
- [`v2_notes.md`](v2_notes.md): underlying ridge implementation.
- [`metadata_audit.md`](../data/metadata_audit.md): experimental calibration evidence.
- Results are development validation; independent new shots are needed to establish prospective accuracy.
