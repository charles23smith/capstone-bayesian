# Siamese ridge waveform experiment

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

## Run

From the repository root:

```powershell
.venv\Scripts\python.exe simease_ridge.py 10 --documented
```

## Research used

- [Hoerl and Kennard, *Ridge Regression: Biased Estimation for Nonorthogonal Problems*, Technometrics 12(1), 55-67 (1970)](https://doi.org/10.1080/00401706.1970.10488634): foundational ridge research. Adding `alpha * ||w||²` to squared prediction error shrinks coefficients and stabilizes fitting when predictors are correlated. This experiment applies that regularization in a kernel feature space, with unpenalized global or diode-family means; inner validation selects `alpha` from `0.01`, `0.1`, `1`, and `10`.
- [Bonilla et al., *Multi-task Gaussian Process Prediction* (2007)](https://proceedings.neurips.cc/paper/2007/hash/66368270ffd51418ec58bd793f2d9b1b-Abstract.html): diode-task kernels and condition similarity.
- [Rasmussen and Williams, *Gaussian Processes for Machine Learning*, Chapter 2, Section 2.7 (2006)](https://gaussianprocess.org/gpml/chapters/RW2.pdf): explicit mean functions plus kernel residuals motivate jointly fitting diode-family means and condition-dependent voltage changes. This is a deterministic kernel-ridge adaptation, without Bayesian uncertainty intervals.
- [Cawley and Talbot, *Model Selection Bias* (2010)](https://www.jmlr.org/papers/v11/cawley10a.html): nested model selection and held-out evaluation.
- **Siamese ridge derivation:** predict `g(x_i) - g(x_j)` against observed knot-voltage differences `y_i - y_j`. The identity `sum_(i != j) (r_i - r_j)^2 = 2*N * sum_i (r_i - mean(r))^2` makes ordered-pair regression equivalent to centered ridge with the appropriate penalty scaling. Family-mean fits apply the identity within each family and weight its pair loss by `1 / (2*n_g)`. This algebra explains the compressed solve; the pair count does not create additional independent data.
- **Experimental motivation:** documented cable reflections and oscillations motivated denser waveform knots. Cached documented voltages include upstream scope calibration; the dense experiment consumes those calibrated measurements rather than recalibrating them.
- Results are development validation; independent new shots are needed to establish prospective accuracy.
