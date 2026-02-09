"""
PyMC Bayesian Analysis of Waveform Data
Fits a sinusoidal model to voltage vs time data
"""

import pandas as pd
import numpy as np
import pymc as pm
import arviz as az
import matplotlib.pyplot as plt

# Configure PyTensor to avoid C compilation issues on Windows
import pytensor
pytensor.config.cxx = ""  # Disable C++ compilation

# Load the waveform data
print("Loading waveform data...")
data = pd.read_csv('sampleData.csv')

# Subsample the data to speed up computation (use every 10th point)
# To use all data points for more accuracy, set subsample_factor = 1
subsample_factor = 10
data = data.iloc[::subsample_factor].reset_index(drop=True)

time = data['Time (s)'].values
voltage = data['Voltage (V)'].values

print(f"Loaded {len(time)} data points (subsampled for faster computation)")
print(f"Time range: {time.min():.3f} to {time.max():.3f} seconds")
print(f"Voltage range: {voltage.min():.3f} to {voltage.max():.3f} volts")

# Define the PyMC model
print("\nBuilding PyMC model...")
with pm.Model() as waveform_model:
    # Priors for the sinusoidal waveform parameters
    # amplitude ~ Normal(5, 2)
    amplitude = pm.Normal('amplitude', mu=5, sigma=2)
    
    # frequency ~ Normal(5, 1) - in Hz
    frequency = pm.Normal('frequency', mu=5, sigma=1)
    
    # phase ~ Uniform(-pi, pi)
    phase = pm.Uniform('phase', lower=-np.pi, upper=np.pi)
    
    # noise standard deviation ~ HalfNormal(1)
    sigma = pm.HalfNormal('sigma', sigma=1)
    
    # Expected voltage as a function of time
    expected_voltage = amplitude * pm.math.sin(2 * np.pi * frequency * time + phase)
    
    # Likelihood (observed data)
    likelihood = pm.Normal('voltage', mu=expected_voltage, sigma=sigma, observed=voltage)
    
    # Sample from the posterior
    # For faster demo: 500 samples, 250 tune, 2 chains
    # For production: increase to 2000 samples, 1000 tune, 4 chains
    print("\nSampling from posterior distributions...")
    print("Using reduced samples for faster demo (should take 1-2 minutes)...")
    trace = pm.sample(500, tune=250, chains=2, random_seed=42, 
                      return_inferencedata=True, progressbar=True,
                      cores=1)  # Use single core to avoid compilation issues

# Print summary statistics
print("\n" + "="*60)
print("POSTERIOR SUMMARY STATISTICS")
print("="*60)
print(az.summary(trace, var_names=['amplitude', 'frequency', 'phase', 'sigma']))

# Create diagnostic plots
print("\nGenerating diagnostic plots...")

# Plot 1: Trace plots
fig, axes = plt.subplots(4, 2, figsize=(12, 10))
az.plot_trace(trace, var_names=['amplitude', 'frequency', 'phase', 'sigma'], axes=axes)
plt.tight_layout()
plt.savefig('trace_plots.png', dpi=150, bbox_inches='tight')
print("Saved: trace_plots.png")

# Plot 2: Posterior distributions
fig, axes = plt.subplots(2, 2, figsize=(12, 8))
az.plot_posterior(trace, var_names=['amplitude', 'frequency', 'phase', 'sigma'], axes=axes)
plt.tight_layout()
plt.savefig('posterior_distributions.png', dpi=150, bbox_inches='tight')
print("Saved: posterior_distributions.png")

# Plot 3: Model fit visualization
posterior_samples = trace.posterior
amplitude_mean = float(posterior_samples['amplitude'].mean())
frequency_mean = float(posterior_samples['frequency'].mean())
phase_mean = float(posterior_samples['phase'].mean())

fitted_voltage = amplitude_mean * np.sin(2 * np.pi * frequency_mean * time + phase_mean)

plt.figure(figsize=(12, 6))
plt.plot(time[:200], voltage[:200], 'o', alpha=0.5, label='Observed Data', markersize=3)
plt.plot(time[:200], fitted_voltage[:200], 'r-', linewidth=2, label='Fitted Model (Posterior Mean)')
plt.xlabel('Time (s)', fontsize=12)
plt.ylabel('Voltage (V)', fontsize=12)
plt.title('Waveform Data with Bayesian Model Fit', fontsize=14, fontweight='bold')
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.savefig('model_fit.png', dpi=150, bbox_inches='tight')
print("Saved: model_fit.png")

print("\n" + "="*60)
print("FITTED PARAMETERS (Posterior Means)")
print("="*60)
print(f"Amplitude: {amplitude_mean:.3f} V")
print(f"Frequency: {frequency_mean:.3f} Hz")
print(f"Phase: {phase_mean:.3f} rad")
print(f"Noise (σ): {float(posterior_samples['sigma'].mean()):.3f} V")
print("="*60)

print("\nAnalysis complete!")
print("Check the generated PNG files for visualizations.")