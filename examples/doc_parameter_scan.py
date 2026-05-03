# <examples/doc_parameter_scan.py>
"""
Parameter Scanning with lmfit
==============================

This example demonstrates how to perform parameter range scanning
to explore how different parameter values affect fitting results.
"""

import numpy as np

from lmfit import (
    Model,
    ParameterScan,
    scan_parameters,
    scan_report,
)
from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel

###############################################################################
# Create synthetic data
# ---------------------
# We'll create a Gaussian peak with noise for demonstration.

np.random.seed(42)
x = np.linspace(0, 10, 101)
y = gaussian(x, amplitude=10.0, center=5.0, sigma=1.0)
y += np.random.normal(scale=0.1, size=x.size)

###############################################################################
# Method 1: Using ParameterScan class
# ------------------------------------
# The ParameterScan class provides full control over the scanning process.

print("=" * 70)
print("Method 1: Using ParameterScan class")
print("=" * 70)

# Create model and scanner
model = GaussianModel()
scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

# Add scan range for 'center' parameter (5 points from 4.0 to 6.0)
scanner.add_scan_range('center', start=4.0, end=6.0, num=5)

# Run the scan
results = scanner.run(verbose=False)

# Get the best result
best = scanner.get_best_result(metric='chisqr')
print(f"\nBest fit at center = {best.param_values['center']:.2f}")
print(f"chi-square = {best.chisqr:.4f}")

# Generate a report
print("\n" + scan_report(scanner))

###############################################################################
# Method 2: Using scan_parameters convenience function
# -----------------------------------------------------
# For quick scans, use the scan_parameters() function.

print("\n" + "=" * 70)
print("Method 2: Using scan_parameters() convenience function")
print("=" * 70)

# One-line parameter scan
scanner2 = scan_parameters(
    model_or_minimizer=model,
    scan_ranges={
        'center': np.linspace(4.5, 5.5, 3),
        'sigma': [0.9, 1.0, 1.1]
    },
    data=y,
    fit_kws={'x': x}
)

print(f"\nCompleted {len(scanner2.scan_results)} fits (3x3 grid)")

best2 = scanner2.get_best_result()
print(f"Best: center={best2.param_values['center']:.2f}, "
      f"sigma={best2.param_values['sigma']:.2f}")
print(f"chi-square = {best2.chisqr:.4f}")

###############################################################################
# Results for analysis
# --------------------
# Use get_results_array() to get data for plotting or further analysis.

print("\n" + "=" * 70)
print("Results for Analysis")
print("=" * 70)

arrays = scanner.get_results_array()
print(f"\nParameter values: {arrays['center']}")
print(f"chi-square values: {arrays['chisqr']}")
print(f"Success flags: {arrays['success']}")

###############################################################################
# Error Handling
# --------------
# The scanner validates inputs and provides clear error messages.

print("\n" + "=" * 70)
print("Error Handling Examples")
print("=" * 70)

# Example 1: Empty values list raises ValueError
try:
    bad_scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})
    bad_scanner.add_scan_range('center', start=0, end=1, values=[])
except ValueError as e:
    print(f"\nEmpty values error: {e}")

# Example 2: num=0 raises ValueError
try:
    bad_scanner.add_scan_range('center', start=4, end=6, num=0)
except ValueError as e:
    print(f"num=0 error: {e}")

# Example 3: step=0 raises ValueError
try:
    bad_scanner.add_scan_range('center', start=4, end=6, step=0)
except ValueError as e:
    print(f"step=0 error: {e}")

# Example 4: No scan ranges before run() raises ValueError
try:
    empty_scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})
    empty_scanner.run()
except ValueError as e:
    print(f"No scan ranges error: {e}")

print("\n" + "=" * 70)
print("Documentation complete.")
print("=" * 70)
# <end examples/doc_parameter_scan.py>
