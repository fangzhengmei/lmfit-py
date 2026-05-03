"""
Parameter Range Scanning
=========================

This example demonstrates how to use the parameter scanning functionality
in lmfit to explore how different parameter values affect fitting results.

Parameter scanning is useful when:
- You want to understand how a parameter influences the fit quality
- You need to find good initial values for parameters
- You want to visualize the chi-square landscape around the best fit
- You suspect multiple local minima and want to explore the parameter space

"""

###############################################################################
# Overview
# --------
#
# The parameter scanning functionality provides:
#
# 1. **ParameterScan class**: The main class for managing parameter scans
# 2. **scan_parameters function**: A convenience function for quick scans
# 3. **scan_report function**: Generate text reports of scan results
# 4. **scan_report_html_table function**: Generate HTML reports for Jupyter
#
# Let's start with a simple example using Gaussian data.
#

import numpy as np

from lmfit import (
    Model,
    ParameterScan,
    ScanResult,
    scan_parameters,
    scan_report,
    report_scan
)
from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel, LinearModel

###############################################################################
# Example 1: Single Parameter Scan
# ---------------------------------
#
# Let's create some synthetic data and scan the 'center' parameter of a
# Gaussian model to see how it affects the fit quality.
#

print("=" * 70)
print("Example 1: Single Parameter Scan")
print("=" * 70)

# Create synthetic data with noise
np.random.seed(42)
x = np.linspace(0, 10, 101)
y_true = gaussian(x, amplitude=10.0, center=5.0, sigma=1.0)
y = y_true + np.random.normal(scale=0.1, size=x.size)

# Create a Gaussian model
model = GaussianModel()

# Create a ParameterScan instance
# - model: the lmfit Model to use
# - data: the data to fit
# - fit_kws: additional arguments to pass to fit() (e.g., x values)
scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

# Add a scan range for the 'center' parameter
# There are three ways to specify the range:
#
# Method 1: Using start, end, and num (number of points)
scanner.add_scan_range('center', start=4.0, end=6.0, num=5)

# Method 2: Using start, end, and step (step size)
# scanner.add_scan_range('center', start=4.0, end=6.0, step=0.5)

# Method 3: Using explicit values
# scanner.add_scan_range('center', start=0, end=1, values=[4.0, 4.5, 5.0, 5.5, 6.0])

print(f"\nScan ranges: {scanner.scan_ranges}")
print(f"Center values: {scanner.scan_ranges['center']}")

# Run the scan
# - verbose: if True, print progress information
# - base_params: optional Parameters to use as base (will be modified)
results = scanner.run(verbose=True)

print(f"\nScan completed with {len(results)} points")

###############################################################################
# Interpreting the Results
# ------------------------
#
# Each ScanResult contains:
# - param_values: dict of parameter values at this scan point
# - chisqr: chi-square value
# - redchi: reduced chi-square
# - aic: Akaike Information Criterion
# - bic: Bayesian Information Criterion
# - success: whether the fit succeeded
# - params: the optimized Parameters after fitting
# - fit_result: the full ModelResult or MinimizerResult

print("\n--- Interpreting Results ---")

# Get the best result based on a metric (chisqr, redchi, aic, bic)
best = scanner.get_best_result(metric='chisqr')
print(f"\nBest result:")
print(f"  Parameter values: {best.param_values}")
print(f"  chi-square: {best.chisqr:.4f}")
print(f"  reduced chi-square: {best.redchi:.4f}")
print(f"  AIC: {best.aic:.4f}")
print(f"  BIC: {best.bic:.4f}")

# Get results as arrays for analysis or plotting
arrays = scanner.get_results_array()
print(f"\nResults as arrays:")
print(f"  center values: {arrays['center']}")
print(f"  chi-square values: {arrays['chisqr']}")

###############################################################################
# Generating Reports
# ------------------
#
# Use scan_report() to generate a comprehensive text report, or
# scan_report_html_table() for HTML output (useful in Jupyter notebooks).

print("\n--- Generating Reports ---")
print("\n" + "=" * 70)
print("Scan Report")
print("=" * 70)

# Generate a text report
report = scan_report(
    scanner,
    show_best=True,      # Show the best fit result
    show_summary=True,    # Show summary statistics
    show_all=False,       # Don't show all points (use max_rows instead)
    sort_by='chisqr',     # Sort by chi-square
    max_rows=10           # Show up to 10 rows
)
print(report)

# Or use report_scan() to print directly
# report_scan(scanner)

###############################################################################
# Example 2: Multi-Parameter Grid Scan
# -------------------------------------
#
# You can scan multiple parameters simultaneously. The scanner will perform
# a grid search over all combinations of parameter values.

print("\n" + "=" * 70)
print("Example 2: Multi-Parameter Grid Scan")
print("=" * 70)

# Create new scanner
scanner2 = ParameterScan(model=model, data=y, fit_kws={'x': x})

# Scan two parameters: center and sigma
# This will create a 3 x 3 = 9 point grid search
scanner2.add_scan_range('center', start=4.5, end=5.5, num=3)
scanner2.add_scan_range('sigma', start=0.8, end=1.2, num=3)

print(f"\nScanning parameters: {list(scanner2.scan_ranges.keys())}")
print(f"Number of combinations: {len(scanner2.scan_ranges['center']) * len(scanner2.scan_ranges['sigma'])}")

# Run the scan
results2 = scanner2.run(verbose=False)

print(f"\nCompleted {len(results2)} fits")

# Get the best result
best2 = scanner2.get_best_result()
print(f"\nBest result from grid scan:")
print(f"  center = {best2.param_values['center']:.4f}")
print(f"  sigma = {best2.param_values['sigma']:.4f}")
print(f"  chi-square = {best2.chisqr:.4f}")

###############################################################################
# Example 3: Using the Convenience Function scan_parameters
# ----------------------------------------------------------
#
# For quick scans, use the scan_parameters() convenience function.

print("\n" + "=" * 70)
print("Example 3: Using scan_parameters() Convenience Function")
print("=" * 70)

# Quick one-line scan
scanner3 = scan_parameters(
    model_or_minimizer=model,
    scan_ranges={
        'center': np.linspace(4.8, 5.2, 3),
        'sigma': [0.9, 1.0, 1.1]
    },
    data=y,
    fit_kws={'x': x},
    verbose=True
)

print(f"\nQuick scan completed with {len(scanner3.scan_results)} points")

###############################################################################
# Example 4: Scanning with Composite Models
# ------------------------------------------
#
# Parameter scanning works with composite models too.

print("\n" + "=" * 70)
print("Example 4: Scanning with Composite Models")
print("=" * 70)

# Create data with Gaussian peak + linear background
y_gauss = gaussian(x, amplitude=8.0, center=5.0, sigma=1.0)
y_linear = 0.5 * x + 2.0
y_complex = y_gauss + y_linear + np.random.normal(scale=0.1, size=x.size)

# Create composite model: Gaussian + Linear
composite_model = GaussianModel() + LinearModel()

# Create initial parameters
params = composite_model.make_params(
    amplitude=5.0, center=4.5, sigma=0.8,
    slope=0.4, intercept=1.5
)

# Create scanner for composite model
scanner4 = ParameterScan(model=composite_model, data=y_complex, fit_kws={'x': x})

# Scan the center parameter while varying others
scanner4.add_scan_range('center', start=4.0, end=6.0, num=5)

print("\nScanning 'center' parameter in composite model...")
results4 = scanner4.run(base_params=params, verbose=False)

best4 = scanner4.get_best_result()
print(f"\nBest center: {best4.param_values['center']:.4f}")
print(f"chi-square: {best4.chisqr:.4f}")

###############################################################################
# Example 5: Error Handling and Common Pitfalls
# ----------------------------------------------
#
# The parameter scanner has built-in validation for common errors.
# Here are some error scenarios and how to handle them.

print("\n" + "=" * 70)
print("Example 5: Error Handling and Common Pitfalls")
print("=" * 70)

scanner5 = ParameterScan(model=model, data=y, fit_kws={'x': x})

# Error 1: Empty values list
print("\n--- Error: Empty values list ---")
try:
    scanner5.add_scan_range('center', start=0, end=1, values=[])
except ValueError as e:
    print(f"Caught expected error: {e}")

# Error 2: num=0
print("\n--- Error: num=0 ---")
try:
    scanner5.add_scan_range('center', start=4.0, end=6.0, num=0)
except ValueError as e:
    print(f"Caught expected error: {e}")

# Error 3: step=0
print("\n--- Error: step=0 ---")
try:
    scanner5.add_scan_range('center', start=4.0, end=6.0, step=0.0)
except ValueError as e:
    print(f"Caught expected error: {e}")

# Error 4: No scan ranges defined before run()
print("\n--- Error: No scan ranges ---")
try:
    empty_scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})
    empty_scanner.run()
except ValueError as e:
    print(f"Caught expected error: {e}")

# Good practice: Always validate your inputs
print("\n--- Good Practice: Valid Inputs ---")
valid_scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})
valid_scanner.add_scan_range('center', start=4.0, end=6.0, num=3)
print(f"Valid scan range: {valid_scanner.scan_ranges['center']}")

###############################################################################
# Summary
# -------
#
# Key takeaways:
#
# 1. **ParameterScan** is the main class for managing scans
# 2. Use **add_scan_range()** with num, step, or values to define ranges
# 3. **run()** executes the scan and stores results
# 4. **get_best_result()** finds the optimal fit
# 5. **get_results_array()** returns data for analysis/plotting
# 6. **scan_report()** generates comprehensive reports
# 7. **scan_parameters()** is a convenience function for quick scans
#
# Common errors to watch for:
# - Empty values lists
# - num <= 0
# - step == 0
# - Forgetting to add scan ranges before run()
#

print("\n" + "=" * 70)
print("Summary")
print("=" * 70)
print("""
Parameter scanning workflow:

1. Create a ParameterScan:
   scanner = ParameterScan(model=my_model, data=y, fit_kws={'x': x})

2. Add scan ranges:
   scanner.add_scan_range('param1', start=0, end=10, num=5)
   scanner.add_scan_range('param2', start=0, end=1, values=[0.1, 0.5, 0.9])

3. Run the scan:
   results = scanner.run(verbose=True)

4. Analyze results:
   best = scanner.get_best_result(metric='chisqr')
   arrays = scanner.get_results_array()

5. Generate report:
   report = scan_report(scanner)
   print(report)

Or use the one-liner:
   scanner = scan_parameters(
       model_or_minimizer=my_model,
       scan_ranges={'param1': [1, 2, 3], 'param2': np.linspace(0, 1, 5)},
       data=y,
       fit_kws={'x': x}
   )
""")
