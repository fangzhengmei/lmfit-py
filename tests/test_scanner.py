"""Tests for the parameter scanning functionality."""

import numpy as np
from numpy.testing import assert_allclose, assert_almost_equal
import pytest

import lmfit
from lmfit import Model, Parameters, ParameterScan, ScanResult, scan_parameters
from lmfit.lineshapes import gaussian
from lmfit.models import GaussianModel, LinearModel
from lmfit.printfuncs import scan_report, scan_report_html_table


@pytest.fixture()
def gaussian_data():
    """Generate test data with a Gaussian peak."""
    np.random.seed(42)
    x = np.linspace(0, 10, 101)
    y = gaussian(x, amplitude=10.0, center=5.0, sigma=1.0)
    y += np.random.normal(scale=0.1, size=x.size)
    return x, y


class TestParameterScanInitialization:
    """Tests for ParameterScan initialization."""

    def test_init_with_model(self, gaussian_data):
        """Test initialization with a Model."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y)
        assert scanner.model is not None
        assert scanner.minimizer is None
        assert np.array_equal(scanner.data, y)

    def test_init_with_minimizer(self):
        """Test initialization with a Minimizer."""
        def residual(params, x, data):
            amp = params['amplitude']
            cen = params['center']
            sigma = params['sigma']
            model = amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
            return model - data

        x = np.linspace(0, 10, 101)
        y = gaussian(x, amplitude=10.0, center=5.0, sigma=1.0)

        params = Parameters()
        params.add('amplitude', value=8.0)
        params.add('center', value=4.0)
        params.add('sigma', value=0.8)

        minimizer = lmfit.Minimizer(residual, params, fcn_args=(x, y))
        scanner = ParameterScan(minimizer=minimizer)
        assert scanner.minimizer is not None
        assert scanner.model is None

    def test_init_without_model_or_minimizer(self):
        """Test that initialization fails without model or minimizer."""
        with pytest.raises(ValueError, match="Either 'model' or 'minimizer' must be provided"):
            ParameterScan()

    def test_init_with_both_model_and_minimizer(self, gaussian_data):
        """Test that initialization fails with both model and minimizer."""
        x, y = gaussian_data
        model = GaussianModel()
        minimizer = lmfit.Minimizer(lambda p, x, d: d, Parameters())
        with pytest.raises(ValueError, match="Only one of 'model' or 'minimizer' should be provided"):
            ParameterScan(model=model, minimizer=minimizer, data=y)


class TestAddScanRange:
    """Tests for add_scan_range method."""

    def test_add_scan_range_with_num(self, gaussian_data):
        """Test adding scan range with num parameter."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y)

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        assert 'center' in scanner.scan_ranges
        assert len(scanner.scan_ranges['center']) == 5
        assert_allclose(scanner.scan_ranges['center'][0], 4.0)
        assert_allclose(scanner.scan_ranges['center'][-1], 6.0)

    def test_add_scan_range_with_step(self, gaussian_data):
        """Test adding scan range with step parameter."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y)

        scanner.add_scan_range('center', start=4.0, end=6.0, step=0.5)
        assert 'center' in scanner.scan_ranges
        assert_allclose(scanner.scan_ranges['center'][0], 4.0)
        assert_allclose(scanner.scan_ranges['center'][-1], 6.0)

    def test_add_scan_range_with_values(self, gaussian_data):
        """Test adding scan range with explicit values."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y)

        values = [4.5, 5.0, 5.5]
        scanner.add_scan_range('center', start=0, end=1, values=values)
        assert 'center' in scanner.scan_ranges
        assert_allclose(scanner.scan_ranges['center'], values)

    def test_add_scan_range_without_valid_specification(self, gaussian_data):
        """Test that add_scan_range fails without valid specification."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y)

        with pytest.raises(ValueError, match="Must provide either 'values', 'num', or 'step'"):
            scanner.add_scan_range('center', start=4.0, end=6.0)


class TestParameterScanRun:
    """Tests for the run method."""

    def test_single_parameter_scan(self, gaussian_data):
        """Test scanning a single parameter."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        results = scanner.run(verbose=False)

        assert len(results) == 5
        assert all(isinstance(r, ScanResult) for r in results)
        assert scanner.scan_results == results

    def test_multi_parameter_scan(self, gaussian_data):
        """Test scanning multiple parameters (grid search)."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.5, end=5.5, num=3)
        scanner.add_scan_range('sigma', start=0.8, end=1.2, num=3)
        results = scanner.run(verbose=False)

        assert len(results) == 9  # 3 x 3 grid

    def test_run_without_scan_ranges(self, gaussian_data):
        """Test that run fails without scan ranges."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y)

        with pytest.raises(ValueError, match="No scan ranges defined"):
            scanner.run()

    def test_get_best_result(self, gaussian_data):
        """Test getting the best result from scan."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        scanner.run(verbose=False)

        best = scanner.get_best_result(metric='chisqr')
        assert best is not None
        assert isinstance(best, ScanResult)
        assert best.success

    def test_get_best_result_invalid_metric(self, gaussian_data):
        """Test that get_best_result fails with invalid metric."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        scanner.run(verbose=False)

        with pytest.raises(ValueError, match="Invalid metric"):
            scanner.get_best_result(metric='invalid')

    def test_get_results_array(self, gaussian_data):
        """Test getting results as arrays."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        scanner.run(verbose=False)

        arrays = scanner.get_results_array()
        assert 'center' in arrays
        assert 'chisqr' in arrays
        assert 'redchi' in arrays
        assert 'aic' in arrays
        assert 'bic' in arrays
        assert 'success' in arrays
        assert len(arrays['center']) == 5


class TestScanParametersFunction:
    """Tests for the scan_parameters convenience function."""

    def test_scan_parameters_with_model(self, gaussian_data):
        """Test scan_parameters function with Model interface."""
        x, y = gaussian_data
        model = GaussianModel()

        scanner = scan_parameters(
            model_or_minimizer=model,
            scan_ranges={'center': np.linspace(4.5, 5.5, 3)},
            data=y,
            fit_kws={'x': x}
        )

        assert isinstance(scanner, ParameterScan)
        assert len(scanner.scan_results) == 3

    def test_scan_parameters_with_minimizer(self):
        """Test scan_parameters function with Minimizer interface."""
        def residual(params, x, data):
            amp = params['amplitude']
            cen = params['center']
            sigma = params['sigma']
            model = amp * np.exp(-(x - cen)**2 / (2 * sigma**2))
            return model - data

        x = np.linspace(0, 10, 101)
        y = gaussian(x, amplitude=10.0, center=5.0, sigma=1.0)

        params = Parameters()
        params.add('amplitude', value=8.0)
        params.add('center', value=5.0)
        params.add('sigma', value=1.0)

        minimizer = lmfit.Minimizer(residual, params, fcn_args=(x, y))

        scanner = scan_parameters(
            model_or_minimizer=minimizer,
            scan_ranges={'amplitude': np.linspace(8.0, 12.0, 3)}
        )

        assert isinstance(scanner, ParameterScan)
        assert len(scanner.scan_results) == 3


class TestScanReport:
    """Tests for the scan_report function."""

    def test_scan_report_text(self, gaussian_data):
        """Test generating text report."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        scanner.run(verbose=False)

        report = scan_report(scanner)
        assert isinstance(report, str)
        assert 'Parameter Scan Report' in report
        assert 'Scan Configuration' in report
        assert 'Scan Statistics' in report
        assert 'Best Fit Result' in report
        assert 'Scan Results' in report

    def test_scan_report_with_options(self, gaussian_data):
        """Test scan_report with different options."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        scanner.run(verbose=False)

        report = scan_report(
            scanner,
            show_best=False,
            show_summary=False,
            show_all=True,
            sort_by='param',
            max_rows=20
        )
        assert isinstance(report, str)

    def test_scan_report_html(self, gaussian_data):
        """Test generating HTML report."""
        x, y = gaussian_data
        model = GaussianModel()
        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})

        scanner.add_scan_range('center', start=4.0, end=6.0, num=5)
        scanner.run(verbose=False)

        html_report = scan_report_html_table(scanner)
        assert isinstance(html_report, str)
        assert '<table' in html_report
        assert 'Parameter Scan Report' in html_report

    def test_scan_report_invalid_type(self):
        """Test that scan_report fails with invalid type."""
        with pytest.raises(TypeError, match="First argument must be a ParameterScan object"):
            scan_report("not a scanner")


class TestCompositeModelScan:
    """Tests for scanning with composite models."""

    def test_scan_composite_model(self):
        """Test parameter scanning with a composite model."""
        np.random.seed(42)
        x = np.linspace(0, 10, 101)

        y_gauss = gaussian(x, amplitude=10.0, center=5.0, sigma=1.0)
        y_linear = 0.5 * x + 2.0
        y = y_gauss + y_linear + np.random.normal(scale=0.1, size=x.size)

        model = GaussianModel() + LinearModel()
        params = model.make_params(
            amplitude=8.0, center=4.5, sigma=0.8,
            slope=0.4, intercept=1.5
        )

        scanner = ParameterScan(model=model, data=y, fit_kws={'x': x})
        scanner.add_scan_range('center', start=4.0, end=6.0, num=3)
        results = scanner.run(base_params=params, verbose=False)

        assert len(results) == 3
        assert all(r.success for r in results)
