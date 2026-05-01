"""Tests for the iteration trace feature."""

import numpy as np
import pytest

from lmfit.lineshapes import gaussian
from lmfit.minimizer import Minimizer, minimize
from lmfit.models import GaussianModel, LinearModel
from lmfit.parameter import Parameters


np.random.seed(7)
x = np.linspace(0, 20, 401)
y = gaussian(x, amplitude=24.56, center=7.6543, sigma=1.23)
y -= 0.20 * x + 3.333 + np.random.normal(scale=0.23, size=len(x))
mod = GaussianModel(prefix='peak_') + LinearModel(prefix='bkg_')


def residual(pars, x, data):
    parvals = pars.valuesdict()
    gauss = gaussian(x, parvals['peak_amplitude'], parvals['peak_center'],
                     parvals['peak_sigma'])
    linear = parvals['bkg_slope'] * x + parvals['bkg_intercept']
    return data - gauss - linear


pars = mod.make_params(peak_amplitude=21.0, peak_center=7.0,
                       peak_sigma=2.0, bkg_intercept=2, bkg_slope=0.0)


class TestIterationTrace:
    """Tests for the iteration trace feature."""

    def test_trace_minimizer_class(self):
        """Test trace with Minimizer class using store_trace=True."""
        mini = Minimizer(residual, pars, fcn_args=(x, y), store_trace=True)
        out = mini.minimize(method='leastsq')

        assert hasattr(out, 'trace')
        assert out.trace is not None
        assert len(out.trace) == out.nfev

        trace = out.get_trace()
        assert trace is not None
        assert len(trace) == out.nfev

        for i, entry in enumerate(trace):
            assert 'iter' in entry
            assert 'params' in entry
            assert 'residual' in entry
            assert 'chisqr' in entry
            assert entry['iter'] == i + 1
            assert isinstance(entry['params'], dict)

    def test_trace_minimize_function(self):
        """Test trace with minimize() function using store_trace=True."""
        out = minimize(residual, pars, args=(x, y), method='leastsq',
                       store_trace=True)

        assert hasattr(out, 'trace')
        assert out.trace is not None
        assert len(out.trace) == out.nfev

        trace = out.get_trace()
        assert trace is not None
        assert len(trace) == out.nfev

    def test_trace_model_class(self):
        """Test trace with Model class using store_trace=True."""
        out = mod.fit(y, pars, x=x, method='leastsq', store_trace=True)

        assert hasattr(out, 'trace')
        assert out.trace is not None
        assert len(out.trace) == out.nfev

        trace = out.get_trace()
        assert trace is not None
        assert len(trace) == out.nfev

    def test_trace_not_stored_when_false(self):
        """Test that trace is NOT stored when store_trace=False (default)."""
        mini = Minimizer(residual, pars, fcn_args=(x, y), store_trace=False)
        out = mini.minimize(method='leastsq')

        assert not hasattr(out, 'trace') or out.trace is None
        assert out.get_trace() is None

    def test_trace_not_stored_by_default(self):
        """Test that trace is NOT stored by default (store_trace=False)."""
        out = mod.fit(y, pars, x=x, method='leastsq')

        assert not hasattr(out, 'trace') or out.trace is None
        assert out.get_trace() is None

    def test_trace_params_evolution(self):
        """Test that trace shows parameter evolution during fitting."""
        out = mod.fit(y, pars, x=x, method='leastsq', store_trace=True)
        trace = out.get_trace()

        assert len(trace) > 0

        first_params = trace[0]['params']
        last_params = trace[-1]['params']

        for param_name in ['peak_amplitude', 'peak_center', 'peak_sigma',
                           'bkg_intercept', 'bkg_slope']:
            assert param_name in first_params
            assert param_name in last_params

        first_chisqr = trace[0]['chisqr']
        last_chisqr = trace[-1]['chisqr']
        assert last_chisqr <= first_chisqr

    def test_trace_with_different_methods(self):
        """Test trace with different fitting methods."""
        methods_to_test = ['leastsq', 'least_squares', 'nelder', 'lbfgsb']

        for method in methods_to_test:
            mini = Minimizer(residual, pars, fcn_args=(x, y),
                             store_trace=True)
            out = mini.minimize(method=method)

            assert hasattr(out, 'trace')
            assert out.trace is not None
            assert len(out.trace) == out.nfev

            trace = out.get_trace()
            assert trace is not None
            assert len(trace) == out.nfev

    def test_trace_residual_array(self):
        """Test that trace stores residual arrays correctly."""
        out = minimize(residual, pars, args=(x, y), method='leastsq',
                       store_trace=True)
        trace = out.get_trace()

        first_entry = trace[0]
        assert isinstance(first_entry['residual'], np.ndarray)
        assert len(first_entry['residual']) == len(y)

        assert 'chisqr' in first_entry
        assert isinstance(first_entry['chisqr'], (float, np.floating))

    def test_trace_get_trace_returns_copy(self):
        """Test that get_trace() returns a copy, not a reference."""
        out = mod.fit(y, pars, x=x, method='leastsq', store_trace=True)
        trace1 = out.get_trace()
        trace2 = out.get_trace()

        assert trace1 is not trace2
        assert trace1[0]['params'] is not trace2[0]['params']
