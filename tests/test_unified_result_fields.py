"""Regression tests for unified result field access across optimizers.

This module tests that the MinimizerResult class provides consistent
access to the objective function value (`fun`), optimal parameters (`x`),
and number of iterations (`nit`) across all optimization backends.

These tests ensure backward compatibility while providing a unified interface.
"""
import numpy as np
import pytest
from numpy.testing import assert_allclose

import lmfit
from lmfit import Minimizer, Parameters, minimize


def residual_1d(params):
    """Simple 1D test function for global optimizers."""
    x = params['x'].value
    return np.cos(14.5 * x - 0.3) + (x + 0.2) * x


def eggholder(params):
    """Eggholder function - good for testing global optimizers."""
    x0 = params['x0'].value
    x1 = params['x1'].value
    return (-(x1 + 47.0) * np.sin(np.sqrt(abs(x0 / 2.0 + (x1 + 47.0))))
            - x0 * np.sin(np.sqrt(abs(x0 - (x1 + 47.0)))))


def residual_quadratic(params):
    """Simple quadratic function for testing."""
    x = params['x'].value
    y = params['y'].value
    return (x - 1.0)**2 + (y - 2.0)**2


class TestUnifiedFunAccess:
    """Tests for unified access to the objective function value (fun)."""

    def test_basinhopping_fun(self):
        """Test that basinhopping result has unified fun access."""
        pars = Parameters()
        pars.add('x', value=1.0)
        mini = Minimizer(residual_1d, pars)

        result = mini.minimize(method='basinhopping', niter=5, seed=42)

        assert result.fun is not None, "fun should be available for basinhopping"
        assert hasattr(result, 'fun'), "fun attribute should exist"

        assert isinstance(result.fun, (int, float, np.floating)), \
            f"fun should be a number, got {type(result.fun)}"

    def test_shgo_fun(self):
        """Test that shgo result has unified fun access and backward compatibility."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='shgo', n=30, sampling_method='sobol')

        assert result.fun is not None, "fun should be available for shgo"

        assert hasattr(result, 'shgo_fun'), "shgo_fun should still exist (backward compat)"
        assert result.fun == result.shgo_fun, "fun should equal shgo_fun"

    def test_dual_annealing_fun(self):
        """Test that dual_annealing result has unified fun access."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='dual_annealing', seed=42)

        assert result.fun is not None, "fun should be available for dual_annealing"

        assert hasattr(result, 'da_fun'), "da_fun should still exist (backward compat)"
        assert result.fun == result.da_fun, "fun should equal da_fun"

    def test_direct_fun(self):
        """Test that direct result has unified fun access."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='direct')

        assert result.fun is not None, "fun should be available for direct"

        assert hasattr(result, 'direct_fun'), "direct_fun should still exist (backward compat)"
        assert result.fun == result.direct_fun, "fun should equal direct_fun"

    def test_brute_fun(self):
        """Test that brute result has unified fun access."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=5.0, max=10.0, brute_step=0.5)
        pars.add('x1', value=1.0, min=2.5, max=7.5, brute_step=0.5)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='brute', Ns=10)

        assert result.fun is not None, "fun should be available for brute"

        assert hasattr(result, 'brute_fval'), "brute_fval should still exist (backward compat)"
        assert result.fun == result.brute_fval, "fun should equal brute_fval"

    def test_ampgo_fun(self):
        """Test that ampgo result has unified fun access."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=0, max=10)
        pars.add('x1', value=1.0, min=0, max=10)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='ampgo', local='L-BFGS-B', totaliter=5)

        assert result.fun is not None, "fun should be available for ampgo"

        assert hasattr(result, 'ampgo_fval'), "ampgo_fval should still exist (backward compat)"
        assert result.fun == result.ampgo_fval, "fun should equal ampgo_fval"


class TestUnifiedXAccess:
    """Tests for unified access to the optimal parameters (x)."""

    def test_basinhopping_x(self):
        """Test that basinhopping result has unified x access."""
        pars = Parameters()
        pars.add('x', value=1.0)
        mini = Minimizer(residual_1d, pars)

        result = mini.minimize(method='basinhopping', niter=5, seed=42)

        assert result.x is not None, "x should be available for basinhopping"
        assert isinstance(result.x, np.ndarray), "x should be a numpy array"
        assert len(result.x) == 1, "x should have correct length"

    def test_shgo_x(self):
        """Test that shgo result has unified x access and backward compatibility."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='shgo', n=30, sampling_method='sobol')

        assert result.x is not None, "x should be available for shgo"

        assert hasattr(result, 'shgo_x'), "shgo_x should still exist (backward compat)"
        assert_allclose(result.x, result.shgo_x), "x should equal shgo_x"

    def test_dual_annealing_x(self):
        """Test that dual_annealing result has unified x access."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='dual_annealing', seed=42)

        assert result.x is not None, "x should be available for dual_annealing"

        assert hasattr(result, 'da_x'), "da_x should still exist (backward compat)"
        assert_allclose(result.x, result.da_x), "x should equal da_x"

    def test_direct_x(self):
        """Test that direct result has unified x access."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='direct')

        assert result.x is not None, "x should be available for direct"

        assert hasattr(result, 'direct_x'), "direct_x should still exist (backward compat)"
        assert_allclose(result.x, result.direct_x), "x should equal direct_x"

    def test_brute_x(self):
        """Test that brute result has unified x access."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=5.0, max=10.0, brute_step=0.5)
        pars.add('x1', value=1.0, min=2.5, max=7.5, brute_step=0.5)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='brute', Ns=10)

        assert result.x is not None, "x should be available for brute"

        assert hasattr(result, 'brute_x0'), "brute_x0 should still exist (backward compat)"
        assert_allclose(result.x, result.brute_x0), "x should equal brute_x0"

    def test_ampgo_x(self):
        """Test that ampgo result has unified x access."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=0, max=10)
        pars.add('x1', value=1.0, min=0, max=10)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='ampgo', local='L-BFGS-B', totaliter=5)

        assert result.x is not None, "x should be available for ampgo"

        assert hasattr(result, 'ampgo_x0'), "ampgo_x0 should still exist (backward compat)"
        assert_allclose(result.x, result.ampgo_x0), "x should equal ampgo_x0"


class TestUnifiedNitAccess:
    """Tests for unified access to the number of iterations (nit)."""

    def test_basinhopping_nit(self):
        """Test that basinhopping result has unified nit access."""
        pars = Parameters()
        pars.add('x', value=1.0)
        mini = Minimizer(residual_1d, pars)

        result = mini.minimize(method='basinhopping', niter=5, seed=42)

        assert result.nit is not None, "nit should be available for basinhopping"
        assert isinstance(result.nit, (int, np.integer)), \
            f"nit should be an integer, got {type(result.nit)}"

    def test_shgo_nit(self):
        """Test that shgo result has unified nit access and backward compatibility."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='shgo', n=30, sampling_method='sobol')

        assert result.nit is not None, "nit should be available for shgo"

        assert hasattr(result, 'shgo_nit'), "shgo_nit should still exist (backward compat)"
        assert result.nit == result.shgo_nit, "nit should equal shgo_nit"

    def test_dual_annealing_nit(self):
        """Test that dual_annealing result has unified nit access."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='dual_annealing', seed=42, maxiter=100)

        assert result.nit is not None, "nit should be available for dual_annealing"

        assert hasattr(result, 'da_nit'), "da_nit should still exist (backward compat)"
        assert result.nit == result.da_nit, "nit should equal da_nit"

    def test_direct_nit(self):
        """Test that direct result has unified nit access."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='direct')

        assert result.nit is not None, "nit should be available for direct"

        assert hasattr(result, 'direct_nit'), "direct_nit should still exist (backward compat)"
        assert result.nit == result.direct_nit, "nit should equal direct_nit"

    def test_brute_nit_is_none(self):
        """Test that brute result returns None for nit (not reported by this method)."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=5.0, max=10.0, brute_step=0.5)
        pars.add('x1', value=1.0, min=2.5, max=7.5, brute_step=0.5)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='brute', Ns=10)

        assert result.nit is None, "nit should be None for brute (not reported)"

    def test_ampgo_nit_behavior(self):
        """Test ampgo nit behavior (may or may not be available depending on implementation)."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=0, max=10)
        pars.add('x1', value=1.0, min=0, max=10)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='ampgo', local='L-BFGS-B', totaliter=5)

        assert hasattr(result, 'ampgo_eval'), "ampgo_eval should exist"
        assert result.fun is not None, "fun should be available"
        assert result.x is not None, "x should be available"


class TestSetterAndGetter:
    """Tests for the property setters and getters of fun, x, nit."""

    def test_fun_setter(self):
        """Test that setting fun works correctly."""
        result = lmfit.minimizer.MinimizerResult()

        result.fun = 10.5

        assert result.fun == 10.5, "fun getter should return the set value"
        assert '_fun' in result.__dict__, "_fun should be in __dict__"
        assert result.__dict__['_fun'] == 10.5, "_fun should have the set value"

    def test_x_setter(self):
        """Test that setting x works correctly."""
        result = lmfit.minimizer.MinimizerResult()
        expected = np.array([1.0, 2.0, 3.0])

        result.x = expected

        assert_allclose(result.x, expected), "x getter should return the set value"
        assert '_x' in result.__dict__, "_x should be in __dict__"
        assert_allclose(result.__dict__['_x'], expected), "_x should have the set value"

    def test_nit_setter(self):
        """Test that setting nit works correctly."""
        result = lmfit.minimizer.MinimizerResult()

        result.nit = 42

        assert result.nit == 42, "nit getter should return the set value"
        assert '_nit' in result.__dict__, "_nit should be in __dict__"
        assert result.__dict__['_nit'] == 42, "_nit should have the set value"

    def test_setter_takes_precedence(self):
        """Test that explicitly set values take precedence over prefixed attributes."""
        result = lmfit.minimizer.MinimizerResult()

        result.shgo_fun = 100.0
        result.da_x = np.array([1.0, 2.0])
        result.direct_nit = 50

        result.fun = 200.0
        result.x = np.array([3.0, 4.0])
        result.nit = 100

        assert result.fun == 200.0, "fun should return explicitly set value, not shgo_fun"
        assert_allclose(result.x, np.array([3.0, 4.0])), "x should return explicitly set value"
        assert result.nit == 100, "nit should return explicitly set value"


class TestUnifiedFieldConsistency:
    """Tests to ensure consistent behavior across all optimizers for fun/x/nit."""

    @pytest.mark.parametrize('method,expected_fun_type,expected_x_type,expected_nit_type', [
        ('basinhopping', (int, float, np.floating), np.ndarray, (int, np.integer)),
        ('shgo', (int, float, np.floating), np.ndarray, (int, np.integer)),
        ('dual_annealing', (int, float, np.floating), np.ndarray, (int, np.integer)),
        ('direct', (int, float, np.floating), np.ndarray, (int, np.integer)),
    ])
    def test_all_global_optimizers_have_consistent_fields(
        self, method, expected_fun_type, expected_x_type, expected_nit_type
    ):
        """Test that all global optimizers have consistent fun/x/nit access."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -10, 10), ('x1', 0, True, -10, 10))
        mini = Minimizer(residual_quadratic, pars)

        result = mini.minimize(method=method)

        assert result.fun is not None, f"fun should be available for {method}"
        assert isinstance(result.fun, expected_fun_type), \
            f"fun should be {expected_fun_type} for {method}, got {type(result.fun)}"

        assert result.x is not None, f"x should be available for {method}"
        assert isinstance(result.x, expected_x_type), \
            f"x should be {expected_x_type} for {method}, got {type(result.x)}"

        if method not in ['brute']:
            assert result.nit is not None, f"nit should be available for {method}"
            assert isinstance(result.nit, expected_nit_type), \
                f"nit should be {expected_nit_type} for {method}, got {type(result.nit)}"

    def test_brute_has_fun_and_x(self):
        """Test that brute has fun and x (nit may be None)."""
        pars = Parameters()
        pars.add('x0', value=0.0, min=-10, max=10, brute_step=1.0)
        pars.add('x1', value=0.0, min=-10, max=10, brute_step=1.0)
        mini = Minimizer(residual_quadratic, pars)

        result = mini.minimize(method='brute', Ns=5)

        assert result.fun is not None, "fun should be available for brute"
        assert result.x is not None, "x should be available for brute"

    def test_ampgo_has_fun_and_x(self):
        """Test that ampgo has fun and x."""
        pars = Parameters()
        pars.add('x0', value=0.0, min=-10, max=10)
        pars.add('x1', value=0.0, min=-10, max=10)
        mini = Minimizer(residual_quadratic, pars)

        result = mini.minimize(method='ampgo', local='L-BFGS-B', totaliter=3)

        assert result.fun is not None, "fun should be available for ampgo"
        assert result.x is not None, "x should be available for ampgo"


class TestBackwardCompatibility:
    """Tests for backward compatibility with prefixed attribute names."""

    def test_shgo_prefixed_attributes_still_work(self):
        """Test that shgo_* attributes are still accessible."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='shgo', n=30, sampling_method='sobol')

        assert hasattr(result, 'shgo_fun'), "shgo_fun should exist"
        assert hasattr(result, 'shgo_x'), "shgo_x should exist"
        assert hasattr(result, 'shgo_nit'), "shgo_nit should exist"
        assert hasattr(result, 'shgo_nfev'), "shgo_nfev should exist"

    def test_dual_annealing_prefixed_attributes_still_work(self):
        """Test that da_* attributes are still accessible."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='dual_annealing', seed=42, maxiter=100)

        assert hasattr(result, 'da_fun'), "da_fun should exist"
        assert hasattr(result, 'da_x'), "da_x should exist"
        assert hasattr(result, 'da_nit'), "da_nit should exist"
        assert hasattr(result, 'da_nfev'), "da_nfev should exist"

    def test_direct_prefixed_attributes_still_work(self):
        """Test that direct_* attributes are still accessible."""
        pars = Parameters()
        pars.add_many(('x0', 0, True, -512, 512), ('x1', 0, True, -512, 512))
        mini = Minimizer(eggholder, pars)

        result = mini.minimize(method='direct')

        assert hasattr(result, 'direct_fun'), "direct_fun should exist"
        assert hasattr(result, 'direct_x'), "direct_x should exist"
        assert hasattr(result, 'direct_nit'), "direct_nit should exist"
        assert hasattr(result, 'direct_nfev'), "direct_nfev should exist"

    def test_brute_prefixed_attributes_still_work(self):
        """Test that brute_* attributes are still accessible."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=5.0, max=10.0, brute_step=0.5)
        pars.add('x1', value=1.0, min=2.5, max=7.5, brute_step=0.5)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='brute', Ns=10)

        assert hasattr(result, 'brute_fval'), "brute_fval should exist"
        assert hasattr(result, 'brute_x0'), "brute_x0 should exist"
        assert hasattr(result, 'brute_grid'), "brute_grid should exist"
        assert hasattr(result, 'brute_Jout'), "brute_Jout should exist"

    def test_ampgo_prefixed_attributes_still_work(self):
        """Test that ampgo_* attributes are still accessible."""
        pars = Parameters()
        pars.add('x0', value=1.0, min=0, max=10)
        pars.add('x1', value=1.0, min=0, max=10)

        def alpine02(params):
            x0 = params['x0'].value
            x1 = params['x1'].value
            return -(np.sin(x0) * np.sin(1.0 * x0**2 / np.pi)**20 +
                     np.sin(x1) * np.sin(2.0 * x1**2 / np.pi)**20)

        mini = Minimizer(alpine02, pars)
        result = mini.minimize(method='ampgo', local='L-BFGS-B', totaliter=5)

        assert hasattr(result, 'ampgo_fval'), "ampgo_fval should exist"
        assert hasattr(result, 'ampgo_x0'), "ampgo_x0 should exist"
        assert hasattr(result, 'ampgo_eval'), "ampgo_eval should exist"
        assert hasattr(result, 'ampgo_msg'), "ampgo_msg should exist"
        assert hasattr(result, 'ampgo_tunnel'), "ampgo_tunnel should exist"


class TestBasinhoppingRegression:
    """Regression tests specifically for basinhopping fixes.

    Before the fix, basinhopping only set the `message` attribute from the
    scipy result, not copying `fun`, `x`, `nit`, etc. These tests ensure
    that regression is caught.
    """

    def test_basinhopping_has_fun_attribute(self):
        """Regression test: basinhopping should have fun attribute."""
        pars = Parameters()
        pars.add('x', value=1.0)
        mini = Minimizer(residual_1d, pars)

        result = mini.minimize(method='basinhopping', niter=5, seed=42)

        assert hasattr(result, 'fun'), "fun attribute should exist for basinhopping"
        assert result.fun is not None, "fun should not be None for basinhopping"

    def test_basinhopping_has_x_attribute(self):
        """Regression test: basinhopping should have x attribute."""
        pars = Parameters()
        pars.add('x', value=1.0)
        mini = Minimizer(residual_1d, pars)

        result = mini.minimize(method='basinhopping', niter=5, seed=42)

        assert hasattr(result, 'x'), "x attribute should exist for basinhopping"
        assert result.x is not None, "x should not be None for basinhopping"

    def test_basinhopping_has_nit_attribute(self):
        """Regression test: basinhopping should have nit attribute."""
        pars = Parameters()
        pars.add('x', value=1.0)
        mini = Minimizer(residual_1d, pars)

        result = mini.minimize(method='basinhopping', niter=5, seed=42)

        assert hasattr(result, 'nit'), "nit attribute should exist for basinhopping"
        assert result.nit is not None, "nit should not be None for basinhopping"
        assert result.nit == 5, f"nit should be 5 (niter=5), got {result.nit}"

    def test_basinhopping_nfev_attribute(self):
        """Test that basinhopping nfev is tracked correctly."""
        pars = Parameters()
        pars.add('x', value=1.0)
        mini = Minimizer(residual_1d, pars)

        result = mini.minimize(method='basinhopping', niter=5, seed=42)

        assert hasattr(result, 'nfev'), "nfev attribute should exist"
        assert result.nfev > 0, "nfev should be > 0"
