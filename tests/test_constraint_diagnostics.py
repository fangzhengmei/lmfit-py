"""Tests for constraint diagnostics functionality.

This module tests the constraint validation features that check for
conflicts between parameter bounds, fixed values, and expression
constraints before fitting begins.
"""
import numpy as np
import pytest

from lmfit import (
    BoundsConflictError,
    CircularDependencyError,
    ConstraintError,
    ConstraintViolations,
    ExprResultOutOfBoundsError,
    InitialValueOutOfBoundsError,
    Minimizer,
    MinimizerException,
    Parameters,
    UndefinedVarInExprError,
    minimize,
)
from lmfit.lineshapes import gaussian, lorentzian


def residual(pars, x, sigma=None, data=None):
    """Define objective function for tests."""
    yg = gaussian(x, pars['amp_g'], pars['cen_g'], pars['wid_g'])
    yl = lorentzian(x, pars['amp_l'], pars['cen_l'], pars['wid_l'])
    model = yg + yl + pars['line_off'] + x * pars['line_slope']
    if data is None:
        return model
    if sigma is None:
        return model - data
    return (model - data) / sigma


class TestConstraintExceptions:
    """Test that constraint exception classes work correctly."""

    def test_constraint_error_base_class(self):
        """Test that ConstraintError is a subclass of MinimizerException."""
        assert issubclass(ConstraintError, MinimizerException)

    def test_bounds_conflict_error(self):
        """Test BoundsConflictError exception."""
        err = BoundsConflictError('param1', 10, 5)
        assert err.param_name == 'param1'
        assert err.min_val == 10
        assert err.max_val == 5
        assert 'param1' in str(err)
        assert 'min=10 > max=5' in str(err)

    def test_initial_value_out_of_bounds_error_low(self):
        """Test InitialValueOutOfBoundsError when value < min."""
        err = InitialValueOutOfBoundsError('param1', -5, 0, 10)
        assert err.param_name == 'param1'
        assert err.value == -5
        assert err.min_val == 0
        assert err.max_val == 10
        assert 'below lower bound' in str(err)

    def test_initial_value_out_of_bounds_error_high(self):
        """Test InitialValueOutOfBoundsError when value > max."""
        err = InitialValueOutOfBoundsError('param1', 15, 0, 10)
        assert err.param_name == 'param1'
        assert err.value == 15
        assert 'above upper bound' in str(err)

    def test_expr_result_out_of_bounds_error(self):
        """Test ExprResultOutOfBoundsError exception."""
        err = ExprResultOutOfBoundsError('param1', '2*a', 15, 0, 10)
        assert err.param_name == 'param1'
        assert err.expr == '2*a'
        assert err.result == 15
        assert '2*a' in str(err)
        assert 'evaluates to 15' in str(err)

    def test_circular_dependency_error(self):
        """Test CircularDependencyError exception."""
        err = CircularDependencyError(['a', 'b', 'a'])
        assert err.cycle == ['a', 'b', 'a']
        assert 'a -> b -> a' in str(err)
        assert 'Circular dependency' in str(err)

    def test_undefined_var_in_expr_error(self):
        """Test UndefinedVarInExprError exception."""
        err = UndefinedVarInExprError('param1', 'a + undefined_var', 'undefined_var')
        assert err.param_name == 'param1'
        assert err.expr == 'a + undefined_var'
        assert err.var_name == 'undefined_var'
        assert 'undefined_var' in str(err)
        assert 'references undefined variable' in str(err)

    def test_constraint_violations_single(self):
        """Test ConstraintViolations with a single violation."""
        err1 = InitialValueOutOfBoundsError('a', 15, 0, 10)
        exc = ConstraintViolations([err1])
        assert len(exc.violations) == 1
        assert exc.violations[0] is err1
        assert '1 constraint violation' in str(exc)

    def test_constraint_violations_multiple(self):
        """Test ConstraintViolations with multiple violations."""
        err1 = InitialValueOutOfBoundsError('a', 15, 0, 10)
        err2 = UndefinedVarInExprError('b', 'a + c', 'c')
        exc = ConstraintViolations([err1, err2])
        assert len(exc.violations) == 2
        assert '2 constraint violations' in str(exc)


class TestCheckConstraintsMethod:
    """Test the check_constraints method on Parameters class."""

    def test_no_violations(self):
        """Test that valid parameters return empty violations list."""
        pars = Parameters()
        pars.add('a', value=5.0, min=0, max=10)
        pars.add('b', value=3.0)
        pars.add('c', expr='a + b', min=0, max=20)

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 0

    def test_initial_value_out_of_bounds(self):
        """Test detection of initial value outside bounds."""
        pars = Parameters()
        pars.add('a', value=15.0, min=0, max=10)

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 1
        assert isinstance(violations[0], InitialValueOutOfBoundsError)
        assert violations[0].param_name == 'a'
        assert violations[0].value == 15.0

    def test_initial_value_below_min(self):
        """Test detection of initial value below min bound."""
        pars = Parameters()
        pars.add('a', value=-5.0, min=0, max=10)

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 1
        assert isinstance(violations[0], InitialValueOutOfBoundsError)
        assert 'below lower bound' in str(violations[0])

    def test_expr_with_undefined_variable_existing_behavior(self):
        """Test that undefined variable raises error during add().

        Note: This is existing behavior - undefined variables are detected
        immediately when the parameter is added, not during check_constraints.
        """
        pars = Parameters()
        pars.add('a', value=5.0)
        pars.add('b', value=3.0)

        with pytest.raises(NameError):
            pars.add('c', expr='a + undefined_var')

    def test_circular_dependency_two_params(self):
        """Test detection of circular dependency between two parameters."""
        pars = Parameters()
        pars.add('a', value=1.0)
        pars.add('b', value=2.0)
        pars['b'].expr = 'a + 1'
        pars['a'].expr = 'b + 1'

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 1
        assert isinstance(violations[0], CircularDependencyError)
        cycle = violations[0].cycle
        assert 'a' in cycle
        assert 'b' in cycle

    def test_circular_dependency_three_params(self):
        """Test detection of circular dependency between three parameters."""
        pars = Parameters()
        pars.add('a', value=1.0)
        pars.add('b', value=2.0)
        pars.add('c', value=3.0)
        pars['b'].expr = 'a + 1'
        pars['c'].expr = 'b + 1'
        pars['a'].expr = 'c + 1'

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 1
        assert isinstance(violations[0], CircularDependencyError)

    def test_self_reference_circular_dependency(self):
        """Test detection of self-referencing expression."""
        pars = Parameters()
        pars.add('a', value=1.0)
        pars['a'].expr = 'a * 2'

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 1
        assert isinstance(violations[0], CircularDependencyError)

    def test_expr_result_out_of_bounds(self):
        """Test detection of expression result outside bounds."""
        pars = Parameters()
        pars.add('base', value=20.0)
        pars.add('scaled', expr='base * 0.5', min=0, max=5)

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 1
        assert isinstance(violations[0], ExprResultOutOfBoundsError)
        assert violations[0].param_name == 'scaled'
        assert violations[0].result == 10.0

    def test_expr_result_below_min(self):
        """Test detection of expression result below min bound."""
        pars = Parameters()
        pars.add('base', value=5.0)
        pars.add('scaled', expr='base - 10', min=0, max=100)

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 1
        assert isinstance(violations[0], ExprResultOutOfBoundsError)
        assert 'below lower bound' in str(violations[0])

    def test_multiple_violations(self):
        """Test detection of multiple constraint violations."""
        pars = Parameters()
        pars.add('a', value=15.0, min=0, max=10)
        pars.add('b', value=-5.0, min=0, max=10)
        pars.add('base', value=20.0)
        pars.add('scaled', expr='base * 0.5', min=0, max=5)

        violations = pars.check_constraints(silent=True)
        assert len(violations) >= 2

        types = {type(v).__name__ for v in violations}
        assert 'InitialValueOutOfBoundsError' in types

    def test_raise_immediately(self):
        """Test that raise_immediately raises the first violation."""
        pars = Parameters()
        pars.add('a', value=15.0, min=0, max=10)
        pars.add('b', value=20.0, min=0, max=10)

        with pytest.raises(InitialValueOutOfBoundsError):
            pars.check_constraints(raise_immediately=True)

    def test_raise_constraint_violations(self):
        """Test that non-silent mode raises ConstraintViolations."""
        pars = Parameters()
        pars.add('a', value=15.0, min=0, max=10)

        with pytest.raises(ConstraintViolations) as exc_info:
            pars.check_constraints(silent=False)

        assert len(exc_info.value.violations) == 1
        assert isinstance(exc_info.value.violations[0], InitialValueOutOfBoundsError)


class TestFitTimeConstraintDiagnostics:
    """Test that constraint diagnostics are called during fit preparation."""

    def setup_method(self):
        """Set up test data."""
        np.random.seed(42)
        self.n = 101
        self.xmin = 0.
        self.xmax = 20.0
        self.x = np.linspace(self.xmin, self.xmax, self.n)
        self.data = (gaussian(self.x, 21, 8.1, 1.2) +
                     lorentzian(self.x, 10, 9.6, 2.4) +
                     np.random.normal(scale=0.23, size=self.n) + self.x * 0.5)

    def create_good_params(self):
        """Create valid parameters for testing."""
        pars = Parameters()
        pars.add(name='amp_g', value=10)
        pars.add(name='cen_g', value=9)
        pars.add(name='wid_g', value=1)
        pars.add(name='amp_tot', value=20)
        pars.add(name='amp_l', expr='amp_tot - amp_g')
        pars.add(name='cen_l', expr='1.5+cen_g')
        pars.add(name='wid_l', expr='2*wid_g')
        pars.add(name='line_slope', value=0.0)
        pars.add(name='line_off', value=0.0)
        return pars

    def test_fit_with_valid_params(self):
        """Test that a valid fit still works."""
        pars = self.create_good_params()
        mini = Minimizer(residual, pars, fcn_args=(self.x,),
                         fcn_kws={'data': self.data})

        result = mini.minimize(method='leastsq')
        assert result.success

    def test_fit_with_circular_dependency(self):
        """Test that circular dependency is caught before fit."""
        pars = self.create_good_params()
        pars['amp_g'].expr = 'amp_l + 1'

        mini = Minimizer(residual, pars, fcn_args=(self.x,),
                         fcn_kws={'data': self.data})

        with pytest.raises(ConstraintViolations) as exc_info:
            mini.minimize(method='leastsq')

        violations = exc_info.value.violations
        assert len(violations) >= 1
        circular_errors = [v for v in violations
                          if isinstance(v, CircularDependencyError)]
        assert len(circular_errors) >= 1

    def test_fit_with_expr_out_of_bounds(self):
        """Test that expression out of bounds is caught before fit."""
        pars = self.create_good_params()
        pars['amp_g'].set(value=30)
        pars['amp_tot'].set(value=10)
        pars['amp_l'].set(min=0, max=5)

        mini = Minimizer(residual, pars, fcn_args=(self.x,),
                         fcn_kws={'data': self.data})

        with pytest.raises(ConstraintViolations):
            mini.minimize(method='leastsq')

    def test_fit_with_undefined_var_in_expr(self):
        """Test that undefined variable raises error when setting expr.

        Note: This is existing behavior - undefined variables are detected
        immediately when the expression is set, not during fit preparation.
        """
        pars = self.create_good_params()

        with pytest.raises(NameError):
            pars['wid_l'].expr = '2*undefined_param'

    def test_minimize_function_with_constraint_violation(self):
        """Test that minimize() function also catches constraint violations."""
        pars = self.create_good_params()
        pars['amp_g'].expr = 'amp_l + 1'

        with pytest.raises(ConstraintViolations):
            minimize(residual, pars, args=(self.x,), kws={'data': self.data},
                     method='leastsq')


class TestDependencyGraph:
    """Test the dependency graph building functionality."""

    def test_build_dependency_graph(self):
        """Test building dependency graph."""
        pars = Parameters()
        pars.add('a', value=1.0)
        pars.add('b', value=2.0)
        pars.add('c', expr='a + b')
        pars.add('d', expr='c * 2')

        graph = pars._build_dependency_graph()

        assert 'a' in graph
        assert 'b' in graph
        assert 'c' in graph
        assert 'd' in graph

        assert graph['a'] == []
        assert graph['b'] == []
        assert 'a' in graph['c']
        assert 'b' in graph['c']
        assert 'c' in graph['d']


class TestEdgeCases:
    """Test edge cases for constraint diagnostics."""

    def test_infinite_bounds_not_violation(self):
        """Test that infinite bounds don't cause violations."""
        pars = Parameters()
        pars.add('a', value=5.0, min=-float('inf'), max=float('inf'))

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 0

    def test_value_at_bound_not_violation(self):
        """Test that value exactly at bound is not a violation."""
        pars = Parameters()
        pars.add('a', value=0.0, min=0, max=10)
        pars.add('b', value=10.0, min=0, max=10)

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 0

    def test_expr_value_at_bound_not_violation(self):
        """Test that expression value at bound is not a violation."""
        pars = Parameters()
        pars.add('a', value=5.0)
        pars.add('b', expr='a * 2', min=0, max=10)

        violations = pars.check_constraints(silent=True)
        assert len(violations) == 0

    def test_partially_defined_expr_deps(self):
        """Test expressions that reference both defined and undefined vars.

        Note: This is existing behavior - undefined variables are detected
        immediately when the parameter is added, not during check_constraints.
        """
        pars = Parameters()
        pars.add('a', value=5.0)
        pars.add('b', value=3.0)

        with pytest.raises(NameError):
            pars.add('c', expr='a + undefined + b')

    def test_original_params_unmodified(self):
        """Test that check_constraints doesn't modify original parameters."""
        pars = Parameters()
        pars.add('a', value=5.0)
        pars.add('b', value=3.0)
        pars.add('c', expr='a + b', min=0, max=10)

        orig_values = {name: par._val for name, par in pars.items()}

        pars.check_constraints(silent=True)

        for name, orig_val in orig_values.items():
            assert pars[name]._val == orig_val
