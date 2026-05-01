"""Integration test to ensure existing functionality still works."""
import numpy as np
from lmfit import Parameters, Minimizer, minimize
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


def test_valid_fit_still_works():
    """Test that a valid fit still works with the new constraint diagnostics."""
    np.random.seed(42)
    n = 101
    xmin = 0.
    xmax = 20.0
    x = np.linspace(xmin, xmax, n)
    data = (gaussian(x, 21, 8.1, 1.2) +
             lorentzian(x, 10, 9.6, 2.4) +
             np.random.normal(scale=0.23, size=n) + x * 0.5)

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

    mini = Minimizer(residual, pars, fcn_args=(x,), fcn_kws={'data': data})
    result = mini.minimize(method='leastsq')

    assert result.success, f"Fit failed: {result.message}"
    print(f"✓ Valid fit succeeded: success={result.success}")
    print(f"  chisqr={result.chisqr:.4f}, redchi={result.redchi:.4f}")
    return True


def test_circular_dependency_caught_early():
    """Test that circular dependency is caught before fit starts."""
    from lmfit import CircularDependencyError, ConstraintViolations

    pars = Parameters()
    pars.add('a', value=1.0)
    pars.add('b', value=2.0)
    pars['b'].expr = 'a + 1'
    pars['a'].expr = 'b + 1'

    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    assert isinstance(violations[0], CircularDependencyError)
    print(f"✓ Circular dependency caught early: {violations[0]}")
    return True


def test_initial_value_out_of_bounds_caught():
    """Test that initial value outside bounds is caught."""
    from lmfit import InitialValueOutOfBoundsError

    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)

    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    assert isinstance(violations[0], InitialValueOutOfBoundsError)
    print(f"✓ Initial value out of bounds caught: {violations[0]}")
    return True


def test_expr_result_out_of_bounds_caught():
    """Test that expression result outside bounds is caught."""
    from lmfit import ExprResultOutOfBoundsError

    pars = Parameters()
    pars.add('base', value=20.0)
    pars.add('scaled', expr='base * 0.5', min=0, max=5)

    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    assert isinstance(violations[0], ExprResultOutOfBoundsError)
    print(f"✓ Expression result out of bounds caught: {violations[0]}")
    return True


def test_multiple_violations_collected():
    """Test that multiple violations are collected in a single exception."""
    from lmfit import (
        ConstraintViolations,
        ExprResultOutOfBoundsError,
        InitialValueOutOfBoundsError,
    )

    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)
    pars.add('b', value=-5.0, min=0, max=10)
    pars.add('base', value=20.0)
    pars.add('scaled', expr='base * 0.5', min=0, max=5)

    try:
        pars.check_constraints(silent=False)
        assert False, "Expected ConstraintViolations exception"
    except ConstraintViolations as e:
        assert len(e.violations) >= 2
        types = {type(v).__name__ for v in e.violations}
        assert 'InitialValueOutOfBoundsError' in types
        print(f"✓ Multiple violations collected: {len(e.violations)} violations")
        for v in e.violations:
            print(f"    - {v}")
    return True


def test_silent_mode_returns_violations():
    """Test that silent mode returns violations without raising."""
    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)

    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    print(f"✓ Silent mode returns violations: {len(violations)} violations")
    return True


def test_raise_immediately():
    """Test that raise_immediately raises the first violation."""
    from lmfit import InitialValueOutOfBoundsError

    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)
    pars.add('b', value=20.0, min=0, max=10)

    try:
        pars.check_constraints(raise_immediately=True)
        assert False, "Expected exception"
    except InitialValueOutOfBoundsError as e:
        print(f"✓ raise_immediately raises first violation: {e}")
    return True


def main():
    """Run all integration tests."""
    print("=" * 60)
    print("Integration Test for Constraint Diagnostics")
    print("=" * 60)

    tests = [
        test_valid_fit_still_works,
        test_circular_dependency_caught_early,
        test_initial_value_out_of_bounds_caught,
        test_expr_result_out_of_bounds_caught,
        test_multiple_violations_collected,
        test_silent_mode_returns_violations,
        test_raise_immediately,
    ]

    passed = 0
    failed = 0

    for test in tests:
        try:
            if test():
                passed += 1
            print()
        except Exception as e:
            failed += 1
            print(f"\n✗ FAILED: {e}")
            import traceback
            traceback.print_exc()
            print()

    print("=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)

    return failed == 0


if __name__ == '__main__':
    import sys
    success = main()
    sys.exit(0 if success else 1)
