"""Verify that constraint diagnostics work correctly."""
import sys
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
)


def test_exception_classes():
    """Test that exception classes are correctly defined."""
    print("Testing exception classes...")
    
    assert issubclass(ConstraintError, MinimizerException)
    assert issubclass(BoundsConflictError, ConstraintError)
    assert issubclass(InitialValueOutOfBoundsError, ConstraintError)
    assert issubclass(ExprResultOutOfBoundsError, ConstraintError)
    assert issubclass(CircularDependencyError, ConstraintError)
    assert issubclass(UndefinedVarInExprError, ConstraintError)
    assert issubclass(ConstraintViolations, MinimizerException)
    
    print("  All exception classes correctly inherit from base classes.")
    return True


def test_check_constraints_no_violations():
    """Test that valid parameters have no violations."""
    print("\nTesting check_constraints with valid parameters...")
    
    pars = Parameters()
    pars.add('a', value=5.0, min=0, max=10)
    pars.add('b', value=3.0)
    pars.add('c', expr='a + b', min=0, max=20)
    
    violations = pars.check_constraints(silent=True)
    assert len(violations) == 0, f"Expected 0 violations, got {len(violations)}"
    
    print("  No violations detected for valid parameters.")
    return True


def test_initial_value_out_of_bounds():
    """Test detection of initial value outside bounds."""
    print("\nTesting initial value out of bounds detection...")
    
    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)
    
    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    assert isinstance(violations[0], InitialValueOutOfBoundsError)
    assert violations[0].param_name == 'a'
    assert violations[0].value == 15.0
    
    print(f"  Detected: {violations[0]}")
    return True


def test_expr_with_undefined_variable():
    """Test that undefined variable in expression raises error during add().
    
    Note: This is existing behavior - undefined variables are detected
    immediately when the parameter is added, not during check_constraints.
    """
    print("\nTesting undefined variable in expression (existing behavior)...")
    
    pars = Parameters()
    pars.add('a', value=5.0)
    
    try:
        pars.add('b', expr='a + undefined_var')
        assert False, "Expected NameError when adding parameter with undefined variable"
    except NameError as e:
        print(f"  Raised NameError during add(): {e}")
        print("  Note: This is existing behavior, not from check_constraints")
    
    return True


def test_circular_dependency():
    """Test detection of circular dependency."""
    print("\nTesting circular dependency detection...")
    
    pars = Parameters()
    pars.add('a', value=1.0)
    pars.add('b', value=2.0)
    pars['b'].expr = 'a + 1'
    pars['a'].expr = 'b + 1'
    
    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    assert isinstance(violations[0], CircularDependencyError)
    
    print(f"  Detected: {violations[0]}")
    return True


def test_self_reference():
    """Test detection of self-referencing expression."""
    print("\nTesting self-reference detection...")
    
    pars = Parameters()
    pars.add('a', value=1.0)
    pars['a'].expr = 'a * 2'
    
    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    assert isinstance(violations[0], CircularDependencyError)
    
    print(f"  Detected: {violations[0]}")
    return True


def test_expr_result_out_of_bounds():
    """Test detection of expression result outside bounds."""
    print("\nTesting expression result out of bounds detection...")
    
    pars = Parameters()
    pars.add('base', value=20.0)
    pars.add('scaled', expr='base * 0.5', min=0, max=5)
    
    violations = pars.check_constraints(silent=True)
    assert len(violations) == 1
    assert isinstance(violations[0], ExprResultOutOfBoundsError)
    
    print(f"  Detected: {violations[0]}")
    return True


def test_multiple_violations():
    """Test detection of multiple violations."""
    print("\nTesting multiple violations detection...")
    
    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)
    pars.add('b', value=-5.0, min=0, max=10)
    pars.add('base', value=20.0)
    pars.add('scaled', expr='base * 0.5', min=0, max=5)
    
    violations = pars.check_constraints(silent=True)
    assert len(violations) >= 2
    
    print(f"  Detected {len(violations)} violations:")
    for v in violations:
        print(f"    - {v}")
    
    return True


def test_raise_constraint_violations():
    """Test that non-silent mode raises ConstraintViolations."""
    print("\nTesting ConstraintViolations exception...")
    
    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)
    
    try:
        pars.check_constraints(silent=False)
        assert False, "Expected ConstraintViolations exception"
    except ConstraintViolations as e:
        assert len(e.violations) == 1
        assert isinstance(e.violations[0], InitialValueOutOfBoundsError)
        print(f"  Raised: {e}")
    
    return True


def test_raise_immediately():
    """Test raise_immediately option."""
    print("\nTesting raise_immediately option...")
    
    pars = Parameters()
    pars.add('a', value=15.0, min=0, max=10)
    pars.add('b', value=20.0, min=0, max=10)
    
    try:
        pars.check_constraints(raise_immediately=True)
        assert False, "Expected exception"
    except InitialValueOutOfBoundsError as e:
        print(f"  Raised immediately: {e}")
    
    return True


def test_dependency_graph():
    """Test dependency graph building."""
    print("\nTesting dependency graph building...")
    
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
    
    print("  Dependency graph built correctly.")
    return True


def main():
    """Run all tests."""
    print("=" * 60)
    print("Verifying Constraint Diagnostics Implementation")
    print("=" * 60)
    
    tests = [
        test_exception_classes,
        test_check_constraints_no_violations,
        test_initial_value_out_of_bounds,
        test_expr_with_undefined_variable,
        test_circular_dependency,
        test_self_reference,
        test_expr_result_out_of_bounds,
        test_multiple_violations,
        test_raise_constraint_violations,
        test_raise_immediately,
        test_dependency_graph,
    ]
    
    passed = 0
    failed = 0
    
    for test in tests:
        try:
            if test():
                passed += 1
        except Exception as e:
            failed += 1
            print(f"\n  FAILED: {e}")
            import traceback
            traceback.print_exc()
    
    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    
    return failed == 0


if __name__ == '__main__':
    success = main()
    sys.exit(0 if success else 1)
