"""Tests for circular dependency detection in Parameters with expr."""

import pytest
import lmfit


class TestCircularDependency:
    """Test cases for circular dependency detection."""

    def test_direct_circular_dependency(self):
        """Test that direct circular dependency (A -> B -> A) is detected."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'b'
            params['b'].expr = 'a'

    def test_indirect_circular_dependency(self):
        """Test that indirect circular dependency (A -> B -> C -> A) is detected."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)
        params.add('c', value=3.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'b'
            params['b'].expr = 'c'
            params['c'].expr = 'a'

    def test_self_referential_dependency(self):
        """Test that self-referential dependency (A -> A) is detected."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'a'

    def test_complex_circular_dependency(self):
        """Test detection of more complex circular dependencies."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)
        params.add('c', value=3.0)
        params.add('d', value=4.0)

        # a -> b -> c -> d -> b (creates a cycle b->c->d->b)
        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'b'
            params['b'].expr = 'c'
            params['c'].expr = 'd'
            params['d'].expr = 'b'

    def test_valid_chain_dependency(self):
        """Test that valid chain dependencies (A -> B -> C) work correctly."""
        params = lmfit.Parameters()
        params.add('c', value=3.0)
        params.add('b', expr='2*c')
        params.add('a', expr='2*b')

        assert params['a'].value == 12.0
        assert params['b'].value == 6.0
        assert params['c'].value == 3.0

    def test_circular_dependency_on_add(self):
        """Test circular dependency detection when adding parameters with expr."""
        params = lmfit.Parameters()
        params.add('b', value=2.0)
        params.add('a', value=1.0)

        params['a'].expr = 'b'

        with pytest.raises(ValueError, match="circular dependency"):
            params['b'].expr = 'a'

    def test_circular_dependency_with_complex_expr(self):
        """Test circular dependency detection with more complex expressions."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'sin(b) + exp(b)'
            params['b'].expr = 'cos(a) + log(a + 1)'


class TestCircularDependencyWithArithmetic:
    """Test cases for circular dependency detection with arithmetic expressions."""

    def test_direct_circular_with_arithmetic(self):
        """Test direct circular dependency with arithmetic: a -> 2*b+1, b -> 3*a-2."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = '2*b + 1'
            params['b'].expr = '3*a - 2'

    def test_indirect_circular_with_arithmetic(self):
        """Test indirect circular dependency with arithmetic: a -> b/2, b -> c*2, c -> a+1."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)
        params.add('c', value=3.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'b / 2'
            params['b'].expr = 'c * 2'
            params['c'].expr = 'a + 1'

    def test_circular_with_multiplication(self):
        """Test circular dependency with multiplication: scale -> amp + base, amp -> scale * 2."""
        params = lmfit.Parameters()
        params.add('scale', value=1.0)
        params.add('amp', value=2.0)
        params.add('base', value=0.5)

        with pytest.raises(ValueError, match="circular dependency"):
            params['scale'].expr = 'amp + base'
            params['amp'].expr = 'scale * 2'

    def test_valid_chain_with_arithmetic(self):
        """Test valid chain with arithmetic: a -> 2*b, b -> 3*c, c is free."""
        params = lmfit.Parameters()
        params.add('c', value=2.0)
        params.add('b', expr='3*c')
        params.add('a', expr='2*b')

        assert params['a'].value == 12.0
        assert params['b'].value == 6.0
        assert params['c'].value == 2.0

    def test_self_referential_with_arithmetic(self):
        """Test self-referential with arithmetic: a -> a + 1."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'a + 1'

    def test_circular_with_division(self):
        """Test circular dependency with division: a -> 100 / b, b -> 50 / a."""
        params = lmfit.Parameters()
        params.add('a', value=10.0)
        params.add('b', value=5.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = '100 / b'
            params['b'].expr = '50 / a'

    def test_circular_with_power(self):
        """Test circular dependency with power: a -> b ** 2, b -> sqrt(a)."""
        params = lmfit.Parameters()
        params.add('a', value=4.0)
        params.add('b', value=2.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'b ** 2'
            params['b'].expr = 'sqrt(a)'


class TestCircularDependencyWithFunctions:
    """Test cases for circular dependency detection with function calls."""

    def test_circular_with_trig_functions(self):
        """Test circular dependency with trigonometric functions."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=0.5)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'sin(b)'
            params['b'].expr = 'cos(a)'

    def test_circular_with_nested_functions(self):
        """Test circular dependency with nested function calls."""
        params = lmfit.Parameters()
        params.add('a', value=2.0)
        params.add('b', value=4.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'exp(log(b))'
            params['b'].expr = 'sqrt(a ** 2)'

    def test_valid_chain_with_functions(self):
        """Test valid chain with functions: a -> exp(b), b -> sqrt(c), c is free."""
        import numpy as np

        params = lmfit.Parameters()
        params.add('c', value=4.0)
        params.add('b', expr='sqrt(c)')
        params.add('a', expr='exp(b)')

        assert params['b'].value == 2.0
        assert abs(params['a'].value - np.exp(2.0)) < 1e-10

    def test_circular_with_multiple_dependencies(self):
        """Test circular dependency where one parameter depends on multiple others."""
        params = lmfit.Parameters()
        params.add('x', value=1.0)
        params.add('y', value=2.0)
        params.add('z', value=3.0)

        # x depends on y and z, y depends on x - creates a cycle
        with pytest.raises(ValueError, match="circular dependency"):
            params['x'].expr = 'y + z'
            params['y'].expr = 'x * 2'

    def test_valid_multiple_dependencies(self):
        """Test valid case with multiple dependencies: x -> y + z, y and z are free."""
        params = lmfit.Parameters()
        params.add('y', value=2.0)
        params.add('z', value=3.0)
        params.add('x', expr='y + z')

        assert params['x'].value == 5.0
        assert params['y'].value == 2.0
        assert params['z'].value == 3.0


class TestCircularDependencyRealWorld:
    """Test cases that simulate real-world scenarios."""

    def test_circular_with_scale_and_amp(self):
        """Test real-world scenario: scale * amp creates cycle."""
        params = lmfit.Parameters()
        params.add('scale', value=1.0)
        params.add('amp', value=10.0)
        params.add('offset', value=0.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['scale'].expr = 'amp / 10'
            params['amp'].expr = 'scale * 10'

    def test_circular_with_ratio(self):
        """Test real-world scenario: ratio parameters creating cycle."""
        params = lmfit.Parameters()
        params.add('total', value=100.0)
        params.add('ratio', value=0.5)
        params.add('part_a', value=50.0)
        params.add('part_b', value=50.0)

        with pytest.raises(ValueError, match="circular dependency"):
            params['part_a'].expr = 'ratio * total'
            params['ratio'].expr = 'part_a / total'

    def test_valid_ratio_scenario(self):
        """Test valid real-world ratio scenario."""
        params = lmfit.Parameters()
        params.add('total', value=100.0)
        params.add('ratio', value=0.3)
        params.add('part_a', expr='ratio * total')
        params.add('part_b', expr='(1 - ratio) * total')

        assert params['part_a'].value == 30.0
        assert params['part_b'].value == 70.0
        assert params['total'].value == 100.0
        assert params['ratio'].value == 0.3
