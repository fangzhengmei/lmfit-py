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

        # Set up circular dependency: a depends on b, b depends on a
        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'b'
            params['b'].expr = 'a'

    def test_indirect_circular_dependency(self):
        """Test that indirect circular dependency (A -> B -> C -> A) is detected."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)
        params.add('c', value=3.0)

        # Set up indirect circular dependency
        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'b'
            params['b'].expr = 'c'
            params['c'].expr = 'a'

    def test_self_referential_dependency(self):
        """Test that self-referential dependency (A -> A) is detected."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)

        # Set up self-referential dependency
        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'a'

    def test_complex_circular_dependency(self):
        """Test detection of more complex circular dependencies."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)
        params.add('c', value=3.0)
        params.add('d', value=4.0)

        # Set up complex circular dependency
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

        # This should work without errors
        assert params['a'].value == 12.0
        assert params['b'].value == 6.0
        assert params['c'].value == 3.0

    def test_circular_dependency_on_add(self):
        """Test circular dependency detection when adding parameters with expr."""
        params = lmfit.Parameters()
        params.add('b', value=2.0)
        params.add('a', value=1.0)

        # First set a to depend on b
        params['a'].expr = 'b'

        # Now try to make b depend on a - should raise error
        with pytest.raises(ValueError, match="circular dependency"):
            params['b'].expr = 'a'

    def test_circular_dependency_with_complex_expr(self):
        """Test circular dependency detection with more complex expressions."""
        params = lmfit.Parameters()
        params.add('a', value=1.0)
        params.add('b', value=2.0)

        # Complex expressions that still form a cycle
        with pytest.raises(ValueError, match="circular dependency"):
            params['a'].expr = 'sin(b) + exp(b)'
            params['b'].expr = 'cos(a) + log(a + 1)'
