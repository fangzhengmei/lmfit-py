"""Test script to explore current constraint conflict behavior."""

import numpy as np
from lmfit import Parameters, Minimizer, minimize


def residual(pars, x, data=None):
    model = pars['a'] * x + pars['b']
    if data is None:
        return model
    return model - data


x = np.linspace(0, 10, 100)
np.random.seed(42)
data = 2.0 * x + 1.0 + np.random.normal(0, 0.1, size=x.shape)


def test_scenario_1():
    """Scenario 1: Bounds conflict (min > max)"""
    print("=" * 60)
    print("Scenario 1: min > max bounds conflict")
    print("=" * 60)
    pars = Parameters()
    try:
        pars.add('a', value=1.0, min=10, max=5)
        print(f"  Parameter 'a': min={pars['a'].min}, max={pars['a'].max}")
        print("  No error raised - bounds were swapped!")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")


def test_scenario_2():
    """Scenario 2: Initial value outside bounds"""
    print("\n" + "=" * 60)
    print("Scenario 2: Initial value outside bounds")
    print("=" * 60)
    pars = Parameters()
    pars.add('a', value=100, min=0, max=10)
    print(f"  Parameter 'a': value={pars['a'].value}, min={pars['a'].min}, max={pars['a'].max}")
    print("  Value was clipped to bounds!")


def test_scenario_3():
    """Scenario 3: Expression result outside bounds"""
    print("\n" + "=" * 60)
    print("Scenario 3: Expression result outside bounds")
    print("=" * 60)
    pars = Parameters()
    pars.add('a', value=10.0)
    pars.add('b', value=5.0, min=0, max=8)  # b is bounded [0, 8]
    pars.add('c', expr='a + b', min=0, max=10)  # c = 10 + 5 = 15, but max=10!
    print(f"  Parameter 'a': value={pars['a'].value}")
    print(f"  Parameter 'b': value={pars['b'].value}, min={pars['b'].min}, max={pars['b'].max}")
    print(f"  Parameter 'c': expr='{pars['c'].expr}', value={pars['c'].value}, min={pars['c'].min}, max={pars['c'].max}")
    print("  Expression result was clipped to bounds!")


def test_scenario_4():
    """Scenario 4: Circular dependency in expressions"""
    print("\n" + "=" * 60)
    print("Scenario 4: Circular dependency in expressions")
    print("=" * 60)
    pars = Parameters()
    try:
        pars.add('a', value=1.0)
        pars.add('b', expr='a + 1')
        pars['a'].expr = 'b + 1'  # Now a = b+1 and b = a+1 - circular!
        pars.update_constraints()
        print(f"  Parameter 'a': value={pars['a'].value}, expr='{pars['a'].expr}'")
        print(f"  Parameter 'b': value={pars['b'].value}, expr='{pars['b'].expr}'")
        print("  Circular dependency might cause infinite loop or wrong values!")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")


def test_scenario_5():
    """Scenario 5: Try to fit with conflicting constraints"""
    print("\n" + "=" * 60)
    print("Scenario 5: Try to fit with conflicting constraints")
    print("=" * 60)
    pars = Parameters()
    pars.add('a', value=1.0, min=0, max=10)
    pars.add('b', value=100.0, min=0, max=10)  # b should be < 10
    pars.add('c', expr='a + b')  # c depends on both a and b
    
    # Now change a so that c = a + b would exceed some implicit limit
    # But actually let's try a more realistic conflict:
    # What if b's expression evaluates to outside its bounds?
    pars2 = Parameters()
    pars2.add('base', value=20.0)
    pars2.add('scaled', expr='base * 0.5', min=0, max=5)  # 20 * 0.5 = 10 > 5!
    
    print(f"  Parameter 'base': value={pars2['base'].value}")
    print(f"  Parameter 'scaled': expr='{pars2['scaled'].expr}', value={pars2['scaled'].value}, min={pars2['scaled'].min}, max={pars2['scaled'].max}")
    
    try:
        result = minimize(residual, pars2, args=(x,), kws={'data': data})
        print(f"  Fit result: success={result.success}, message={result.message}")
    except Exception as e:
        print(f"  Error during fit: {type(e).__name__}: {e}")


def test_scenario_6():
    """Scenario 6: Self-referencing expression"""
    print("\n" + "=" * 60)
    print("Scenario 6: Self-referencing expression")
    print("=" * 60)
    pars = Parameters()
    try:
        pars.add('a', value=1.0)
        pars['a'].expr = 'a * 2'  # Self-reference!
        pars.update_constraints()
        print(f"  Parameter 'a': value={pars['a'].value}, expr='{pars['a'].expr}'")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")


def test_scenario_7():
    """Scenario 7: Expression with undefined variable"""
    print("\n" + "=" * 60)
    print("Scenario 7: Expression with undefined variable")
    print("=" * 60)
    pars = Parameters()
    try:
        pars.add('a', value=1.0)
        pars.add('b', expr='a + c')  # 'c' is not defined!
        print(f"  Parameter 'b': value={pars['b'].value}, expr='{pars['b'].expr}'")
    except Exception as e:
        print(f"  Error: {type(e).__name__}: {e}")


if __name__ == '__main__':
    test_scenario_1()
    test_scenario_2()
    test_scenario_3()
    test_scenario_4()
    test_scenario_5()
    test_scenario_6()
    test_scenario_7()
