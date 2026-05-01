class MinimizerException(Exception):
    """General Purpose Exception."""

    def __init__(self, msg):
        Exception.__init__(self)
        self.msg = msg

    def __str__(self):
        """string"""
        return f"{self.msg}"


class AbortFitException(MinimizerException):
    """Raised when a fit is aborted by the user."""


class ConstraintError(MinimizerException):
    """Base exception for parameter constraint violations."""

    def __init__(self, param_name, msg):
        self.param_name = param_name
        super().__init__(msg)


class BoundsConflictError(ConstraintError):
    """Raised when parameter bounds are invalid (min > max)."""

    def __init__(self, param_name, min_val, max_val):
        self.min_val = min_val
        self.max_val = max_val
        msg = (f"Parameter '{param_name}' has invalid bounds: "
               f"min={min_val} > max={max_val}")
        super().__init__(param_name, msg)


class InitialValueOutOfBoundsError(ConstraintError):
    """Raised when parameter initial value is outside its bounds."""

    def __init__(self, param_name, value, min_val, max_val):
        self.value = value
        self.min_val = min_val
        self.max_val = max_val
        if value < min_val:
            msg = (f"Parameter '{param_name}' initial value {value} "
                   f"is below lower bound {min_val}")
        else:
            msg = (f"Parameter '{param_name}' initial value {value} "
                   f"is above upper bound {max_val}")
        super().__init__(param_name, msg)


class ExprResultOutOfBoundsError(ConstraintError):
    """Raised when expression evaluation result is outside parameter bounds."""

    def __init__(self, param_name, expr, result, min_val, max_val):
        self.expr = expr
        self.result = result
        self.min_val = min_val
        self.max_val = max_val
        if result < min_val:
            msg = (f"Parameter '{param_name}' expression '{expr}' "
                   f"evaluates to {result}, which is below lower bound {min_val}")
        else:
            msg = (f"Parameter '{param_name}' expression '{expr}' "
                   f"evaluates to {result}, which is above upper bound {max_val}")
        super().__init__(param_name, msg)


class CircularDependencyError(ConstraintError):
    """Raised when parameter expressions form a circular dependency."""

    def __init__(self, cycle):
        self.cycle = cycle
        cycle_str = " -> ".join(cycle)
        msg = f"Circular dependency detected in parameter expressions: {cycle_str}"
        super().__init__(cycle[0] if cycle else None, msg)


class UndefinedVarInExprError(ConstraintError):
    """Raised when an expression references an undefined variable."""

    def __init__(self, param_name, expr, var_name):
        self.expr = expr
        self.var_name = var_name
        msg = (f"Parameter '{param_name}' expression '{expr}' "
               f"references undefined variable '{var_name}'")
        super().__init__(param_name, msg)


class ConstraintViolations(MinimizerException):
    """Raised when multiple constraint violations are detected.

    Contains a list of all constraint errors found during validation.
    """

    def __init__(self, violations):
        self.violations = violations
        if len(violations) == 1:
            msg = f"1 constraint violation detected:\n  - {violations[0]}"
        else:
            msg = (f"{len(violations)} constraint violations detected:\n" +
                   "\n".join(f"  - {v}" for v in violations))
        super().__init__(msg)
