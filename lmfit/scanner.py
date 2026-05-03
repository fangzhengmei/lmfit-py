"""Parameter range scanning for lmfit models.

This module provides functionality to perform parameter range scanning
for lmfit models, allowing users to explore how different parameter
values affect the fitting results.
"""

from copy import deepcopy
from dataclasses import dataclass, field
from itertools import product
from typing import Any, Callable, Dict, List, Optional, Tuple, Union

import numpy as np

from .model import Model, ModelResult
from .minimizer import Minimizer, MinimizerResult
from .parameter import Parameters


@dataclass
class ScanResult:
    """Class to store results from a single parameter scan point.

    Attributes
    ----------
    param_values : Dict[str, float]
        Dictionary mapping parameter names to their values at this scan point.
    fit_result : Optional[Union[ModelResult, MinimizerResult]]
        The fit result object from lmfit.
    chisqr : Optional[float]
        Chi-square value of the fit.
    redchi : Optional[float]
        Reduced chi-square value.
    aic : Optional[float]
        Akaike Information Criterion.
    bic : Optional[float]
        Bayesian Information Criterion.
    success : bool
        Whether the fit was successful.
    params : Optional[Parameters]
        The Parameters object after fitting.
    """
    param_values: Dict[str, float]
    fit_result: Optional[Union[ModelResult, MinimizerResult]] = None
    chisqr: Optional[float] = None
    redchi: Optional[float] = None
    aic: Optional[float] = None
    bic: Optional[float] = None
    success: bool = False
    params: Optional[Parameters] = None


@dataclass
class ParameterScan:
    """Class to manage parameter range scanning for lmfit models.

    This class allows users to define parameter ranges to scan,
    perform the scans, and collect results for analysis.

    Attributes
    ----------
    model : Optional[Model]
        The lmfit Model to use for fitting (if using Model interface).
    minimizer : Optional[Minimizer]
        The lmfit Minimizer to use for fitting (if using Minimizer interface).
    data : Optional[np.ndarray]
        The data to fit (for Model interface).
    weights : Optional[np.ndarray]
        Weights for the fit (for Model interface).
    fit_kws : Dict[str, Any]
        Additional keyword arguments to pass to the fit method.
    scan_ranges : Dict[str, Union[np.ndarray, List, Tuple]]
        Dictionary mapping parameter names to their scan ranges.
    scan_results : List[ScanResult]
        List of results from each scan point.
    """
    model: Optional[Model] = None
    minimizer: Optional[Minimizer] = None
    data: Optional[np.ndarray] = None
    weights: Optional[np.ndarray] = None
    fit_kws: Dict[str, Any] = field(default_factory=dict)
    scan_ranges: Dict[str, Union[np.ndarray, List, Tuple]] = field(default_factory=dict)
    scan_results: List[ScanResult] = field(default_factory=list)

    def __post_init__(self):
        """Validate initialization parameters."""
        if self.model is None and self.minimizer is None:
            raise ValueError("Either 'model' or 'minimizer' must be provided.")
        if self.model is not None and self.minimizer is not None:
            raise ValueError("Only one of 'model' or 'minimizer' should be provided.")

    def add_scan_range(
        self,
        param_name: str,
        start: float,
        end: float,
        num: Optional[int] = None,
        step: Optional[float] = None,
        values: Optional[Union[List, np.ndarray]] = None
    ) -> None:
        """Add a parameter scan range.

        Parameters
        ----------
        param_name : str
            Name of the parameter to scan.
        start : float
            Start of the range (ignored if 'values' is provided).
        end : float
            End of the range (ignored if 'values' is provided).
        num : int, optional
            Number of points in the range (must provide either 'num' or 'step').
        step : float, optional
            Step size between points (must provide either 'num' or 'step').
        values : list or np.ndarray, optional
            Explicit list of values to use (overrides start/end/num/step).

        Raises
        ------
        ValueError
            If neither 'num' nor 'step' is provided when using start/end,
            or if no valid range specification is provided.
        """
        if values is not None:
            self.scan_ranges[param_name] = np.asarray(values)
        elif num is not None:
            self.scan_ranges[param_name] = np.linspace(start, end, num)
        elif step is not None:
            self.scan_ranges[param_name] = np.arange(start, end + step, step)
        else:
            raise ValueError(
                "Must provide either 'values', 'num', or 'step' to define the scan range."
            )

    def _get_param_combinations(self) -> List[Tuple[Tuple[str, float], ...]]:
        """Generate all combinations of parameter values for scanning.

        Returns
        -------
        list
            List of tuples, each containing (param_name, value) pairs for one scan point.
        """
        if not self.scan_ranges:
            return []

        param_names = list(self.scan_ranges.keys())
        param_values = [self.scan_ranges[name] for name in param_names]

        combinations = []
        for value_combination in product(*param_values):
            combinations.append(
                tuple(zip(param_names, value_combination))
            )

        return combinations

    def _fit_with_params(
        self,
        param_values: Dict[str, float],
        base_params: Optional[Parameters] = None
    ) -> ScanResult:
        """Perform a fit with the specified parameter values.

        Parameters
        ----------
        param_values : Dict[str, float]
            Dictionary mapping parameter names to their values for this scan point.
        base_params : Parameters, optional
            Base parameters to use (will be modified with scan values).

        Returns
        -------
        ScanResult
            Result of the fit at this scan point.
        """
        result = ScanResult(param_values=param_values.copy())

        try:
            if self.model is not None:
                if base_params is None:
                    params = self.model.make_params()
                else:
                    params = deepcopy(base_params)

                for name, value in param_values.items():
                    if name in params:
                        params[name].value = value
                        params[name].vary = False

                fit_result = self.model.fit(
                    self.data,
                    params=params,
                    weights=self.weights,
                    **self.fit_kws
                )

                result.fit_result = fit_result
                result.chisqr = getattr(fit_result, 'chisqr', None)
                result.redchi = getattr(fit_result, 'redchi', None)
                result.aic = getattr(fit_result, 'aic', None)
                result.bic = getattr(fit_result, 'bic', None)
                result.success = getattr(fit_result, 'success', True)
                result.params = getattr(fit_result, 'params', None)

            else:
                if base_params is None:
                    params = deepcopy(self.minimizer.params)
                else:
                    params = deepcopy(base_params)

                for name, value in param_values.items():
                    if name in params:
                        params[name].value = value
                        params[name].vary = False

                fit_method = self.fit_kws.get('method', 'leastsq')
                fit_result = self.minimizer.minimize(
                    method=fit_method,
                    params=params
                )

                result.fit_result = fit_result
                result.chisqr = getattr(fit_result, 'chisqr', None)
                result.redchi = getattr(fit_result, 'redchi', None)
                result.aic = getattr(fit_result, 'aic', None)
                result.bic = getattr(fit_result, 'bic', None)
                result.success = getattr(fit_result, 'success', True)
                result.params = getattr(fit_result, 'params', None)

        except Exception as e:
            result.success = False

        return result

    def run(
        self,
        base_params: Optional[Parameters] = None,
        verbose: bool = False
    ) -> List[ScanResult]:
        """Run the parameter scan.

        Parameters
        ----------
        base_params : Parameters, optional
            Base parameters to use for the fit. If not provided,
            default parameters from the model or minimizer will be used.
        verbose : bool, optional
            If True, print progress information.

        Returns
        -------
        List[ScanResult]
            List of results from each scan point.
        """
        if not self.scan_ranges:
            raise ValueError("No scan ranges defined. Use add_scan_range() first.")

        combinations = self._get_param_combinations()
        total_points = len(combinations)

        if verbose:
            print(f"Starting parameter scan with {total_points} points...")
            print(f"Scanning parameters: {list(self.scan_ranges.keys())}")

        self.scan_results = []

        for i, combo in enumerate(combinations):
            param_values = dict(combo)

            if verbose:
                progress = f"[{i+1}/{total_points}] "
                param_str = ", ".join([f"{k}={v:.4g}" for k, v in param_values.items()])
                print(f"{progress}{param_str}")

            result = self._fit_with_params(param_values, base_params)
            self.scan_results.append(result)

            if verbose and not result.success:
                print(f"  Warning: Fit failed for this point")

        if verbose:
            print(f"\nScan complete. Processed {len(self.scan_results)} points.")
            successful = sum(1 for r in self.scan_results if r.success)
            print(f"Successful fits: {successful}/{len(self.scan_results)}")

        return self.scan_results

    def get_best_result(self, metric: str = 'chisqr') -> Optional[ScanResult]:
        """Get the best result based on a specified metric.

        Parameters
        ----------
        metric : str, optional
            Metric to use for determining 'best'. Options are:
            'chisqr', 'redchi', 'aic', 'bic'.
            Lower values are considered better for all metrics.

        Returns
        -------
        ScanResult or None
            The best result, or None if no successful results.
        """
        if not self.scan_results:
            return None

        valid_metrics = ['chisqr', 'redchi', 'aic', 'bic']
        if metric not in valid_metrics:
            raise ValueError(f"Invalid metric. Choose from: {valid_metrics}")

        successful_results = [r for r in self.scan_results if r.success]
        if not successful_results:
            return None

        def get_metric_value(result: ScanResult) -> float:
            value = getattr(result, metric, None)
            return float('inf') if value is None else value

        return min(successful_results, key=get_metric_value)

    def get_results_array(self) -> Dict[str, np.ndarray]:
        """Get scan results as arrays for analysis or plotting.

        Returns
        -------
        Dict[str, np.ndarray]
            Dictionary containing arrays for each scanned parameter
            and for each metric (chisqr, redchi, aic, bic).
        """
        if not self.scan_results:
            return {}

        param_names = list(self.scan_ranges.keys())

        arrays = {name: [] for name in param_names}
        arrays.update({
            'chisqr': [],
            'redchi': [],
            'aic': [],
            'bic': [],
            'success': []
        })

        for result in self.scan_results:
            for name in param_names:
                arrays[name].append(result.param_values.get(name, np.nan))
            arrays['chisqr'].append(result.chisqr if result.chisqr is not None else np.nan)
            arrays['redchi'].append(result.redchi if result.redchi is not None else np.nan)
            arrays['aic'].append(result.aic if result.aic is not None else np.nan)
            arrays['bic'].append(result.bic if result.bic is not None else np.nan)
            arrays['success'].append(result.success)

        return {k: np.array(v) for k, v in arrays.items()}


def scan_parameters(
    model_or_minimizer: Union[Model, Minimizer],
    scan_ranges: Dict[str, Union[np.ndarray, List, Tuple]],
    data: Optional[np.ndarray] = None,
    weights: Optional[np.ndarray] = None,
    base_params: Optional[Parameters] = None,
    fit_kws: Optional[Dict[str, Any]] = None,
    verbose: bool = False
) -> ParameterScan:
    """Convenience function to perform a parameter scan.

    Parameters
    ----------
    model_or_minimizer : Model or Minimizer
        The lmfit Model or Minimizer to use for fitting.
    scan_ranges : Dict[str, Union[np.ndarray, List, Tuple]]
        Dictionary mapping parameter names to their scan ranges.
        Each range can be an array, list, or tuple of values.
    data : np.ndarray, optional
        The data to fit (required for Model interface).
    weights : np.ndarray, optional
        Weights for the fit (for Model interface).
    base_params : Parameters, optional
        Base parameters to use for the fit.
    fit_kws : Dict[str, Any], optional
        Additional keyword arguments to pass to the fit method.
    verbose : bool, optional
        If True, print progress information.

    Returns
    -------
    ParameterScan
        The ParameterScan object containing results.
    """
    if isinstance(model_or_minimizer, Model):
        scanner = ParameterScan(
            model=model_or_minimizer,
            data=data,
            weights=weights,
            fit_kws=fit_kws or {}
        )
    else:
        scanner = ParameterScan(
            minimizer=model_or_minimizer,
            fit_kws=fit_kws or {}
        )

    for param_name, values in scan_ranges.items():
        scanner.add_scan_range(param_name, start=0, end=1, values=values)

    scanner.run(base_params=base_params, verbose=verbose)

    return scanner
