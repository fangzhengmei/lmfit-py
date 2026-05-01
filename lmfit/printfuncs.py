"""Functions to display fitting results and confidence intervals."""

from math import log10
import re

import numpy as np

try:
    import numdifftools  # noqa: F401
    HAS_NUMDIFFTOOLS = True
except ImportError:
    HAS_NUMDIFFTOOLS = False


def alphanumeric_sort(s, _nsre=re.compile('([0-9]+)')):
    """Sort alphanumeric string."""
    return [int(text) if text.isdigit() else text.lower()
            for text in re.split(_nsre, s)]


def getfloat_attr(obj, attr, length=11):
    """Format an attribute of an object for printing."""
    val = getattr(obj, attr, None)
    if val is None:
        return 'unknown'
    if isinstance(val, int):
        return f'{val}'
    if isinstance(val, float):
        return gformat(val, length=length).strip()
    return repr(val)


def gformat(val, length=11):
    """Format a number with '%g'-like format.

    Except that:
        a) the length of the output string will be of the requested length.
        b) positive numbers will have a leading blank.
        b) the precision will be as high as possible.
        c) trailing zeros will not be trimmed.

    The precision will typically be ``length-7``.

    Parameters
    ----------
    val : float
        Value to be formatted.
    length : int, optional
        Length of output string (default is 11).

    Returns
    -------
    str
        String of specified length.

    Notes
    ------
    Positive values will have leading blank.

    """
    if val is None or isinstance(val, bool):
        return f'{repr(val):>{length}s}'
    try:
        expon = int(log10(abs(val)))
    except (OverflowError, ValueError):
        expon = 0
    except TypeError:
        return f'{repr(val):>{length}s}'

    length = max(length, 7)
    form = 'e'
    prec = length - 7
    if abs(expon) > 99:
        prec -= 1
    elif ((expon > 0 and expon < (prec+4)) or
          (expon <= 0 and -expon < (prec-1))):
        form = 'f'
        prec += 4
        if expon > 0:
            prec -= expon
    return f'{val:{length}.{prec}{form}}'


def fit_report(inpars, modelpars=None, show_correl=True, min_correl=0.1,
               sort_pars=False, correl_mode='list'):
    """Generate a report of the fitting results.

    The report contains the best-fit values for the parameters and their
    uncertainties and correlations.

    Parameters
    ----------
    inpars : Parameters
        Input Parameters from fit or MinimizerResult returned from a fit.
    modelpars : Parameters, optional
        Known Model Parameters.
    show_correl : bool, optional
        Whether to show list of sorted correlations (default is True).
    min_correl : float, optional
        Smallest correlation in absolute value to show (default is 0.1).
    sort_pars : bool or callable, optional
        Whether to show parameter names sorted in alphanumerical order. If
        False (default), then the parameters will be listed in the order
        they were added to the Parameters dictionary. If callable, then
        this (one argument) function is used to extract a comparison key
        from each list element.
    correl_mode : {'list', table'} str, optional
        Mode for how to show correlations. Can be either 'list' (default)
        to show a sorted (if ``sort_pars`` is True) list of correlation
        values, or 'table' to show a complete, formatted table of
        correlations.

    Returns
    -------
    str
        Multi-line text of fit report.

    """
    from .parameter import Parameters
    if isinstance(inpars, Parameters):
        result, params = None, inpars
    if hasattr(inpars, 'params'):
        result = inpars
        params = inpars.params

    if sort_pars:
        if callable(sort_pars):
            key = sort_pars
        else:
            key = alphanumeric_sort
        parnames = sorted(params, key=key)
    else:
        # dict.keys() returns a KeysView in py3, and they're indexed
        # further down
        parnames = list(params.keys())

    buff = []
    add = buff.append
    namelen = max(len(n) for n in parnames)
    if result is not None:
        add("[[Fit Statistics]]")
        add(f"    # fitting method   = {result.method}")
        add(f"    # function evals   = {getfloat_attr(result, 'nfev')}")
        add(f"    # data points      = {getfloat_attr(result, 'ndata')}")
        add(f"    # variables        = {getfloat_attr(result, 'nvarys')}")
        add(f"    chi-square         = {getfloat_attr(result, 'chisqr')}")
        add(f"    reduced chi-square = {getfloat_attr(result, 'redchi')}")
        add(f"    Akaike info crit   = {getfloat_attr(result, 'aic')}")
        add(f"    Bayesian info crit = {getfloat_attr(result, 'bic')}")
        if hasattr(result, 'rsquared'):
            add(f"    R-squared          = {getfloat_attr(result, 'rsquared')}")
        if not result.errorbars:
            add("##  Warning: uncertainties could not be estimated:")
            if result.method in ('leastsq', 'least_squares') or HAS_NUMDIFFTOOLS:
                parnames_varying = [par for par in result.params
                                    if result.params[par].vary]
                for name in parnames_varying:
                    par = params[name]
                    space = ' '*(namelen-len(name))
                    if par.init_value and np.allclose(par.value, par.init_value):
                        add(f'    {name}:{space}  at initial value')
                    if (np.allclose(par.value, par.min) or np.allclose(par.value, par.max)):
                        add(f'    {name}:{space}  at boundary')
            else:
                add("    this fitting method does not natively calculate uncertainties")
                add("    and numdifftools is not installed for lmfit to do this. Use")
                add("    `pip install numdifftools` for lmfit to estimate uncertainties")
                add("    with this fitting method.")

    add("[[Variables]]")
    for name in parnames:
        par = params[name]
        space = ' '*(namelen-len(name))
        nout = f"{name}:{space}"
        inval = '(init = ?)'
        if par.init_value is not None:
            inval = f'(init = {par.init_value:.7g})'
        if modelpars is not None and name in modelpars:
            inval = f'{inval}, model_value = {modelpars[name].value:.7g}'
        try:
            sval = gformat(par.value)
        except (TypeError, ValueError):
            sval = ' Non Numeric Value?'
        if par.stderr is not None:
            serr = gformat(par.stderr)
            try:
                spercent = f'({abs(float(par.stderr)/float(par.value)):.2%})'
            except ZeroDivisionError:
                spercent = ''
            sval = f'{sval} +/-{serr} {spercent}'

        if par.vary:
            add(f"    {nout} {sval} {inval}")
        elif par.expr is not None:
            add(f"    {nout} {sval} == '{par.expr}'")
        else:
            add(f"    {nout} {par.value: .7g} (fixed)")

    if show_correl and correl_mode.startswith('tab'):
        add('[[Correlations]] ')
        for line in correl_table(params).split('\n'):
            buff.append('  %s' % line)
    elif show_correl:
        correls = {}
        for i, name in enumerate(parnames):
            par = params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if (name != name2 and name2 in par.correl and
                            abs(par.correl[name2]) > min_correl):
                        correls[f"{name}, {name2}"] = par.correl[name2]

        sort_correl = sorted(correls.items(), key=lambda it: abs(it[1]))
        sort_correl.reverse()
        if len(sort_correl) > 0:
            add('[[Correlations]] (unreported correlations are < '
                f'{min_correl:.3f})')
            maxlen = max(len(k) for k in list(correls.keys()))
        for name, val in sort_correl:
            lspace = max(0, maxlen - len(name))
            add(f"    C({name}){(' '*30)[:lspace]} = {val:+.4f}")
    return '\n'.join(buff)


def lcol(s, cat='td'):
    "html left column"
    return f"<{cat} style='text-align:left'>{s}</{cat}>"


def rcol(s, cat='td'):
    "html right column"
    return f"<{cat} style='text-align:right'>{s}</{cat}>"


def trow(columns, cat='td'):
    "html row"
    nlast = len(columns)-1
    rows = []
    for i, col in enumerate(columns):
        cform = rcol if i == nlast else lcol
        rows.append(cform(col, cat=cat))
    return rows


def fitreport_html_table(result, show_correl=True, min_correl=0.1):
    """Generate a report of the fitting result as an HTML table.

    Parameters
    ----------
    result : MinimizerResult or ModelResult
        Object containing the optimized parameters and several
        goodness-of-fit statistics.
    show_correl : bool, optional
        Whether to show list of sorted correlations (default is True).
    min_correl : float, optional
        Smallest correlation in absolute value to show (default is 0.1).

    Returns
    -------
    str
        Multi-line HTML code of fit report.

    """
    html = []
    add = html.append

    def stat_row(label, val, val2=None, cat='td'):
        if val2 is None:
            rows = trow([label, val], cat=cat)
        else:
            rows = trow([label, val, val2], cat=cat)
        add(f"<tr>{''.join(rows)}</tr>")

    add('<table class="jp-toc-ignore">')
    add('<caption class="jp-toc-ignore">Fit Statistics</caption>')
    stat_row('fitting method', result.method)
    stat_row('# function evals', result.nfev)
    stat_row('# data points', result.ndata)
    stat_row('# variables', result.nvarys)
    stat_row('chi-square', gformat(result.chisqr))
    stat_row('reduced chi-square', gformat(result.redchi))
    stat_row('Akaike info crit.', gformat(result.aic))
    stat_row('Bayesian info crit.', gformat(result.bic))
    if hasattr(result, 'rsquared'):
        stat_row('R-squared', gformat(result.rsquared))
    add('</table>')
    add(params_html_table(result.params))
    if show_correl:
        correls = []
        parnames = list(result.params.keys())
        for i, name in enumerate(result.params):
            par = result.params[name]
            if not par.vary:
                continue
            if hasattr(par, 'correl') and par.correl is not None:
                for name2 in parnames[i+1:]:
                    if (name != name2 and name2 in par.correl and
                            abs(par.correl[name2]) > min_correl):
                        correls.append((name, name2, par.correl[name2]))
        if len(correls) > 0:
            sort_correls = sorted(correls, key=lambda val: abs(val[2]))
            sort_correls.reverse()
            extra = f'(unreported values are < {min_correl:.3f})'
            add('<table class="jp-toc-ignore">')
            add(f'<caption>Correlations {extra}</caption>')
            stat_row('Parameter1', 'Parameter 2', 'Correlation', cat='th')
            for name1, name2, val in sort_correls:
                stat_row(name1, name2, f"{val:+.4f}")
            add('</table>')
    return ''.join(html)


def correl_table(params):
    """Return a printable correlation table for a Parameters object."""
    varnames = [vname for vname in params if params[vname].vary]
    nwid = max(8, max([len(vname) for vname in varnames])) + 1

    def sfmt(a):
        return f" {a:{nwid}s}"

    def ffmt(a):
        return sfmt(f"{a:+.4f}")

    title = ['', sfmt('Variable')]
    title.extend([sfmt(vname) for vname in varnames])

    title = '|'.join(title) + '|'
    bar = [''] + ['-'*(nwid+1) for i in range(len(varnames)+1)] + ['']
    bar = '+'.join(bar)

    buff = [bar, title, bar]

    for vname, par in params.items():
        if not par.vary:
            continue
        line = ['', sfmt(vname)]
        for vother in varnames:
            if vother == vname:
                line.append(ffmt(1))
            elif vother in par.correl:
                line.append(ffmt(par.correl[vother]))
            else:
                line.append('unknown')
        buff.append('|'.join(line) + '|')
    buff.append(bar)
    return '\n'.join(buff)


def params_html_table(params):
    """Return an HTML representation of Parameters.

    Parameters
    ----------
    params : Parameters
        Object containing the Parameters of the model.

    Returns
    -------
    str
        Multi-line HTML code of fitting parameters.

    """
    has_err = any(p.stderr is not None for p in params.values())
    has_expr = any(p.expr is not None for p in params.values())
    has_brute = any(p.brute_step is not None for p in params.values())

    html = []
    add = html.append

    add('<table class="jp-toc-ignore"><caption>Parameters</caption>')
    headers = ['name', 'value']
    if has_err:
        headers.extend(['standard error', 'relative error'])
    headers.extend(['initial value', 'min', 'max', 'vary'])
    if has_expr:
        headers.append('expression')
    if has_brute:
        headers.append('brute step')

    hrow = trow(headers, cat='th')
    add(f"<tr>{''.join(hrow)}</tr>")

    for par in params.values():
        rows = [par.name, gformat(par.value)]
        if has_err:
            serr = ''
            spercent = ''
            if par.stderr is not None:
                serr = gformat(par.stderr)
                try:
                    spercent = f'({abs(float(par.stderr)/float(par.value)):.2%})'
                except ZeroDivisionError:
                    pass
            rows.extend([serr, spercent])
        rows.extend((par.init_value, gformat(par.min),
                     gformat(par.max), f'{par.vary}'))
        if has_expr:
            expr = ''
            if par.expr is not None:
                expr = par.expr
            rows.append(expr)
        if has_brute:
            brute_step = 'None'
            if par.brute_step is not None:
                brute_step = gformat(par.brute_step)
            rows.append(brute_step)

        hrow = trow(rows, cat='td')
        add(f"<tr>{''.join(hrow)}</tr>")
    add('</table>')
    return ''.join(html)


def report_fit(params, **kws):
    """Print a report of the fitting results."""
    print(fit_report(params, **kws))


def ci_report(ci, with_offset=True, ndigits=5):
    """Return text of a report for confidence intervals.

    Parameters
    ----------
    ci : dict
        The result of :func:`~lmfit.confidence.conf_interval`: a dictionary
        containing a list of ``(sigma, vals)``-tuples for each parameter.
    with_offset : bool, optional
        Whether to subtract best value from all other values (default is
        True).
    ndigits : int, optional
        Number of significant digits to show (default is 5).

    Returns
    -------
    str
        Text of formatted report on confidence intervals.

    """
    maxlen = max(len(i) for i in ci)
    buff = []
    add = buff.append

    def convp(x):
        """Convert probabilities into header for CI report."""
        if abs(x[0]) < 1.e-2:
            return "_BEST_"
        return f"{x[0] * 100:.2f}%"

    title_shown = False
    fmt_best = fmt_diff = "{0:.%if}" % ndigits
    if with_offset:
        fmt_diff = "{0:+.%if}" % ndigits
    for name, row in ci.items():
        if not title_shown:
            add("".join([''.rjust(maxlen+1)] + [i.rjust(ndigits+5)
                                                for i in map(convp, row)]))
            title_shown = True
        thisrow = [f" {name.ljust(maxlen)}:"]
        offset = 0.0
        if with_offset:
            for cval, val in row:
                if abs(cval) < 1.e-2:
                    offset = val
        for cval, val in row:
            if cval < 1.e-2:
                sval = fmt_best.format(val)
            else:
                sval = fmt_diff.format(val-offset)
            thisrow.append(sval.rjust(ndigits+5))
        add("".join(thisrow))

    return '\n'.join(buff)


def report_ci(ci):
    """Print a report for confidence intervals."""
    print(ci_report(ci))


def compare_fit_results(result1, result2, sort_pars=False,
                         show_statistics=True, show_parameters=True,
                         show_model_comparison=True, min_correl=0.1,
                         title1='Result 1', title2='Result 2'):
    """Generate a report comparing two fitting results.

    The report contains comparisons of fit statistics, parameter values
    with uncertainties, and model comparison metrics.

    Parameters
    ----------
    result1 : MinimizerResult, ModelResult, or Parameters
        First fitting result to compare. Can be a MinimizerResult,
        ModelResult, or Parameters object.
    result2 : MinimizerResult, ModelResult, or Parameters
        Second fitting result to compare. Can be a MinimizerResult,
        ModelResult, or Parameters object.
    sort_pars : bool or callable, optional
        Whether to show parameter names sorted in alphanumerical order.
        If False (default), then the parameters will be listed in the
        order they were added to the Parameters dictionary. If callable,
        then this (one argument) function is used to extract a comparison
        key from each list element.
    show_statistics : bool, optional
        Whether to show fit statistics comparison (default is True).
    show_parameters : bool, optional
        Whether to show parameter comparison (default is True).
    show_model_comparison : bool, optional
        Whether to show model comparison metrics like AIC/BIC weights
        (default is True).
    min_correl : float, optional
        Smallest correlation in absolute value to show when showing
        correlations (default is 0.1). Note: correlation comparison is
        not yet implemented.
    title1 : str, optional
        Title for the first result (default is 'Result 1').
    title2 : str, optional
        Title for the second result (default is 'Result 2').

    Returns
    -------
    str
        Multi-line text of fit comparison report.

    Notes
    -----
    The comparison report includes:
    - Fit statistics: chi-square, reduced chi-square, AIC, BIC, R-squared
    - Parameter values: comparison of best-fit values and uncertainties
    - Model comparison: AIC/BIC differences and weights, F-test for nested models

    Examples
    --------
    >>> result1 = model1.fit(data1, params, x=x)
    >>> result2 = model2.fit(data2, params, x=x)
    >>> print(compare_fit_results(result1, result2,
    ...                           title1='Model A', title2='Model B'))

    """
    from .parameter import Parameters

    def get_result_data(result):
        """Extract data from result object."""
        data = {'params': None, 'stats': {}}

        if isinstance(result, Parameters):
            data['params'] = result
        elif hasattr(result, 'params'):
            data['params'] = result.params
            stats = {
                'method': getattr(result, 'method', None),
                'nfev': getattr(result, 'nfev', None),
                'ndata': getattr(result, 'ndata', None),
                'nvarys': getattr(result, 'nvarys', None),
                'chisqr': getattr(result, 'chisqr', None),
                'redchi': getattr(result, 'redchi', None),
                'aic': getattr(result, 'aic', None),
                'bic': getattr(result, 'bic', None),
                'rsquared': getattr(result, 'rsquared', None),
                'success': getattr(result, 'success', None),
                'errorbars': getattr(result, 'errorbars', None),
            }
            data['stats'] = {k: v for k, v in stats.items() if v is not None}

        return data

    data1 = get_result_data(result1)
    data2 = get_result_data(result2)

    params1 = data1['params']
    params2 = data2['params']
    stats1 = data1['stats']
    stats2 = data2['stats']

    if params1 is None or params2 is None:
        raise ValueError("Both results must have Parameters to compare.")

    buff = []
    add = buff.append

    add("=" * 70)
    add("[[Fit Results Comparison]]")
    add("=" * 70)
    add(f"  {title1} vs {title2}")
    add("")

    if show_statistics and (stats1 or stats2):
        add("-" * 70)
        add("[[Fit Statistics Comparison]]")
        add("-" * 70)
        add("")

        stat_labels = {
            'method': 'fitting method',
            'nfev': 'function evals',
            'ndata': 'data points',
            'nvarys': 'variables',
            'chisqr': 'chi-square',
            'redchi': 'reduced chi-square',
            'aic': 'Akaike info crit',
            'bic': 'Bayesian info crit',
            'rsquared': 'R-squared',
            'success': 'success',
            'errorbars': 'error bars estimated',
        }

        all_stats = set(stats1.keys()) | set(stats2.keys())
        display_order = ['method', 'nfev', 'ndata', 'nvarys', 'chisqr',
                         'redchi', 'aic', 'bic', 'rsquared', 'success',
                         'errorbars']

        col1_width = max(len(title1), 15)
        col2_width = max(len(title2), 15)
        label_width = max(len(stat_labels.get(s, s)) for s in all_stats)

        header = f"    {'Statistic':<{label_width}}  {title1:^{col1_width}}  {title2:^{col2_width}}  {'Difference':^{15}}"
        add(header)
        add("    " + "-" * (label_width + col1_width + col2_width + 19))

        for stat in display_order:
            if stat not in all_stats:
                continue

            label = stat_labels.get(stat, stat)
            val1 = stats1.get(stat, 'N/A')
            val2 = stats2.get(stat, 'N/A')

            if isinstance(val1, float):
                val1_str = gformat(val1)
            elif val1 is None:
                val1_str = 'N/A'
            else:
                val1_str = str(val1)

            if isinstance(val2, float):
                val2_str = gformat(val2)
            elif val2 is None:
                val2_str = 'N/A'
            else:
                val2_str = str(val2)

            diff_str = ''
            if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                diff = val2 - val1
                if abs(diff) < 1e10 and abs(diff) > 1e-10:
                    diff_str = f"{diff:+.4e}"
                elif abs(diff) >= 1e10 or abs(diff) <= 1e-10:
                    diff_str = f"{diff:+.4g}"
                else:
                    diff_str = f"{diff:+.6f}"

            line = f"    {label:<{label_width}}  {val1_str:>{col1_width}}  {val2_str:>{col2_width}}  {diff_str:^{15}}"
            add(line)

        add("")

    if show_parameters:
        add("-" * 70)
        add("[[Parameter Comparison]]")
        add("-" * 70)
        add("")

        all_params = set(params1.keys()) | set(params2.keys())

        if sort_pars:
            if callable(sort_pars):
                key = sort_pars
            else:
                key = alphanumeric_sort
            parnames = sorted(all_params, key=key)
        else:
            parnames = list(params1.keys())
            for p in params2.keys():
                if p not in parnames:
                    parnames.append(p)

        if not parnames:
            add("    No parameters found.")
            add("")
        else:
            namelen = max(len(n) for n in parnames)

            has_err1 = any(p.stderr is not None for p in params1.values()
                          if hasattr(p, 'stderr'))
            has_err2 = any(p.stderr is not None for p in params2.values()
                          if hasattr(p, 'stderr'))
            has_any_err = has_err1 or has_err2

            header = f"    {'Parameter':<{namelen}}  "
            if has_any_err:
                header += f"{title1:^25}  {title2:^25}  {'Value Diff':^18}  {'Rel Diff':^12}"
            else:
                header += f"{title1:^12}  {title2:^12}  {'Value Diff':^18}  {'Rel Diff':^12}"
            add(header)
            add("    " + "-" * (namelen + 70 if has_any_err else namelen + 56))

            for name in parnames:
                par1 = params1.get(name, None)
                par2 = params2.get(name, None)

                val1 = getattr(par1, 'value', None)
                val2 = getattr(par2, 'value', None)
                err1 = getattr(par1, 'stderr', None)
                err2 = getattr(par2, 'stderr', None)
                vary1 = getattr(par1, 'vary', True) if par1 else None
                vary2 = getattr(par2, 'vary', True) if par2 else None
                expr1 = getattr(par1, 'expr', None) if par1 else None
                expr2 = getattr(par2, 'expr', None) if par2 else None

                val1_str = 'N/A'
                val2_str = 'N/A'
                err1_str = ''
                err2_str = ''
                diff_str = ''
                rel_diff_str = ''

                if val1 is not None:
                    if isinstance(val1, (int, float)):
                        val1_str = gformat(val1)
                    else:
                        val1_str = str(val1)

                if val2 is not None:
                    if isinstance(val2, (int, float)):
                        val2_str = gformat(val2)
                    else:
                        val2_str = str(val2)

                if err1 is not None:
                    if isinstance(err1, (int, float)):
                        err1_str = f" +/- {gformat(err1)}"

                if err2 is not None:
                    if isinstance(err2, (int, float)):
                        err2_str = f" +/- {gformat(err2)}"

                if isinstance(val1, (int, float)) and isinstance(val2, (int, float)):
                    diff = val2 - val1
                    if abs(diff) < 1e10 and abs(diff) > 1e-10:
                        diff_str = f"{diff:+.4e}"
                    elif abs(diff) >= 1e10 or abs(diff) <= 1e-10:
                        diff_str = f"{diff:+.4g}"
                    else:
                        diff_str = f"{diff:+.6f}"

                    if abs(val1) > 1e-15:
                        rel_diff = (val2 - val1) / abs(val1)
                        rel_diff_str = f"{rel_diff:+.2%}"
                    else:
                        rel_diff_str = 'N/A'

                status1 = ''
                status2 = ''
                if par1:
                    if expr1:
                        status1 = " (expr)"
                    elif not vary1:
                        status1 = " (fixed)"
                if par2:
                    if expr2:
                        status2 = " (expr)"
                    elif not vary2:
                        status2 = " (fixed)"

                line = f"    {name:<{namelen}}  "
                if has_any_err:
                    line += f"{val1_str}{err1_str}{status1:>8s}  "
                    line += f"{val2_str}{err2_str}{status2:>8s}  "
                else:
                    line += f"{val1_str:>12}  "
                    line += f"{val2_str:>12}  "
                line += f"{diff_str:^18}  {rel_diff_str:^12}"
                add(line)

            add("")

    if show_model_comparison and stats1 and stats2:
        add("-" * 70)
        add("[[Model Comparison]]")
        add("-" * 70)
        add("")

        aic1 = stats1.get('aic', None)
        aic2 = stats2.get('aic', None)
        bic1 = stats1.get('bic', None)
        bic2 = stats2.get('bic', None)
        chisqr1 = stats1.get('chisqr', None)
        chisqr2 = stats2.get('chisqr', None)
        nvarys1 = stats1.get('nvarys', None)
        nvarys2 = stats2.get('nvarys', None)
        ndata1 = stats1.get('ndata', None)
        ndata2 = stats2.get('ndata', None)

        if aic1 is not None and aic2 is not None:
            add("  [AIC Comparison]")
            aic_min = min(aic1, aic2)
            delta_aic1 = aic1 - aic_min
            delta_aic2 = aic2 - aic_min

            rel_lik1 = np.exp(-0.5 * delta_aic1)
            rel_lik2 = np.exp(-0.5 * delta_aic2)
            aic_sum = rel_lik1 + rel_lik2
            weight1 = rel_lik1 / aic_sum if aic_sum > 0 else 0
            weight2 = rel_lik2 / aic_sum if aic_sum > 0 else 0

            add(f"    {title1:15s}: AIC = {aic1:10.4f}, ΔAIC = {delta_aic1:10.4f}, weight = {weight1:.4f}")
            add(f"    {title2:15s}: AIC = {aic2:10.4f}, ΔAIC = {delta_aic2:10.4f}, weight = {weight2:.4f}")
            add("")

            if delta_aic1 < 2 and delta_aic2 < 2:
                add("    Conclusion: Both models have substantial support.")
            elif delta_aic1 < 2:
                add(f"    Conclusion: {title1} has substantial support.")
            elif delta_aic2 < 2:
                add(f"    Conclusion: {title2} has substantial support.")
            elif delta_aic1 < 10:
                add(f"    Conclusion: {title1} has considerably less support.")
            elif delta_aic2 < 10:
                add(f"    Conclusion: {title2} has considerably less support.")
            else:
                winner = title1 if aic1 < aic2 else title2
                add(f"    Conclusion: {winner} is strongly preferred.")
            add("")

        if bic1 is not None and bic2 is not None:
            add("  [BIC Comparison]")
            bic_min = min(bic1, bic2)
            delta_bic1 = bic1 - bic_min
            delta_bic2 = bic2 - bic_min

            rel_lik1_bic = np.exp(-0.5 * delta_bic1)
            rel_lik2_bic = np.exp(-0.5 * delta_bic2)
            bic_sum = rel_lik1_bic + rel_lik2_bic
            weight1_bic = rel_lik1_bic / bic_sum if bic_sum > 0 else 0
            weight2_bic = rel_lik2_bic / bic_sum if bic_sum > 0 else 0

            add(f"    {title1:15s}: BIC = {bic1:10.4f}, ΔBIC = {delta_bic1:10.4f}, weight = {weight1_bic:.4f}")
            add(f"    {title2:15s}: BIC = {bic2:10.4f}, ΔBIC = {delta_bic2:10.4f}, weight = {weight2_bic:.4f}")
            add("")

            if delta_bic1 < 2 and delta_bic2 < 2:
                add("    Conclusion: Evidence against the model with higher BIC is not worth more than a bare mention.")
            elif delta_bic1 < 6 and delta_bic2 < 6:
                add("    Conclusion: Evidence against the model with higher BIC is positive.")
            elif delta_bic1 < 10 and delta_bic2 < 10:
                add("    Conclusion: Evidence against the model with higher BIC is strong.")
            else:
                winner = title1 if bic1 < bic2 else title2
                add(f"    Conclusion: Evidence against the model with higher BIC is very strong ({winner} preferred).")
            add("")

        if (chisqr1 is not None and chisqr2 is not None and
            nvarys1 is not None and nvarys2 is not None and
            ndata1 is not None and ndata2 is not None and
            ndata1 == ndata2):

            add("  [F-test for Nested Models]")

            df1 = ndata1 - nvarys1
            df2 = ndata2 - nvarys2

            if nvarys1 != nvarys2 and df1 > 0 and df2 > 0:
                if nvarys2 > nvarys1:
                    chisqr_reduced = chisqr1
                    chisqr_full = chisqr2
                    df_reduced = df1
                    df_full = df2
                    model_reduced = title1
                    model_full = title2
                else:
                    chisqr_reduced = chisqr2
                    chisqr_full = chisqr1
                    df_reduced = df2
                    df_full = df1
                    model_reduced = title2
                    model_full = title1

                df_diff = df_reduced - df_full

                if df_diff > 0 and df_full > 0:
                    try:
                        from scipy.stats import f, chi2

                        f_stat = ((chisqr_reduced - chisqr_full) / df_diff) / (chisqr_full / df_full)
                        p_value = 1 - f.cdf(f_stat, df_diff, df_full)

                        lr_stat = chisqr_reduced - chisqr_full
                        lr_pvalue = 1 - chi2.cdf(lr_stat, df_diff)

                        add(f"    Comparing nested models:")
                        add(f"      Reduced model ({model_reduced}): {nvarys1 if nvarys2 > nvarys1 else nvarys2} parameters")
                        add(f"      Full model ({model_full}): {nvarys2 if nvarys2 > nvarys1 else nvarys1} parameters")
                        add(f"      Difference: {df_diff} degrees of freedom")
                        add("")
                        add(f"      F-statistic: {f_stat:.4f}")
                        add(f"      F-test p-value: {p_value:.4e}")
                        add(f"      Likelihood ratio statistic: {lr_stat:.4f}")
                        add(f"      Likelihood ratio p-value: {lr_pvalue:.4e}")
                        add("")

                        if p_value < 0.05:
                            add(f"    Conclusion: Full model ({model_full}) is significantly better (p < 0.05).")
                        else:
                            add(f"    Conclusion: Reduced model ({model_reduced}) is preferred (not significantly worse).")
                    except Exception:
                        add("    Note: Could not perform F-test (scipy.stats required).")
                else:
                    add("    Note: F-test not applicable (models have same complexity or invalid df).")
            else:
                add("    Note: F-test requires models with different numbers of parameters fitted to the same data.")
            add("")

    add("=" * 70)
    return '\n'.join(buff)


def report_compare_fit(result1, result2, **kws):
    """Print a report comparing two fitting results.

    Parameters
    ----------
    result1 : MinimizerResult, ModelResult, or Parameters
        First fitting result to compare.
    result2 : MinimizerResult, ModelResult, or Parameters
        Second fitting result to compare.
    **kws : dict, optional
        Additional keyword arguments passed to :func:`compare_fit_results`.

    See Also
    --------
    compare_fit_results : Generate comparison report as string.

    """
    print(compare_fit_results(result1, result2, **kws))
