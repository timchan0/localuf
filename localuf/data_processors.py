"""Module for functions to process numerical data, mainly from `sim`."""

import numpy as np
from pandas import DataFrame
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from localuf.constants import STANDARD_ERROR_ALPHA


def get_failure_data(
        data: DataFrame,
        p_slice=slice(None),
        alpha=STANDARD_ERROR_ALPHA,
        method='wilson',
) -> DataFrame:
    """Get failure stats from output of `sim.make_threshold_data`.

    Input:
    * `data` output from `sim.make_threshold_data`.
    * `p_slice` slice of probabilities to restrict output to.
    * `alpha` significance level of confidence intervals.
    * `method` method to compute confidence intervals.
    For details on confidence intervals,
    see https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html.

    Output:
    * `dT` a DataFrame where each row a (distance, probability);
    columns are:
    * `f` logical failure rate
    * `lo` lower confidence bound of `f`
    * `hi` upper confidence bound of `f`
    * `x` log10(`p`)
    * `y` log10(`f`)
    * `yerr` half the confidence interval of `y`.
    """
    dT = data.loc[:, (slice(None), p_slice)].T # type: ignore
    dT['f'] = dT.m / dT.n
    dT['lo'], dT['hi'] = proportion_confint(dT.m, dT.n, alpha=alpha, method=method)
    dT['x'] = np.log10(dT.index.get_level_values('p'))
    dT['y']  = np.log10(dT.f)
    dT['yerr'] = (np.log10(dT.hi) - np.log10(dT.lo)) / 2
    del dT['m']
    del dT['n']
    return dT


def get_log_runtime_data(data: DataFrame):
    """Get log runtime data from runtime data.

    Input:
    * `data` output from `make_runtime_data`.

    Output:
    * `log_data` a DataFrame where each
    column a stat: `x`, `y`, or `yerr`;
    row, a (probability, distance).
    """
    df = np.log10(data)
    log_data = DataFrame({
        'x': np.log10(data.columns.get_level_values('d')),
        'y': df.mean(),
        'yerr': df.sem(), # type: ignore
    })
    return log_data.swaplevel(0, 1, axis=0).sort_index(axis=0)


def get_stats(log_data: DataFrame, missing='drop', **kwargs):
    """Get WLS (weighted least squares) stats from output of either
    `get_log_runtime_data` OR `get_failure_data`.

    Input:
    * `log_data` output from `get_log_runtime_data` or `get_failure_data`.
    * `missing, **kwargs` passed to `statsmodels.regression.linear_model.WLS`.

    Output: `stats` a DataFrame where each row an x-value
    (probability OR distance);
    columns are:
    * `intercept`
    * `se_intercept`
    * `gradient`
    * `se_gradient`
    * `r_squared`
    """
    records = []
    index_name = log_data.index.names[0]
    for idx, df in log_data.groupby(level=0):
        df=df.droplevel(index_name, axis=0)
        results = sm.WLS(
            endog=df['y'],
            exog=sm.add_constant(df['x']),
            weights=1/df['yerr']**2, # type: ignore
            missing=missing,
            **kwargs,
        ).fit()
        intercept, gradient = results.params
        se_intercept, se_gradient = results.bse
        r_squared = results.rsquared
        records.append((
            idx,
            intercept, se_intercept,
            gradient, se_gradient,
            r_squared,
        ))
    stats = DataFrame.from_records(
        records,
        index=index_name,
        columns=[
            index_name,
            'intercept', 'se_intercept',
            'gradient', 'se_gradient',
            'r_squared',
        ]
    )
    return stats