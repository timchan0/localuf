"""Module for functions to process numerical data, mainly from `sim`."""

from typing import Type

import numpy as np
import numpy.typing as npt
import pandas as pd
from pandas import DataFrame
import statsmodels.api as sm
from statsmodels.stats.proportion import proportion_confint

from localuf._base_classes import Code
from localuf.type_aliases import NoiseModel
from localuf.constants import STANDARD_ERROR_ALPHA


def get_failure_data(
        data: DataFrame,
        p_slice=slice(None),
        alpha=STANDARD_ERROR_ALPHA,
        method='wilson',
) -> DataFrame:
    """Get failure stats from output of `sim.accuracy.monte_carlo`.

    Input:
    * `data` output from `sim.accuracy.monte_carlo`.
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
    dT: DataFrame = data.loc[:, (slice(None), p_slice)].T # type: ignore
    dT['f'] = dT.m / dT.n
    dT['lo'], dT['hi'] = proportion_confint(dT.m, dT.n, alpha=alpha, method=method)
    dT['x'] = np.log10(dT.index.get_level_values('p'))
    dT['y']  = np.log10(dT.f)
    dT['yerr'] = (np.log10(dT.hi) - np.log10(dT.lo)) / 2
    del dT['m']
    del dT['n']
    return dT


def get_failure_data_from_SS(
        data: DataFrame,
        code_class: Type[Code],
        noise: NoiseModel,
        ps: npt.NDArray[np.float_],
        alpha: float = STANDARD_ERROR_ALPHA,
        method='normal',
):
    """Get failure stats from output of `sim.accuracy.subset_sample`.
    
    Output: a DataFrame indexed by (distance, probability),
    with columns:
    * `f` logical error probability.
    * `lo` lower bound of `f`.
    * `hi` upper bound of `f`.

    Side effects: adds columns `f`, `SE_lo`, `SE_hi` to `data`.
    """
    data['f'] = data.m / data.n
    lo, hi = proportion_confint(data.m, data.n, alpha=alpha, method=method)
    data['SE_lo'] = data.f - lo
    data['SE_hi'] = hi - data.f
    data.fillna(0, inplace=True)
    ls = []
    for d, df in data.groupby(level='d'): # type: ignore
        d: int
        df = df.droplevel('d')
        code = code_class(d, noise=noise)
        f, lo, hi = [], [], []
        for p in ps:
            subset_probs = code.NOISE.subset_probabilities(p, survival=False)
            considered = df.join(subset_probs, rsuffix=' current')
            ignored = subset_probs[~subset_probs.index.isin(df.index)]

            mean = np.dot(considered['subset prob current'], considered['f'])
            se_lo = np.linalg.norm(considered['subset prob current'] * considered['SE_lo'])
            se_hi = np.linalg.norm(considered['subset prob current'] * considered['SE_hi'])
            cutoff = ignored['subset prob'].sum()

            f.append(mean)
            lo.append(mean - se_lo)
            hi.append(mean + se_hi + cutoff)
        ls.append(DataFrame({
            'f': f,
            'lo': lo,
            'hi': hi,
        }, index=ps))
    return pd.concat(
        ls,
        keys=data.index.get_level_values('d').unique(),
        names=['d', 'p'],
    )


def get_log_runtime_data(data: DataFrame):
    """Get log runtime data from runtime data.

    Input:
    `data` output from `sim.runtime.batch`.

    Output:
    `log_data` a DataFrame where each
    column a stat: `x`, `y`, or `yerr`;
    row, a (probability, distance).
    """
    df: DataFrame = np.log10(data) # type: ignore
    log_data = DataFrame({
        'x': np.log10(data.columns.get_level_values('d')),
        'y': df.mean(),
        'yerr': df.sem(),
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
    records: list[tuple[
        int | float,
        float, float,
        float, float,
    ]] = []
    index_name, _ = log_data.index.names
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
            idx, # type: ignore
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


def from_merging_to_all(merging_data: DataFrame, _extra_steps_per_layer=2):
    """Convert output of `sum.runtime.frugal` with `time_only='merging'` to `time_only='all'`."""
    dc = {(d, p): merging_data[d, p] + _extra_steps_per_layer*d for (d, p) in merging_data.columns}
    return DataFrame(dc, columns=merging_data.columns)