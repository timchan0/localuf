"""Plot failure probability to deduce accuracy thresholds.

Available functions:
* monte_carlo
* subset_sample
"""

import itertools

from matplotlib import pyplot as plt
from matplotlib.container import ErrorbarContainer
import numpy as np
from pandas import DataFrame
from statsmodels.stats.proportion import proportion_confint

from localuf.constants import STANDARD_ERROR_ALPHA


def monte_carlo(
        data: DataFrame,
        title: str = '',
        xlabel: None | str = None,
        ylabel: None | str = None,
        legend: None | bool = None,
        alpha: float = STANDARD_ERROR_ALPHA,
        method: str = 'wilson',
        base_color: None | tuple[float, float, float] | str = None,
        capsize: float = 2,
        **kwargs_for_errorbar,
):
    """Plot threshold data in ``data``.
    
    
    :param data: a DataFrame where each
        column a (distance, probability);
    rows m, n indicate number of logical errors and samples, respectively.
    :param title: plot title.
    :param xlabel: x-axis label.
    :param ylabel: y-axis label.
    :param legend: whether to show legend.
    :param alpha: significance level of confidence intervals.
    :param method: method to compute confidence intervals.
        For details on confidence intervals,
    see https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html.
    :param base_color: a single color for all errorbars and their connecting lines.
        Increasing distance is then shown by increasing opacity.
    If ``None``, each distance is shown by a different, fully opaque color.
    :param capsize: length of error bar caps in points.
    :param kwargs_for_errorbar: passed to ``pyplot.errorbar``.
    
    
    :returns:
    * ``plotted`` transposed ``data`` with additional columns ``f, lo, hi`` storing respectively the mean, lower and upper confidence bounds of failure probability.
    * ``containers`` a dictionary where each key a distance; value, the ``ErrorbarContainer`` for that distance.
    """
    containers: dict[int, ErrorbarContainer] = {}
    if base_color is None:
        colors = itertools.repeat(None)
        if legend is None:
            legend = True
    else:
        if legend is None:
            legend = False
        d_count = len(data.columns.unique('d'))
        colors = [(base_color, k/d_count) for k in range(1, d_count+1)]
    plotted = data.T
    plotted['f'] = plotted.m / plotted.n
    plotted['lo'], plotted['hi'] = proportion_confint(
        plotted.m,
        plotted.n,
        alpha=alpha,
        method=method,
    )
    # next line is a fix for when m = 0 then sometimes lo > 0
    plotted['lo'] = np.min(plotted[['lo', 'f']], axis=1)
    for (d, df), color in zip(plotted.groupby(level='d'), colors):
        df = df.droplevel('d')
        container = plt.errorbar(
            x=df.index,
            y=df.f,
            yerr=(df.f-df.lo, df.hi-df.f),
            capsize=capsize,
            label=str(d),
            color=color,
            **kwargs_for_errorbar,
        )
        containers[d] = container # type: ignore
    if legend:
        plt.legend(title=r'$d =\dots$')
    if title:
        plt.title(title)
    if xlabel is None:
        xlabel = 'noise level'
    plt.xlabel(xlabel)
    if ylabel is None:
        ylabel = 'logical error probability'
    plt.ylabel(ylabel)
    return plotted, containers


def subset_sample(
        data: DataFrame,
        legend: bool = True,
        alpha: float = 0.3,
        title: str = '',
):
    """Plot failure probability from output of ``get_failure_data_from_subset_sample``.
    
    
    :param data: output of ``get_failure_data_from_subset_sample``.
    :param legend: whether to show legend.
    :param alpha: transparency of confidence region.
    :param title: plot title.
    """
    for d, df in data.groupby(level='d'):
        df = df.droplevel('d')
        plt.plot(df.index, df.f, label=str(d))
        plt.fill_between(
            x=df.index,
            y1=df.lo,
            y2=df.hi,
            alpha=alpha,
        )
    plt.loglog()
    if legend:
        plt.legend(title=r'$d =\dots$')
    plt.xlabel('noise level')
    plt.ylabel('logical error probability')
    if title:
        plt.title(title)