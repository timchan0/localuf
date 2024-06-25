"""Module for plotting numeric data from `sim`."""

import itertools
from typing import Iterable, Literal
from string import ascii_lowercase

from matplotlib import pyplot as plt
from pandas import DataFrame, Index, Series
from statsmodels.stats.proportion import proportion_confint

from localuf.constants import STANDARD_ERROR_ALPHA

def threshold_data(
        data: DataFrame,
        title: str = '',
        xlabel: None | str = None,
        ylabel: None | str = None,
        legend: bool = True,
        alpha: float = STANDARD_ERROR_ALPHA,
        method: str = 'wilson',
        **kwargs,
):
    """Plot threshold data in `data`.

    Input:
    * `data` a DataFrame where each
    column a (distance, probability);
    rows m, n indicate number of logical errors and samples, respectively.
    * `title, xlabel, ylabel, legend` for plot.
    * `alpha` significance level of confidence intervals.
    * `method` method to compute confidence intervals.
    For details on confidence intervals,
    see https://www.statsmodels.org/dev/generated/statsmodels.stats.proportion.proportion_confint.html.
    * `kwargs` passed to `pyplot.errorbar`.
    """
    for d, df in data.T.groupby(level='d'):
        df = df.droplevel('d')
        mean: Series[float] = df.m / df.n
        lo, hi = proportion_confint(df.m, df.n, alpha=alpha, method=method)
        plt.errorbar(
            x=df.index,
            y=mean,
            yerr=(mean-lo, hi-mean),
            capsize=2,
            label=str(d),
            **kwargs,
        )
    if legend:
        plt.legend(title=r'$d =\dots$')
    if title:
        plt.title(title)
    if xlabel is None:
        xlabel = 'physical error probability'
    plt.xlabel(xlabel)
    if ylabel is None:
        ylabel = 'logical error probability'
    plt.ylabel(ylabel)

def subset_sampled(
        data: DataFrame,
        legend: bool = True,
        alpha: float = 0.3,
        title: str = '',
):
    """Plot failure probability from output of `get_failure_data_from_SS`.

    Input:
    * `data` output of `get_failure_data_from_SS`.
    * `legend` whether to show legend.
    * `alpha` transparency of confidence region.
    * `title` plot title.
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
    plt.xlabel('physical error probability')
    plt.ylabel('logical error probability')
    if title:
        plt.title(title)

def mean_runtime(
        data: DataFrame,
        title: str = '',
        per_measurement_round: bool = False,
        yerr_shows: Literal['sem', 'std'] = 'sem',
        ps: Iterable[float] | None = None,
        legend: bool = True,
        grid: bool = True,
        xlabel: None | str = None,
        ylabel: None | str = None,
):
    """Plot mean timestep count against code distance.
    
    Input:
    * `data` a DataFrame where each
    column a (distance, probability);
    row, a runtime sample.
    * `title` plot title.
    * `per_measurement_round` whether to divide runtime by measurement round count.
    * `yerr_shows` what errorbars show:
    either `'sem'` for standard error,
    or `'std'` for standard deviation.
    * `ps` iterable specifying which probabilities to plot, in case want to omit any.

    Output:
    `data_copy` a copy of `data` with runtimes divided by distance
    if `per_measurement_round`
    else an exact copy of `data`.
    """
    ds: Index[int] = data.columns.get_level_values('d').unique()
    data_copy = data.copy()
    if per_measurement_round:
        for d in ds:
            data_copy[d] = data[d] / d
        default_ylabel = 'mean runtime per measurement round'
    else:
        default_ylabel = 'mean runtime'
    if ps is None:
        ps = data.columns.get_level_values('p').unique()
    for p in ps:
        df: DataFrame = data_copy.xs(p, level='p', axis=1) # type: ignore

        if yerr_shows == 'sem':
            yerr: Series[float] = df.sem()
        elif yerr_shows == 'std':
            yerr: Series[float] = df.std()
        else:
            raise ValueError(f'invalid yerr_shows: {yerr_shows}')
        
        plt.errorbar(
            x=ds,
            y=df.mean(),
            yerr=yerr,
            capsize=2,
            label=f'{p:.1e}',
        )
    plt.xticks(ds);
    if legend: plt.legend(title=r'$p =\dots$', reverse=True)
    if grid: plt.grid(which='both')
    if xlabel is None: xlabel = 'code distance'
    if ylabel is None: ylabel = default_ylabel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title: plt.title(title)
    return data_copy

def runtime_distribution(
        data: DataFrame,
        p: float,
        bins=80,
        horizontal=True,
        figsize: None | tuple[float, float] = None,
        log_scale=True,
        grid=False,
        global_range=True,
):
    df: DataFrame = data.xs(p, level='p', axis=1) # type: ignore
    ds = df.columns
    min_runtime = min(df.min())
    max_runtime = max(df.max())
    range_ = (min_runtime, max_runtime) if global_range else None
    if horizontal:
        if figsize is None:
            figsize=(15, 4)
        f = plt.figure(figsize=figsize)
        for k, d in enumerate(ds, start=1):
            ax = plt.subplot(1, len(ds), k)
            if bins:
                df[d].hist(
                    bins=bins,
                    range=range_,
                    orientation='horizontal',
                )
            else:
                vc = df[d].value_counts(); plt.barh(vc.index, vc, height=1)
            plt.grid(grid)
            plt.ylim(0, max_runtime)
            if k == 1:
                plt.ylabel('timestep count')
            else:
                ax.set_yticklabels([])
            if log_scale:
                plt.xscale('log')
            plt.xlabel(f'$d = {d}$')
            plt.axhline(y=df[d].mean(), color='magenta')
            plt.axhline(y=df[d].max(), color='r')
    else:
        if figsize is None:
            figsize=(5, 10)
        f = plt.figure(figsize=figsize)
        for k, d in enumerate(ds, start=1):
            ax = plt.subplot(len(ds), 1, k)
            if bins:
                df[d].hist(
                    bins=bins,
                    range=range_,
                )
            else:
                vc = df[d].value_counts(); plt.bar(vc.index, vc, width=1)
            plt.grid(grid)
            plt.xlim(0, max_runtime)
            if k == len(ds):
                plt.xlabel('timestep count')
            else:
                ax.set_xticklabels([])
            if log_scale:
                plt.yscale('log')
            plt.ylabel(f'$d = {d}$')
            plt.axvline(x=df[d].mean(), color='magenta')
            plt.axvline(x=df[d].max(), color='r')
    return f

def runtime_distributions(
        data: list[DataFrame],
        p: float,
        bins: int | Iterable[int] = 80,
        figsize: None | tuple[float, float] = None,
        log_scale=True,
        grid=False,
        global_range=True,
        **kwargs,
):
    """Histogram runtime distributions for each DataFrame in `data`.
    
    Input:
    * `data` list of DataFrames. In each DataFrame, each
    column a (distance, probability);
    row, a runtime sample.
    * `p` physical error probability associated to the runtimes histogrammed.
    * `bins` bin count in each histogram.
    If an int, use same bin count for all entries in `data`.
    * `global_range` whether to use same bins for all distances within a DataFrame.
    * `**kwargs` passed to `plt.ylabel` for the first subplot.
    """
    if isinstance(bins, int):
        bins = itertools.repeat(bins)
    ds: Index[int] = next(iter(data)).xs(p, level='p', axis=1).columns # type: ignore
    if figsize is None:
        figsize=(15, 4)
    w, h = figsize
    figsize = (w, h*len(data))
    f = plt.figure(figsize=figsize)
    for i, (dfmi, bin_count, letter) in enumerate(zip(data, bins, ascii_lowercase)):
        df: DataFrame = dfmi.xs(p, level='p', axis=1) # type: ignore
        min_runtime = min(df.min())
        max_runtime = max(df.max())
        range_ = (min_runtime, max_runtime) if global_range else None
        for j, d in enumerate(ds, start=1):
            ax = plt.subplot(len(data), len(ds), i*len(ds) + j)
            if bin_count:
                df[d].hist(
                    bins=bin_count,
                    range=range_,
                    orientation='horizontal',
                )
            else:
                vc = df[d].value_counts(); plt.barh(vc.index, vc, height=1)
            plt.grid(grid)
            plt.ylim(0, max_runtime)
            if j == 1:
                plt.ylabel(f'({letter})', **kwargs)
            else:
                ax.set_yticklabels([])
            if log_scale:
                plt.xscale('log')
            if i == len(data)-1:
                plt.xlabel(f'$d = {d}$')
            plt.axhline(y=df[d].mean(), color='magenta')
            plt.axhline(y=df[d].max(), color='r')
    return f

def runtime_violin(
        data: DataFrame,
        p: float,
        title='',
        widths=1,
        showextrema=False,
        errorbar_kwargs: None | dict = None,
        **kwargs,
):
    df: DataFrame = data.xs(p, level='p', axis=1) # type: ignore
    if errorbar_kwargs is None: errorbar_kwargs = {}
    parts = plt.violinplot(
        df,
        positions=list(df.columns),
        widths=widths,
        showextrema=showextrema,
        **kwargs,
    )
    for pc in parts['bodies']: # type: ignore
        pc.set_alpha(1)
    plt.errorbar(
        x=df.columns,
        y=df.mean(),
        yerr=df.std(),
        fmt='.',
        capsize=3,
        linestyle='none',
        **errorbar_kwargs,
    )
    plt.xticks(df.columns)
    plt.grid(which='both')
    plt.xlabel('code distance')
    plt.ylabel('timestep count')
    if title:
        plt.title(title)