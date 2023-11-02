"""Module for plotting numeric data from `sim`."""

import itertools
from typing import Iterable
from string import ascii_lowercase

from matplotlib import pyplot as plt
from pandas import DataFrame
from statsmodels.stats.proportion import proportion_confint

from localuf.constants import STANDARD_ERROR_ALPHA

def threshold_data(
        data: DataFrame,
        title='',
        xlabel=None,
        ylabel=None,
        legend=True,
        alpha=STANDARD_ERROR_ALPHA,
        method='wilson',
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
    """
    for d, df in data.T.groupby(level=0):
        df = df.droplevel('d')
        mean = df.m / df.n
        lo, hi = proportion_confint(df.m, df.n, alpha=alpha, method=method)
        plt.errorbar(
            x=df.index,
            y=mean,
            yerr=(mean-lo, hi-mean),
            capsize=2,
            label=f'$d={d}$',
            **kwargs,
        )
    if legend:
        plt.legend()
    if title:
        plt.title(title)
    if xlabel is None:
        xlabel = 'physical error probability'
    plt.xlabel(xlabel)
    if ylabel is None:
        ylabel = 'logical error probability'
    plt.ylabel(ylabel)

def mean_runtime(
        data: DataFrame,
        title='',
        per_measurement_round=False,
        yerr_shows='sem',
        ps: Iterable[float] | None = None,
        legend=True,
        grid=True,
        xlabel=None,
        ylabel=None,
):
    """Plot mean number of timesteps against code distance.
    
    Inputs:
    * `data` a DataFrame where each
    column a (distance, probability);
    row, a runtime sample.
    * `title` plot title.
    * `per_measurement_round` whether to divide runtime by number of measurement rounds.
    * `yerr_shows` what errorbars show:
    either `'sem'` for standard error,
    or `'std'` for standard deviation.
    * `ps` iterable specifying which probabilities to plot, in case want to omit any.
    """
    ds = data.columns.get_level_values('d').unique()
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
        df = data_copy.xs(p, level='p', axis=1)

        if yerr_shows == 'sem':
            yerr = df.sem()
        elif yerr_shows == 'std':
            yerr = df.std()
        else:
            raise ValueError(f'invalid yerr_shows: {yerr_shows}')
        
        plt.errorbar(
            x=ds,
            y=df.mean(),
            yerr=yerr, # type: ignore
            capsize=2,
            label=f'$p = ${p:.1e}',
        )
    plt.xticks(ds);
    if legend:
        plt.legend()
    if grid:
        plt.grid(which='both')
    if xlabel is None:
        xlabel = 'code distance'
    if ylabel is None:
        ylabel = default_ylabel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title:
        plt.title(title)

def runtime_distributions(
        data: list[DataFrame],
        p: float,
        bins: int | Iterable[int] = 80,
        figsize=None,
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
    ds = next(iter(data)).xs(p, level='p', axis=1).columns
    if figsize is None:
        figsize=(15, 4)
    w, h = figsize
    figsize = (w, h*len(data))
    f = plt.figure(figsize=figsize)
    for i, (dfmi, bin_count, letter) in enumerate(zip(data, bins, ascii_lowercase)):
        df = dfmi.xs(p, level='p', axis=1)
        min_runtime = min(df.min())
        max_runtime = max(df.max())
        range = (min_runtime, max_runtime) if global_range else None
        for j, d in enumerate(ds, start=1):
            ax = plt.subplot(len(data), len(ds), i*len(ds) + j)
            if bin_count:
                df[d].hist(
                    bins=bin_count,
                    range=range,
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