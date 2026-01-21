"""Plot runtime data from ``sim.runtime``.

Available functions:
* mean
* distribution
* distributions
* violin
"""

from collections.abc import Callable
import itertools
from string import ascii_lowercase
from typing import Iterable, Literal, Sequence

from matplotlib import pyplot as plt
from matplotlib.lines import Line2D
from matplotlib.container import ErrorbarContainer
from pandas import DataFrame, Index, Series


DEFAULT_FILL_ALPHA = 0.15


def mean(
        data: DataFrame,
        title: str = '',
        per_measurement_round: bool = False,
        layers_per_sample: Callable[[int], int] = lambda d: d,
        yerr_shows: Literal['sem', 'std'] = 'sem',
        noise_levels: Sequence[float] | None = None,
        legend: None | bool = None,
        grid: bool = False,
        xlabel: None | str = None,
        ylabel: None | str = None,
        base_color: None | tuple[float, float, float] | str = None,
        fill_between: bool = True,
        fill_alpha: float = DEFAULT_FILL_ALPHA,
        capsize: float = 2,
        quantile: None | float = None,
        quantile_linestyle: str = '--',
        **kwargs,
):
    """Plot mean timestep count against code distance.
    
    
    :param data: a DataFrame where each
        column a (distance, probability);
    row, a runtime sample.
    :param title: plot title.
    :param per_measurement_round: whether to divide runtime by measurement round count.
    :param layers_per_sample: a function with input ``d`` that outputs
        the measurement round count per row of ``data``.
    Affects output only if ``per_measurement_round``.
    :param yerr_shows: what errorbars show:
        either ``'sem'`` for standard error,
    or ``'std'`` for standard deviation.
    :param noise_levels: sequence specifying which noise levels to plot, in case want to omit any.
    :param base_color: a single color for all errorbars and their connecting lines.
        Decreasing noise level is then shown by increasing opacity.
    If ``None``, each noise level is shown by a different, fully opaque color.
    :param fill_between: whether to use ``fill_between`` instead of ``errorbar``.
    :param fill_alpha: alpha value for the filled area.
    :param capsize: length of error bar caps in points.
    :param quantile: optional quantile (in the interval [0, 1]) to line plot.
    :param quantile_linestyle: linestyle for the quantile line.
    :param kwargs: passed to either ``errorbar`` or ``fill_between``
        depending on which is used.
    
    
    :returns:
    * ``data_copy`` a copy of ``data`` with runtimes divided by distance if ``per_measurement_round`` else an exact deep copy of ``data``.
    * ``containers`` a dictionary where each key a noise level; value, the ``ErrorbarContainer`` for that noise level.
    """
    containers: dict[float, ErrorbarContainer | list[Line2D]] = {}
    ds: Index[int] = data.columns.get_level_values('d').unique()
    data_copy = data.copy()
    if per_measurement_round:
        for d in ds:
            data_copy[d] = data[d] / layers_per_sample(d)
        default_ylabel = 'mean runtime per measurement round'
    else:
        default_ylabel = 'mean runtime'
    ps: Sequence[float] = data.columns.get_level_values('p').unique() \
        if noise_levels is None else noise_levels # type: ignore
    if base_color is None:
        colors = itertools.repeat(None)
        if legend is None:
            legend = True
    else:
        if legend is None:
            legend = False
        p_count = len(ps)
        colors = [(base_color, k/p_count) for k in range(p_count, 0, -1)]
    for p, color in zip(ps, colors):
        df: DataFrame = data_copy.xs(p, level='p', axis=1) # type: ignore

        if yerr_shows == 'sem':
            yerr: Series[float] = df.sem()
        elif yerr_shows == 'std':
            yerr: Series[float] = df.std()
        else:
            raise ValueError(f'invalid yerr_shows: {yerr_shows}')

        if fill_between:
            container = plt.plot(
                ds,
                df.mean(),
                label=f'{p:.1e}',
                color=color,
            )
            lo = df.mean() - yerr
            hi = df.mean() + yerr
            plt.fill_between(
                x=ds,
                y1=lo,
                y2=hi,
                color=container[0].get_color(),
                alpha=fill_alpha,
                linewidth=0,
                **kwargs,
            )
        else:
            container = plt.errorbar(
                x=ds,
                y=df.mean(),
                yerr=yerr,
                capsize=capsize,
                label=f'{p:.1e}',
                color=color,
                **kwargs,
            )
        if quantile is not None:
            plt.plot(
                ds,
                df.quantile(quantile),
                color=container[0].get_color(),
                linestyle=quantile_linestyle,
            )
        containers[p] = container
    plt.xticks(ds);
    if legend: plt.legend(title=r'$p =\dots$', reverse=True)
    if grid: plt.grid(which='both')
    if xlabel is None: xlabel = 'code distance'
    if ylabel is None: ylabel = default_ylabel
    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    if title: plt.title(title)
    return data_copy, containers


def distribution(
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


def distributions(
        data: Sequence[DataFrame],
        p: float,
        bins: int | Iterable[int] = 80,
        figsize: None | tuple[float, float] = None,
        log_scale=True,
        grid=False,
        global_range=True,
        show_xticks=False,
        subplots_hspace=0.05,
        quantile: float = 1,
        supxlabel_y: float = 0.15,
        **kwargs_for_ylabel,
):
    """Histogram runtime distributions for each DataFrame in ``data``.
    
    
    :param data: sequence of DataFrames. In each DataFrame, each
        column a (distance, probability);
    row, a runtime sample.
    :param p: noise level associated to the runtimes histogrammed.
    :param bins: bin count in each histogram.
        If an int, use same bin count for all entries in ``data``.
    If any bin count is 0, set bin width to 1.
    :param global_range: whether to use same bins for all distances within a DataFrame.
    :param quantile: the quantile (in the interval [0, 1]) to draw as a horizontal red line.
        Default is 1 i.e. the maximum of the sample.
    :param supxlabel_y: y-coordinate for the figure x-label.
    :param kwargs_for_ylabel: passed to ``plt.ylabel`` for the leftmost subplot in each row.
    """
    if isinstance(bins, int):
        bins = itertools.repeat(bins)
    ds: Index[int] = next(iter(data)).xs(p, level='p', axis=1).columns # type: ignore
    if figsize is None:
        figsize=(len(ds), 3)
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
                plt.ylabel(f'({letter})', **kwargs_for_ylabel)
            else:
                ax.set_yticklabels([])
            if log_scale:
                plt.xscale('log')
            if i == len(data)-1:
                plt.xlabel(str(d))
            plt.axhline(y=df[d].mean(), color='magenta')
            plt.axhline(y=df[d].quantile(quantile), color='r')
    f.supxlabel('code distance $d$', size='medium', y=supxlabel_y)
    f.tight_layout()
    if not show_xticks:
        for ax in f.axes:
            ax.set_xticks([])
            ax.set_xticks([], minor=True)
        f.subplots_adjust(hspace=subplots_hspace, wspace=0)
    return f


def violin(
        data: DataFrame,
        p: float,
        title='',
        widths=1,
        showextrema=False,
        capsize: float = 3,
        errorbar_kwargs: None | dict = None,
        **kwargs_for_violinplot,
):
    """Violin plot of runtime distributions for a given ``p``.
    
    
    :param capsize: length of error bar caps in points.
    """
    df: DataFrame = data.xs(p, level='p', axis=1) # type: ignore
    if errorbar_kwargs is None: errorbar_kwargs = {}
    parts = plt.violinplot(
        df,
        positions=list(df.columns),
        widths=widths,
        showextrema=showextrema,
        **kwargs_for_violinplot,
    )
    for pc in parts['bodies']: # type: ignore
        pc.set_alpha(1)
    plt.errorbar(
        x=df.columns,
        y=df.mean(),
        yerr=df.std(),
        fmt='.',
        capsize=capsize,
        linestyle='none',
        **errorbar_kwargs,
    )
    plt.xticks(df.columns)
    plt.grid(which='both')
    plt.xlabel('code distance')
    plt.ylabel('timestep count')
    if title:
        plt.title(title)