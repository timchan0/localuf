"""Emulate logical success/failure to deduce accuracy thresholds.

Available functions:
* monte_carlo
* monte_carlo_pymatching
* monte_carlo_special
* subset_sample
"""

from collections.abc import Callable
import itertools
from typing import Sequence, Type

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex

from localuf.sim._height_calculator import get_heights
from localuf.type_aliases import NoiseModel, DecodingScheme, Parametrization, FloatSequence, IntSequence
from localuf._base_classes import Code, Decoder

def monte_carlo(
        sample_counts: dict[int, list[tuple[float, int]]],
        code_class: Type[Code],
        decoder_class: Type[Decoder],
        noise: NoiseModel,
        scheme: DecodingScheme = 'batch',
        get_commit_height: Callable[[int], int] | None = None,
        get_buffer_height: Callable[[int], int] | None = None,
        parametrization: Parametrization = 'balanced',
        demolition: bool = False,
        monolingual: bool = False,
        merge_redundant_edges: bool = True,
        **kwargs_for_decoder_class,
):
    """Make threshold data for any decoder.

    Input:
    * `sample_counts` a dictionary where each
    key a code distance;
    value, a list of (noise level, sample count) pairs.
    If `scheme` is 'global batch',
    sample count must be the same for all noise levels of a given distance.
    For a more detailed definition of 'sample', see `_base_classes.Scheme.run`.
    * `code_class` the class of the code.
    * `decoder_class` the class of the decoder.
    * inputs with same name as for `Code.__init__` serve the same purpose.
    In global batch scheme, the decoding graph is `d*n` layers tall i.e.
    as tall as the (entire) decoding graph would be in forward scheme with commit height `d`.
    * `kwargs_for_decoder_class` are for `decoder_class`.

    The following 2 inputs affect only forward and frugal decoding schemes:
    * `get_commit_height` a function with input `d` that outputs commit height
    e.g. `lambda d: 2*(d//2)`.
    If `None`, commit height is `d` for forward scheme and `1` for frugal scheme.
    * `get_buffer_height` a function with input `d` that outputs buffer height.
    If `None`, buffer height is `d` for forward scheme and `2*(d//2)` for frugal scheme.

    Output:
    `df` a DataFrame where each
    column a (distance, probability);
    rows m, n indicate number of logical errors and samples, respectively.
    """
    dc = {}
    for d, list_ in sample_counts.items():
        window_height, commit_height, buffer_height = get_heights(
            d,
            list_[0][1],
            scheme,
            get_commit_height,
            get_buffer_height,
        )
        code = code_class(
            d,
            noise,
            scheme=scheme,
            window_height=window_height,
            commit_height=commit_height,
            buffer_height=buffer_height,
            parametrization=parametrization,
            demolition=demolition,
            monolingual=monolingual,
            merge_redundant_edges=merge_redundant_edges,
        )
        decoder = decoder_class(code, **kwargs_for_decoder_class)
        for p, n in list_:
            dc[d, p] = code.SCHEME.run(decoder, p, n)
    df = DataFrame(dc, index=('m', 'n'))
    df.columns.set_names(['d', 'p'], inplace=True)
    return df


def monte_carlo_pymatching(
        ds: Sequence[int],
        ps: FloatSequence,
        ns: int | IntSequence,
        code_class: Type[Code],
        noise: NoiseModel,
        **kwargs_for_code_class,
):
    """Make threshold data for PyMatching decoder.

    Input:
    * `ds, ps, ns, code_class, noise` same as for `monte_carlo`.
    * `**kwargs_for_code_class` for `code_class`.

    Output same as for `monte_carlo`.
    """
    if isinstance(ns, int):
        ns = [ns] * len(ps)
    elif len(ns) != len(ps):
        raise ValueError('Sequence `ns` must have same length as `ps`')
    mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
    df = DataFrame(columns=mi)
    for d in ds:
        code = code_class(
            d=d,
            noise=noise,
            **kwargs_for_code_class,
        )
        dc: dict[float, tuple[int, int]] = {}
        for p, n in zip(ps, ns):
            matching = code.get_matching_graph(p)
            m = 0
            for _ in itertools.repeat(None, n):
                error, syndrome = matching.add_noise() # type: ignore
                correction = matching.decode(syndrome)
                m += not np.array_equal(error, correction) # type: ignore
            dc[p] = (m, n)
        df[d] = DataFrame(dc, index=('m', 'n'))
    return df

def monte_carlo_special(
        ds: Sequence[int],
        ps: FloatSequence,
        ns: int | IntSequence,
        code_class: Type[Code],
        decoder_class: Type[Decoder],
        parametrization: Parametrization = 'balanced',
        demolition: bool = False,
        monolingual: bool = False,
        merge_redundant_edges: bool = True,
        **kwargs_for_decoder_class,
):
    """Make threshold data where
    code noise model is circuit-level;
    decoder noise model, phenomenological.

    Input & output same as for `monte_carlo`.
    """
    if isinstance(ns, int):
        ns = [ns] * len(ps)
    elif len(ns) != len(ps):
        raise ValueError('Sequence `ns` must have same length as `ps`')
    mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
    df = DataFrame(columns=mi)
    for d in ds:
        code = code_class(
            d,
            noise='circuit-level',
            parametrization=parametrization,
            demolition=demolition,
            monolingual=monolingual,
            merge_redundant_edges=merge_redundant_edges,
        )
        decoder = decoder_class(
            code_class(d, noise='phenomenological'),
            **kwargs_for_decoder_class,
        )
        dc = {p: code.SCHEME.run(decoder, p, n) for p, n in zip(ps, ns)}
        df[d] = DataFrame(dc, index=('m', 'n'))
    return df

def subset_sample(
        ds: Sequence[int],
        p: float,
        n: int,
        code_class: Type[Code],
        decoder_class: Type[Decoder],
        noise: NoiseModel,
        parametrization: Parametrization = 'balanced',
        tol: float = 5e-1,
):
    """Make threshold data for any decoder using subset sampling.
    
    Output:
    `df` a DataFrame indexed by (distance, error weight), with columns:
    * 'subset prob' probability of error of that weight, given distance.
    * 'survival prob' complement of cumulative sum of 'subset prob'.
    * 'm' failure count.
    * 'n' shot count.
    """
    ls = []
    for d in ds:
        code = code_class(
            d,
            noise=noise,
            parametrization=parametrization,
        )
        decoder = decoder_class(code)
        ls.append(decoder.subset_sample(p, n, tol=tol))
    df = pd.concat(ls, keys=ds, names=['d'])
    return df