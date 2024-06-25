"""Emulate logical success/failure to deduce accuracy thresholds.

Available functions:
* monte_carlo
* monte_carlo_pymatching
* monte_carlo_special
* subset_sample
"""

import itertools
from typing import Iterable, Type

import numpy as np
import pandas as pd
from pandas import DataFrame, MultiIndex

from localuf.type_aliases import NoiseModel, DecodingScheme, Parametrization
from localuf._base_classes import Code, Decoder

def monte_carlo(
        ds: Iterable[int],
        ps: Iterable[float],
        ns: int | Iterable[int],
        code_class: Type[Code],
        decoder_class: Type[Decoder],
        noise: NoiseModel,
        scheme: DecodingScheme = 'batch',
        commit_multiplier: int | None = None,
        buffer_multiplier: int | None = None,
        parametrization: Parametrization = 'balanced',
        demolition: bool = False,
        monolingual: bool = False,
        merge_redundant_edges: bool = True,
        **kwargs,
):
    """Make threshold data for any decoder.

    Input:
    * `ds` an iterable of surface code distances.
    * `ps` an iterable of physical error probabilities.
    * `ns` sample count, or iterable of sample counts for each `p` in `ps`.
    For a more detailed definition of 'sample', see `_base_classes.Scheme.run`.
    * `code_class` the class of the code.
    * `decoder_class` the class of the decoder.
    * inputs with same name as for `Code.__init__` serve the same purpose.
    In global batch scheme, the decoding graph is `d*n` layers tall i.e.
    as tall as the (entire) decoding graph would be in forward scheme with commit height `d`.
    * `kwargs` are for `decoder_class`.

    The following 2 inputs affect only forward and frugal decoding schemes:
    * `commit_multiplier` commit height in units of `d//2`.
    If `None`, commit height is `d` for forward scheme and `1` for frugal scheme.
    * `buffer_multiplier` buffer height in units of `d//2`.
    If `None`, buffer height is `d` for forward scheme and `2*(d//2)` for frugal scheme.

    Output:
    `df` a DataFrame where each
    column a (distance, probability);
    rows m, n indicate number of logical errors and samples, respectively.
    """
    if isinstance(ns, int):
        ns = itertools.repeat(ns)
    elif scheme == 'global batch':
        raise ValueError('global batch scheme only supports fixed sample count')
        
    mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
    df = DataFrame(columns=mi)
    for d in ds:
        window_height, commit_height, buffer_height = _get_heights(
            d,
            ns,
            scheme,
            commit_multiplier,
            buffer_multiplier,
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
        decoder = decoder_class(code, **kwargs)
        dc = {p: code.SCHEME.run(decoder, p, n) for p, n in zip(ps, ns)}
        df[d] = DataFrame(dc, index=('m', 'n'))
    return df

def _get_heights(
        d: int,
        ns: Iterable[int],
        scheme: DecodingScheme,
        commit_multiplier: int | None,
        buffer_multiplier: int | None,
):
    """Get height inputs for `Code.__init__` based on decoding scheme and multipliers."""
    window_height = d*next(iter(ns)) if scheme == 'global batch' else None

    if commit_multiplier is None:
        if scheme == 'forward':
            commit_height = d
        elif scheme == 'frugal':
            commit_height = 1
        else:  # 'batch' in scheme
            commit_height = None
    else:
        commit_height = commit_multiplier*(d//2)

    if buffer_multiplier is None:
        if scheme == 'forward':
            buffer_height = d
        elif scheme == 'frugal':
            buffer_height = 2*(d//2)
        else:  # 'batch' in scheme
            buffer_height = None
    else:
        buffer_height = buffer_multiplier*(d//2)

    return window_height, commit_height, buffer_height

def monte_carlo_pymatching(
        ds: Iterable[int],
        ps: Iterable[float],
        ns: int | Iterable[int],
        code_class: Type[Code],
        noise: NoiseModel,
        **kwargs,
):
    """Make threshold data for PyMatching decoder.

    Input:
    * `ds, ps, ns, code_class, noise` same as for `monte_carlo`.
    * `**kwargs` for `code_class`.

    Output same as for `monte_carlo`.
    """
    if isinstance(ns, int):
        ns = itertools.repeat(ns)
    mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
    df = DataFrame(columns=mi)
    for d in ds:
        code = code_class(
            d=d,
            noise=noise,
            **kwargs,
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
        ds: Iterable[int],
        ps: Iterable[float],
        ns: int | Iterable[int],
        code_class: Type[Code],
        decoder_class: Type[Decoder],
        parametrization: Parametrization = 'balanced',
        demolition: bool = False,
        monolingual: bool = False,
        merge_redundant_edges: bool = True,
        **kwargs,
):
    """Make threshold data where
    code noise model is circuit-level;
    decoder noise model, phenomenological.

    Input & output same as for `monte_carlo`.
    """
    if isinstance(ns, int):
        ns = itertools.repeat(ns)
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
            **kwargs,
        )
        dc = {p: code.SCHEME.run(decoder, p, n) for p, n in zip(ps, ns)}
        df[d] = DataFrame(dc, index=('m', 'n'))
    return df

def subset_sample(
        ds: Iterable[int],
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