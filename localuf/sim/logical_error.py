"""Simulate logical success/failure to deduce thresholds."""

import itertools
from typing import Iterable, Type

import numpy as np
from pandas import DataFrame, MultiIndex

from localuf.codes import Surface, Repetition
from localuf.decoders.base_decoder import BaseDecoder
from localuf.type_aliases import ErrorModel, Parametrization

def make_threshold_data(
        ds: Iterable[int],
        ps: Iterable[float],
        ns: int | Iterable[int],
        code_class: Type[Surface | Repetition],
        decoder_class: Type[BaseDecoder],
        error_model: ErrorModel,
        gate_noise: Parametrization = 'balanced',
        demolition: bool = False,
        monolingual: bool = False,
        merge_redundant_edges: bool = True,
        decoder_error_model: ErrorModel | None = None,
        **kwargs,
):
    """Make threshold data for any decoder.

    Inputs:
    * `ds` an iterable of surface code distances.
    * `ps` an iterable of physical error probabilities.
    * `ns` sample count, or iterable of sample counts for each `p` in `ps`.
    * `code_class` the class of the code.
    * `decoder_class` the class of the decoder.
    * `error_model` the error model.
    * `gate_noise` the relative fault probabilities
    of 1- and 2-qubit gates, and prep/measurement.
    * `demolition` whether measurements are demolition.
    * `monolingual` whether can prep/measure in only one basis of X and Z.
    * `merge_redundant_edges` whether to merge redundant boundary edges.
    * `decoder_error_model` the error model of the decoder.
    Use only as `phenomenological` when `error_model = 'circuit-level`.
    * `**kwargs` for `decoder_class`.

    Outputs:
    * `dfmi` a DataFrame where each
    column a (distance, probability);
    rows m, n indicate number of logical errors and samples, respectively.
    """
    if isinstance(ns, int):
        ns = itertools.repeat(ns)
    mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
    dfmi = DataFrame(columns=mi)
    if decoder_error_model is None:
        decoder_error_model = error_model
    for d in ds:
        code = code_class(
            d,
            error_model=decoder_error_model,
            gate_noise=gate_noise,
            demolition=demolition,
            monolingual=monolingual,
            merge_redundant_edges=merge_redundant_edges,
        )
        decoder = decoder_class(code, **kwargs)
        dc: dict[float, tuple[int, int]] = {}
        if error_model != decoder_error_model:
            distinct_code = code_class(
                d,
                error_model=error_model,
                gate_noise=gate_noise,
                demolition=demolition,
                monolingual=monolingual,
            )
            for p, n in zip(ps, ns):
                dc[p] = cycles_with_distinct_code(distinct_code, decoder, p, n)
        else:
            for p, n in zip(ps, ns):
                dc[p] = decoder.sim_cycles_given_p(p, n)
        dfmi[d] = DataFrame(dc, index=('m', 'n'))
    return dfmi

def make_pymatching_threshold_data(
        ds: Iterable[int],
        ps: Iterable[float],
        ns: Iterable[int] | int,
        code_class: Type[Surface | Repetition],
        error_model: ErrorModel,
        **kwargs,
):
    """Make threshold data for PyMatching decoder.

    Inputs:
    * `ds, ps, ns, code_class, error_model` same as for `make_threshold_data`.
    * `**kwargs` for `code_class`.

    Outputs same as for `make_threshold_data`.
    """
    if isinstance(ns, int):
        ns = itertools.repeat(ns)
    mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
    df = DataFrame(columns=mi)
    for d in ds:
        code = code_class(
            d=d,
            error_model=error_model,
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

def cycles_with_distinct_code(
        distinct_code: Surface | Repetition,
        decoder: BaseDecoder,
        p: float,
        n: int,
):
    """Decoding cycle where decoder error model differs from code error model."""
    m = 0
    for _ in itertools.repeat(None, n):
        error = distinct_code.make_error(p)
        syndrome = distinct_code.get_syndrome(error)
        decoder.reset()
        decoder.decode(syndrome)
        leftover = error ^ decoder.correction
        logical_error = distinct_code.get_logical_error(leftover)
        m += logical_error
    return (m, n)