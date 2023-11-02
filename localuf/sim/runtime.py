"""Emulate runtime for `LUF`."""

import itertools
from typing import Iterable

from pandas import DataFrame, MultiIndex
from localuf.codes import Surface
from localuf.decoders.luf import LUF
from localuf.type_aliases import ErrorModel


def make_runtime_data(
        ds: Iterable[int],
        ps: Iterable[float],
        n: int,
        error_model: ErrorModel,
        visible=True,
        validate_only=True,
):
    """Make runtime data of Local UF decoder.

    Inputs:
    * `ds` an iterable of surface code distances.
    * `ps` an iterable of physical error probabilities.
    * `n` sample count.
    * `error_model` error model.
    * `visible` whether controller connects directly to each node or only node 0.
    * `validate_only` whether to time only syndrome validation or the full decoding cycle.

    Outputs:
    * `df` a DataFrame where each
    column a (probability, distance);
    row, a runtime sample.
    If `validate_only` is `False`, then each column is a (stage, probability, distance)
    where stage is either `SV` or `BP`
    which stand for 'syndrome validation' and 'burning & peeling'.
    """
    if validate_only:
        mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
        df = DataFrame(columns=mi)
        for d in ds:
            sf = Surface(d=d, error_model=error_model)
            luf = LUF(sf, visible)
            dc = {}
            for p in ps:
                dc[p] = []
                for _ in itertools.repeat(None, times=n):
                    error = sf.make_error(p)
                    syndrome = sf.get_syndrome(error)
                    tSV = luf.validate(syndrome)
                    dc[p].append(tSV)
                    luf.reset()
            df[d] = DataFrame(dc)
    else:
        stages = ('SV', 'BP')
        mi = MultiIndex.from_product([stages, ds, ps], names=['stage', 'd', 'p'])
        df = DataFrame(columns=mi)
        for d in ds:
            sf = Surface(d=d, error_model=error_model)
            luf = LUF(sf, visible)
            dcSV, dcBP = {}, {}
            for p in ps:
                dcSV[p], dcBP[p] = [], []
                for _ in itertools.repeat(None, times=n):
                    error = sf.make_error(p)
                    syndrome = sf.get_syndrome(error)
                    tSV, tBP = luf.decode(syndrome)
                    dcSV[p].append(tSV)
                    dcBP[p].append(tBP)
                    luf.reset()
            df['SV', d] = DataFrame(dcSV)
            df['BP', d] = DataFrame(dcBP)
    return df