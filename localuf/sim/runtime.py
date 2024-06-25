"""Module for emulating runtime for Macar, Actis, Snowflake.

Available functions:
* batch
* frugal
"""

import itertools
from typing import Iterable, Literal, Type

from pandas import DataFrame, MultiIndex

from localuf import Surface
from localuf.decoders import Macar, Actis
from localuf.decoders.snowflake import Snowflake
from localuf.type_aliases import NoiseModel
from localuf._base_classes import Code
from localuf._schemes import Frugal


def batch(
        ds: Iterable[int],
        ps: Iterable[float],
        n: int,
        noise: NoiseModel,
        decoder_class: Type[Macar | Actis] = Macar,
        validate_only=True,
        **kwargs,
):
    """Make runtime data of Local UF decoder.

    Input:
    * `ds` an iterable of surface code distances.
    * `ps` an iterable of physical error probabilities.
    * `n` batch count.
    * `noise` noise model.
    * `decoder_class` the class of the decoder.
    * `validate_only` whether to time only syndrome validation or the full decoding cycle.
    * `kwargs` passed to `Surface`.

    Output: `df` a DataFrame where each
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
            sf = Surface(d, noise, **kwargs)
            decoder = decoder_class(sf)
            dc: dict[float, list[int]] = {}
            for p in ps:
                dc[p] = []
                for _ in itertools.repeat(None, times=n):
                    error = sf.make_error(p)
                    syndrome = sf.get_syndrome(error)
                    tSV = decoder.validate(syndrome)
                    dc[p].append(tSV)
                    decoder.reset()
            df[d] = DataFrame(dc)
    else:
        stages = ('BP', 'SV')  # this order ensures `mi.is_lexsorted()`
        mi = MultiIndex.from_product([stages, ds, ps], names=['stage', 'd', 'p'])
        df = DataFrame(columns=mi)
        for d in ds:
            sf = Surface(d, noise, **kwargs)
            decoder = decoder_class(sf)
            dcSV, dcBP = {}, {}
            for p in ps:
                dcSV[p], dcBP[p] = [], []
                for _ in itertools.repeat(None, times=n):
                    error = sf.make_error(p)
                    syndrome = sf.get_syndrome(error)
                    tSV, tBP = decoder.decode(syndrome)
                    dcSV[p].append(tSV)
                    dcBP[p].append(tBP)
                    decoder.reset()
            df['SV', d] = DataFrame(dcSV)
            df['BP', d] = DataFrame(dcBP)
    return df


def frugal(
        ds: Iterable[int],
        ps: Iterable[float],
        n: int,
        code_class: Type[Code],
        noise: Literal['phenomenological', 'circuit-level'],
        time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
        **kwargs,
):
    """Make runtime data Snowflake.

    Input:
    * `ds` an iterable of surface code distances.
    * `ps` an iterable of physical error probabilities.
    * `n` sample count.
    * `code_class` the class of the code.
    * `noise` the noise model.
    * `time_only` whether runtime includes a timestep
    for each drop, each grow, and each merging step ('all');
    each merging step only ('merging');
    or each unrooting step only ('unrooting').
    * `kwargs` passed to Snowflake
    e.g. `merger` decides whether Snowflake's nodes
    flood before syncing (fast) or vice versa (slow) in a merging step.

    Output: `df` a DataFrame where each
    column a (probability, distance);
    row, a runtime sample.
    """
    mi = MultiIndex.from_product([ds, ps], names=['d', 'p'])
    df = DataFrame(columns=mi)
    for d in ds:
        code = code_class(d, noise, scheme='frugal')
        frugal: Frugal = code.SCHEME # type: ignore
        decoder = Snowflake(code, **kwargs)
        dc: dict[float, tuple[int, ...]] = {}
        for p in ps:
            frugal.run(decoder, p, n, time_only=time_only)
            dc[p] = tuple(frugal.step_counts)
        df[d] = DataFrame(dc)
    return df