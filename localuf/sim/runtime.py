"""Module for emulating runtime for Macar, Actis, Snowflake.

Available functions:
* batch
* frugal
"""

from collections import defaultdict
from collections.abc import Callable
import itertools
from typing import Iterable, Literal, Type

from pandas import DataFrame

from localuf import Surface
from localuf.decoders import Macar, Actis
from localuf.decoders.snowflake import Snowflake
from localuf.type_aliases import NoiseModel, StreamingNoiseModel
from localuf._base_classes import Code
from localuf._schemes import Forward, Frugal
from localuf.sim._height_calculator import get_heights


def batch(
        ds: Iterable[int],
        ps: Iterable[float],
        n: int,
        noise: NoiseModel,
        decoder_class: Type[Macar | Actis] = Macar,
        validate_only=True,
        **kwargs_for_Surface,
):
    """Make runtime data of Local UF decoder under batch decoding scheme.
    
    
    :param ds: an iterable of surface code distances.
    :param ps: an iterable of noise levels.
    :param n: sample count.
    :param noise: noise model.
    :param decoder_class: the class of the decoder.
    :param validate_only: whether to time only syndrome validation or the full decoding cycle.
    :param kwargs_for_Surface: passed to ``Surface``.
    
    Output: ``data`` a DataFrame where each
    column a (stage, distance, probability);
    row, a runtime sample.
    Stage is either ``SV`` or ``BP``
    which stand for 'syndrome validation' and 'burning & peeling'.
    """
    scheme = 'batch'
    dc: defaultdict[tuple[str, int, float], list[int]] = defaultdict(list)
    for d in ds:
        code = Surface(
            d,
            noise,
            scheme=scheme,
            window_height=None,
            **kwargs_for_Surface,
        )
        decoder = decoder_class(code)
        for p in ps:
            for _ in itertools.repeat(None, times=n):
                error = code.make_error(p)
                syndrome = code.get_syndrome(error)
                if validate_only:
                    tSV = decoder.validate(syndrome)
                else:
                    tSV, tBP = decoder.decode(syndrome)
                    dc['BP', d, p].append(tBP)
                dc['SV', d, p].append(tSV)
                decoder.reset()
    data = DataFrame(dc).sort_index(axis=1)
    data.columns.set_names(['stage', 'd', 'p'], inplace=True)
    return data


def forward(
        ds: Iterable[int],
        ps: Iterable[float],
        n: int,
        noise: StreamingNoiseModel,
        get_commit_height: Callable[[int], int] | None = None,
        get_buffer_height: Callable[[int], int] | None = None,
        **kwargs_for_Surface,
):
    """Make runtime data of Macar under forward decoding scheme.
    
    Use this function to analyse throughput.
    Latency can be analysed using ``sim.runtime.batch``,
    since the final decoding window of a memory experiment
    resembles that of a batch decoding window.
    
    
        :param ds: same as for ``batch``.
        :param ps: same as for ``batch``.
        :param n: same as for ``batch``.
        :param noise: same as for ``batch``.
        :param kwargs_for_Surface: same as for ``batch``.
    :param get_commit_height: a function with input ``d`` that outputs commit height
        e.g. ``lambda d: 2*(d//2)``.
    If ``None``, commit height is ``d``.
    :param get_buffer_height: a function with input ``d`` that outputs buffer height.
        If ``None``, buffer height is ``d``.
    
    Output same as for ``batch`` where ``validate_only=True``.
    """
    scheme = 'forward'
    dc: dict[tuple[str, int, float], list[int]] = {}
    for d in ds:
        _, commit_height, buffer_height = get_heights(
            d,
            n,
            scheme,
            get_commit_height=get_commit_height,
            get_buffer_height=get_buffer_height,
        )
        code = Surface(
            d,
            noise,
            scheme=scheme,
            window_height=None,
            commit_height=commit_height,
            buffer_height=buffer_height,
            **kwargs_for_Surface,
        )
        decoder = Macar(code)
        for p in ps:
            forward: Forward = code.SCHEME # type: ignore
            forward.run(decoder, p, n)
            step_counts: list[tuple[int, int]] = forward.step_counts # type: ignore
            dc['SV', d, p] = [sv for sv, _ in step_counts]
            dc['BP', d, p] = [bp for _, bp in step_counts]
    data = DataFrame(dc).sort_index(axis=1)
    data.columns.set_names(['stage', 'd', 'p'], inplace=True)
    return data


def frugal(
        ds: Iterable[int],
        ps: Iterable[float],
        n: int,
        code_class: Type[Code],
        noise: StreamingNoiseModel,
        time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
        get_commit_height: Callable[[int], int] | None = None,
        get_buffer_height: Callable[[int], int] | None = None,
        **kwargs_for_Snowflake,
):
    """Make runtime data Snowflake.
    
    
    :param ds: an iterable of surface code distances.
    :param ps: an iterable of noise levels.
    :param n: sample count.
    :param code_class: the class of the code.
    :param noise: the noise model.
    :param time_only: whether runtime includes a timestep
        for each drop, each grow, and each merging step ('all');
    each merging step only ('merging');
    or each unrooting step only ('unrooting').
    :param get_commit_height: a function with input ``d`` that outputs commit height.
        If ``None``, commit height is ``1``.
    :param get_buffer_height: a function with input ``d`` that outputs buffer height.
        If ``None``, buffer height is ``2*(d//2)``.
    :param kwargs_for_Snowflake: passed to Snowflake
        e.g. ``merger`` decides whether Snowflake's nodes
    flood before syncing (fast) or vice versa (slow) in a merging step.
    
    Output: ``data`` a DataFrame where each
    column a (distance, probability);
    row, a runtime sample.
    """
    scheme = 'frugal'
    dc: dict[tuple[int, float], tuple[int, ...]] = {}
    for d in ds:
        _, commit_height, buffer_height = get_heights(
            d,
            n,
            scheme,
            get_commit_height=get_commit_height,
            get_buffer_height=get_buffer_height,
        )
        code = code_class(
            d,
            noise,
            scheme=scheme,
            window_height=None,
            commit_height=commit_height,
            buffer_height=buffer_height,
        )
        frugal: Frugal = code.SCHEME # type: ignore
        decoder = Snowflake(code, **kwargs_for_Snowflake)
        for p in ps:
            frugal.run(decoder, p, n, time_only=time_only)
            dc[d, p] = tuple(frugal.step_counts)
    data = DataFrame(dc).sort_index(axis=1)
    data.columns.set_names(['d', 'p'], inplace=True)
    return data