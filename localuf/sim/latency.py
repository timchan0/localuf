from collections.abc import Callable
from typing import Iterable, Literal, Type

from pandas import DataFrame

from localuf._schemes import Frugal
from localuf.type_aliases import StreamingNoiseModel
from localuf._base_classes import Code
from localuf.sim._height_calculator import get_heights
from localuf.decoders.snowflake import Snowflake


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
    """Make latency data Snowflake.
    
    
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
    row, a latency sample.
    """
    scheme = 'frugal'
    dc: dict[tuple[int, float], list[int]] = {}
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
            dc[d, p] = [frugal.sample_latency(
                decoder, p, time_only=time_only) for _ in range(n)]
    data = DataFrame(dc).sort_index(axis=1)
    data.columns.set_names(['d', 'p'], inplace=True)
    return data