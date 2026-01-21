import abc
from abc import abstractmethod
import itertools
import math
from typing import Literal, TYPE_CHECKING

from localuf import constants
from localuf.noise import CodeCapacity
from localuf.type_aliases import Edge, Node
from localuf._pairs import LogicalCounter, Pairs
from localuf._determinants import Determinant, SpaceDeterminant, SpaceTimeDeterminant


if TYPE_CHECKING:
    from localuf._base_classes import Code, Decoder
    from localuf.decoders.snowflake.main import Snowflake


class Scheme(abc.ABC):
    """Abstract base class for decoding scheme of a CSS code.
    
    Attributes:
    * ``_CODE`` the CSS code.
    * ``_DETERMINANT`` object to determine whether a node is a boundary.
    * ``WINDOW_HEIGHT`` total height of decoding graph G or sliding window W.
    """

    def __init__(self, code: 'Code'):
        """Input: ``code`` the CSS code."""
        self._CODE = code
        self._DETERMINANT: Determinant

    @property
    @abc.abstractmethod
    def WINDOW_HEIGHT(self) -> int:
        """Total height of decoding graph G or sliding window W."""

    @abc.abstractmethod
    def get_logical_error(self, leftover: set[Edge]) -> int:
        """See ``Code.get_logical_error``."""

    def is_boundary(self, v: Node):
        """See ``Code.is_boundary``."""
        return self._DETERMINANT.is_boundary(v)

    @abc.abstractmethod
    def run(
        self,
        decoder: 'Decoder',
        p: float,
        n: int,
        **kwargs,
    ) -> tuple[int, int | float]:
        """Simulate ``n`` (equivalent) decoding cycles given ``p``.
        
        
        :param decoder: the decoder.
        :param p: noise level.
        :param n: depends on the decoding scheme.
            If the scheme is 'batch',
            then ``n`` is decoding cycle count.
            If the scheme is 'global batch',
            then ``n`` is slenderness := (layer count / code distance),
            where layer count := measurement round count + 1.
            If the scheme is 'forward',
            then ``n`` is the number of decoding cycles in the steady state.
            If the scheme is 'frugal',
            then ``n`` is decoding cycle count in the steady state, divided by the code distance.
            For Snowflake, the there is 1 decoding cycle per stabiliser measurement round.
        
        :returns:
        * logical error count.
        * decoding cycle count
        if noise is code capacity; else, (total layer count / code distance).
        """

    @abc.abstractmethod
    def sim_cycles_given_weight(
            self,
            decoder: 'Decoder',
            weight: int | tuple[int, ...],
            n: int,
    ) -> tuple[int, int]:
        """Simulate ``n`` decoding cycles given ``weight``.
        
        
        :param decoder: the decoder.
        :param weight: the weight of the error.
        :param n: decoding cycle count.
        
        Output: tuple of (failure count, ``n``).
        """


class Batch(Scheme):
    """Batch decoding scheme.
    
    Extends ``Scheme``.
    
    Overriden methods:
    * ``run``
    * ``sim_cycles_given_weight``
    * ``get_logical_error``
    """

    @staticmethod
    def __str__() -> str:
        return 'batch'

    def __init__(self, code: 'Code', window_height: int):
        """Additional inputs: ``window_height`` the height of the batch."""
        super().__init__(code)
        self._WINDOW_HEIGHT = window_height
        self._DETERMINANT = SpaceDeterminant(code.D, code.LONG_AXIS)

    @property
    def WINDOW_HEIGHT(self): return self._WINDOW_HEIGHT

    def run(self, decoder: 'Decoder', p: float, n: int):
        # this assumes logical error count per batch << 1
        m = sum(self._sim_cycle_given_p(decoder, p) for _ in itertools.repeat(None, n))
        return (m, n if isinstance(self._CODE.NOISE, CodeCapacity)
            else self.WINDOW_HEIGHT * n / self._CODE.D)

    def get_logical_error(self, leftover):
        """Return logical error count parity in ``leftover``."""
        flip_count: int = 0
        for u, _ in leftover:
            flip_count += (u[self._CODE.LONG_AXIS] == -1)
        return flip_count % 2

    def _sim_cycle_given_p(self, decoder: 'Decoder', p: float) -> int:
        """Simulate a decoding cycle given ``p``.
        
        
        :param decoder: the decoder.
        :param p: noise level.
        
        Output: ``0`` if success else ``1``.
        """
        error = self._CODE.make_error(p)
        return self._sim_cycle_given_error(decoder, error)

    def _sim_cycle_given_error(self, decoder: 'Decoder', error: set[Edge]):
        """Simulate a decoding cycle given ``error``.
        
        
        :param error: the set of bitflipped edges.
        
        Output: ``0`` if success else ``1``.
        """
        syndrome = self._CODE.get_syndrome(error)
        decoder.reset()
        decoder.decode(syndrome)
        leftover = error ^ decoder.correction
        return self.get_logical_error(leftover)

    def sim_cycles_given_weight(self, decoder, weight, n):
        m = 0
        for _ in itertools.repeat(None, n):
            error = self._CODE.NOISE.force_error(weight)
            m += self._sim_cycle_given_error(decoder, error)
        return m, n


class Global(Batch):
    """Global batch decoding scheme.
    
    Extends ``Batch``.
    
    Additional attributes:
    * ``pairs`` a set of node pairs defining free anyon strings.
        Used to count logical error strings.
    
    Overriden methods:
    * ``get_logical_error``
    * ``run``
    """

    @staticmethod
    def __str__() -> str:
        return 'global batch'

    def __init__(self, code: 'Code', window_height: int):
        self.pairs = Pairs()
        super().__init__(code, window_height)

    def reset(self):
        """Factory reset."""
        self.pairs.reset()

    def run(self, decoder: 'Decoder', p: float, n: int):
        # `slenderness = n` assumes the following:
        assert self.WINDOW_HEIGHT == self._CODE.D * n
        self.reset()
        m = self._sim_cycle_given_p(decoder, p)
        return m, n

    def get_logical_error(self, leftover: set[Edge]):
        """Count logical errors in ``leftover``."""
        for e in leftover:
            self.pairs.load(e)
        error_count: int = 0
        for u, v in self.pairs.as_set:
            pair_separation = abs(u[self._CODE.LONG_AXIS] - v[self._CODE.LONG_AXIS])
            if pair_separation == self._CODE.D:
                error_count += 1
        return error_count


class _Streaming(Scheme):
    """Abstract base class for stream decoding schemes.
    
    Extends ``Scheme``.
    
    Additional attributes:
    * ``_COMMIT_HEIGHT`` the height of the commit region.
    * ``_BUFFER_HEIGHT`` the height of the buffer region.
    * ``_COMMIT_EDGES`` the edges of the commit region.
    * ``_BUFFER_EDGES`` the edges of the buffer region.
    * ``pairs`` a set of node pairs defining free anyon strings.
        Used to count logical error strings.
    * ``_LOGICAL_COUNTER`` for ``get_logical_error``.
    * ``step_counts`` a list in which each entry is the decoder timestep count of ``d`` decoding cycles.
        Populated when ``run`` is called.
    
    Overriden methods:
    * ``get_logical_error``
    """

    def __init__(
            self,
            code: 'Code',
            commit_height: int,
            buffer_height: int,
            commit_edges: tuple[Edge, ...],
    ):
        """
        :param commit_height: the height of the commit region.
        :param buffer_height: the height of the buffer region.
        :param commit_edges: the edges in the commit region.
        """
        super().__init__(code)
        self._COMMIT_HEIGHT = commit_height
        self._BUFFER_HEIGHT = buffer_height
        self._COMMIT_EDGES = commit_edges
        self._BUFFER_EDGES = tuple(set(code.EDGES) - set(commit_edges))
        self._DETERMINANT = SpaceTimeDeterminant(
            d=code.D,
            long_axis=code.LONG_AXIS,
            time_axis=code.TIME_AXIS,
            window_height=self.WINDOW_HEIGHT,
        )
        self.pairs = Pairs()
        self._LOGICAL_COUNTER = LogicalCounter(
            d=code.D,
            commit_height=commit_height,
            long_axis=code.LONG_AXIS,
            time_axis=code.TIME_AXIS,
        )
        self.step_counts = []

    @property
    def WINDOW_HEIGHT(self): return self._COMMIT_HEIGHT + self._BUFFER_HEIGHT

    @abstractmethod
    def reset(self):
        """Factory reset."""
        self.pairs.reset()
        self.step_counts.clear()

    def get_logical_error(self):
        """Count logical errors completed in current commit.
        
        
        :returns: The number of logical errors, i.e. paths between opposite boundaries, completed by the leftover in the current commit region.
        
        Side effect:
        Update ``self.pairs`` with error strings ending
        at the future boundary of the commit region.
        """
        error_count, self.pairs = self._LOGICAL_COUNTER.count(self.pairs)
        return error_count
    
    def sim_cycles_given_weight(self, decoder, weight, n):
        raise NotImplementedError("Not implemented for streaming schemes.")


class Forward(_Streaming):
    """Forward decoding scheme. Also known as overlapping recovery method.
    
    Extends ``_Streaming``.
    
    Additional attributes:
    * ``history`` a list of tuples (error, leftover, artificial defects) for each cycle.
    
    Overriden methods:
    * ``reset``
    * ``get_logical_error``
    * ``run``
    """

    step_counts: list[int | tuple[int, int] | None]

    @staticmethod
    def __str__() -> str:
        return 'forward'

    def reset(self):
        super().reset()
        try: del self.history
        except: pass

    def _make_error(
            self,
            buffer_leftover: set[Edge],
            p: float,
            exclude_future_boundary: bool = False,
    ):
        """Lower ``buffer_leftover`` by commit height
        and sample edges from freshly discovered region with probability ``p``.
        
        
        :param buffer_leftover: the current error in the buffer region.
        :param p: probability for an edge to bitflip.
        :param exclude_future_boundary: passed to ``self._CODE.make_error``.
        
        Output: The set of bitflipped edges.
        """
        # lower `buffer_leftover` by commit height
        seen: set[Edge] = set()
        for e in buffer_leftover:
            seen.add(self._CODE.raise_edge(e, -self._COMMIT_HEIGHT))
        # populate freshly discovered region with new errors
        unseen = self._CODE.make_error(p, exclude_future_boundary=exclude_future_boundary)
        return seen | unseen

    def _get_leftover(self, error: set[Edge], correction: set[Edge]):
        """Sequentially compose ``error`` and commit region of ``correction``.
        
        
        :param error: the set of bitflipped edges in the window.
        :param correction: the decoder output for the whole window.
        
        
        :returns:
        * ``commit_leftover`` the leftover, i.e. sequential composition of ``error`` and ``correction``, in the commit region.
        * ``buffer_leftover`` the part of ``error`` in the buffer region.
        """
        commit_leftover = error.intersection(self._COMMIT_EDGES) ^ correction.intersection(self._COMMIT_EDGES)
        buffer_leftover = error.intersection(self._BUFFER_EDGES)
        return commit_leftover, buffer_leftover

    def get_logical_error(self, commit_leftover: set[Edge]):
        """Count logical errors completed in current commit.
        
        Additional input over ``_Streaming.get_logical_error``:
        ``commit_leftover`` the leftover in the commit region.
        """
        for e in commit_leftover:
            self.pairs.load(e)
        return super().get_logical_error()

    def _get_syndrome(
            self,
            commit_leftover: set[Edge],
            error: set[Edge],
    ):
        """Get syndrome of ``error`` accounting for artificial defects due to previous commit."""
        artificial_defects = self._get_artificial_defects(commit_leftover)
        return self._CODE.get_syndrome(error) ^ artificial_defects

    def _get_artificial_defects(self, commit_leftover: set[Edge]):
        """Get artificial defects due to ``commit_leftover``.
        
        See Skoric et al. [arXiv:2209.08552v2, Section I B]
        for artificial defect definition.
        """
        commit_syndrome = self._CODE.get_syndrome(commit_leftover)
        return {self._CODE.raise_node(v, -self._COMMIT_HEIGHT) for v in commit_syndrome
                if v[self._CODE.TIME_AXIS] == self._COMMIT_HEIGHT}

    def _make_error_in_buffer_region(self, p: float):
        """Sample edges from buffer region.
        
        
        ``p`` characteristic probability if circuit-level noise;
        else, bitflip probability.
        
        
        :returns: The set of bitflipped edges in the buffer region. Each edge bitflips with probability defined by its multiplicity if circuit-level noise; else, probability ``p``.
        
        TODO: test this method.
        """
        if self._COMMIT_HEIGHT >= self._BUFFER_HEIGHT:
            error = self._CODE.make_error(p)
        else:
            rep_count = math.ceil(self._BUFFER_HEIGHT / self._COMMIT_HEIGHT)
            error: set[Edge] = set()
            for _ in itertools.repeat(None, rep_count):
                error = self._make_error(error, p)
        return error & set(self._BUFFER_EDGES)
    
    def run(
            self,
            decoder: 'Decoder',
            p: float,
            n: int,
            draw=False,
            log_history=False,
            **kwargs_for_draw_run,
    ):
        """Simulate ``n`` decoding cycles (in the steady state) given ``p``, for analysing accuracy and throughput.
        
        
        :param decoder: the decoder.
        :param p: noise level.
        :param n: is decoding cycle count in the steady state.
        
        Output: tuple of (failure count, slenderness).
        
        Side effect: Populate ``self.step_counts`` with ``n`` entries,
        each being the step count of one decoding cycle in the steady state.
        """
        log_history |= draw
        self.reset()
        if n < 1: raise ValueError("n must be positive integer.")
        m = 0
        commit_leftover: set[Edge] = set()
        buffer_leftover = self._make_error_in_buffer_region(p)
        # `cleanse_count` additional decoding cycles ensures window is free of defects
        cleanse_count = self.WINDOW_HEIGHT // self._COMMIT_HEIGHT
        # If F is the last freshly discovered region of edges that can flip (that excludes a future boundary),
        # then `cleanse_count` is the number of decoding cycles needed for F to be fully in the commit region.
        # If F includes the future boundary, then need 1 more than this.
        if log_history:
            self.history: list[tuple[set[Edge], set[Edge], set[Node]]] = []
        for prob, time, exclude_future_boundary in itertools.chain(
            itertools.repeat((p, True, False), n),
            itertools.repeat((p, False, True), 1),
            itertools.repeat((0, False, False), cleanse_count),
        ):
            error = self._make_error(buffer_leftover, prob, exclude_future_boundary=exclude_future_boundary)
            artificial_defects = self._get_artificial_defects(commit_leftover)
            syndrome = self._CODE.get_syndrome(error) ^ artificial_defects
            decoder.reset()
            step_count = decoder.decode(syndrome)
            if time:
                self.step_counts.append(step_count)
            commit_leftover, buffer_leftover = self._get_leftover(error, decoder.correction)
            if log_history:
                self.history.append((error, commit_leftover | buffer_leftover, artificial_defects))
            m += self.get_logical_error(commit_leftover)
        if draw:
            self._draw_run(**kwargs_for_draw_run)
        return m, (self._BUFFER_HEIGHT + (n+1)*self._COMMIT_HEIGHT) / self._CODE.D

    def _draw_run(
        self,
        fig_width: float | None = None,
        x_offset=constants.STREAM_X_OFFSET,
        subplot_hspace=-0.1,
    ):
        """Draw the history of ``self.run``."""
        import matplotlib.pyplot as plt
        column_count, row_count = len(self.history), 2
        if fig_width is None:
            fig_width = 1.5 if self._CODE.DIMENSION==2 else self._CODE.D*3/5
        plt.figure(figsize=(
            fig_width * column_count,
            fig_width * row_count * self.WINDOW_HEIGHT/self._CODE.D
        ))
        for k, (error, leftover, artificial_defects) in enumerate(self.history, start=1):
            for l, edges in enumerate((error, leftover)):
                plt.subplot(row_count, column_count, column_count*l + k)
                self._CODE.draw(
                    edges,
                    syndrome=self._CODE.get_syndrome(edges) ^ artificial_defects,
                    x_offset=x_offset,
                    with_labels=False,
                )
        plt.tight_layout()
        plt.subplots_adjust(hspace=subplot_hspace)


class Frugal(_Streaming):
    """Frugal decoding scheme.
    
    Extends ``_Streaming``.
    
    Additional attributes:
    * ``error`` a set of edges.
    * ``_future_boundary_syndrome`` the set of defects in the future boundary of the viewing window.
    
    Overriden methods:
    * ``reset``
    * ``run``
    """

    @staticmethod
    def __str__() -> str:
        return 'frugal'

    def __init__( self, code, commit_height, buffer_height, commit_edges):
        super().__init__(code, commit_height, buffer_height, commit_edges)
        self.error: set[Edge] = set()
        self._future_boundary_syndrome: set[Node] = set()
        self.step_counts: list[int]

    def reset(self):
        super().reset()
        self.error.clear()
        self._future_boundary_syndrome.clear()

    def advance(
            self,
            prob: float,
            decoder: 'Decoder',
            exclude_future_boundary: bool = False,
            **kwargs,
    ) -> int:
        """Advance 1 decoding cycle.
        
        
        :param prob: instantaneous noise level.
        :param decoder: the frugal-compatible decoder to use.
        :param exclude_future_boundary: passed to ``self._CODE.make_error``.
        :param log_history: forwarded to ``decoder.decode``.
        :param time_only: forwarded to ``decoder.decode``.
        :param defects_possible: forwarded to ``decoder.decode``.
        
        Output: number of decoder timesteps to complete decoding cycle.
        """
        self._raise_window()
        error = self._CODE.make_error(
            prob,
            exclude_future_boundary=exclude_future_boundary,
        )
        syndrome = self._load(error)
        return decoder.decode(syndrome, **kwargs) # type: ignore

    def _raise_window(self):
        """Raise window by ``self._COMMIT_HEIGHT`` layers.
        
        TODO: store ``lowness, next_edge`` as attributes of each edge.
        """
        next_error: set[Edge] = set()
        for e in self.error:
            lowness = sum(v[self._CODE.TIME_AXIS] == 0 for v in e)
            if lowness == 0:
                next_edge = self._CODE.raise_edge(e, delta_t=-self._COMMIT_HEIGHT)
                next_error.add(next_edge)
            else:
                self.pairs.load(e)
        self.error = next_error

    def _load(self, error: set[Edge]):
        """Load incremental ``error``.
        
        Input: ``error`` the incremental error, which should never intersect ``self.error``.
        
        Output: ``syndrome`` the incremental syndrome due to ``error``.
        
        Side effect: Update ``self.error`` and ``self.future_boundary_syndrome``.
        """
        self.error |= error
        syndrome = {self._CODE.raise_node(v, delta_t=-self._COMMIT_HEIGHT)
            for v in self._future_boundary_syndrome}
        self._future_boundary_syndrome.clear()
        for e in error:
            for v in e:
                if not self.is_boundary(v):
                    syndrome.symmetric_difference_update({v})
                elif v[self._CODE.TIME_AXIS] == self.WINDOW_HEIGHT:  # in future boundary
                    self._future_boundary_syndrome.symmetric_difference_update({v})
        return syndrome

    def run(
            self,
            decoder: 'Decoder',
            p: float,
            n: int,
            draw: Literal[False, 'fine', 'coarse'] = False,
            log_history: Literal[False, 'fine', 'coarse'] = False,
            time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
            **kwargs_for_draw_decode,
    ):
        """Simulate ``n*d`` decoding cycles (in the steady state) given ``p``.
        
        
        :param decoder: the decoder.
        :param p: noise level.
        :param n: is decoding cycle count in the steady state, divided by the code distance.
            For Snowflake, the there is 1 decoding cycle per stabiliser measurement round.
        :param draw: whether to draw.
        :param log_history: whether to populate ``history`` attribute.
        :param time_only: whether runtime includes a timestep
            for each drop, each grow, and each merging step ('all');
        each merging step only ('merging');
        or each unrooting step only ('unrooting').
        Note: if commit height is 1 and window height is d,
        then changing from 'merging' to 'all' simply increases each step count by 2d
        [for a total step count increase of 2d(n-1)]
        in the case of Snowflake with the 1:1 schedule.
        In the case of the 2:1 schedule, the increase of each step count is 3d.
        This can be done post-run via ``add_ignored_timesteps``.
        :param kwargs_for_draw_decode: passed to ``decoder.draw_decode``
            e.g. ``margins=(0.1, 0.1)``.
        
        Output: tuple of (failure count, slenderness := (total layer count) / (code distance)).
        
        Side effect: Populate ``self.step_counts`` with ``n`` entries,
        each being the step count of ``d`` decoding cycles in the steady state.
        """
        d = self._CODE.D
        self.reset()
        if n < 1: raise ValueError("n must be positive integer.")
        decoder.reset()
        if draw:
            log_history = draw
        if log_history:
            decoder.init_history() # type: ignore
        m = 0
        transient_count = math.ceil(self.WINDOW_HEIGHT / self._COMMIT_HEIGHT)
        # require `transient_count` decoding cycles to reach steady state
        cleanse_count = 2 * self.WINDOW_HEIGHT
        # require `cleanse_count` decoding cycles after receiving last sheet of syndrome
        # to guarantee last defect is annihilated
        # TODO: tighten this bound
        # Note: cleanse_count = math.ceil(self.WINDOW_HEIGHT / self._COMMIT_HEIGHT) does NOT work
        # as the committed correction will be incomplete
        for prob, advance_count, time, exclude_future_boundary in itertools.chain(
            ((p, transient_count, False, False),),
            itertools.repeat((p, d, True, False), n),
            ((p, d-1, False, False),),
            ((p, 1, False, True),),
            ((0, cleanse_count, False, False),),
        ):
            step_count = 0
            for _ in itertools.repeat(None, advance_count):
                step_count += self.advance(
                    prob,
                    decoder,
                    exclude_future_boundary=exclude_future_boundary,
                    log_history=log_history,
                    time_only=time_only,
                )
                m += self.get_logical_error()
            if time:
                self.step_counts.append(step_count)
        if draw:
            decoder.draw_decode(**kwargs_for_draw_decode)
        return m, transient_count / d + n + 1
    
    def sample_latency(
            self,
            decoder: 'Snowflake',
            p: float,
            draw: Literal[False, 'fine', 'coarse'] = False,
            log_history: Literal[False, 'fine', 'coarse'] = False,
            time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
            **kwargs_for_draw_decode,
    ):
        """Sample the latency of the decoder in the frugal decoding scheme.
        
        This runs a memory experiment of ``self.WINDOW_HEIGHT+1`` measurement rounds
        i.e. ``self.WINDOW_HEIGHT+2`` sheets of syndrome data are produced.
        Assumes commit height is 1.
        
        Inputs same as in ``run``.
        
        
        :returns: ``latency`` the number of timesteps from receiving the last measurement round to outputting the final correction.
        """
        self.reset()
        decoder.reset()
        if draw:
            log_history = draw
        if log_history:
            decoder.init_history()
        latency = 0
        defects_possible = True
        for prob, include_in_latency in itertools.chain(
            itertools.repeat((p, False), self.WINDOW_HEIGHT+1),
            ((p, True),),
            itertools.repeat((0, True), self.WINDOW_HEIGHT-1),
        ):
            if defects_possible and prob==0:
                defects_possible = bool(decoder.syndrome)
            step_count = self.advance(
                prob,
                decoder,
                exclude_future_boundary=include_in_latency,
                log_history=log_history,
                time_only=time_only,
                defects_possible=defects_possible,
            )
            if include_in_latency:
                latency += step_count
        if draw:
            decoder.draw_decode(**kwargs_for_draw_decode)
        return latency