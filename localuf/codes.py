"""Module for CSS codes.

Available codes:
* Repetition
* Surface
"""

from functools import cache
import itertools

from localuf import constants
from localuf.type_aliases import Coord, Edge, EdgeType, Node
from localuf._inner_init_helper import InnerInitHelper
from localuf._base_classes import Code


class Repetition(Code):
    """The decoding graph of a repetition code.

    Extends `Code` class.
    """

    _LONG_AXIS = 0

    def _inner_init(self, **kwargs):
        self._DIMENSION = 1
        node_ranges = [range(-1, self.D)]
        InnerInitHelper.help_(
                code=self,
                node_ranges=node_ranges,
                **kwargs,
            )

    def _code_capacity_edges(self) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        return d, tuple(((j,), (j+1,)) for j in range(-1, d-1))

    def _phenomenological_edges(self, h, temporal_boundary, t_start=0) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        layer_count = h if temporal_boundary else h-1
        n_edges = h*d + layer_count*(d-1)
        j_edges = (((j, t), (j+1, t)) for j in range(-1, d-1) for t in range(t_start, t_start+h))
        t_edges = (((j, t), (j, t+1)) for j in range(d-1) for t in range(t_start, t_start+layer_count))
        return n_edges, (*j_edges, *t_edges)

    def _temporal_boundary_nodes(self, h) -> list[Node]:
        return [(j, h) for j in range(self.D-1)]
    
    def _redundant_boundary_nodes(self, h):
        raise NotImplementedError("Yet to implement circuit-level noise for repetition code.")
    
    def _circuit_level_edges(self, **_):
        raise NotImplementedError("Yet to implement circuit-level noise for repetition code.")

    def __repr__(self) -> str:
        return f'Repetition(d={self.D}, noise={str(self.NOISE)})'

    @cache
    def index_to_id(self, index: Node):
        """Return ID of node at index (j,) or (j, t)."""
        d = self.D
        if len(index) == 2:
            j, t = index
            if j == -1:  # on west boundary
                return 2*t
            elif j == d-1:  # on east boundary
                return 2*t + 1
            else:  # not a boundary
                n_boundaries = 2 * self.SCHEME.WINDOW_HEIGHT
                return n_boundaries + (d-1)*t + j
        else:  # 1
            j, = index
            return j+1

    def get_pos(
            self,
            x_offset: float = constants.DEFAULT_X_OFFSET,
    ) -> dict[Node, Coord]:
        if self.DIMENSION == 1:
            pos = {(j,): (j, 0) for j, in self.NODES}
        else:
            pos = dict(zip(self.NODES, self.NODES))
        return pos


class Surface(Code):
    """The decoding graph of an unrotated surface code with boundaries.

    Extends `Code` class.

    Additional instance properties:
    * `DATA_QUBIT_COUNT` number of data qubits.
    """

    _LONG_AXIS = 1

    @property
    def DATA_QUBIT_COUNT(self) -> int:
        d = self.D
        return d**2 + (d-1)**2

    def _inner_init(self, **kwargs):
        d = self.D
        self._DIMENSION = 2
        node_ranges = [range(d), range(-1, d)]
        InnerInitHelper.help_(
            code=self,
            node_ranges=node_ranges,
            **kwargs,
        )

    def _code_capacity_edges(self) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        i_edges = (((i, j), (i+1, j)) for i in range(d-1) for j in range(    d-1))
        j_edges = (((i, j), (i, j+1)) for i in range(d  ) for j in range(-1, d-1))
        return self.DATA_QUBIT_COUNT, (*i_edges, *j_edges)

    def _phenomenological_edges(self, h, temporal_boundary, t_start=0) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        layer_count = h if temporal_boundary else h-1
        n_edges = h * self.DATA_QUBIT_COUNT + layer_count * d*(d-1)
        i_edges = (((i, j, t), (i+1, j, t)) for i in range(d-1) for j in range(    d-1) for t in range(t_start, t_start+h))
        j_edges = (((i, j, t), (i, j+1, t)) for i in range(d  ) for j in range(-1, d-1) for t in range(t_start, t_start+h))
        t_edges = (((i, j, t), (i, j, t+1)) for i in range(d  ) for j in range(    d-1) for t in range(t_start, t_start+layer_count))
        return n_edges, (*i_edges, *j_edges, *t_edges)

    def _temporal_boundary_nodes(self, h) -> list[Node]:
        d = self.D
        return [(i, j, h) for i in range(d) for j in range(d-1)]
    
    def _redundant_boundary_nodes(self, h) -> list[Node]:
        d = self.D
        return [(i, j, t) for i in range(d) for j, t in ((-1, -1), (d-1, h))]

    def _circuit_level_edges(
            self,
            h: int,
            temporal_boundary: bool,
            merge_redundant_edges: bool,
            t_start=0,
    ) -> tuple[
        int,
        tuple[Edge, ...],
        dict[EdgeType, tuple[Edge, ...]],
        dict[Edge, Edge] | None,
    ]:
        d = self.D
        j_ranges = ((-1,), range(d-2), (d-2,))
        if temporal_boundary:
            layer_count = h
            t_ranges = (
                range(t_start-1, t_start-1+layer_count),
                *itertools.repeat(range(t_start, t_start+layer_count), 2)
            )
        else:
            layer_count = h-1
            t_ranges = tuple(itertools.repeat(range(t_start, t_start+layer_count), 3))
        s = tuple(((i, j, t), (i+1, j, t)) for i in range(d-1) for j in range(d-1) for t in range(t_start, t_start+h))
        e_wm, e_bulk, e_em = (tuple(
            ((i, j, t), (i, j+1, t)) for i in range(d) for j in js for t in range(t_start, t_start+h)
        ) for js in j_ranges)
        u3, u4 = (tuple(
            ((i, j, t), (i, j, t+1)) for i in is_ for j in range(d-1) for t in range(t_start, t_start+layer_count)
        ) for is_ in ((0, d-1), range(1, d-1)))
        sd = tuple(((i, j, t), (i+1, j, t-1)) for i in range(d-1) for j in range(d-1) for t in range(t_start+1, t_start+1+layer_count))
        eu_wc, eu_edge_NS, eu_ec = (tuple(
            ((i, j, t), (i, j+1, t+1)) for i in (0, d-1) for j in js for t in ts
        ) for js, ts in zip(j_ranges, t_ranges))
        eu_edge_W, eu_centre, eu_edge_E = (tuple(
                ((i, j, t), (i, j+1, t+1)) for i in range(1, d-1) for j in js for t in ts
        ) for js, ts in zip(j_ranges, t_ranges))
        eu_edge_EW = (*eu_edge_W, *eu_edge_E)
        eu_edge = (*eu_edge_NS, *eu_edge_EW)
        seu_W, seu_bulk, seu_E = (tuple(
                ((i, j, t), (i+1, j+1, t+1)) for i in range(d-1) for j in js for t in ts
        ) for js, ts in zip(j_ranges, t_ranges))
        seu_boundary = (*seu_W, *seu_E)
        seu = (*seu_boundary, *seu_bulk)
        if merge_redundant_edges:
            n_edges = h * self.DATA_QUBIT_COUNT + layer_count * (2*d-1)*(2*d-3)
            edges = (
                *s,
                *e_wm, *e_bulk, *e_em,
                *u3, *u4,
                *sd,
                *eu_edge_NS, *eu_centre,
                *seu_bulk,
            )
            merges = {e: self._substitute(e)
                for e in (*eu_wc, *eu_ec, *eu_edge_EW, *seu_boundary)}
        else:
            n_edges = (h+layer_count) * self.DATA_QUBIT_COUNT + 2 * d * (d-1) * layer_count
            edges = (
                *s,
                *e_wm, *e_bulk, *e_em,
                *u3, *u4,
                *sd,
                *eu_wc, *eu_ec, *eu_edge, *eu_centre,
                *seu,
            )
            merges = None
        edge_dict: dict[EdgeType, tuple[Edge, ...]] = {
            'S': s,
            'E westmost': e_wm,
            'E bulk': e_bulk,
            'E eastmost': e_em,
            'U 3': u3,
            'U 4': u4,
            'SD': sd,
            'EU west corners': eu_wc,
            'EU east corners': eu_ec,
            'EU edge': eu_edge,
            'EU centre': eu_centre,
            'SEU': seu,
        }
        return n_edges, edges, edge_dict, merges

    def _substitute(self, e: Edge) -> Edge:
        """Return substitute edge for redundant edge `e`."""
        d = self.D
        u, v = e
        if u[self.LONG_AXIS] == -1:
            w = (*v[:self.LONG_AXIS], -1, *v[self.LONG_AXIS+1:])
            return (w, v)
        elif v[self.LONG_AXIS] == d-1:
            w = (*u[:self.LONG_AXIS], d-1, *u[self.LONG_AXIS+1:])
            return (u, w)
        else:
            raise ValueError(f'Edge {e} must have a boundary node.')

    def __repr__(self) -> str:
        return f'Surface(d={self.D}, noise={str(self.NOISE)})'

    def index_to_label(self, index: Node):
        """Return node label of measure-Z qubit at index (i, j)."""
        if self.DIMENSION == 2:
            i, j = index
            return (self.D+1)*i + j + 1
        else:
            raise NotImplementedError('Only implemented for code capacity.')

    def label_to_index(self, a: int) -> Node:
        """Return measure-Z qubit index (i, j) of node label a."""
        d = self.D
        if self.DIMENSION == 2:
            return (a // (d+1), a % (d+1) - 1)
        else:
            raise NotImplementedError('Only implemented for code capacity.')

    @cache
    def index_to_id(self, index: Node):
        """Return ID of node at index (i, j) or (i, j, t)."""
        d = self.D
        if len(index) == 2:
            i, j = index
            if j == -1:  # on west boundary
                return 2*i
            elif j == d-1:  # on east boundary
                return 2*i + 1
            else:  # not a boundary
                n_boundaries = 2 * d
                return n_boundaries + (d-1)*i + j
        else:  # 3
            h = self.SCHEME.WINDOW_HEIGHT
            i, j, t = index
            if j == -1:
                return 2*h*i + t
            elif j == d-1:
                return (2*i + 1) * h + t
            else:
                n_boundaries = 2 * d * h
                return n_boundaries + h*(d-1)*i + h*j + t

    def get_pos(self, x_offset: float = constants.DEFAULT_X_OFFSET) -> dict[Node, Coord]:
        if self.DIMENSION == 2:
            pos = {(i, j): (j, -i) for i, j in self.NODES}
        else:
            pos = {(i, j, t): (j+x_offset*i, t-x_offset*i)
                for i, j, t in self.NODES}
        return pos