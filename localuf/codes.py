"""Module for CSS codes.

Available codes:
* Repetition
* Surface
"""

from collections.abc import Iterable
from functools import cache
from itertools import repeat, chain

from localuf import constants
from localuf.type_aliases import Coord, Edge, EdgeType, Node
from localuf._base_classes import Code
from localuf._schemes import Batch


class Repetition(Code):
    """The decoding graph of a repetition code.
    
    Extends ``Code`` class.
    """

    _LONG_AXIS = 0
    _CODE_DIMENSION = 1

    def _code_capacity_edges(
            self,
            merge_equivalent_boundary_nodes: bool,
    ) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        return d, tuple(((j,), (j+1,)) for j in range(-1, d-1))

    def _phenomenological_edges(
            self,
            h,
            future_boundary,
            t_start=0,
            merge_equivalent_boundary_nodes=False,
    ) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        layer_count = h if future_boundary else h-1
        n_edges = h*d + layer_count*(d-1)
        if merge_equivalent_boundary_nodes:
            j_edges = [((-1, 0), (0, t)) for t in range(t_start, t_start+h)] \
            + [((j, t), (j+1, t)) for j in range(d-2) for t in range(t_start, t_start+h)] \
            + [((d-2, t), (d-1, 0)) for t in range(t_start, t_start+h)]
        else:
            j_edges = (((j, t), (j+1, t)) for j in range(-1, d-1) for t in range(t_start, t_start+h))
        t_edges = (((j, t), (j, t+1)) for j in range(d-1) for t in range(t_start, t_start+layer_count))
        return n_edges, (*j_edges, *t_edges)

    def _future_boundary_nodes(self, h) -> list[Node]:
        return [(j, h) for j in range(self.D-1)]
    
    def _redundant_boundary_nodes(self, h):
        raise NotImplementedError("Yet to implement circuit-level noise for repetition code.")
    
    def _circuit_level_edges(self, **_):
        raise NotImplementedError("Yet to implement circuit-level noise for repetition code.")

    @cache
    def index_to_id(self, index: Node):
        """Return ID of node at index (j,) or (j, t)."""
        d = self.D
        h = self.SCHEME.WINDOW_HEIGHT
        if len(index) == 2:
            j, t = index
            if j == -1:  # on west boundary
                return 2*t
            elif j == d-1:  # on east boundary
                return 2*t + 1
            elif t == h:  # on future boundary
                spatial_boundary_count = 2*h
                return spatial_boundary_count + j
            else:  # not a boundary
                boundary_count = 2*h if isinstance(self.SCHEME, Batch) else 2*h + d-1
                return boundary_count + (d-1)*t + j
        else:  # 1
            j, = index
            return j+1

    def get_pos(
            self,
            x_offset: float = constants.DEFAULT_X_OFFSET,
            nodelist: None | Iterable[Node] = None,
    ) -> dict[Node, Coord]:
        if nodelist is None:
            nodelist = self.NODES
        if self.DIMENSION == 1:
            pos = {(j,): (j, 0) for j, in nodelist}
        else:
            pos = dict(zip(nodelist, nodelist))
        return pos


class Surface(Code):
    """The decoding graph of an unrotated surface code with boundaries.
    
    Extends ``Code`` class.
    
    Additional instance properties:
    * ``DATA_QUBIT_COUNT`` number of data qubits.
    """

    _LONG_AXIS = 1
    _CODE_DIMENSION = 2

    @property
    def DATA_QUBIT_COUNT(self) -> int:
        d = self.D
        return d**2 + (d-1)**2

    def _code_capacity_edges(
            self,
            merge_equivalent_boundary_nodes: bool,
    ) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        i_edges = (((i, j), (i+1, j)) for i in range(d-1) for j in range(    d-1))
        if merge_equivalent_boundary_nodes:
            j_edges = [((0, -1), (i, 0)) for i in range(d)] \
                + [((i, j), (i, j+1)) for i in range(d  ) for j in range(d-2)] \
                + [((i, d-2), (0, d-1)) for i in range(d)]
        else:
            j_edges = (((i, j), (i, j+1)) for i in range(d  ) for j in range(-1, d-1))
        return self.DATA_QUBIT_COUNT, (*i_edges, *j_edges)

    def _phenomenological_edges(
            self,
            h,
            future_boundary,
            t_start=0,
            merge_equivalent_boundary_nodes=False,
    ) -> tuple[int, tuple[Edge, ...]]:
        d = self.D
        layer_count = h if future_boundary else h-1
        n_edges = h * self.DATA_QUBIT_COUNT + layer_count * d*(d-1)
        i_edges = (((i, j, t), (i+1, j, t)) for i in range(d-1) for j in range(    d-1) for t in range(t_start, t_start+h))
        if merge_equivalent_boundary_nodes:
            j_edges = [((0, -1, 0), (i, 0, t)) for i in range(d) for t in range(t_start, t_start+h)] \
                + [((i, j, t), (i, j+1, t)) for i in range(d  ) for j in range(d-2) for t in range(t_start, t_start+h)] \
                + [((i, d-2, t), (0, d-1, 0)) for i in range(d) for t in range(t_start, t_start+h)]
        else:
            j_edges = (((i, j, t), (i, j+1, t)) for i in range(d  ) for j in range(-1, d-1) for t in range(t_start, t_start+h))
        t_edges = (((i, j, t), (i, j, t+1)) for i in range(d  ) for j in range(    d-1) for t in range(t_start, t_start+layer_count))
        return n_edges, (*i_edges, *j_edges, *t_edges)

    def _future_boundary_nodes(self, h) -> list[Node]:
        d = self.D
        return [(i, j, h) for i in range(d) for j in range(d-1)]
    
    def _redundant_boundary_nodes(self, h) -> list[Node]:
        d = self.D
        return [(i, j, t) for i in range(d) for j, t in ((-1, -1), (d-1, h))]

    def _circuit_level_edges(
            self,
            h: int,
            future_boundary: bool,
            _merge_redundant_edges: bool,
            t_start=0,
            merge_equivalent_boundary_nodes=False,
    ) -> tuple[
        int,
        tuple[Edge, ...],
        dict[EdgeType, tuple[Edge, ...]],
        dict[Edge, Edge] | None,
    ]:
        d = self.D
        j_ranges = ((-1,), range(d-2), (d-2,))
        if future_boundary:
            layer_count = h
            t_ranges = (
                range(t_start-1, t_start-1+layer_count),
                *repeat(range(t_start, t_start+layer_count), 2)
            )
        else:
            layer_count = h-1
            t_ranges = tuple(repeat(range(t_start, t_start+layer_count), 3))
        # south
        s = tuple(((i, j, t), (i+1, j, t)) for i in range(d-1) for j in range(d-1) for t in range(t_start, t_start+h))
        # east [westmost, bulk, eastmost]
        e_wm, e_bulk, e_em = (tuple(
            ((i, j, t), (i, j+1, t)) for i in range(d) for j in js for t in range(t_start, t_start+h)
        ) for js in j_ranges)
        if merge_equivalent_boundary_nodes:
            e_wm = tuple(((0, -1, 0), (i, 0, t)) for i in range(d) for t in range(t_start, t_start+h))
            e_em = tuple(((i, d-2, t), (0, d-1, 0)) for i in range(d) for t in range(t_start, t_start+h))
        # north- or southmost up, up
        u3, u4 = (tuple(
            ((i, j, t), (i, j, t+1)) for i in is_ for j in range(d-1) for t in range(t_start, t_start+layer_count)
        ) for is_ in ((0, d-1), range(1, d-1)))
        # south down
        sd = tuple(((i, j, t), (i+1, j, t-1)) for i in range(d-1) for j in range(d-1) for t in range(t_start+1, t_start+1+layer_count))
        # east up [west corners, on north or south wall, east corners]
        eu_wc, eu_edge_NS, eu_ec = (tuple(
            ((i, j, t), (i, j+1, t+1)) for i in (0, d-1) for j in js for t in ts
        ) for js, ts in zip(j_ranges, t_ranges))
        # east up [on west wall, in centre, on east wall]
        eu_edge_W, eu_centre, eu_edge_E = (tuple(
                ((i, j, t), (i, j+1, t+1)) for i in range(1, d-1) for j in js for t in ts
        ) for js, ts in zip(j_ranges, t_ranges))
        # east up on east or west wall
        eu_edge_EW = (*eu_edge_W, *eu_edge_E)
        # east up on a wall
        eu_edge = (*eu_edge_NS, *eu_edge_EW)
        # south east up [westmost, bulk, eastmost]
        seu_W, seu_bulk, seu_E = (tuple(
                ((i, j, t), (i+1, j+1, t+1)) for i in range(d-1) for j in js for t in ts
        ) for js, ts in zip(j_ranges, t_ranges))
        # south east up west- or eastmost
        seu_boundary = (*seu_W, *seu_E)
        # south east up
        seu = (*seu_boundary, *seu_bulk)
        if _merge_redundant_edges or merge_equivalent_boundary_nodes:
            n_edges = h * self.DATA_QUBIT_COUNT + layer_count * (2*d-1)*(2*d-3)
            edges = (
                *s,
                *e_wm, *e_bulk, *e_em,
                *u3, *u4,
                *sd,
                *eu_edge_NS, *eu_centre,
                *seu_bulk,
            )
            merges = {e: self._substitute(e, merge_equivalent_boundary_nodes)
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

    def _substitute(self, e: Edge, merge_equivalent_boundary_nodes: bool = False) -> Edge:
        """Return substitute edge for redundant edge ``e``."""
        d = self.D
        a = self.LONG_AXIS
        u, v = e
        remaining_dimensions = len(u) - a - 1
        if u[a] == -1:
            w = tuple(chain(repeat(0, a), (-1,), repeat(0, remaining_dimensions))
            ) if merge_equivalent_boundary_nodes else ((*v[:a], -1, *v[a+1:]))
            return (w, v)
        elif v[a] == d-1:
            w = tuple(chain(repeat(0, a), (d-1,), repeat(0, remaining_dimensions))
            ) if merge_equivalent_boundary_nodes else (*u[:a], d-1, *u[a+1:])
            return (u, w)
        else:
            raise ValueError(f'Edge {e} must have a boundary node.')

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
                boundary_count = 2 * d
                return boundary_count + (d-1)*i + j
        else:  # 3
            h = self.SCHEME.WINDOW_HEIGHT
            i, j, t = index
            if j == -1:  # on west boundary
                return 2*h*i + t
            elif j == d-1:  # on east boundary
                return (2*i + 1) * h + t
            elif t == h:  # on future boundary
                spatial_boundary_count = 2*d*h
                return spatial_boundary_count + (d-1)*i + j
            else:  # not a boundary
                boundary_count = 2*d*h if isinstance(self.SCHEME, Batch) else 2*d*h + d*(d-1)
                return boundary_count + h*(d-1)*i + h*j + t

    def get_pos(
            self,
            x_offset: float = constants.DEFAULT_X_OFFSET,
            nodelist: None | Iterable[Node] = None,
    ) -> dict[Node, Coord]:
        if nodelist is None:
            nodelist = self.NODES
        if self.DIMENSION == 2:
            pos = {(i, j): (j, -i) for i, j in nodelist}
        else:
            pos = {(i, j, t): (j+x_offset*i, t-x_offset*i)
                for i, j, t in nodelist}
        return pos