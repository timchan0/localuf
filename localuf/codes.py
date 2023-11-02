"""Module for CSS codes.

Available codes:
* Repetition
* Surface
"""

import abc
from functools import cache, cached_property
import itertools
from typing import Iterable

from localuf import constants
from localuf.error_models import CircuitLevel, CodeCapacity, Phenomenological
from localuf.type_aliases import Coord, Edge, EdgeType, Parametrization, ErrorModel, Node


class Code(abc.ABC):
    """The decoding graph G = (V, E) of a CSS code.

    Atttributes (all are constants):
    * `D` code distance.
    * `SCHEME` the decoding scheme.
    * `N_EDGES` number of edges in G.
    * `EDGES` a tuple of edges of G.
    Use tuple instead of generator so can repeatedly iterate through.
    * `NODES` a tuple of nodes of G.
    * `ERROR_MODEL` error model.
    * `TIME_AXIS` that which represents time.
    * `LONG_AXIS` that whose index runs from -1 to d-1.
    * `DIMENSION` of G.
    * `INCIDENT_EDGES` a dictionary where each
    key a node;
    value, a set of incident edges.
    (Tried construction via manually changing indices by 1 so need not iterate through `self.EDGES`,
    but this is ~twice as slow.)
    * `GRAPH` a NetworkX graph of G.

    G represents: if `ERROR_MODEL`...
    
    `'code capacity'`: the code, where each
    * bulk node represents a measure-Z qubit;
    * edge, a data qubit i.e. possible bitflip location.

    `'phenomenological'`:
    `window_height+1` measurement rounds of the code,
    where each
    * bulk node represents the difference between two consecutive measurements
    of a measure-Z qubit at a given point in time;
    * horizontal edge, a possible time at which a data qubit could bitflip;
    * vertical edge, a possible faulty measurement
    (i.e. measure-Z qubit recording the wrong parity with some probability `q`)
    location.
    First AND last round assumed to be perfect (hence no temporal boundary edges).

    `'circuit-level'`: same as `'phenomenological'`
    but each edge represents a possible pair of defects
    that could have resulted from one fault.
    """

    _TIME_AXIS = -1
    _LONG_AXIS: int

    def __init__(self, d: int):
        """Input:
        * `d` code distance.
        """
        self._EDGES: tuple[Edge, ...]
        self._N_EDGES: int
        self._DIMENSION: int
        self._SCHEME: _Scheme
        self._ERROR_MODEL: CodeCapacity | Phenomenological | CircuitLevel
        self._NODES: tuple[Node, ...]

        self._D = d
        self._INCIDENT_EDGES = {v: {
            e for e in self._EDGES if v in e
        } for v in self._NODES}

    @property
    def D(self): return self._D

    @property
    def SCHEME(self): return self._SCHEME

    @property
    def N_EDGES(self): return self._N_EDGES

    @property
    def EDGES(self): return self._EDGES

    @property
    def NODES(self): return self._NODES

    @property
    def ERROR_MODEL(self): return self._ERROR_MODEL

    @property
    def TIME_AXIS(self): return self._TIME_AXIS

    @property
    def LONG_AXIS(self): return self._LONG_AXIS

    @property
    def DIMENSION(self): return self._DIMENSION

    @property
    def INCIDENT_EDGES(self): return self._INCIDENT_EDGES

    def is_boundary(self, v: Node):
        """Determine whether `v` a boundary node."""
        return self.SCHEME.is_boundary(v)

    def neighbors(self, v: Node):
        """Return neighbors of `v`.
        
        Never actually used but if do,
        change to @cached_property."""
        ans: set[Node] = set()
        for e in self.INCIDENT_EDGES[v]:
            ans.update(e)
        ans.remove(v)
        return ans

    @staticmethod
    def traverse_edge(e: Edge, u: Node):
        """Return node at other end of edge `e`."""
        v = e[1] if u == e[0] else e[0]
        return v
    
    def raise_node(self, v: Node, delta_t: int = 1) -> Node:
        """Move `v` up by `delta_t`."""
        new_v = list(v)
        new_v[self.TIME_AXIS] += delta_t
        return tuple(new_v)

    def make_error(self, p: float):
        """Make error on G.
        
        Input: `p` probability for an edge to bitflip.
        Output: The set of bitflipped edges.
        """
        return self.ERROR_MODEL.make_error(p)

    def get_syndrome(self, error: set[Edge]):
        """Get syndrome from error configuration.
        
        Input:
        `error` a set of bitflipped edges.

        Output:
        `syndrome` a set of defects.

        Notes:
        Implementing `syndrome` as a set we add to and remove from is
        empirically faster than as a dictionary w/ a key for each measure-Z qubit
        and Booleans as values, which we flip back and forth (for d=29, p=0.11:
        3.03(4) ms < 3.26(5) ms).
        """
        # verbose as treats boundary nodes as if they were bulk nodes
        verbose_syndrome: set[Node] = set()
        for e in error:
            verbose_syndrome.symmetric_difference_update(e)
        syndrome = {v for v in verbose_syndrome if not self.is_boundary(v)}
        return syndrome

    @staticmethod
    def compose_errors(*args: set[Edge]):
        """Sequentially compose any number of errors.

        Input: `args` a tuple (error1, error2, ...) where each
        error a set of bitflipped edges.

        Output:
        A set of edges representing the sequential composition
        of all errors in args.
        """
        composition: set[Edge] = set()
        for error in args:
            composition ^= error
        return composition

    def get_logical_error(self, leftover: set[Edge]):
        """Whether leftover implements logical X.

        Input: `leftover` a set of bitflipped edges.
        Output: `1` if logical error; else `0`.
        """
        return self.SCHEME.get_logical_error(leftover)
    
    @abc.abstractmethod
    def index_to_id(self, index: Node) -> int:
        """Return unique ID of node."""

    @cached_property
    def GRAPH(self):
        """Return a NetworkX graph of G.
        
        Notes: caching so only need make it once provides noticeable speedup!
        """
        import networkx as nx
        g = nx.Graph(
            self.EDGES,
            D=self.D,
            SCHEME=str(self.SCHEME),
        )
        for v in g.nodes:
            g.nodes[v]['is_boundary'] = self.is_boundary(v)
        return g

    def draw(
        self,
        error=None,
        syndrome=None,
        x_offset=constants.DEFAULT_X_OFFSET,
        with_labels=None,
        node_size=None,
        width=None,
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        **kwargs
    ):
        """Draw G using matplotlib.

        Input:
        * `error` a set of edges.
        * `syndrome` a set of defects.
        * `x_offset` the ratio of out-of-screen to along-screen distance.
        * `with_labels` whether to draw labels on each node.
        * use `kwargs` to modify/add any keyword arguments to `networkx.draw()`.

        Draws: G, where
        * bitflipped edges thick red; else, thin black
        * boundary nodes blue; defects red; else, green.
        """
        import networkx as nx
        # get error and syndrome
        if error is None: error = set()
        if syndrome is None: syndrome = self.get_syndrome(error)

        # get graph and kwargs for nx.draw()
        g = self.GRAPH
        pos = self._get_pos(x_offset)
        # if d small enough, can draw labels on each node
        if with_labels is None:
            with_labels = (self.D <= 3) if self.DIMENSION == 3 else (self.D <= 7)
        if node_size is None:
            node_size = constants.DEFAULT if with_labels else constants.SMALL
        node_color = self._get_node_color(
            syndrome,
            boundary_color=boundary_color,
            defect_color=defect_color,
            nondefect_color=nondefect_color,
        )
        if width is None:
            width = [
                constants.WIDE if e in error else
                constants.MEDIUM if with_labels else
                constants.THIN if self.DIMENSION == 3 else
                0 for e in self.EDGES
            ]
        edge_color = [
            'r' if e in error else
            'k' for e in self.EDGES
        ]
        nx.draw(
            g,
            pos=pos,
            with_labels=with_labels,
            nodelist=self.NODES,
            edgelist=self.EDGES,
            node_size=node_size,
            node_color=node_color,
            width=width,
            edge_color=edge_color,
            **kwargs
        )

    @abc.abstractmethod
    def _get_pos(self, x_offset: float = constants.DEFAULT_X_OFFSET) -> dict[Node, Coord]:
        """Compute coordinates of each node G for `draw()`.

        Input: `x_offset` the ratio of out-of-screen to along-screen distance.
        
        Output: `pos` a dictionary where each key a node index; value, position coordinate.
        E.g. for surface code w/ perfect measurements,
        convert each matrix index to position coords via
        (i, j) -> (x, y) = (j, -i).
        """

    def _get_node_color(
            self,
            syndrome: set[Node],
            boundary_color=constants.BLUE,
            defect_color=constants.RED,
            nondefect_color=constants.GREEN,
            nodelist: Iterable[Node] | None = None,
    ):
        """Return a list of colors each node should be for `draw()`.
        
        Input:
        * `syndrome` a set of defects.

        Output:
        * list of colors for
        each node in `nodelist` if `nodelist` not `None`
        else each node in `self.NODES`.
        """
        if nodelist is None:
            nodelist = self.NODES
        node_color = [
            defect_color if v in syndrome else
            boundary_color if self.is_boundary(v) else
            nondefect_color for v in nodelist
        ]
        # change to return a tuple instead?
        return node_color
    
    def get_matching_graph(self, p: float):
        """Return PyMatching matching graph whose edge weights depend on `p`."""
        import pymatching
        import numpy as np
        edge_probabilities = self.ERROR_MODEL.get_edge_probabilities(p)
        matching = pymatching.Matching()
        for (u, v), p in edge_probabilities:
            matching.add_edge(
                node1=self.index_to_id(u),
                node2=self.index_to_id(v),
                fault_ids=0 if (u[self.LONG_AXIS]==-1) else None, # type: ignore
                weight=np.log((1-p)/p),
                error_probability=p,
            )
        matching.set_boundary_nodes({self.index_to_id(v)
            for v in self.NODES if self.is_boundary(v)})
        return matching


class Repetition(Code):
    """The decoding graph of a repetition code.
    
    Extends `Code` class.
    """

    _LONG_AXIS = 0

    def __init__(
            self,
            d: int,
            error_model: ErrorModel,
            gate_noise: Parametrization = 'balanced',
            demolition: bool = False,
            monolingual: bool = False,
            window_height: int | None = None,
            merge_redundant_edges: bool = True,
    ):
        """Input:
        * `d` code distance.
        * `error_model` error model.
        * `gate_noise` defines relative fault probabilities of
        of 1- and 2-qubit gates, and prep/measurement.
        Affects only circuit-level noise.
        * `demolition` whether measurements are demolition.
        Affects only circuit-level noise.
        * `monolingual` whether can prep/measure in only one basis of X and Z.
        Affects only circuit-level noise.
        * `window_height` total layer count in the time direction.
        * `merge_redundant_edges` whether to merge redundant boundary edges.
        """
        node_ranges = [range(-1, d)]
        if error_model == 'code capacity':
            wh = 1
            self._DIMENSION = 1
            self._EDGES = tuple(((j,), (j+1,)) for j in range(-1, d-1))
            self._N_EDGES = d
            self._ERROR_MODEL = CodeCapacity(self.EDGES)
        else:
            wh = d if window_height is None else window_height
            node_ranges.append(range(wh))
            self._DIMENSION = 2
            if error_model == 'phenomenological':
                j_edges = (((j, t), (j+1, t)) for j in range(-1, d-1) for t in range(wh))
                t_edges = (((j, t), (j, t+1)) for j in range(d-1) for t in range(wh-1))
                self._EDGES = (*j_edges, *t_edges)
                self._N_EDGES = d*wh + (d-1)*(wh-1)
                self._ERROR_MODEL = Phenomenological(self.EDGES)
            else: # error_model == 'circuit-level'
                raise NotImplementedError("Yet to implement circuit-level noise for repetition code.")
        self._SCHEME = _Batch(self, wh)
        self._NODES = tuple(itertools.product(*node_ranges))
        super().__init__(d)

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

    def _get_pos(
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
    """

    _LONG_AXIS = 1

    def __init__(
            self,
            d: int,
            error_model: ErrorModel,
            gate_noise: Parametrization = 'balanced',
            demolition: bool = False,
            monolingual: bool = False,
            window_height: int | None = None,
            merge_redundant_edges: bool = True,
    ):
        """Input:
        * `d` code distance.
        * `error_model` error model.
        * `gate_noise` defines relative fault probabilities of
        1- and 2-qubit gates, and prep/measurement.
        Affects only circuit-level noise.
        * `demolition` whether measurement destroys the ancilla the qubit state
        which hence needs to be initialized for next measurement cycle.
        Affects only circuit-level noise.
        * `monolingual` whether can prep/measure in only Z basis
        hence X-basis prep/measurement needs Hadamard gates.
        Affects only circuit-level noise.
        * `window_height` total layer count in the time direction.
        * `merge_redundant_edges` whether to merge redundant boundary edges.
        """
        n_DQ = d**2 + (d-1)**2  # number of data qubits
        node_ranges = [range(d), range(-1, d)]
        if error_model == 'code capacity':
            wh = 1
            self._DIMENSION = 2
            self._EDGES = self._make_code_capacity_edges(d)
            self._N_EDGES = n_DQ
            self._ERROR_MODEL = CodeCapacity(self.EDGES)
        else:
            wh = d if window_height is None else window_height
            node_ranges.append(range(wh))
            self._DIMENSION = 3
            if error_model == 'phenomenological':
                self._EDGES = self._make_phenomenological_edges(d, wh)
                self._N_EDGES = wh * n_DQ + d * (d-1) * (wh-1)
                self._ERROR_MODEL = Phenomenological(self.EDGES)
            else:  # error_model == 'circuit-level'
                self._EDGES, edge_dict, merges = self._make_circuit_level_inputs(
                    d=d,
                    wh=wh,
                    merge_redundant_edges=merge_redundant_edges,
                )
                self._N_EDGES = wh * n_DQ + (wh-1) * (2*d-1)*(2*d-3) \
                    if merge_redundant_edges else (2*wh-1) * n_DQ + 2 * d * (d-1) * (wh-1)
                self._ERROR_MODEL = CircuitLevel(
                    edge_dict=edge_dict,
                    parametrization=gate_noise,
                    demolition=demolition,
                    monolingual=monolingual,
                    merges=merges,
                )
        self._SCHEME = _Batch(self, wh)
        self._NODES = tuple(itertools.product(*node_ranges))
        super().__init__(d)

    def _make_code_capacity_edges(self, d: int) -> tuple[Edge, ...]:
        """Return edges of G for code capacity noise model."""
        i_edges = (((i, j), (i+1, j)) for i in range(d-1) for j in range(    d-1))
        j_edges = (((i, j), (i, j+1)) for i in range(d  ) for j in range(-1, d-1))
        return (*i_edges, *j_edges)
    
    def _make_phenomenological_edges(self, d: int, wh: int) -> tuple[Edge, ...]:
        """Return edges of G for phenomenological noise model.
        
        Inputs:
        * `d` code distance.
        * `wh` window height.
        """
        i_edges = (((i, j, t), (i+1, j, t)) for i in range(d-1) for j in range(    d-1) for t in range(wh  ))
        j_edges = (((i, j, t), (i, j+1, t)) for i in range(d  ) for j in range(-1, d-1) for t in range(wh  ))
        t_edges = (((i, j, t), (i, j, t+1)) for i in range(d  ) for j in range(    d-1) for t in range(wh-1))
        return (*i_edges, *j_edges, *t_edges)

    def _make_circuit_level_inputs(
            self,
            d: int,
            wh: int,
            merge_redundant_edges: bool,
    ) -> tuple[
        tuple[Edge, ...],
        dict[EdgeType, tuple[Edge, ...]],
        dict[Edge, Edge] | None,
    ]:
        """Return inputs for `_CircuitLevel`.
        
        Additional inputs over `_make_phenomenological_edges`:
        * `merge_redundant_edges` whether to merge redundant boundary edges.

        Outputs:
        * `edges` a tuple of edges of G which excludes the redundant edges.
        * `edge_dict` maps from edge type
        (i.e. orientation and location)
        to tuple of all edges of that type.
        Includes redundant edge types.
        * `merges` maps each redundant edge to its substitute.
        """
        s = tuple(((i, j, t), (i+1, j, t)) for i in range(d-1) for j in range(d-1) for t in range(wh))
        e_wm, e_bulk, e_em = (
            tuple(((i, j, t), (i, j+1, t)) for i in range(d) for j in js for t in range(wh))
            for js in ((-1,), range(d-2), (d-2,))
        )
        u3, u4 = (
            tuple(((i, j, t), (i, j, t+1)) for i in is_ for j in range(d-1) for t in range(wh-1))
            for is_ in ((0, d-1), range(1, d-1))
        )
        sd = tuple(((i, j, t), (i+1, j, t-1)) for i in range(d-1) for j in range(d-1) for t in range(1, wh))
        eu_wc, eu_edge_NS, eu_ec = (
            tuple((((i, j, t), (i, j+1, t+1)) for i in (0, d-1) for j in js for t in range(wh-1)))
            for js in ((-1,), range(d-2), (d-2,))
        )
        eu_edge_EW, eu_centre = (
            tuple(((i, j, t), (i, j+1, t+1)) for i in range(1, d-1) for j in js for t in range(wh-1))
            for js in ((-1, d-2), range(d-2))
        )
        eu_edge = (*eu_edge_NS, *eu_edge_EW)
        seu_boundary, seu_bulk = (
            tuple(((i, j, t), (i+1, j+1, t+1)) for i in range(d-1) for j in js for t in range(wh-1))
            for js in ((-1, d-2), range(d-2))
        )
        seu = (*seu_boundary, *seu_bulk)
        if merge_redundant_edges:
            edges = (
                *s,
                *e_wm, *e_bulk, *e_em,
                *u3, *u4,
                *sd,
                *eu_edge_NS, *eu_centre,
                *seu_bulk,
            )
            merges = {e: self._substitute(d, e)
                for e in (*eu_wc, *eu_ec, *eu_edge_EW, *seu_boundary)}
        else:
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
        return edges, edge_dict, merges
    
    def _substitute(self, d: int, e: Edge) -> Edge:
        """Return substitute edge for redundant edge `e`."""
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
        return f'Surface(d={self.D}, scheme={str(self.SCHEME)})'

    def index_to_label(self, index):
        """Return node label of measure-Z qubit at index (i, j)."""
        if self.DIMENSION == 2:
            i, j = index
            return (self.D+1)*i + j + 1
        else:
            raise NotImplementedError('Only implemented for code capacity.')

    def label_to_index(self, a):
        """Return measure-Z qubit index (i, j) of node label a."""
        if self.DIMENSION == 2:
            return (a // (self.D+1), a % (self.D+1) - 1)
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
            wh = self.SCHEME.WINDOW_HEIGHT
            i, j, t = index
            if j == -1:
                return 2*wh*i + t
            elif j == d-1:
                return (2*i + 1) * wh + t
            else:
                n_boundaries = 2 * d * wh
                return n_boundaries + wh*(d-1)*i + wh*j + t
    
    def _get_pos(self, x_offset: float = constants.DEFAULT_X_OFFSET) -> dict[Node, Coord]:
        if self.DIMENSION == 2:
            pos = {(i, j): (j, -i) for i, j in self.NODES}
        else:
            pos = {(i, j, t): (j+x_offset*i, t-x_offset*i)
                for i, j, t in self.NODES}
        return pos


class _Scheme(abc.ABC):
    """Abstract base class for decoding scheme of a CSS code.
    
    Attributes:
    * `CODE` the CSS code.
    * `WINDOW_HEIGHT` total height of sliding window.
    """

    def __init__(
            self,
            code: Code,
    ):
        """Input: `code` the CSS code."""
        self.WINDOW_HEIGHT: int
        self._DETERMINANT: _Determinant

        self._CODE = code

    @property
    def CODE(self): return self._CODE
    
    def get_logical_error(self, leftover: set[Edge]):
        """See `Code.get_logical_error()`."""
        ct: int = 0
        for u, _ in leftover:
            ct += (u[self.CODE.LONG_AXIS] == -1)
        return ct % 2
    
    def is_boundary(self, v: Node):
        """See `Code.is_boundary()`."""
        return self._DETERMINANT.is_boundary(v)


class _Batch(_Scheme):
    """Batch decoding scheme.
    
    Extends `_Scheme`.
    """

    def __init__(self, code: Code, window_height: int):
        """Input: `code` the CSS code."""
        self._CODE = code
        self._WINDOW_HEIGHT = window_height
        self._DETERMINANT = SpaceDeterminant(code)

    @property
    def WINDOW_HEIGHT(self): return self._WINDOW_HEIGHT

    @staticmethod
    def __str__() -> str:
        return 'batch'


class _Determinant:
    """Determines whether a node is a boundary.
    
    Instance attributes:
    * `CODE` the CSS code.
    """

    def __init__(self, code: Code) -> None:
        self._CODE = code

    @property
    def CODE(self): return self._CODE

    def is_boundary(self, v: Node):
        """See `Code.is_boundary()`."""
        # `node` either (i, j) or (j, t) or (i, j, t).
        return v[self.CODE.LONG_AXIS] in {-1, self.CODE.D-1}


class SpaceDeterminant(_Determinant):
    """Determines only space boundaries.
    
    Extends `_Determinant`.
    """