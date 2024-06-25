import abc
from functools import cached_property
from typing import Iterable

import numpy as np
import pandas as pd

from localuf import constants
from localuf.noise.main import Noise
from localuf.type_aliases import Coord, Edge, EdgeType, Parametrization, NoiseModel, Node, DecodingScheme
from localuf._determinants import Determinant


class Code(abc.ABC):
    """The decoding graph G = (V, E) of a CSS code.

    Atttributes (all are constants):
    * `D` code distance.
    * `SCHEME` the decoding scheme.
    * `N_EDGES` number of edges in G.
    * `EDGES` a tuple of edges of G.
    Use tuple instead of generator so can repeatedly iterate through.
    * `NODES` a tuple of nodes of G.
    * `NOISE` noise model.
    * `TIME_AXIS` that which represents time.
    * `LONG_AXIS` that whose index runs from -1 to d-1 inclusive.
    * `DIMENSION` of G.
    * `INCIDENT_EDGES` a dictionary where each
    key a node;
    value, a set of incident edges.
    (Tried construction via manually changing indices by 1 so need not iterate through `self.EDGES`,
    but this is ~twice as slow.)
    * `GRAPH` a NetworkX graph of G.

    G represents: if `NOISE`...
    
    `'code capacity'`: the code, where each
    * detector represents a measure-Z qubit;
    * edge, a data qubit i.e. possible bitflip location.

    `'phenomenological'`:
    `window_height+1` measurement rounds of the code,
    where each
    * detector represents the difference between two consecutive measurements
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

    def __init__(
            self,
            d: int,
            noise: NoiseModel,
            scheme: DecodingScheme = 'batch',
            window_height: int | None = None,
            commit_height: int | None = None,
            buffer_height: int | None = None,
            parametrization: Parametrization = 'balanced',
            demolition: bool = False,
            monolingual: bool = False,
            merge_redundant_edges: bool = True,
    ):
        """Input:
        * `d` code distance.
        * `noise` noise model.
        * `scheme` the decoding scheme.
        * `window_height` total layer count in the time direction.
        Affects only batch and global batch decoding schemes.
        Default value is `d`.
        * `commit_height` (`buffer_height`) the layer count in the time direction
        that is committed (not committed).
        Affects only forward and frugal decoding schemes.
        Default value is `d` for forward scheme
        and `1` (`2*(d//2)`) for frugal scheme.

        The following 4 inputs affect only circuit-level noise:
        * `parametrization` defines relative fault probabilities of
        1- and 2-qubit gates, and prep/measurement.
        * `demolition` whether measurement destroys the ancilla the qubit state
        which hence needs to be initialized for next measurement cycle.
        * `monolingual` whether can prep/measure in only Z basis
        hence X-basis prep/measurement needs Hadamard gates.
        * `merge_redundant_edges` whether to merge redundant boundary edges.
        """
        self._D = d
        self._inner_init(
            noise=noise,
            scheme=scheme,
            window_height=window_height,
            commit_height=commit_height,
            buffer_height=buffer_height,
            parametrization=parametrization,
            demolition=demolition,
            monolingual=monolingual,
            merge_redundant_edges=merge_redundant_edges,
        )
        self._EDGES: tuple[Edge, ...]
        self._N_EDGES: int
        self._DIMENSION: int
        self._SCHEME: Scheme
        self._NOISE: Noise
        self._NODES: tuple[Node, ...]
        
        ie: dict[Node, set[Edge]] = {}
        for e in self.EDGES:
            for v in e:
                if v not in ie:
                    ie[v] = set()
                ie[v].add(e)
        self._INCIDENT_EDGES = ie

    @abc.abstractmethod
    def _inner_init(self, **kwargs):
        """Inner init for subclasses.
        
        For inputs, see `Code.__init__`.
        """

    @abc.abstractmethod
    def _code_capacity_edges(self) -> tuple[int, tuple[Edge, ...]]:
        """Edges of G for code capacity noise model.

        Output:
        * number of edges in G.
        * tuple of edges of G.
        """

    @abc.abstractmethod
    def _phenomenological_edges(
        self,
        h: int,
        temporal_boundary: bool,
        t_start=0,
    ) -> tuple[int, tuple[Edge, ...]]:
        """Edges of decoding graph G or viewing window W for phenomenological noise model.
        
        Input:
        * `h` window height.
        * `temporal_boundary` whether top of graph has temporal boundary.
        True for W, False for G.
        * `t_start` the time index of the bottom layer.

        Output:
        * number of edges in the graph.
        * a tuple of edges of the graph.
        """

    @abc.abstractmethod
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
        """Return inputs for `noise.CircuitLevel`.

        Additional inputs over `_phenomenological_edges`:
        * `merge_redundant_edges` whether to merge redundant boundary edges.

        Output:
        * `n_edges` number of edges in the graph.
        * `edges` a tuple of edges of the graph which excludes the redundant edges
        if `merge_redundant_edges` is `True`.
        * `edge_dict` maps from edge type
        (i.e. orientation and location)
        to tuple of all edges of that type.
        Always includes redundant edges.
        Inter-key order matters as used by `noise.forcers.ForceByEdge.force_error`.
        * `merges` maps each redundant edge to its substitute.
        """

    @abc.abstractmethod
    def _temporal_boundary_nodes(self, h: int) -> list[Node]:
        """Return list of boundary nodes at top of viewing window.
        
        Input: `h` window height.
        """

    @abc.abstractmethod
    def _redundant_boundary_nodes(self, h: int) -> list[Node]:
        """Return list of additional boundary nodes due to unmerged redundant edges.

        Used only by circuit-level noise.
        
        Input: `h` window height.
        """

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
    def NOISE(self): return self._NOISE

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
        change to @cached_property.
        """
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
        # tuple unpacking is marginally slower
        new_v = list(v)
        new_v[self.TIME_AXIS] += delta_t
        return tuple(new_v)
    
    def raise_edge(self, e: Edge, delta_t: int = 1) -> Edge:
        """Move `e` up by `delta_t`."""
        return tuple(self.raise_node(v, delta_t) for v in e) # type: ignore

    def make_error(self, p: float):
        """Sample edges from freshly discovered region.

        Input:
        `p` characteristic probability if circuit-level noise;
        else, bitflip probability.

        Output:
        The set of bitflipped edges in the freshly discovered region.
        Each edge bitflips with
        probability defined by its multiplicity if circuit-level noise; else,
        probability `p`.
        """
        return self.NOISE.make_error(p)

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
        # verbose as treats boundary nodes as if they were detectors
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
        Output: logical error count parity if scheme is batch
        else logical error count if scheme is scheme is global batch
        else logical error count in commit region if scheme is forward.
        """
        return self.SCHEME.get_logical_error(leftover)
    
    @abc.abstractmethod
    def index_to_id(self, index: Node) -> int:
        """Return unique ID of node.
        
        Not implemented for non-batch schemes.
        """

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
        error: set[Edge] | None = None,
        syndrome: set[Node] | None = None,
        x_offset=constants.DEFAULT_X_OFFSET,
        with_labels: bool | None = None,
        nodelist: Iterable[Node] | None = None,
        node_size: float | Iterable[float] | None = None,
        width: float | Iterable[float] | None = None,
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        error_color=constants.RED,
        **kwargs,
    ):
        """Draw G using matplotlib.

        Input:
        * `error` a set of edges. Default is empty set.
        * `syndrome` a set of defects. Default is that produced by `error`.
        * `x_offset` the ratio of out-of-screen to along-screen distance.
        * `with_labels` whether to draw labels on each node.
        * `nodelist` the nodes to draw. Default is all nodes i.e. `self.NODES`.
        * `node_size` size of nodes.
        * `width` line width of edges.
        * `{boundary, defect, nondefect, error}_color` string specifying
        color of {boundary nodes, defects, nondefects, bitflipped edges}.
        * use `kwargs` to modify/add any keyword arguments to `networkx.draw()`.

        Draws: G, where
        * bitflipped edges thick red; else, thin black
        * boundary nodes blue; defects red; else, green.

        Output: The NetworkX graph of G.
        """
        import networkx as nx
        # get error and syndrome
        if error is None: error = set()
        if syndrome is None: syndrome = self.get_syndrome(error)

        # get graph and kwargs for nx.draw()
        g = self.GRAPH
        pos = self.get_pos(x_offset)
        # if d small enough, can draw labels on each node
        if with_labels is None:
            with_labels = (self.D <= 3) if self.DIMENSION == 3 else (self.D <= 7)
        if nodelist is None:
            nodelist = self.NODES
        if node_size is None:
            node_size = constants.DEFAULT if with_labels else constants.SMALL
        node_color = self.get_node_color(
            syndrome,
            boundary_color=boundary_color,
            defect_color=defect_color,
            nondefect_color=nondefect_color,
            nodelist=nodelist,
        )
        if width is None:
            width = [
                constants.WIDE if e in error else
                constants.MEDIUM_THIN if with_labels else
                constants.THIN if self.DIMENSION == 3 else
                0 for e in self.EDGES
            ]
        edge_color = [
            error_color if e in error else
            'k' for e in self.EDGES
        ]
        nx.draw(
            g,
            pos=pos,
            with_labels=with_labels,
            nodelist=nodelist,
            edgelist=self.EDGES,
            node_size=node_size,
            node_color=node_color,
            width=width,
            edge_color=edge_color,
            **kwargs,
        )
        return g

    @abc.abstractmethod
    def get_pos(self, x_offset: float = constants.DEFAULT_X_OFFSET) -> dict[Node, Coord]:
        """Compute coordinates of each node G for `draw()`.

        Input: `x_offset` the ratio of out-of-screen to along-screen distance.
        
        Output: `pos` a dictionary where each key a node index; value, position coordinate.
        E.g. for surface code w/ perfect measurements,
        convert each matrix index to position coords via
        (i, j) -> (x, y) = (j, -i).
        """

    def get_node_color(
            self,
            syndrome: set[Node],
            boundary_color=constants.BLUE,
            defect_color=constants.RED,
            nondefect_color=constants.GREEN,
            nodelist: Iterable[Node] | None = None,
            show_boundary_defects=True,
    ):
        """Return a list of colors each node should be for `draw()`.
        
        Input:
        * `syndrome` a set of defects.
        * `{boundary, defect, nondefect}_color` string specifying
        color of {boundary nodes, defects, nondefects}.
        * `nodelist` the nodes to draw. Default is all nodes i.e. `self.NODES`.
        * `show_boundary_defects` whether to boundary nodes can be defects.

        Output: List of colors for each node in `nodelist`.
        """
        if nodelist is None:
            nodelist = self.NODES
        if show_boundary_defects:
            node_color = [
                defect_color if v in syndrome else
                boundary_color if self.is_boundary(v) else
                nondefect_color for v in nodelist
            ]
        else:
            node_color = [
                boundary_color if self.is_boundary(v) else
                defect_color if v in syndrome else
                nondefect_color for v in nodelist
            ]
        # TODO: change to return a tuple instead?
        return node_color
    
    def get_matching_graph(self, p: float):
        """Return PyMatching matching graph whose edge weights depend on `p`."""
        import pymatching
        import numpy as np
        edge_probabilities = self.NOISE.get_edge_probabilities(p)
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


class Decoder(abc.ABC):
    """Base class for decoders.

    Instance attributes (1 constant):
    * `CODE` the code to be decoded.
    * `correction` a set of edges comprising the correction.
    """

    @property
    def CODE(self): return self._CODE

    def __init__(self, code: Code):
        """Input: `code` the code to be decoded."""
        self._CODE = code
        self.correction: set[Edge]

    def reset(self):
        """Factory reset."""
        try: del self.correction
        except AttributeError: pass

    @abc.abstractmethod
    def decode(
            self,
            syndrome: set[Node],
            **kwargs,
    ) -> None | int | tuple[int, int]:
        """Decode syndrome.

        Input:
        * `syndrome` the set of defects.
        * `draw` whether to draw.
        * `log_history` whether keep track of state history.

        Correction stored in `self.correction`.
        """

    @abc.abstractmethod
    def draw_decode(self, **kwargs):
        """Draw all stages of decoding.
    
        Input: `kwargs` passed to `NetworkX.draw`
        e.g. `margins=(0.1, 0.1)`.
        """

    def subset_sample(self, p: float, n: int, tol: float = 5e-1):
        """Simulate decoding cycles for each error subset.

        Input:
        * `p` physical error probability.
        * `n` number of decoding cycles per subset.
        * `tol` how much cutoff error we can tolerate,
        as a fraction of the mean.

        Output:
        * A pandas DataFrame indexed by `weight`
        with columns `['subset prob', 'survival prob', 'm', 'n']`.
        """
        subset_probs = self.CODE.NOISE.subset_probabilities(p)
        mean, mn, weights = 0, [], []
        for weight in subset_probs.index:
            weights.append(weight)
            weightless = weight==0 if isinstance(weight, int) else all(w==0 for w in weight)
            if weightless:
                mn.append((np.nan, np.nan))
            else:
                m, shots = self.CODE.SCHEME.sim_cycles_given_weight(self, weight, n)
                mn.append((m, shots))
                mean += subset_probs.loc[weight, 'subset prob'] * m / shots # type: ignore
                if mean * tol > subset_probs.loc[weight, 'survival prob']: # type: ignore
                    break
        if type(subset_probs.index) is pd.MultiIndex:
            index = pd.MultiIndex.from_tuples(weights)
        else:
            index = pd.Index(weights)
        subset_failures = pd.DataFrame(mn, index=index, columns=['m', 'n'])
        return pd.concat([subset_probs, subset_failures], axis=1, join='inner')


class Scheme(abc.ABC):
    """Abstract base class for decoding scheme of a CSS code.
    
    Attributes:
    * `_CODE` the CSS code.
    * `_DETERMINANT` object to determine whether a node is a boundary.
    * `WINDOW_HEIGHT` total height of decoding graph G or sliding window W.
    """

    def __init__(self, code: Code):
        """Input: `code` the CSS code."""
        self._CODE = code
        self._DETERMINANT: Determinant

    @property
    @abc.abstractmethod
    def WINDOW_HEIGHT(self) -> int:
        """Total height of decoding graph G or sliding window W."""
    
    @abc.abstractmethod
    def get_logical_error(self, leftover: set[Edge]) -> int:
        """See `Code.get_logical_error`."""
    
    def is_boundary(self, v: Node):
        """See `Code.is_boundary`."""
        return self._DETERMINANT.is_boundary(v)

    @abc.abstractmethod
    def run(
        self,
        decoder: Decoder,
        p: float,
        n: int,
        **kwargs,
    ) -> tuple[int, int | float]:
        """Simulate `n` (equivalent) decoding cycles given `p`.
        
        Input:
        * `decoder` the decoder.
        * `p` physical error probability.
        * positive integer `n` is
        decoding cycle count if scheme is 'batch' or 'forward'
        else slenderness := (layer count / code distance),
        where layer count := measurement round count + 1.

        Output: tuple of (failure count, decoding cycle count
        if noise is code capacity else slenderness).
        """

    @abc.abstractmethod
    def sim_cycles_given_weight(
            self,
            decoder: Decoder,
            weight: int | tuple[int, ...],
            n: int,
    ) -> tuple[int, int]:
        """Simulate `n` decoding cycles given `weight`.

        Input:
        * `decoder` the decoder.
        * `weight` the weight of the error.
        * `n` decoding cycle count.

        Output: tuple of (failure count, `n`).
        """