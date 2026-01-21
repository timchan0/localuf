import abc
from functools import cached_property
from collections import defaultdict
from collections.abc import Collection, Iterable, Container, Sequence

import numpy as np
import pandas as pd

from localuf import constants
from localuf.noise.main import Noise
from localuf.type_aliases import Coord, Edge, EdgeType, Parametrization, NoiseModel, Node, DecodingScheme
from localuf._schemes import Scheme, Batch, Global, Forward, Frugal
from localuf.noise import CircuitLevel, CodeCapacity, Phenomenological


class Code(abc.ABC):
    """The decoding graph G = (V, E) of a CSS code.
    
    Attributes (all are constants):
    * ``D`` code distance.
    * ``MERGED_EQUIVALENT_BOUNDARY_NODES`` whether
        all nodes that represent the same boundary are merged.
    This results in a decoding graph with
    as many boundary nodes as there are boundaries.
    * ``SCHEME`` the decoding scheme.
    * ``N_EDGES`` number of edges in G.
    * ``EDGES`` a tuple of edges of G.
    Use tuple instead of generator so can repeatedly iterate through.
    * ``NODES`` a tuple of nodes of G.
    * ``NODE_COUNT`` number of nodes in G.
    * ``NOISE`` noise model.
    * ``TIME_AXIS`` that which represents time.
    * ``LONG_AXIS`` that whose index runs from -1 to d-1 inclusive.
    * ``DIMENSION`` of G.
    * ``INCIDENT_EDGES`` maps each node to a set of incident edges.
    * ``DETECTORS`` a tuple of all detectors of G.
    * ``DETECTOR_COUNT`` number of detectors in G.
    * ``BOUNDARY_NODES`` a tuple of all boundary nodes of G.
    * ``GRAPH`` a NetworkX graph of G.
    
    G represents: if ``NOISE`` is...
    
    ``'code capacity'``: the code, where each
    * detector represents a measure-Z qubit;
    * edge, a data qubit i.e. possible bitflip location.
    
    ``'phenomenological'``:
    ``window_height+1`` measurement rounds of the code,
    where each
    * detector represents the difference between two consecutive measurements
        of a measure-Z qubit at a given point in time;
    * horizontal edge, a possible time at which a data qubit could bitflip;
    * vertical edge, a possible faulty measurement
        (i.e. measure-Z qubit recording the wrong parity with some probability ``q``)
        location.
        First AND last round assumed to be perfect (hence no future boundary edges).
    
    ``'circuit-level'``: same as ``'phenomenological'``
    but each edge represents a possible pair of defects
    that could have resulted from one fault.
    """

    _TIME_AXIS = -1
    _LONG_AXIS: int
    _CODE_DIMENSION: int

    def __init__(
            self,
            d: int,
            noise: NoiseModel,
            scheme: DecodingScheme = 'batch',
            window_height: int | None = None,
            commit_height: int | None = None,
            buffer_height: int | None = None,
            merge_equivalent_boundary_nodes: bool = False,
            parametrization: Parametrization = 'balanced',
            demolition: bool = False,
            monolingual: bool = False,
            _merge_redundant_edges: bool = True,
    ):
        """
        :param d: code distance.
        :param noise: noise model.
        :param scheme: the decoding scheme.
        :param window_height: total layer count in the time direction.
            Affects only batch and global batch decoding schemes.
        Default value is ``d``.
        :param commit_height: the layer count in the time direction
            that is committed.
            Affects only forward and frugal decoding schemes.
            Default value is ``d`` for forward scheme
            and ``1`` (``2*(d//2)``) for frugal scheme.
        :param buffer_height: the layer count in the time direction
            that is not committed.
            Affects only forward and frugal decoding schemes.
            Default value is ``d`` for forward scheme
            and ``1`` (``2*(d//2)``) for frugal scheme.
        :param merge_equivalent_boundary_nodes: whether to merge
            all nodes that represent the same boundary.
            This results in a decoding graph with
            as many boundary nodes as there are boundaries.
        :param parametrization: defines relative fault probabilities of
            1- and 2-qubit gates, and prep/measurement.
            Affects only circuit-level noise.
        :param demolition: whether measurement destroys the ancilla qubit state
            which hence needs to be initialized for next measurement cycle.
            Affects only circuit-level noise.
        :param monolingual: whether can prep/measure in only Z basis
            hence X-basis prep/measurement needs Hadamard gates.
            Affects only circuit-level noise.
        :param _merge_redundant_edges: whether to merge redundant boundary edges.
            Affects only circuit-level noise.
        """
        self._D = d
        self._MERGED_EQUIVALENT_BOUNDARY_NODES = merge_equivalent_boundary_nodes

        # INNER INIT START
        if 'batch' in scheme:
            for height in (commit_height, buffer_height):
                if height is not None:
                    raise ValueError(f"Cannot specify `{height=}` for `{scheme}` scheme.")
            if noise == 'code capacity':
                if window_height is not None:
                    raise ValueError(f"Cannot specify `window_height` for code capacity noise model.")
                h = 1
                self._N_EDGES, self._EDGES = self._code_capacity_edges(merge_equivalent_boundary_nodes)
                self._NOISE = CodeCapacity(self.EDGES)
            else:
                h = d if window_height is None else window_height
                if noise == 'phenomenological':
                    self._N_EDGES, self._EDGES = self._phenomenological_edges(
                        h,
                        False,
                        merge_equivalent_boundary_nodes=merge_equivalent_boundary_nodes,
                    )
                    self._NOISE = Phenomenological(self.EDGES)
                else:  # noise == 'circuit-level'
                    self._N_EDGES, self._EDGES, edge_dict, merges = self._circuit_level_edges(
                        h=h,
                        future_boundary=False,
                        _merge_redundant_edges=_merge_redundant_edges,
                        merge_equivalent_boundary_nodes=merge_equivalent_boundary_nodes,
                    )
                    self._NOISE = CircuitLevel(
                        edge_dict=edge_dict,
                        parametrization=parametrization,
                        demolition=demolition,
                        monolingual=monolingual,
                        merges=merges,
                    )
            self._SCHEME = Batch(self, h) if scheme == 'batch' else Global(self, h)
        else:
            if merge_equivalent_boundary_nodes:
                raise NotImplementedError("Yet to implement merging equivalent boundary nodes for the stream decoding schemes.")
            if window_height is not None:
                raise ValueError(f"Cannot specify `window_height` for the {scheme} decoding scheme.")
            if scheme == 'forward':
                if commit_height is None: commit_height = d
                if buffer_height is None: buffer_height = d
                scheme_class = Forward
            else:  # scheme == 'frugal'
                if commit_height is None: commit_height = 1
                if buffer_height is None: buffer_height = 2*(d//2)
                scheme_class = Frugal
            h = commit_height + buffer_height

            if noise == 'code capacity':
                raise TypeError(f"Code capacity incompatible with the {scheme} decoding scheme.")
            elif noise == 'phenomenological':
                self._N_EDGES, self._EDGES = self._phenomenological_edges(h, True)
                _, commit_edges = self._phenomenological_edges(commit_height, True)
                _,  fresh_edges = self._phenomenological_edges(commit_height, True, t_start=buffer_height)
                self._NOISE = Phenomenological(fresh_edges)
            else:  # noise == 'circuit-level'
                self._N_EDGES, self._EDGES, *_ = self._circuit_level_edges(
                    h=h,
                    future_boundary=True,
                    _merge_redundant_edges=_merge_redundant_edges,
                )
                _, commit_edges, *_ = self._circuit_level_edges(
                    h=commit_height,
                    future_boundary=True,
                    _merge_redundant_edges=_merge_redundant_edges,
                )
                *_, fresh_edge_dict, fresh_merges = self._circuit_level_edges(
                    h=commit_height,
                    future_boundary=True,
                    _merge_redundant_edges=_merge_redundant_edges,
                    t_start=buffer_height,
                )
                self._NOISE = CircuitLevel(
                    edge_dict=fresh_edge_dict,
                    parametrization=parametrization,
                    demolition=demolition,
                    monolingual=monolingual,
                    merges=fresh_merges,
                )
            
            self._SCHEME = scheme_class(
                    self,
                    commit_height,
                    buffer_height,
                    commit_edges,
                )
        # INNER INIT END

        self._EDGES: tuple[Edge, ...]
        self._N_EDGES: int
        self._SCHEME: Scheme
        self._NOISE: Noise
        self._NODES: tuple[Node, ...] = tuple(set().union(*self.EDGES))
        
        ie: defaultdict[Node, set[Edge]] = defaultdict(set)
        for e in self.EDGES:
            for v in e:
                ie[v].add(e)
        self._INCIDENT_EDGES = dict(ie)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}({self.D}, '{str(self.NOISE)}', scheme='{str(self.SCHEME)}')"
    
    @abc.abstractmethod
    def _code_capacity_edges(
        self,
        merge_equivalent_boundary_nodes: bool,
    ) -> tuple[int, tuple[Edge, ...]]:
        """Edges of G for code capacity noise model.
        
        
        :returns:
        * number of edges in G.
        * tuple of edges of G.
        """

    @abc.abstractmethod
    def _phenomenological_edges(
        self,
        h: int,
        future_boundary: bool,
        t_start=0,
        merge_equivalent_boundary_nodes=False,
    ) -> tuple[int, tuple[Edge, ...]]:
        """Edges of decoding graph G or viewing window W for phenomenological noise model.
        
        
        :param h: window height.
        :param future_boundary: whether top of graph has future boundary.
            True for W, False for G.
        :param t_start: the time index of the bottom layer.
        :param merge_equivalent_boundary_nodes: whether to merge
            all nodes that represent the same boundary.
        
        
        :returns:
        * number of edges in the graph.
        * a tuple of edges of the graph.
        """

    @abc.abstractmethod
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
        """Return inputs for ``noise.CircuitLevel``.
        
        Additional inputs over ``_phenomenological_edges``:
        * ``_merge_redundant_edges`` whether to merge redundant boundary edges.
            Automatically ``True`` if ``merge_equivalent_boundary_nodes`` is ``True``.
        
        
        :returns:
        * ``n_edges`` number of edges in the graph.
        * ``edges`` a tuple of edges of the graph which excludes the redundant edges if ``_merge_redundant_edges`` is ``True``.
        * ``edge_dict`` maps from edge type (i.e. orientation and location) to tuple of all edges of that type. Always includes redundant edges. Inter-key order matters as used by ``noise.forcers.ForceByEdge.force_error``.
        * ``merges`` maps each redundant edge to its substitute.
        """

    @abc.abstractmethod
    def _future_boundary_nodes(self, h: int) -> list[Node]:
        """Return list of boundary nodes at top of viewing window.
        
        Input: ``h`` window height.
        """

    @abc.abstractmethod
    def _redundant_boundary_nodes(self, h: int) -> list[Node]:
        """Return list of additional boundary nodes due to unmerged redundant edges.
        
        Used only by circuit-level noise.
        
        Input: ``h`` window height.
        """

    @property
    def D(self): return self._D

    @property
    def MERGED_EQUIVALENT_BOUNDARY_NODES(self):
        return self._MERGED_EQUIVALENT_BOUNDARY_NODES

    @property
    def SCHEME(self): return self._SCHEME

    @property
    def N_EDGES(self): return self._N_EDGES

    @property
    def EDGES(self): return self._EDGES

    @property
    def NODES(self): return self._NODES

    @cached_property
    def NODE_COUNT(self):
        return len(self.NODES)

    @property
    def NOISE(self): return self._NOISE

    @property
    def TIME_AXIS(self): return self._TIME_AXIS

    @property
    def LONG_AXIS(self): return self._LONG_AXIS

    @cached_property
    def DIMENSION(self):
        return self._CODE_DIMENSION + (str(self.NOISE) != 'code capacity')

    @property
    def INCIDENT_EDGES(self): return self._INCIDENT_EDGES
    # Tried construction via manually changing indices by 1
    # so need not iterate through `self.EDGES`,
    # but this is ~twice as slow.
    
    @cached_property
    def DETECTORS(self):
        return tuple(v for v in self.NODES if not self.is_boundary(v))

    @cached_property
    def DETECTOR_COUNT(self):
        return len(self.DETECTORS)

    @cached_property
    def BOUNDARY_NODES(self):
        return tuple(v for v in self.NODES if self.is_boundary(v))

    def is_boundary(self, v: Node):
        """Determine whether ``v`` a boundary node."""
        return self.SCHEME.is_boundary(v)

    def neighbors(self, v: Node):
        """Return neighbors of ``v``.
        
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
        """Return node at other end of edge ``e``."""
        v = e[1] if u == e[0] else e[0]
        return v
    
    def raise_node(self, v: Node, delta_t: int = 1) -> Node:
        """Move ``v`` up by ``delta_t``."""
        # tuple unpacking is marginally slower
        new_v = list(v)
        new_v[self.TIME_AXIS] += delta_t
        return tuple(new_v)
    
    def raise_edge(self, e: Edge, delta_t: int = 1) -> Edge:
        """Move ``e`` up by ``delta_t``."""
        return tuple(self.raise_node(v, delta_t) for v in e) # type: ignore

    def make_error(self, p: float, exclude_future_boundary: bool = False):
        """Sample edges from freshly discovered region.
        
        
        :param p: characteristic noise level if circuit-level noise;
            else, bitflip probability.
            Should be in [0, 1], though no check is done to ensure this.
        :param exclude_future_boundary: whether to exclude future boundary edges
            from being sampled for the new error in the freshly discovered region.
            Set to ``True`` if you want to emulate the end of a memory experiment,
            where the data qubits are measured and the last syndrome sheet is obtained
            from classically taking parities of these measurement outcomes
            (so there is no measurement error).
        
        
        :returns: The set of bitflipped edges in the freshly discovered region. Each edge bitflips with probability defined by its multiplicity if circuit-level noise; else, probability ``p``.
        """
        error = self.NOISE.make_error(p)
        if exclude_future_boundary:
            excluded_edges: set[Edge] = set()
            for e in error:
                for v in e:
                    if v[self.TIME_AXIS] == self.SCHEME.WINDOW_HEIGHT:
                        excluded_edges.add(e)
            error.difference_update(excluded_edges)
        return error

    def get_syndrome(self, error: set[Edge]):
        """Get syndrome from error configuration.
        
        
        ``error`` a set of bitflipped edges.
        
        
        :returns: ``syndrome`` a set of defects.
        """
        return self.get_verbose_syndrome(error).difference(self.BOUNDARY_NODES)

    def get_verbose_syndrome(self, error: set[Edge]):
        """Get syndrome, treating boundary nodes as detectors too.
        
        
        ``error`` a set of bitflipped edges.
        
        
        :returns: ``verbose_syndrome`` a set of defects.
        """
        # Note: Implementing `verbose_syndrome` as a set we add to and remove from is
        # empirically faster than as a dictionary w/ a key for each measure-Z qubit
        # and Booleans as values, which we flip back and forth (for d=29, p=0.11:
        # 3.03(4) ms < 3.26(5) ms).
        verbose_syndrome: set[Node] = set()
        for e in error:
            verbose_syndrome.symmetric_difference_update(e)
        return verbose_syndrome

    @staticmethod
    def compose_errors(*errors: set[Edge]):
        """Sequentially compose any number of errors.
        
        :param errors: a tuple (error1, error2, ...) where each
        error a set of bitflipped edges.
        
        
        :returns: A set of edges representing the sequential composition of all errors in ``errors``.
        """
        composition: set[Edge] = set()
        for error in errors:
            composition ^= error
        return composition

    def get_logical_error(self, leftover: set[Edge]):
        """Whether leftover implements logical X.
        
        
        :param leftover: a set of bitflipped edges.
        
        
        :returns: logical error count parity if scheme is batch else logical error count if scheme is scheme is global batch else logical error count in commit region if scheme is forward.
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
        error: set[Edge] | None = None,
        syndrome: set[Node] | None = None,
        x_offset=constants.DEFAULT_X_OFFSET,
        with_labels: bool | None = None,
        nodelist: Sequence[Node] | None = None,
        node_size: float | Collection[float] | None = None,
        width: float | Collection[float] | None = None,
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        error_color=constants.RED,
        non_error_color='k',
        **kwargs_for_networkx_draw,
    ):
        """Draw G using matplotlib.
        
        
        :param error: a set of edges. Default is empty set.
        :param syndrome: a set of defects. Default is that produced by ``error``.
        :param x_offset: the ratio of out-of-screen to along-screen distance.
        :param with_labels: whether to draw labels on each node.
        :param nodelist: the subset of nodes to draw. Default is all nodes i.e. ``self.NODES``.
        :param node_size: size of nodes.
        :param width: line width of edges.
        :param boundary_color: string specifying color of boundary nodes.
        :param defect_color: string specifying color of defects.
        :param nondefect_color: string specifying color of nondefects.
        :param error_color: string specifying color of bitflipped edges.
        :param non_error_color: string specifying color of unflipped edges.
        :param kwargs_for_networkx_draw: modifies/adds keyword arguments to ``networkx.draw()``.
        
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
            non_error_color for e in self.EDGES
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
            **kwargs_for_networkx_draw,
        )
        return g

    @abc.abstractmethod
    def get_pos(
        self,
        x_offset: float = constants.DEFAULT_X_OFFSET,
        nodelist: None | Iterable[Node] = None,
    ) -> dict[Node, Coord]:
        """Compute coordinates of each node G for ``draw()``.
        
        
        :param x_offset: the ratio of out-of-screen to along-screen distance.
        :param nodelist: the nodes to draw. Default is ``self.NODES``.
        
        Output: ``pos`` a dictionary where each key a node index; value, position coordinate.
        E.g. for surface code w/ perfect measurements,
        convert each matrix index to position coords via
        (i, j) -> (x, y) = (j, -i).
        """

    def get_node_color(
            self,
            syndrome: Container[Node],
            boundary_color=constants.BLUE,
            defect_color=constants.RED,
            nondefect_color=constants.GREEN,
            nodelist: Iterable[Node] | None = None,
            show_boundary_defects=True,
    ):
        """Return a list of colors each node should be for ``draw()``.
        
        
        :param syndrome: a set of defects.
        :param boundary_color: string specifying color of boundary nodes.
        :param defect_color: string specifying color of defects.
        :param nondefect_color: string specifying color of nondefects.
        :param nodelist: the nodes to draw. Default is all nodes i.e. ``self.NODES``.
        :param show_boundary_defects: whether to boundary nodes can be defects.
        
        Output: List of colors for each node in ``nodelist``.
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
        return node_color
    

class Decoder(abc.ABC):
    """Base class for decoders.
    
    Instance attributes (1 constant):
    * ``CODE`` the code to be decoded.
    * ``correction`` a set of edges comprising the correction.
    """

    @property
    def CODE(self): return self._CODE

    def __init__(self, code: Code):
        """Input: ``code`` the code to be decoded."""
        self._CODE = code
        self.correction: set[Edge]

    def reset(self):
        """Factory reset."""
        self.correction.clear()

    @abc.abstractmethod
    def decode(
            self,
            syndrome: set[Node],
            **kwargs,
    ) -> None | int | tuple[int, int]:
        """Decode syndrome.
        
        
        :param syndrome: the set of defects.
        :param draw: whether to draw.
        :param log_history: whether keep track of state history.
        
        Correction stored in ``self.correction``.
        """

    @abc.abstractmethod
    def draw_decode(self, **kwargs_for_networkx_draw):
        """Draw all stages of decoding.
        
        Input: ``kwargs_for_networkx_draw`` passed to ``NetworkX.draw``
        e.g. ``margins=(0.1, 0.1)``.
        """

    def subset_sample(self, p: float, n: int, tol: float = 5e-1):
        """Simulate decoding cycles for each error subset.
        
        
        :param p: noise level.
        :param n: number of decoding cycles per subset.
        :param tol: how much cutoff error we can tolerate,
            as a fraction of the mean.
        
        
        :returns: A pandas DataFrame indexed by ``weight`` with columns ``['subset prob', 'survival prob', 'm', 'n']``.
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