from abc import abstractmethod
import copy
from functools import cache, cached_property
from typing import Literal

import networkx as nx

from localuf import constants
from localuf._base_classes import Code, Decoder
from localuf.constants import Growth
from localuf.type_aliases import Edge, Node
from itertools import chain, repeat


direction = Literal[
    'C',
    'N',
    'W',
    'E',
    'S',
    'D',
    'U',
    'SD',
    'NU',
    'EU',
    'WD',
    'SEU',
    'NWD',
]
"""Possible pointer value."""


class BaseUF(Decoder):
    """The abstract class representing the UF decoder (Union--Find).
    
    Extends ``BaseDecoder``.
    """

    _ACTIVE_GROWTH_VALUES = {Growth.UNGROWN, Growth.HALF}
    """The set of growth values for which an edge is active i.e. can grow."""
    _INACTIVE_GROWTH_VALUES = {Growth.BURNT, Growth.FULL}
    """The set of growth values for which an edge is inactive."""

    def __init__(self, code: Code):
        """Input: ``code`` the code to be decoded."""
        super().__init__(code)
        self._growth: dict[Edge, Growth] = {e: Growth.UNGROWN for e in self.CODE.EDGES}
        self.syndrome: set[Node]
        """The set of defects."""
        self.history: list[BaseUF]
        """A list of past self snapshots at each growth round
        (all with the same ``history`` attribute to prevent infinite loop).
        """

    @property
    def growth(self):
        """A map from each edge in the decoding graph to an integer representing its growth value."""
        return self._growth

    @property
    @abstractmethod
    def _FIG_WIDTH(self) -> float:
        """Width of figure for drawer."""
    
    @property
    @abstractmethod
    def _FIG_HEIGHT(self) -> float:
        """Height of figure for drawer."""

    @cached_property
    def erasure(self):
        """The set of fully grown edges, computed from ``self.growth``.
        Only computed if ``erasure`` not yet an attribute.
        """
        return {e for e, growth in self.growth.items() if growth is Growth.FULL}

    def reset(self):
        """Factory reset."""
        # use `_growth` instead of `growth` as the child class `Snowflake`
        # defines latter as a property that must be computed every time it is accessed
        # meaning the property would have to be redundantly computed |E| times!
        for e in self._growth.keys():
            self._growth[e] = Growth.UNGROWN
        super().reset()
        try: del self.erasure
        except AttributeError: pass
        try: del self.history
        except AttributeError: pass

    def init_history(self):
        """Initialize ``history`` attribute."""
        self.history = []

    def append_history(self):
        """Append a snapshot of the current ``self`` to ``self.history``."""
        self.history.append(copy.deepcopy(
            self, memo={id(self.history): self.history}
        ))

    @cache
    def _total_edge_weight(self, noise_level: None | float = None) -> float:
        """Compute the sum of all edge weights in the decoding window."""
        edge_weights = self.CODE.NOISE.get_edge_weights(noise_level)
        return sum(edge_weights[edge][1] for edge in self.growth.keys())
    
    def unclustered_edge_fraction(self, noise_level: None | float = None):
        """Compute the Unclustered Edge Fraction DCS from ``self.growth``.
        
        
        :param noise_level: A probability representing the noise strength.
            This is needed to define nonuniform edge weights of the decoding graph
            in the circuit-level noise model.
            If not specified, all edges are assumed to have weight 1.
        
        
        :returns fraction: The fraction of edges in the decoding window
            that are not in a cluster, weighted by their weights.
        """
        edge_weights = self.CODE.NOISE.get_edge_weights(noise_level)
        numerator = sum(
            edge_weights[edge][1] * growth_value.as_float
            for edge, growth_value in self.growth.items()
        )
        return 1 - numerator / self._total_edge_weight(noise_level=noise_level)
    
    def draw_growth(
        self,
        growth: dict[Edge, Growth] | None = None,
        syndrome: set[Node] | None = None,
        highlighted_edges: set[Edge] | None = None,
        highlighted_edge_color='k',
        unhighlighted_edge_color=constants.GRAY,
        x_offset=constants.DEFAULT_X_OFFSET,
        with_labels=True,
        labels: dict[Node, str] | None = None,
        node_size=constants.SMALL,
        width=constants.WIDE_MEDIUM,
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        show_boundary_defects=True,
        **kwargs_for_networkx_draw,
    ):
        """Draw growth of edges using matplotlib.
        
        
        :param growth: A dictionary where each key an edge index; value, its growth value.
        :param syndrome: The set of defects.
        :param highlighted_edges: The set of edges to be highlighted in drawing.
        :param x_offset: The ratio of out-of-screen to along-screen distance.
        :param with_labels: Whether to show node labels.
        :param labels: A dictionary where each key a node index; value, its label as a string.
        :param width: Line width of edges.
        :param arrows: Whether to draw arrows on edges used by pointers.
            For ``Macar, Actis, Snowflake`` decoders only.
        :param kwargs_for_networkx_draw: Passed to ``networkx.draw()``.
            E.g. ``linewidths`` line width of node symbol border.
        """
        g = self.CODE.GRAPH
        pos = self.CODE.get_pos(x_offset)
        if growth is None:
            growth = self.growth
        if syndrome is None:
            syndrome = set()
        if highlighted_edges is None:
            highlighted_edges = set()

        # node-related kwargs
        if labels is None:
            with_labels = False
        node_color = self.CODE.get_node_color(
            syndrome,
            boundary_color=boundary_color,
            defect_color=defect_color,
            nondefect_color=nondefect_color,
            show_boundary_defects=show_boundary_defects,
        )

        # edge-related kwargs
        edgelist = [
            e for e in self.CODE.EDGES
            if growth[e] in {Growth.HALF, Growth.FULL}
        ]
        edge_color, style = self._get_edge_color_and_style(
            edgelist,
            highlighted_edges,
            highlighted_edge_color,
            unhighlighted_edge_color,
            growth=growth,
        )
        return nx.draw(
            g,
            pos=pos,
            with_labels=with_labels,
            nodelist=self.CODE.NODES,
            edgelist=edgelist,
            node_size=node_size,
            node_color=node_color,
            width=width,
            edge_color=edge_color,
            style=style,
            labels=labels,
            **kwargs_for_networkx_draw,
        )

    def _get_edge_color_and_style(
            self,
            edgelist: list[Edge],
            highlighted_edges: set[Edge],
            highlighted_edge_color: str,
            unhighlighted_edge_color: str,
            growth: dict[Edge, Growth] | None = None,
    ):
        """Return ``edge_color``, ``style`` kwargs for ``networkx.draw()``."""
        if growth is None:
            growth = self.growth
        edge_color = [
            highlighted_edge_color if e in highlighted_edges else
            unhighlighted_edge_color for e in edgelist
        ]
        style = [
            ':' if growth[e] is Growth.HALF else
            '-' for e in edgelist
        ]
        return edge_color, style

    def _get_node_color_and_edgecolors(
            self,
            outlined_nodes: set[Node],
            nodelist: list[Node],
            outline_color: str = 'k',
            **kwargs_for_get_node_color,
    ):
        """Return node color and node edge color kwargs for ``networkx.draw()``.
        
        
        :param outlined_nodes: Set of nodes to be outlined.
        :param nodelist: List of nodes (needed to specify the node order in the outputs).
        :param kwargs_for_get_node_color: Passed to ``self.CODE.get_node_color()``.

        :return node_color: List of node colors.
        :return edgecolors: List of node edge colors.
        """
        node_color = self.CODE.get_node_color(
            self.syndrome,
            nodelist=nodelist,
            **kwargs_for_get_node_color,
        )
        edgecolors = [outline_color if v in outlined_nodes else color
                        for v, color in zip(nodelist, node_color)]
        return node_color, edgecolors
    
    def swim_distance(
            self,
            noise_level: None | float = None,
            draw=False,
            **kwargs_for_draw_swim_graph,
    ) -> float:
        """Compute the swim distance DCS from ``self.growth``.
        
        Assumes ``self.validate()`` or ``self.decode()`` has already been called.
        swim distance is the shortest distance from one boundary to the other
        where travelling within clusters is free.
        Introduced in [doi.org/10.1038/s42005-024-01883-4] and [arXiv:2405.07433].
        
        This method works regardless of whether ``self.CODE`` was instantiated with
        ``merge_equivalent_boundary_nodes=False`` or ``True``
        but works slightly faster in the latter case.
        
        :param noise_level: An optional probability representing the noise strength.
            This defines nonuniform edge weights of the decoding graph
            in the circuit-level noise model.
            If not specified, all edges are assumed to have weight 1.
        :param draw: Whether to draw the search graph and the shortest path.
        :param kwargs_for_draw_swim_graph: Passed to ``self._draw_swim_graph()``.
        
        :return length: The swim distance.
        """
        graph = self._swim_graph(noise_level=noise_level)
        length, path = nx.bidirectional_dijkstra(graph, *self._META_BOUNDARIES)
        if draw:
            self._draw_swim_graph(path, **kwargs_for_draw_swim_graph)
        return length
    
    def _swim_graph(self, noise_level: None | float = None):
        """Return the search graph and endpoints for the swim distance.
        
        :param noise_level: As described in ``swim_distance()``.
        
        :return graph: A NetworkX graph representing
            the search graph for the swim distance DCS.
            The edges not in ``self.growth`` have weight 0 ALWAYS.
            The weight of the other edges is set according to ``self.growth``.
        """
        # TODO: use `networkx.quotient_graph`
        graph = self._cached_swim_graph
        for e, (_, weight) in self.CODE.NOISE.get_edge_weights(noise_level).items():
            if e not in self.growth or self.growth[e] is Growth.UNGROWN:
                # this can happen when the decoder omits some edges in W
                # e.g. Snowflake omitting the future boundary
                graph.edges[e]['weight'] = weight
            elif self.growth[e] is Growth.HALF:
                graph.edges[e]['weight'] = weight / 2
            else:  # growth is Growth.FULL or Growth.BURNT
                graph.edges[e]['weight'] = 0
        return self._cached_swim_graph
    
    @cached_property
    def _META_BOUNDARIES(self) -> tuple[Node, Node]:
        """The two extra nodes connecting the west and east boundary nodes.
        
        :return west: The node connected to the west boundary nodes.
        :return east: The node connected to the east boundary nodes.
        """
        d = self.CODE.D
        a = self.CODE.LONG_AXIS
        n = self.CODE.DIMENSION
        return (
            tuple(chain(repeat(d//2, a), (-2,), repeat(d//2, n-a-1))),
            tuple(chain(repeat(d//2, a), ( d,), repeat(d//2, n-a-1))),
        )
    
    @cached_property
    def _cached_swim_graph(self):
        """Return the search graph and endpoints for the swim distance.
        
        :return graph: A NetworkX graph representing
            the search graph for the swim distance DCS.
            The only time it is retrieved is in the property ``_swim_graph``,
            which always updates the edge weights before returning the graph.
            The edges not in ``self.growth`` have weight 0 ALWAYS.
            There is no guarantee about the weight of the other edges.
        """
        graph = self.CODE.GRAPH.copy()
        d = self.CODE.D
        a = self.CODE.LONG_AXIS
        west, east = self._META_BOUNDARIES
        graph.add_edges_from(((west, v) for v in self.CODE.NODES if v[a]==-1), weight=0)
        graph.add_edges_from(((v, east) for v in self.CODE.NODES if v[a]==d-1), weight=0)
        return graph

    def _draw_swim_graph(
            self,
            path: list[Node],
            weightless_edge_width=constants.THIN,
            max_edge_width=constants.V_WIDE,
            node_size=100,
            node_alpha=1,
            edge_alpha=0.5,
            **kwargs_for_networkx_draw,
    ):
        """Draw the search graph for the swim distance.
        
        :param path: The shortest west--east path.
        :param weightless_edge_width: The width of zero-weight edges.
        :param max_edge_width: The maximum width of edges.
        :param kwargs_for_networkx_draw: Passed to ``networkx.draw()``.
        """
        graph = self._cached_swim_graph
        max_weight: float = max(weight for _, _, weight in graph.edges.data('weight')) # type: ignore
        width_multiplier = (max_edge_width - weightless_edge_width) / max_weight
        nx.draw_networkx_nodes(
            graph,
            pos=self.CODE.get_pos(nodelist=graph.nodes),
            node_size=node_size,
            node_color=self.CODE.get_node_color(path, nodelist=graph.nodes), # type: ignore
            alpha=node_alpha,
        )
        nx.draw_networkx_edges(
            graph,
            pos=self.CODE.get_pos(nodelist=graph.nodes),
            node_size=node_size,
            width=[width_multiplier * weight + weightless_edge_width
                   for _, _, weight in graph.edges.data('weight')], # type: ignore
            edge_color=[constants.RED if set(edge)<=set(path)
                        else 'k' for edge in graph.edges], # type: ignore
            alpha=edge_alpha,
            **kwargs_for_networkx_draw,
        )