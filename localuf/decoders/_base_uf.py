from abc import abstractmethod
import copy
from functools import cached_property
from typing import Literal

import networkx as nx

from localuf import constants
from localuf._base_classes import Code, Decoder
from localuf.constants import Growth
from localuf.type_aliases import Edge, Node


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
    """The abstract graph used by the UF decoder (Union--Find).
    Extends `BaseDecoder`.

    Class attributes:
    * `_ACTIVE_GROWTH_VALUES` the set of growth values for which an edge is active i.e. can grow.
    * `_INACTIVE_GROWTH_VALUES` the set of growth values for which an edge is inactive.

    Additional instance attributes:
    * `growth` a dictionary where each key an edge; value, an integer representing its growth stage.
    * `syndrome` the set of defects.
    * `erasure` the set of fully grown edges.
    Only computed if `erasure` not yet an attribute.
    * `history` a list of past self snapshots @ each growth round
    (all w/ same `history` attribute to prevent infinite loop).
    """

    _ACTIVE_GROWTH_VALUES = {Growth.UNGROWN, Growth.HALF}
    _INACTIVE_GROWTH_VALUES = {Growth.BURNT, Growth.FULL}

    def __init__(self, code: Code):
        """Input: `code` the code to be decoded."""
        super().__init__(code)
        self._growth = {e: Growth.UNGROWN for e in self.CODE.EDGES}
        self.syndrome: set[Node]

    @property
    def growth(self): return self._growth

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
        """Compute erasure from growth."""
        return {e for e in self.CODE.EDGES if self.growth[e] is Growth.FULL}

    def reset(self):
        """Factory reset."""
        for e in self.growth.keys():
            self.growth[e] = Growth.UNGROWN
        super().reset()
        try: del self.erasure
        except AttributeError: pass
        try: del self.history
        except AttributeError: pass

    def init_history(self):
        """Initialize `history` attribute."""
        self.history: list[BaseUF] = []

    def append_history(self):
        """Append a snapshot of the current `self` to `self.history`."""
        self.history.append(copy.deepcopy(
            self, memo={id(self.history): self.history}
        ))

    def draw_growth(
        self,
        growth: dict[Edge, Growth] | None = None,
        syndrome: set[Node] | None = None,
        highlighted_edges: set[Edge] | None = None,
        highlighted_edge_color='k',
        unhighlighted_edge_color=constants.GRAY,
        x_offset=constants.DEFAULT_X_OFFSET,
        labels: dict[Node, str] | None = None,
        node_size=constants.SMALL,
        width=constants.WIDE_MEDIUM,
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        show_boundary_defects=True,
        **kwargs,
    ):
        """Draw growth of edges using matplotlib.
        
        Input:
        * `growth` a dictionary where each key an edge index; value, its growth value.
        * `syndrome` the set of defects.
        * `highlighted_edges` the set of edges to be highlighted in drawing.
        * `x_offset` the ratio of out-of-screen to along-screen distance.
        * `labels` a dictionary where each key a node index; value, its label as a string.
        * `width` line width of edges.
        * `kwargs` passed to `networkx.draw()`.
        E.g. `linewidths` line width of node symbol border.
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
        with_labels = False if labels is None else True
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
            **kwargs,
        )

    def _get_edge_color_and_style(
            self,
            edgelist: list[Edge],
            highlighted_edges: set[Edge],
            highlighted_edge_color: str,
            unhighlighted_edge_color: str,
            growth: dict[Edge, Growth] | None = None,
    ):
        """Return `edge_color`, `style` kwargs for `networkx.draw()`."""
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
            **kwargs,
    ):
        """Return `node_color`, `edgecolors` kwargs for `networkx.draw()`.

        Input:
        * `outlined_nodes` the set of nodes to be outlined.
        * `nodelist` the list of nodes (needed to specify the node order in the outputs).
        * `kwargs` passed to `self.CODE.get_node_color()`.
        """
        node_color = self.CODE.get_node_color(
            self.syndrome,
            nodelist=nodelist,
            **kwargs,
        )
        edgecolors = [outline_color if v in outlined_nodes else color
                        for v, color in zip(nodelist, node_color)]
        return node_color, edgecolors