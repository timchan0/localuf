from typing import Literal, TYPE_CHECKING

import networkx as nx

from localuf.constants import Growth
from localuf.type_aliases import Node, Edge

if TYPE_CHECKING:
    from localuf.decoders._base_uf import BaseUF
    from localuf.decoders.luf.main import MacarNode, ActisNode, _Node

class _PolicyMixin:
    """Provides ``_NODES`` and ``_growth`` attributes."""

    def __init__(
            self,
            nodes: 'dict[Node, MacarNode] | dict[Node, ActisNode]',
            growth: dict[Edge, Growth],
    ):
        self._NODES = nodes
        self._growth = growth


class DigraphMaker(_PolicyMixin):
    """Maker of NetworkX digraph.
    
    Instance attributes:
    * ``_NODES`` dictionary of nodes.
    * ``_growth`` dictionary of growths.
    """

    @property
    def pointer_digraph(self):
        """Return a NetworkX digraph representing the fully grown edges used by pointers,
        the set of its edges as directed edges,
        the set of its edges as undirected edges.
        """
        dig = nx.DiGraph()
        dig.add_nodes_from(self._NODES.keys())
        dig_diedges: list[Edge] = []
        dig_edges: list[Edge] = []
        for u, node in self._NODES.items():
            if node.pointer != 'C':
                e, index = node.NEIGHBORS[node.pointer]
                if self._growth[e] is Growth.FULL:
                    v = e[index]
                    dig.add_edge(u, v)
                    dig_diedges.append((u, v))
                    dig_edges.append(e)
        return dig, dig_diedges, dig_edges
    

class DecodeDrawer:
    """Provides ``draw_decode``."""

    def __init__(
            self,
            fig_width: float,
            fig_height: None | float = None,
    ):
        self._FIG_WIDTH = fig_width
        self._FIG_HEIGHT = fig_width if fig_height is None else fig_height

    def draw(
            self,
            history: list['BaseUF'],
            style: Literal['interactive', 'horizontal', 'vertical'] = 'interactive',
            fig_width: None | float = None,
            fig_height: None | float = None,
            **kwargs_for_networkx_draw,
    ):
        """Draw the decoder's history."""
        from matplotlib import pyplot as plt
        if fig_width is None: fig_width = self._FIG_WIDTH
        if fig_height is None: fig_height = self._FIG_HEIGHT
        n_plots = len(history)
        if style == 'interactive':
            from ipywidgets import interact, BoundedIntText
            @interact(timestep=BoundedIntText(
                value=1,
                min=1,
                max=n_plots,
                step=1,
                description="timestep:",
                disabled=False,
            ))
            def f(timestep: int):
                plt.figure(figsize=(fig_width, fig_height))
                history[timestep-1].draw_growth(**kwargs_for_networkx_draw)
        elif style == 'horizontal':
            plt.figure(figsize=(fig_width*n_plots, fig_height))
            for k, older_self in enumerate(history, start=1):
                plt.subplot(1, n_plots, k)
                older_self.draw_growth(**kwargs_for_networkx_draw)
            plt.tight_layout()
        else:
            plt.figure(figsize=(fig_width, fig_height*n_plots))
            for k, older_self in enumerate(history, start=1):
                plt.subplot(n_plots, 1, k)
                older_self.draw_growth(**kwargs_for_networkx_draw)
            plt.tight_layout()


class AccessUpdater(_PolicyMixin):
    """Updater of access.
    
    Currently unused!
    """

    def update(self, node: '_Node'):
        node.access = {
            pointer: self._NODES[e[index]]
            for pointer, (e, index) in node.NEIGHBORS.items()
            if self._growth[e] is Growth.FULL
        }