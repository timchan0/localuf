from typing import Any

import networkx as nx

from localuf.constants import Growth
from localuf.type_aliases import Node, Edge

class _PolicyMixin:
    """Provides `_NODES` and `_growth` attributes."""

    def __init__(
            self,
            nodes: dict[Node, Any],
            growth: dict[Edge, Growth],
    ):
        self._NODES = nodes
        self._growth = growth


class DigraphMaker(_PolicyMixin):
    """Maker of NetworkX digraph.
    
    Instance attributes:
    * `_NODES` dictionary of nodes.
    * `_growth` dictionary of growths.
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
    """Provides `draw_decode`."""

    def __init__(self, fig_width: float):
        self._FIG_WIDTH = fig_width

    def draw(
            self,
            history: list,
            interactive=True,
            fig_width=None,
            **kwargs,
    ):
        """Draw the decoder's history."""
        from matplotlib import pyplot as plt
        if fig_width is None:
            fig_width = self._FIG_WIDTH
        n_plots = len(history)
        if interactive:
            from ipywidgets import interact, BoundedIntText
            @interact(timestep=BoundedIntText(
                value=1,
                min=1,
                max=n_plots,
                step=1,
                description="timestep:",
                disabled=False,
            ))
            def f(timestep):
                plt.figure(figsize=(fig_width, fig_width))
                history[timestep-1].draw_growth(**kwargs)
        else:
            plt.figure(figsize=(fig_width, fig_width*n_plots))
            for k, older_self in enumerate(history, start=1):
                plt.subplot(n_plots, 1, k)
                older_self.draw_growth(**kwargs)
            plt.tight_layout()