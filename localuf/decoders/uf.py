import collections
import copy
from functools import cached_property
from typing import Literal

import networkx as nx

from localuf.type_aliases import Edge, Node
from localuf.codes import Code
from localuf import constants
from localuf.constants import Growth
from localuf.decoders.base_decoder import BaseDecoder

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


class _BaseUF(BaseDecoder):
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

    def draw_growth(
        self,
        growth: dict[Edge, Growth] | None = None,
        syndrome: set[Node] | None = None,
        highlighted_edges: set[Edge] | None = None,
        highlighted_edge_color='k',
        unhighlighted_edge_color=constants.GRAY,
        x_offset=constants.DEFAULT_X_OFFSET,
        labels=None,
        node_size=constants.SMALL,
        width=constants.WIDE_MEDIUM,
    ):
        """Draw growth of edges using matplotlib."""
        g = self.CODE.GRAPH
        pos = self.CODE._get_pos(x_offset)
        if growth is None:
            growth = self.growth
        if syndrome is None:
            syndrome = set()
        if highlighted_edges is None:
            highlighted_edges = set()

        # node-related kwargs
        with_labels = False if labels is None else True
        node_color = self.CODE._get_node_color(syndrome)

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


class UF(_BaseUF):
    """The graph used by UF decoder.
    Extends _BaseUF.

    Instance attributes:
    * `parents` a dictionary where each key a node; value, its parent node.
    * `clusters` a dictionary where each key a root node; value, the cluster of that root node.
    Initially, EVERY node its own cluster; then, only grow active clusters.
    * `active_clusters` the set of clusters w/ odd # defects & no surface boundary node.
    * `history` a list of past self snapshots @ each growth round
    (all w/ same `history` attribute to prevent infinite loop).
    * `forest` the spanning forest after syndrome validation.
    8 `digraph` a NetworkX digraph of `forest`.
    * `correction` only exists after calling `decode()`.
    """

    def __init__(self, code: Code):
        super().__init__(code)
        self.reset()

    def __repr__(self):
        return f'UF(code={self.CODE}, syndrome={self.syndrome})'
    
    def reset(self):
        super().reset()
        self.syndrome = set()
        self.parents = {v: v for v in self.CODE.NODES}
        self.clusters = {v: _Cluster(self, v) for v in self.CODE.NODES}
        self.active_clusters: set[_Cluster] = set()
        try: del self.history
        except AttributeError: pass
        try: del self.changed_edges
        except AttributeError: pass
        try: del self.forest
        except AttributeError: pass
        try: del self.digraph
        except AttributeError: pass

    def load(self, syndrome: set[Node]):
        """Load a new syndrome."""
        self.syndrome = syndrome
        for v in syndrome:
            cluster = self.clusters[v]
            cluster.odd = True
            self.active_clusters.add(cluster)
    
    def decode(
            self,
            syndrome: set[Node],
            dynamic=False,
            draw=False,
            fig_width: float | None = None,
    ):
        """Additional inputs over `_BaseUF.decode()`:
        * `dynamic` whether forest dynamic or static.
        * `fig_width` figure width.
        """
        self.validate(
            syndrome,
            dynamic=dynamic,
            log_history=draw,
        )
        self.peel()
        if draw:
            self._draw_decode(fig_width=fig_width)

    # SYNDROME VALIDATION

    def validate(
            self,
            syndrome: set[Node],
            dynamic=False,
            log_history=False,
    ):
        """Grow clusters until they are all valid."""
        self.load(syndrome)
        if log_history: self.history: list[UF] = []
        while self.active_clusters:
            self._growth_round(dynamic, log_history)

    def _growth_round(
            self,
            dynamic: bool,
            log_history: bool,
            clusters_to_grow: 'set[_Cluster] | None' = None,
    ):
        """Grow each active cluster once."""
        if clusters_to_grow is None:
            clusters_to_grow = self.active_clusters
        merge_ls: list[Edge] = []
        changed_edges: set[Edge] = set()
        for cluster in clusters_to_grow:
            self._grow(cluster, merge_ls, changed_edges)
        for u, v in merge_ls:
            cu = self.clusters[self._find(u)]
            cv = self.clusters[self._find(v)]
            # remove edge (u, v) from all visions
            cu.vision.remove((u, v))
            cv.vision.discard((u, v))  # discard instead of remove as cu may be cv
            self._merge(cu, cv, (u, v), dynamic)
        if log_history:
            self.changed_edges: set[Edge] = changed_edges
            self.history.append(copy.deepcopy(
                self,
                memo={id(self.history): self.history}
            ))

    def _grow(
            self,
            cluster: '_Cluster',
            merge_ls: list[Edge],
            changed_edges: set[Edge],
    ):
        """Grow a cluster once."""
        changed_edges.update(cluster.vision)
        for e in cluster.vision:
            self.growth[e] += Growth.INCREMENT
            if self.growth[e] is Growth.FULL:
                merge_ls.append(e)

    def _find(self, v: Node):
        """Find root of `v` (w/ path compression).
        
        From https://www.geeksforgeeks.org/union-by-rank-and-path-compression-in-union-find-algorithm/.
        """
        if self.parents[v] != v:
            self.parents[v] = self._find(self.parents[v])
        return self.parents[v]

    def _merge(self, cu: '_Cluster', cv: '_Cluster', uv: Edge, dynamic: bool):
        """Merge clusters `cu` and `cv` along edge `uv`."""
        u, v = uv
        if dynamic:
            if cu == cv or (cu.boundary and cv.boundary):
                self.growth[u, v] = Growth.BURNT
            else:
                self._union(cu, cv)
        else:
            if cu != cv:
                self._union(cu, cv)

    def _union(self, cu: '_Cluster', cv: '_Cluster'):
        """Union clusters `cu` w/ `cv` (by weight)."""
        if cu.size >= cv.size:
            larger, smaller = cu, cv
        else:
            larger, smaller = cv, cu
        self.parents[smaller.root] = larger.root
        larger.size += smaller.size
        larger.odd ^= smaller.odd  # logical XOR
        old_larger_vision_length = len(larger.vision) + 1  # +1 as we already removed (u, v) from all visions
        larger.vision.update(smaller.vision)
        if (not larger.boundary) and smaller.boundary:
            larger.boundary = smaller.boundary
        self._update_self_after_union(larger, smaller, old_larger_vision_length)

    def _update_self_after_union(
            self,
            larger: '_Cluster',
            smaller: '_Cluster',
            old_larger_vision_length: int,
    ):
        """Update attributes `clusters` and `active_clusters` after union of larger w/ smaller."""

        # DELETE SMALLER CLUSTER
        del self.clusters[smaller.root]
        # discard instead of remove as smaller may be a node not in syndrome
        self.active_clusters.discard(smaller)

        # UPDATE active_clusters SET
        if larger.odd and not larger.boundary:
            # need this line as an even cluster could become odd again by union w/ smaller odd cluster
            self.active_clusters.add(larger)
        else:
            # discard instead of remove as larger could have had boundary before union, hence was not active
            self.active_clusters.discard(larger)

    # PEELING DECODER
    
    def peel(self):
        """Peel the validated erasure."""
        self.forest = self._make_forest()
        modified_syndrome = self.syndrome.copy()
        self.correction: set[Edge] = set()
        for e, u in reversed(self.forest):
            if u in modified_syndrome:
                self.correction.add(e)
                modified_syndrome.symmetric_difference_update(e)
    
    def _make_forest(self):
        """Return a spanning forest of the validated erasure."""
        forest: list[tuple[Edge, Node]] = []
        for cluster in self.clusters.values():
            tree_root = cluster.boundary if cluster.boundary else cluster.root
            forest += self._make_tree(tree_root)
        return forest

    def _make_tree(self, tree_root: Node):
        """Return a spanning tree of the validated cluster, from `tree_root`."""
        discovered = {tree_root}
        stack = collections.deque([tree_root])
        tree: list[tuple[Edge, Node]] = []
        while stack:  # each edge is considered twice due to its two endpoints
            u = stack.pop()
            for e in self.erasure.intersection(self.CODE.INCIDENT_EDGES[u]):
                v = self.CODE.traverse_edge(e, u)
                if v not in discovered:
                # could also check nb not surface boundary (when not dynamic), but not needed
                    discovered.add(v)
                    stack.append(v)
                    tree.append((e, v))
        return tree

    # DRAWERS

    def draw_growth(self, **kwargs):
        return super().draw_growth(**kwargs, syndrome=self.syndrome)

    @cached_property
    def digraph(self):
        """Return a NetworkX digraph representing the spanning forest after syndrome validation.
        
        Note: Requires calling `uf_graph.peel()` first.
        """
        dig = nx.DiGraph()
        dig.add_nodes_from(self.CODE.NODES)
        for e, u in self.forest:
            v = self.CODE.traverse_edge(e, u)
            dig.add_edge(u, v)
        return dig

    def draw_forest(
        self,
        x_offset=constants.DEFAULT_X_OFFSET,
        node_size=constants.SMALL,
        width=constants.WIDE_MEDIUM,
        edge_color = 'g',
        **kwargs,
    ):
        """Draw spanning forest.

        `kwargs` passed to `networkx.draw()`.
        
        Note: Not as informative as draw_peel().
        """
        dig = self.digraph.reverse()
        pos = self.CODE._get_pos(x_offset)
        node_color = self.CODE._get_node_color(self.syndrome)
        return nx.draw(
            dig,
            pos=pos,
            node_size=node_size,
            node_color=node_color,
            width=width,
            edge_color=edge_color,
            **kwargs,
        )

    def draw_peel(
        self,
        x_offset=constants.DEFAULT_X_OFFSET,
        node_size=constants.SMALL,
        width=constants.WIDE_MEDIUM,
        **kwargs,
    ):
        """Draw forest and correction from peeling.

        `kwargs` passed to `networkx.draw()`.
        
        Note: Requires calling `uf_graph.peel()` first.
        """
        dig = self.digraph
        pos = self.CODE._get_pos(x_offset)
        node_color = self.CODE._get_node_color(self.syndrome)
        edge_color = [
            'r' if ((u, v) in self.correction) or ((v, u) in self.correction) else
            constants.GRAY for u, v in dig.edges
        ]
        return nx.draw(
            dig,
            pos=pos,
            with_labels=False,
            node_size=node_size,
            node_color=node_color,
            width=width,
            edge_color=edge_color,
            **kwargs,
        )

    def _draw_decode(self, fig_width: float | None = None):
        from matplotlib import pyplot as plt
        if fig_width is None:
            fig_width = max(
                1,
                self.CODE.D * (2*self.CODE.DIMENSION - 3) / 3
            )
        if self.CODE.DIMENSION == 1:
            fig_height = fig_width / self.CODE.D
        elif self.CODE.DIMENSION == 2:
            fig_height = fig_width * max(
                self.CODE.D, self.CODE.SCHEME.WINDOW_HEIGHT
            ) / self.CODE.D
        else:
            fig_height = fig_width * self.CODE.SCHEME.WINDOW_HEIGHT / self.CODE.D
        n_plots = len(self.history) + 1
        plt.figure(figsize=(fig_width, fig_height*n_plots))
        for k, older_self in enumerate(self.history, start=1):
            plt.subplot(n_plots, 1, k)
            older_self.draw_growth(
                highlighted_edges=older_self.changed_edges
            )
        plt.subplot(n_plots, 1, n_plots)
        self.draw_peel()
        plt.tight_layout()


class _Cluster:
    """A tree data structure to represent a cluster of nodes of UF.
    
    Attributes:
    * `root` the representative of the cluster.
    * `size` the number of nodes in the cluster.
    * `odd` a Boolean to say if the number of defects in the cluster is odd.
    * `vision` a set of active edges incident to the cluster i.e. edges grown by the cluster in next growth round.
    * `boundary` the node in the cluster that is the surface boundary if it exists; else, None.
    If `cluster.boundary` not None, cluster will never be active in future.
    If syndrome validation not dynamic, boundary may not be unique!
    """

    def __init__(self, uf: UF, root: Node) -> None:
        self.root = root
        self.size = 1
        self.odd = False
        self.vision = uf.CODE.INCIDENT_EDGES[root].copy()
        self.boundary = root if uf.CODE.is_boundary(root) else None

    def __str__(self) -> str:
        return f'cluster(root={self.root}, size={self.size}, odd={self.odd})'


class BUF(UF):
    """The graph used by Bucket UF.

    Attributes:
    * `mvl` an integer of the minimum vision length of
    all active clusters at the start of a growth round.

    Methods:
    * `_update_mvl`.
    * `_demote_cluster`.
    
    Overriden attributes:
    * `buckets` replaces `active_clusters`
    and is a dictionary where each key an integer;
    value, a set of all active clusters whose vision length is that integer.

    Overriden methods:
    * `reset`.
    * `load`.
    * `validate_syndrome`.
    * `_merge`.
    * `_update_self_after_union`.
    """

    def reset(self):
        super().reset()
        self.buckets: dict[int, set[_Cluster]] = {
            vision_length: set() for vision_length
            in range(1, self.CODE.N_EDGES+1)
        }
        self.mvl = None

    def load(self, syndrome):
        super().load(syndrome)
        for cluster in self.active_clusters:
            self.buckets[len(cluster.vision)].add(cluster)
        del self.active_clusters
        self._update_mvl()

    def _update_mvl(self):
        """Update `mvl` attribute."""
        changed = False
        for vision_length, bucket in self.buckets.items():
            if bucket:
                self.mvl = vision_length
                changed = True
                break
        if not changed:
            self.mvl = None

    # SYNDROME VALIDATION (need tests)

    def validate(
            self,
            syndrome: set[Node],
            dynamic=False,
            log_history=False
    ):
        """Grow clusters (always shortest vision first) until they are all valid."""
        self.load(syndrome)  # TODO: test this is called
        if log_history: self.history: list[UF] = []
        while self.mvl is not None:
            self._growth_round(
                dynamic,
                log_history,
                clusters_to_grow=self.buckets[self.mvl]
            )
            self._update_mvl()

    def _merge(self, cu, cv, uv, dynamic):
        u, v = uv
        if dynamic:
            if cu == cv:
                self.growth[u, v] = Growth.BURNT
                self._demote_cluster(cu)
            elif (cu.boundary and cv.boundary):
                self.growth[u, v] = Growth.BURNT
                self._demote_cluster(cu)
                self._demote_cluster(cv)
            else:
                self._union(cu, cv)
        else:
            if cu == cv:
                self._demote_cluster(cu)
            else:
                self._union(cu, cv)

    def _demote_cluster(self, cluster: _Cluster):
        """If cluster active, move down a bucket as an edge has been removed from its vision. Else do nothing."""
        vision_length = len(cluster.vision)
        try:
            self.buckets[vision_length+1].remove(cluster)
            self.buckets[vision_length].add(cluster)
        except KeyError: pass

    def _update_self_after_union(self, larger, smaller, old_larger_vision_length):
        """Update attributes clusters and buckets after union of larger w/ smaller."""

        # DELETE SMALLER CLUSTER
        del self.clusters[smaller.root]
        # discard instead of remove as smaller may be a node not in syndrome
        self.buckets[len(smaller.vision)+1].discard(smaller)  # +1 as we already removed (u, v) from all visions

        # UPDATE buckets DICTIONARY
        # discard instead of remove as larger could have had boundary before union, hence was not active
        self.buckets[old_larger_vision_length].discard(larger)
        if larger.odd and not larger.boundary:
            self.buckets[len(larger.vision)].add(larger)