import collections
from functools import cached_property

import networkx as nx

from localuf import constants
from localuf.constants import Growth
from localuf.type_aliases import Edge, Node
from localuf._base_classes import Code
from localuf._schemes import Frugal
from localuf.decoders._base_uf import BaseUF


class UF(BaseUF):
    """The graph used by UF decoder.
    
    Extends `BaseUF`.
    Incompatible with frugal scheme.

    Additional instance attributes:
    * `parents` a dictionary where each key a node; value, its parent node.
    * `clusters` a dictionary where each key a root node; value, the cluster of that root node.
    Initially, EVERY node its own cluster; then, only grow active clusters.
    * `active_clusters` the set of clusters w/ odd # defects & no surface boundary node.
    * `forest` the spanning forest after syndrome validation.
    8 `digraph` a NetworkX digraph of `forest`.
    * `correction` only exists after calling `decode()`.
    """

    def __init__(self, code: Code):
        if isinstance(code.SCHEME, Frugal):
            raise ValueError('UF incompatible with frugal scheme.')
        self.history: list[UF]
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
        """Additional inputs over `Decoder.decode()`:
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
            self.draw_decode(fig_width=fig_width)

    # SYNDROME VALIDATION

    def validate(
            self,
            syndrome: set[Node],
            dynamic=False,
            log_history=False,
    ):
        """Grow clusters until they are all valid."""
        self.load(syndrome)
        if log_history: self.init_history()
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
            self.append_history()

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
        
        Note: Requires calling `peel()` first.
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
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        show_boundary_defects=True,
        **kwargs,
    ):
        """Draw spanning forest.

        `kwargs` passed to `networkx.draw()`.
        
        Note: Not as informative as draw_peel().
        """
        dig = self.digraph.reverse()
        pos = self.CODE.get_pos(x_offset)
        node_color = self.CODE.get_node_color(
            self.syndrome,
            boundary_color=boundary_color,
            defect_color=defect_color,
            nondefect_color=nondefect_color,
            show_boundary_defects=show_boundary_defects,
        )
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
        correction_color='k',
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        show_boundary_defects=True,
        **kwargs,
    ):
        """Draw forest and correction from peeling.

        `kwargs` passed to `networkx.draw()`.
        
        Note: Requires calling `peel()` first.
        """
        dig = self.digraph
        pos = self.CODE.get_pos(x_offset)
        node_color = self.CODE.get_node_color(
            self.syndrome,
            boundary_color=boundary_color,
            defect_color=defect_color,
            nondefect_color=nondefect_color,
            show_boundary_defects=show_boundary_defects,
        )
        edge_color = [
            correction_color if ((u, v) in self.correction) or ((v, u) in self.correction) else
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

    def draw_decode(
            self,
            fig_width: float | None = None,
            fig_height: float | None = None,
            **kwargs,
    ):
        from matplotlib import pyplot as plt
        if fig_width is None: fig_width = self._FIG_WIDTH
        if fig_height is None: fig_height = self._FIG_HEIGHT
        n_plots = len(self.history) + 1
        plt.figure(figsize=(fig_width, fig_height*n_plots))
        for k, older_self in enumerate(self.history, start=1):
            plt.subplot(n_plots, 1, k)
            older_self.draw_growth(
                highlighted_edges=older_self.changed_edges,
                **kwargs,
            )
        plt.subplot(n_plots, 1, n_plots)
        self.draw_peel(**kwargs)
        plt.tight_layout()

    @property
    def _FIG_WIDTH(self):
        return max(1, self.CODE.D * (2*self.CODE.DIMENSION - 3) / 3)
    
    @property
    def _FIG_HEIGHT(self):
        d = self.CODE.D
        h = self.CODE.SCHEME.WINDOW_HEIGHT
        w = self._FIG_WIDTH
        if (n:=self.CODE.DIMENSION) == 1:
            return w / d
        elif n == 2:
            return w * max(d, h) / d
        else:
            return w * h / d


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