import abc
import collections
from functools import cached_property
from typing import Literal

import networkx as nx

from localuf import constants
from localuf.constants import Growth
from localuf.type_aliases import Edge, Node
from localuf._base_classes import Code
from localuf._schemes import Frugal
from localuf.decoders._base_uf import BaseUF


class UF(BaseUF):
    """The original, algorithmic UF decoder.
    
    Extends ``BaseUF``.
    Incompatible with the frugal decoding scheme.
    
    Additional instance constants:
    * ``_FORESTER`` decides whether to maintain a static or dynamic forest.
    * ``_INCLINATION`` a class that decides which boundary the root should be on
        when there is a percolating cluster
    i.e. one that spans opposite boundaries.
    Has the ``update_boundary`` method.
    
    Additional instance attributes:
    * ``parents`` maps each node in the decoding graph to its parent node.
    * ``clusters`` maps each root node to the cluster of that root node.
        Initially, EVERY node its own cluster; then, only grow active clusters.
    * ``active_clusters`` the set of clusters
        with odd defect count and no boundary node.
    * ``forest`` the spanning forest after syndrome validation as a list of pairs (edge, node)
        where node is the further endpoint from the root of the tree in the forest.
    * ``digraph`` a NetworkX digraph of ``forest``.
    """

    def __init__(
            self,
            code: Code,
            dynamic: bool = False,
            inclination: Literal['default', 'west'] = 'default',
    ):
        """
        :param code: the code to be decoded.
        :param dynamic: whether the forest is dynamic or static.
            Static forest is when the spanning forest of the erasure
        is grown only after syndrome validation;
        this implementation grows the spanning tree using depth-first search (DFS).
        Dynamic forest is when the spanning forest of the erasure
        is maintained throughout syndrome validation;
        this is equivalent to multi-source breadth-first search (BFS) where each defect is a source.
        Dynamic forests are further explained in https://doi.org/10.13140/RG.2.2.13495.96162, section 5.3.2.
        This ``dynamic`` parameter has a big effect on the complementary gap:
        the gap of dynamic UF closely follows the log odds of success;
        the gap of static UF can be much larger.
        This is because for dynamic UF only, the spanning tree of each cluster
        connects any two defects in a low-weight path.
        :param inclination: decides which boundary the root should be on when there is a percolating cluster.
            'default' means prefer neither east nor west, essentially random.
        'west' defines the root of a percolating cluster always at the west boundary.
        """
        if isinstance(code.SCHEME, Frugal):
            raise ValueError('UF incompatible with frugal scheme.')
        self.history: list[UF]
        self.correction = set()
        self._FORESTER = _DynamicForester(self) if dynamic else _StaticForester(self)
        inclination_class = _DefaultInclination if inclination == 'default' else _WestInclination
        self._INCLINATION = inclination_class(code.LONG_AXIS)
        super().__init__(code)
        self.reset()

    def __repr__(self):
        return f'UF(code={self.CODE}, syndrome={self.syndrome})'
    
    def reset(self, no_boundaries=False):
        """Factory reset.
        
        
        :param no_boundaries: whether to treat all boundary nodes as detectors too.
        """
        super().reset()
        self.syndrome = set()
        # note: following line is faster than dict(zip(nodes, nodes))
        self.parents = {v: v for v in self.CODE.NODES}
        self.clusters = {v: _Cluster(self, v, no_boundaries=no_boundaries) for v in self.CODE.NODES}
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
            draw=False,
            fig_width: float | None = None,
    ):
        """Additional inputs over ``Decoder.decode()``:
        * ``fig_width`` figure width.
        """
        self.validate(syndrome, log_history=draw)
        self.peel()
        if draw:
            self.draw_decode(fig_width=fig_width)

    def complementary_gap(
            self,
            noise_level: None | float = None,
            draw=False,
            fig_width: float | None = None,
            fig_height: float | None = None,
            **kwargs_for_networkx_draw,
    ):
        """Calculate the complementary gap after decoding.
        
        Assumes ``self.decode()`` has already been called.
        If ``self.CODE.MERGED_EQUIVALENT_BOUNDARY_NODES`` is ``False``,
        raises a ``ValueError``.
        
        
        :param noise_level: a probability representing the noise strength.
            This is needed to define nonuniform edge weights of the decoding graph
        in the circuit-level noise model.
        If ``None``, all edges are assumed to have weight 1.
        :param draw: whether to draw the original and complementary corrections.
        :param fig_width: figure width.
        :param fig_height: figure height.
        :param kwargs_for_networkx_draw: passed to ``networkx.draw()``.
        
        
        :returns: ``weight_2 - weight_1`` the complementary gap.
        
        Side effects:
        * ``self`` ends up in the final state after computing the complementary correction.
        * All clusters are blind to boundary nodes
            so will stop growing only once they have an odd defect count.
        """
        if not self.CODE.MERGED_EQUIVALENT_BOUNDARY_NODES:
            raise ValueError('Complementary gap requires all equivalent boundary nodes in the decoding graph be merged.')
        if draw:
            from matplotlib import pyplot as plt
            if fig_width is None: fig_width = self._FIG_WIDTH
            if fig_height is None: fig_height = self._FIG_HEIGHT
            plt.figure(figsize=(2*fig_width, fig_height))
            plt.subplot(121)
            self.draw_peel(**kwargs_for_networkx_draw)
        weight_1 = self._weigh_correction(noise_level=noise_level)
        complementary_syndrome = self.CODE.get_verbose_syndrome(
            self.correction).symmetric_difference(self.CODE.BOUNDARY_NODES)
        self.reset(no_boundaries=True)
        self.decode(complementary_syndrome)
        if draw:
            plt.subplot(122)
            self.draw_peel(**kwargs_for_networkx_draw)
            plt.tight_layout()
        weight_2 = self._weigh_correction(noise_level=noise_level)
        return weight_2 - weight_1
    
    def _weigh_correction(self, noise_level: None | float = None) -> float:
        """Weigh ``self.correction``.
        
        
        :param noise_level: a probability representing the noise strength.
            This is needed to define nonuniform edge weights of the decoding graph
        in the circuit-level noise model.
        If ``None``, all edges are assumed to have weight 1.
        
        
        :returns: The sum of the weights of the edges in ``self.correction``.
        """
        edge_weights = self.CODE.NOISE.get_edge_weights(noise_level)
        return sum(edge_weights[e][1] for e in self.correction)
    
    def unclustered_node_fraction(self):
        """Compute the Unclustered Node Fraction DCS.
        
        
        :returns: ``fraction`` the fraction of nodes that are not in a cluster.
        """
        clustered_node_count = sum(cluster.size
            for cluster in self.clusters.values() if cluster.size > 1)
        return 1 - clustered_node_count/self.CODE.NODE_COUNT

    # SYNDROME VALIDATION

    def validate(
            self,
            syndrome: set[Node],
            log_history=False,
    ):
        """Grow all active clusters until they are inactive."""
        self.load(syndrome)
        if log_history: self.init_history()
        while self.active_clusters:
            self._growth_round(log_history)

    def _growth_round(
            self,
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
        for (u, v) in merge_ls:
            cu = self.clusters[self._find(u)]
            cv = self.clusters[self._find(v)]
            # remove edge (u, v) from all visions
            cu.vision.remove((u, v))
            cv.vision.discard((u, v))  # discard instead of remove as cu may be cv
            self._FORESTER.merge(cu, cv, (u, v))
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
        """Find root of ``v`` (w/ path compression).
        
        From https://www.geeksforgeeks.org/union-by-rank-and-path-compression-in-union-find-algorithm/.
        """
        if self.parents[v] != v:
            self.parents[v] = self._find(self.parents[v])
        return self.parents[v]
    
    def static_merge(self, cu: '_Cluster', cv: '_Cluster'):
        """Merge clusters ``cu`` and ``cv`` along edge ``uv`` for static UF."""
        if cu != cv:
            self._union(cu, cv)

    def dynamic_merge(self, cu: '_Cluster', cv: '_Cluster', e: Edge):
        """Merge clusters ``cu`` and ``cv`` along edge ``uv`` for dynamic UF."""
        if cu == cv or (cu.boundary and cv.boundary):
            self.growth[e] = Growth.BURNT
        else:
            self._union(cu, cv)

    def _union(self, cu: '_Cluster', cv: '_Cluster'):
        """Union clusters ``cu`` with ``cv`` (by weight)."""
        if cu.size >= cv.size:
            larger, smaller = cu, cv
        else:
            larger, smaller = cv, cu
        self.parents[smaller.root] = larger.root
        larger.size += smaller.size
        larger.odd ^= smaller.odd  # logical XOR
        old_larger_vision_length = len(larger.vision) + 1  # +1 as we already removed (u, v) from all visions
        larger.vision.update(smaller.vision)
        self._INCLINATION.update_boundary(larger, smaller)
        self._update_self_after_union(larger, smaller, old_larger_vision_length)

    def _update_self_after_union(
            self,
            larger: '_Cluster',
            smaller: '_Cluster',
            old_larger_vision_length: int,
    ):
        """Update attributes ``clusters`` and ``active_clusters`` after union of larger w/ smaller."""

        # DELETE SMALLER CLUSTER
        del self.clusters[smaller.root]
        # discard instead of remove as smaller may be a node not in syndrome
        self.active_clusters.discard(smaller)

        # UPDATE `active_clusters` SET
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
        for e, u in reversed(self.forest):
            if u in modified_syndrome:
                self.correction.add(e)
                modified_syndrome.symmetric_difference_update(e)
    
    def _make_forest(self):
        """Return a spanning forest of the validated erasure.
        
        
        :returns: ``forest`` a list of pairs (edge, node) where the node is the further endpoint from the tree root.
        """
        forest: list[tuple[Edge, Node]] = []
        for cluster in self.clusters.values():
            tree_root = cluster.boundary if cluster.boundary else cluster.root
            forest += self._make_tree(tree_root)
        return forest

    def _make_tree(self, tree_root: Node):
        """Return a spanning tree of the validated cluster, from ``tree_root``.
        
        
        :returns: ``tree`` a list of pairs (edge, node) where the node is the further endpoint from the root.
        """
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
        
        Note: Requires calling ``peel()`` first.
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
        **kwargs_for_networkx_draw,
    ):
        """Draw spanning forest.
        
        ``kwargs_for_networkx_draw`` passed to ``networkx.draw()``.
        
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
            **kwargs_for_networkx_draw,
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
        **kwargs_for_networkx_draw,
    ):
        """Draw forest and correction from peeling.
        
        ``kwargs_for_networkx_draw`` passed to ``networkx.draw()``.
        
        Note: Requires calling ``peel()`` first.
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
            **kwargs_for_networkx_draw,
        )

    def draw_decode(
            self,
            fig_width: float | None = None,
            fig_height: float | None = None,
            **kwargs_for_networkx_draw,
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
                **kwargs_for_networkx_draw,
            )
        plt.subplot(n_plots, 1, n_plots)
        self.draw_peel(**kwargs_for_networkx_draw)
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


class BaseCluster(abc.ABC):
    """Abstract base class for the cluster in UF.
    
    Mathematically, a cluster is a connected subgraph of the decoding graph.
    
    Attributes:
    * ``root`` the representative of the cluster.
    * ``size`` the number of nodes in the cluster.
    * ``odd`` a Boolean to say if the number of defects in the cluster is odd.
    * ``boundary`` the node in the cluster that is the surface boundary if it exists; else, None.
        If ``cluster.boundary`` not None, cluster will never be active in future.
    If syndrome validation not dynamic, boundary may not be unique!
    """

    def __init__(self, uf: UF, root: Node, no_boundaries=False):
        self.root = root
        self.size = 1
        self.odd = False
        self.boundary = root if (not no_boundaries and uf.CODE.is_boundary(root)) else None

    def __str__(self) -> str:
        return f'cluster(root={self.root}, size={self.size}, odd={self.odd})'


class _Cluster(BaseCluster):
    """A UF cluster that tracks which edges to grow next.
    
    Extends ``BaseCluster``.
    
    Additional attributes:
    * ``vision`` a set of active edges incident to the cluster
        i.e. edges to be grown by the cluster in next growth round.
    """

    def __init__(self, uf: UF, root: Node, no_boundaries=False):
        super().__init__(uf, root, no_boundaries=no_boundaries)
        self.vision = uf.CODE.INCIDENT_EDGES[root].copy()
    

class _Inclination(abc.ABC):
    """Decides which boundary the root should be on when there is a percolating cluster."""

    def __init__(self, long_axis: int):
        self._LONG_AXIS = long_axis

    @abc.abstractmethod
    def update_boundary(self, larger: BaseCluster, smaller: BaseCluster):
        """Update the ``boundary`` attribute of the cluster ``larger``."""


class _DefaultInclination(_Inclination):
    """Prefer neither east nor west."""

    def update_boundary(self, larger: BaseCluster, smaller: BaseCluster):
        if (not larger.boundary) and smaller.boundary:
            larger.boundary = smaller.boundary


class _WestInclination(_Inclination):
    """Define the root of a percolating cluster at the west boundary."""

    def update_boundary(self, larger: BaseCluster, smaller: BaseCluster):
        if smaller.boundary and (
            (not larger.boundary) or
            (smaller.boundary[self._LONG_AXIS] < larger.boundary[self._LONG_AXIS])
        ):
            larger.boundary = smaller.boundary


class _Forester(abc.ABC):
    """Decides whether to maintain a static or dynamic forest.
    
    See the ``dynamic`` kwarg of ``UF.__init__`` for details.
    """

    def __init__(self, uf: UF):
        self.UF = uf

    @abc.abstractmethod
    def merge(self, cu: _Cluster, cv: _Cluster, e: Edge):
        """Merge clusters ``cu`` and ``cv`` along edge ``e``."""


class _StaticForester(_Forester):
    """Maintains a static forest.
    
    Extends ``_Forester``.
    """

    def merge(self, cu, cv, e):
        self.UF.static_merge(cu, cv)


class _DynamicForester(_Forester):
    """Maintains a dynamic forest.
    
    Extends ``_Forester``.
    """

    def merge(self, cu, cv, e):
        self.UF.dynamic_merge(cu, cv, e)