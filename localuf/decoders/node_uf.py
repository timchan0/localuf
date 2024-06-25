from localuf.type_aliases import Node, Edge
from localuf.decoders import uf
from localuf.constants import Growth

class NodeUF(uf.UF):
    """The graph used by original UF decoder.

    Extends `UF` class.
    Overriden attributes:
    * `clusters` and `active_clusters` now use `_NodeCluster` rather than `_Cluster`.

    Overriden methods:
    * `reset`.
    * `_growth_round`.
    * `_grow`.
    * `_union`.
    """

    def reset(self):
        super().reset()
        self.clusters = {v: _NodeCluster(self, v) for v in self.CODE.NODES}
        self.active_clusters: set[_NodeCluster] = set()

    def _growth_round(
            self,
            dynamic,
            log_history,
            clusters_to_grow: 'set[_NodeCluster] | None' = None,
    ):
        if clusters_to_grow is None:
            clusters_to_grow = self.active_clusters
        merge_ls: list[Edge] = []
        changed_edges: set[Edge] = set()
        for cluster in clusters_to_grow:
            self._grow(cluster, merge_ls, changed_edges)
        for u, v in merge_ls:
            cu = self.clusters[self._find(u)]
            cv = self.clusters[self._find(v)]
            self._merge(cu, cv, (u, v), dynamic)
        if log_history:
            self.changed_edges = changed_edges
            self.append_history()

    def _grow(
            self,
            cluster: '_NodeCluster',
            merge_ls,
            changed_edges,
    ):
        nonboundaries = set()
        for v in cluster.boundaries:

            edges_to_grow = {e for e in self.CODE.INCIDENT_EDGES[v]
                if self.growth[e] in self._ACTIVE_GROWTH_VALUES}
            changed_edges.update(edges_to_grow)
            # if node no longer on cluster boundary, mark to remove
            if not edges_to_grow:
                nonboundaries.add(v)

            for e in edges_to_grow:
                self.growth[e] += Growth.INCREMENT
                if self.growth[e] is Growth.FULL:
                    merge_ls.append(e)

        # remove nonbounary nodes from boundary set
        cluster.boundaries.difference_update(nonboundaries)

    def _union(self, cu: '_NodeCluster', cv: '_NodeCluster'):
        if cu.size >= cv.size:
            larger, smaller = cu, cv
        else:
            larger, smaller = cv, cu
        self.parents[smaller.root] = larger.root
        old_larger_size = larger.size
        larger.size += smaller.size
        larger.odd ^= smaller.odd  # logical XOR
        larger.boundaries.update(smaller.boundaries)
        if (not larger.boundary) and smaller.boundary:
            larger.boundary = smaller.boundary
        self._update_self_after_union(larger, smaller, old_larger_size)


class _NodeCluster(uf._Cluster):
    """A tree data structure to represent a cluster of nodes of original UF.
    
    Overriden attributes:
    * `vision` replaced by `boundaries`.
    * `boundaries` the boundary nodes of the cluster
    i.e. not a surface boundary and incident to >=1 active edge.
    """

    def __init__(self, uf, root: Node):
        super().__init__(uf, root)
        del self.vision
        self.boundaries: set[Node] = set() if uf.CODE.is_boundary(root) else {root}