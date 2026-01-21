from localuf.type_aliases import Node, Edge
from localuf.decoders import uf
from localuf.constants import Growth

class NodeUF(uf.UF):
    """The graph used by original UF decoder.
    
    Extends ``UF`` class.
    Overriden attributes:
    * ``clusters`` and ``active_clusters`` now use ``_NodeCluster`` rather than ``_Cluster``.
    
    Overriden methods:
    * ``reset``.
    * ``_growth_round``.
    * ``_grow``.
    * ``_union``.
    """

    def reset(self):
        super().reset()
        self.clusters = {v: _NodeCluster(self, v) for v in self.CODE.NODES}
        self.active_clusters: set[_NodeCluster] = set()

    def _growth_round(
            self,
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
            self._FORESTER.merge(cu, cv, (u, v)) # type: ignore
        if log_history:
            self.changed_edges = changed_edges
            self.append_history()

    def _grow(
            self,
            cluster: '_NodeCluster',
            merge_ls,
            changed_edges,
    ):
        ex_frontier: set[Node] = set()
        for v in cluster.frontier:

            edges_to_grow = {e for e in self.CODE.INCIDENT_EDGES[v]
                if self.growth[e] in self._ACTIVE_GROWTH_VALUES}
            changed_edges.update(edges_to_grow)
            # if node no longer on cluster frontier, mark to remove
            if not edges_to_grow:
                ex_frontier.add(v)

            for e in edges_to_grow:
                self.growth[e] += Growth.INCREMENT
                if self.growth[e] is Growth.FULL:
                    merge_ls.append(e)

        # remove nodes that are no longer on the cluster frontier
        cluster.frontier.difference_update(ex_frontier)

    def _union(self, cu: '_NodeCluster', cv: '_NodeCluster'):
        if cu.size >= cv.size:
            larger, smaller = cu, cv
        else:
            larger, smaller = cv, cu
        self.parents[smaller.root] = larger.root
        old_larger_size = larger.size
        larger.size += smaller.size
        larger.odd ^= smaller.odd  # logical XOR
        larger.frontier.update(smaller.frontier)
        self._INCLINATION.update_boundary(larger, smaller)
        self._update_self_after_union(larger, smaller, old_larger_size) # type: ignore


class _NodeCluster(uf.BaseCluster):
    """A UF cluster that tracks the nodes on its frontier.
    
    Extends ``BaseCluster``.
    
    Additional attributes:
    * ``frontier`` the detectors in the cluster that are incident to >=1 active edge.
    """

    def __init__(self, uf: NodeUF, root: Node):
        super().__init__(uf, root)
        self.frontier: set[Node] = set() if uf.CODE.is_boundary(root) else {root}