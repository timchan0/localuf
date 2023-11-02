import copy

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
            self.changed_edges: set[Edge] = changed_edges
            self.history.append(copy.deepcopy(
                self,
                memo={id(self.history): self.history}
            ))

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

    def __init__(self, uf, root) -> None:
        super().__init__(uf, root)
        del self.vision
        self.boundaries = set() if uf.CODE.is_boundary(root) else {root}


class NodeBUF(NodeUF):
    """The graph used by Bucket UF decoder in Huang2020.
    
    Overriden attributes:
    * `buckets` replaces `active_clusters`
    and is a dictionary where each key a cluster size;
    value, a set of all active clusters of that size.

    Overriden methods:
    * `reset`.
    * `load`.
    * `validate_syndrome`.
    * `_update_self_after_union`.
    """

    def reset(self):
        super().reset()
        d = self.CODE.D
        n_nodes = (d+1) * d
        self.buckets: dict[int, set[_NodeCluster]] = {size: set() for size in range(1, n_nodes+1)}

    def load(self, syndrome: set[Node]):
        super().load(syndrome)
        for cluster in self.active_clusters:
            self.buckets[cluster.size].add(cluster)
        del self.active_clusters

    # SYNDROME VALIDATION

    def validate(
        self,
        syndrome,
        dynamic=False,
        log_history=False,
    ):
        """Grow clusters (always smallest first) until they are all valid."""
        self.load(syndrome)  # TODO: test this is called
        if log_history: self.history: list[NodeBUF] = []
        for bucket in self.buckets.values():
            while bucket:
                self._growth_round(dynamic, log_history, clusters_to_grow=bucket)

    def _update_self_after_union(self, larger: _NodeCluster, smaller: _NodeCluster, old_larger_size: int):
        """Update attributes clusters and buckets after union of larger w/ smaller."""

        # DELETE SMALLER CLUSTER
        del self.clusters[smaller.root]
        # discard instead of remove as smaller may be a node not in syndrome
        self.buckets[smaller.size].discard(smaller)

        # UPDATE buckets DICTIONARY
        # discard instead of remove as larger could have had boundary before union, hence was not active
        self.buckets[old_larger_size].discard(larger)
        if larger.odd and not larger.boundary:
            self.buckets[larger.size].add(larger)