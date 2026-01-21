from localuf.decoders.node_uf import _NodeCluster, NodeUF
from localuf.type_aliases import Node


class NodeBUF(NodeUF):
    """The graph used by Bucket UF decoder in Huang2020.
    
    Extends ``decoders.NodeUF``.
    
    Overriden attributes:
    * ``buckets`` replaces ``active_clusters``
        and is a dictionary where each key a cluster size;
    value, a set of all active clusters of that size.
    
    Overriden methods:
    * ``reset``.
    * ``load``.
    * ``validate_syndrome``.
    * ``_update_self_after_union``.
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
        log_history=False,
    ):
        """Grow clusters (always smallest first) until they are all valid."""
        self.load(syndrome)
        if log_history: self.init_history()
        for bucket in self.buckets.values():
            while bucket:
                self._growth_round(log_history, clusters_to_grow=bucket)

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