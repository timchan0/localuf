from localuf.constants import Growth
from localuf.decoders.uf import UF, _Cluster
from localuf.type_aliases import Node


class BUF(UF):
    """The graph used by Bucket UF.
    
    Extends ``decoders.UF``.
    
    Attributes:
    * ``mvl`` an integer of the minimum vision length of
        all active clusters at the start of a growth round.
    
    Methods:
    * ``_update_mvl``.
    * ``_demote_cluster``.
    
    Overriden attributes:
    * ``buckets`` replaces ``active_clusters`` and maps each integer
        to a set of all active clusters whose vision length is that integer.
    It is important that its keys is in ascending order
    as it is used by ``_update_mvl()``.
    
    Overriden methods:
    * ``reset``.
    * ``load``.
    * ``validate_syndrome``.
    * ``_merge``.
    * ``_update_self_after_union``.
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
        """Update ``mvl`` attribute."""
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
            log_history=False
    ):
        """Grow clusters (always shortest vision first) until they are all valid."""
        self.load(syndrome)
        if log_history: self.init_history()
        while self.mvl is not None:
            self._growth_round(
                log_history,
                clusters_to_grow=self.buckets[self.mvl]
            )
            self._update_mvl()

    def static_merge(self, cu, cv):
        if cu == cv:
            self._demote_cluster(cu)
        else:
            self._union(cu, cv)

    def dynamic_merge(self, cu, cv, e):
        if cu == cv:
            self.growth[e] = Growth.BURNT
            self._demote_cluster(cu)
        elif (cu.boundary and cv.boundary):
            self.growth[e] = Growth.BURNT
            self._demote_cluster(cu)
            self._demote_cluster(cv)
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