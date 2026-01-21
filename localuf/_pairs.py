from localuf.type_aliases import Edge, Node


class Pairs:
    """A set of node pairs defining free anyon strings.
    
    In form of a dictionary where there are two entries, u:v and v:u, per pair uv.
    """

    def __repr__(self) -> str:
        return str(self.as_set)

    @property
    def as_set(self):
        """Return the set of pairs."""
        set_: set[Edge] = set()
        visited: set[Node] = set()
        for u, v in self._dc.items():
            if u not in visited:
                set_.add((u, v))
                visited.add(v)
        return set_

    def __init__(self) -> None:
        self._dc: dict[Node, Node] = {}

    def reset(self):
        """Factory reset."""
        self._dc.clear()

    def __contains__(self, u: Node):
        """Return whether u is in the set."""
        return u in self._dc

    def __getitem__(self, u: Node):
        """Return v if uv is in the set."""
        return self._dc[u]

    def add(self, u: Node, v: Node):
        """Add pair uv."""
        self._dc[u] = v
        self._dc[v] = u

    def remove(self, u: Node):
        """Remove pair containing u."""
        v = self._dc.pop(u)
        del self._dc[v]

    def load(self, e: Edge):
        """Load in edge ``e``."""
        u, v = e
        if u in self:
            w = self[u]
            self.remove(u)
            if v in self:
                x = self[v]
                self.remove(v)
                self.add(w, x)
            elif v != w:
                self.add(v, w)
        elif v in self:
            x = self[v]
            self.remove(v)
            self.add(u, x)
        else:
            self.add(u, v)


class LogicalCounter:
    """Counts logical error strings in, and updates, ``Pairs`` instance."""

    def __init__(
            self,
            d: int,
            commit_height: int,
            long_axis: int,
            time_axis: int,
    ) -> None:
        self._D = d
        self._COMMIT_HEIGHT = commit_height
        self._LONG_AXIS = long_axis
        self._TIME_AXIS = time_axis

    def _lower_node(self, v: Node) -> Node:
        """Move ``v`` down by commit height."""
        new_v = list(v)
        new_v[self._TIME_AXIS] -= self._COMMIT_HEIGHT
        return tuple(new_v)

    def count(self, pairs: Pairs):
        """Count logical error strings in, and update ``pairs``.
        
        
        ``pairs`` a set of node pairs defining free anyon strings.
        
        
        :returns:
        * ``error_count`` the number of logical error strings completed in ``pairs``.
        * ``new_pairs`` the error strings in ``pairs`` ending at the future boundary of the commit region, lowered by commit height.
        """
        error_count: int = 0
        new_pairs = Pairs()
        for u, v in pairs.as_set:
            pair_separation = abs(u[self._LONG_AXIS] - v[self._LONG_AXIS])
            if pair_separation == self._D:
                error_count += 1
            elif not (pair_separation == 0 and u[self._LONG_AXIS] in {-1, self._D-1}):
                new_pairs.load((
                    self._lower_node(u),
                    self._lower_node(v),
                ))
        return error_count, new_pairs