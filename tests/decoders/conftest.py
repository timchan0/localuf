import pytest

from localuf.type_aliases import Node
from localuf.decoders import UF, BUF, NodeBUF
from localuf.decoders.uf import _Cluster
from localuf.constants import Growth
from localuf.type_aliases import Node

@pytest.fixture
def v00():
    return 0, 0

@pytest.fixture
def v000():
    return 0, 0, 0

@pytest.fixture
def syndrome7F() -> set[Node]:
    return {
        (0, 0),
        (0, 1),
        (2, 0),
        (0, 4),
        (1, 4),
        (3, 4),
        (4, 4),
        (4, 3),
    }

@pytest.fixture
def fixture_test_update_self_after_union():
    def f(uf_after_union: tuple[BUF | NodeBUF, _Cluster, _Cluster]):
        buf, larger, smaller = uf_after_union
        # check smaller cluster deleted
        assert smaller.root not in buf.clusters
        assert all(smaller not in bucket for bucket in buf.buckets.values())
        # check larger cluster in clusters but not in buckets
        assert larger.root in buf.clusters
        assert all(larger not in bucket for bucket in buf.buckets.values())
    return f

@pytest.fixture
def fixture_test_grow():
    def f(uf: UF):
        
        # setup
        sq = (2, 0)
        c = uf.clusters[sq]
        sq_incident_edges = {
            ((1, 0), sq),
            ((2, -1), sq),
            (sq, (2, 1)),
            (sq, (3, 0))
        }
        merge_ls, changed_edges = [], set()

        # test core functionality
        uf._grow(c, merge_ls, changed_edges)
        assert merge_ls == []
        assert changed_edges == sq_incident_edges
        assert all(uf.growth[e] is Growth.HALF for e in sq_incident_edges)
        return sq, c
    return f


@pytest.fixture
def uvw() -> tuple[Node, Node, Node]:
    u, v, w = (0, 0), (0, 1), (0, 2)
    return u, v, w


@pytest.fixture
def fixture_test_union():
    def f(uf_after_union: tuple[BUF | NodeBUF, _Cluster, _Cluster]):
        uf, larger, smaller = uf_after_union
        assert uf.parents[smaller.root] == larger.root
        assert larger.size == 2
        assert larger.odd
        assert larger.boundary == larger.root
    return f