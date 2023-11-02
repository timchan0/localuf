import pytest

from localuf.constants import Growth
from localuf.decoders.uf import UF, _Cluster

@pytest.fixture
def first_changed_edges():
    return {
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 1)),
        ((0, 3), (0, 4)),
        ((0, 4), (0, 5)),
        ((0, 4), (1, 4)),
        ((1, 0), (2, 0)),
        ((1, 3), (1, 4)),
        ((1, 4), (1, 5)),
        ((1, 4), (2, 4)),
        ((2, -1), (2, 0)),
        ((2, 0), (2, 1)),
        ((2, 0), (3, 0)),
        ((2, 4), (3, 4)),
        ((3, 3), (3, 4)),
        ((3, 3), (4, 3)),
        ((3, 4), (3, 5)),
        ((3, 4), (4, 4)),
        ((4, 2), (4, 3)),
        ((4, 3), (4, 4)),
        ((4, 3), (5, 3)),
        ((4, 4), (4, 5)),
        ((4, 4), (5, 4))
    }

@pytest.fixture
def uf_after_union(
    uf7F: UF,
    get_uf_after_union,
    syndrome7F
) -> tuple[UF, _Cluster, _Cluster]:
    uf7F.load(syndrome7F)
    return get_uf_after_union(uf7F)

def test_validate_static(uf7F: UF, syndrome7F):
    uf7F.validate(syndrome7F)
    assert len(uf7F.clusters) == 26
    assert uf7F.active_clusters == set()
    # test erasure
    covered_nodes = set()
    for e in uf7F.erasure:
        covered_nodes.update(e)
    assert uf7F.syndrome.issubset(covered_nodes)
    assert len(uf7F.erasure) == 38

def test_validate_dynamic(uf7F: UF, syndrome7F):
    uf7F.validate(syndrome7F, dynamic=True)
    assert len(uf7F.clusters) == 27
    assert uf7F.active_clusters == set()
    # test erasure
    covered_nodes = set()
    for e in uf7F.erasure:
        covered_nodes.update(e)
    assert uf7F.syndrome.issubset(covered_nodes)
    assert len(uf7F.erasure) == 29

def test_growth_round_static(
        uf7F: UF,
        syndrome7F,
        first_changed_edges,
):
    uf7F.load(syndrome7F)
    dynamic, log_history = False, True
    uf7F._growth_round(dynamic, log_history)
    assert uf7F.history[0].changed_edges == first_changed_edges
    c = uf7F.clusters[uf7F._find((0, 0))]
    # test inactive edges removed
    assert c.vision == {
        ((0, -1), (0, 0)),
        ((0, 1), (0, 2)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1))
    }
    while uf7F.active_clusters:
        assert all(val in {
            Growth.UNGROWN,
            Growth.HALF,
            Growth.FULL
        } for val in uf7F.growth.values())
        uf7F._growth_round(dynamic, log_history)

def test_growth_round_dynamic(
        uf7F: UF,
        syndrome7F,
        first_changed_edges,
):
    uf7F.load(syndrome7F)
    dynamic, log_history = True, True
    uf7F._growth_round(dynamic, log_history)
    assert uf7F.history[0].changed_edges == first_changed_edges
    c = uf7F.clusters[uf7F._find((0, 0))]
    # test inactive edges removed
    assert c.vision == {
        ((0, -1), (0, 0)),
        ((0, 1), (0, 2)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1))
    }
    while uf7F.active_clusters:
        assert all(v in {
            Growth.BURNT,
            Growth.UNGROWN,
            Growth.HALF,
            Growth.FULL
        } for v in uf7F.growth.values())
        uf7F._growth_round(dynamic, log_history)

def test_grow(uf7F: UF, fixture_test_grow):
    fixture_test_grow(uf7F)

def test_find(uf7F, uvw):
    # setup
    u, v, w = uvw
    uf7F.parents[u] = v
    uf7F.parents[v] = w
    # test output
    assert uf7F._find(u) == w
    # test path compression
    assert uf7F.parents[u] == w

def test_union(uf_after_union, fixture_test_union):
    fixture_test_union(uf_after_union)

def test_update_self_after_union(uf_after_union):
    uf, larger, smaller = uf_after_union
    # check smaller cluster deleted
    assert smaller.root not in uf.clusters
    assert smaller not in uf.active_clusters
    # check larger cluster in clusters but not in active_clusters
    assert larger.root in uf.clusters
    assert larger not in uf.active_clusters

def test_peel(uf7F: UF, syndrome7F):
    uf7F.validate(syndrome7F)
    uf7F.peel()
    syndrome_from_correction = uf7F.CODE.get_syndrome(uf7F.correction)
    assert syndrome_from_correction == uf7F.syndrome

def test_make_forest(uf7F: UF, syndrome7F):
    uf7F.validate(syndrome7F, dynamic=True)
    forest = uf7F._make_forest()
    assert len(forest) == len(uf7F.erasure)

def test_make_tree(uf7F: UF, syndrome7F):
    uf7F.validate(syndrome7F)
    assert uf7F._make_tree((0, 2)) == []
    assert uf7F._make_tree((0, 0)) == [(((0, 0), (0, 1)), (0, 1))]
    tree = uf7F._make_tree((2, -1))
    assert tree[0] == (((2, -1), (2, 0)), (2, 0))
    assert set(tree[1:]) == {
        (((1, 0), (2, 0)), (1, 0)),
        (((2, 0), (2, 1)), (2, 1)),
        (((2, 0), (3, 0)), (3, 0))
    }
    n_nodes_in_big_cluster = 26
    assert len(uf7F._make_tree((4, 6))) == n_nodes_in_big_cluster - 1