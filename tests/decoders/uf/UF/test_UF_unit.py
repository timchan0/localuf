from unittest import mock

import pytest

from localuf.constants import Growth
from localuf.codes import Surface
from localuf.decoders.uf import UF, _Cluster


@pytest.fixture
def dynamic_uf7F(sf7F: Surface):
    decoder = UF(sf7F, dynamic=True)
    decoder.history = []
    return decoder


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


def test_reset(uf7F: UF):
    uf7F.syndrome = {(0, 0)}
    uf7F.parents = {}
    uf7F.clusters = {}
    del uf7F.active_clusters
    uf7F.changed_edges = set()
    uf7F.forest = []
    with mock.patch('localuf.decoders.uf.BaseUF.reset') as mock_reset:
        uf7F.reset()
        mock_reset.assert_called_once_with()
    assert uf7F.syndrome == set()
    nodes = uf7F.CODE.NODES
    assert uf7F.parents == dict(zip(nodes, nodes))
    assert set(uf7F.clusters.keys()) == set(nodes)
    assert uf7F.active_clusters == set()
    assert not hasattr(uf7F, 'changed_edges')
    assert not hasattr(uf7F, 'forest')


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

def test_validate_dynamic(dynamic_uf7F: UF, syndrome7F):
    dynamic_uf7F.validate(syndrome7F)
    assert len(dynamic_uf7F.clusters) == 27
    assert dynamic_uf7F.active_clusters == set()
    # test erasure
    covered_nodes = set()
    for e in dynamic_uf7F.erasure:
        covered_nodes.update(e)
    assert dynamic_uf7F.syndrome.issubset(covered_nodes)
    assert len(dynamic_uf7F.erasure) == 29

def test_growth_round_static(
        uf7F: UF,
        syndrome7F,
        first_changed_edges,
):
    uf7F.load(syndrome7F)
    log_history = True
    uf7F._growth_round(log_history)
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
        uf7F._growth_round(log_history)

def test_growth_round_dynamic(
        dynamic_uf7F: UF,
        syndrome7F,
        first_changed_edges,
):
    dynamic_uf7F.load(syndrome7F)
    log_history = True
    dynamic_uf7F._growth_round(log_history)
    assert dynamic_uf7F.history[0].changed_edges == first_changed_edges
    c = dynamic_uf7F.clusters[dynamic_uf7F._find((0, 0))]
    # test inactive edges removed
    assert c.vision == {
        ((0, -1), (0, 0)),
        ((0, 1), (0, 2)),
        ((0, 0), (1, 0)),
        ((0, 1), (1, 1))
    }
    while dynamic_uf7F.active_clusters:
        assert all(v in {
            Growth.BURNT,
            Growth.UNGROWN,
            Growth.HALF,
            Growth.FULL
        } for v in dynamic_uf7F.growth.values())
        dynamic_uf7F._growth_round(log_history)

def test_grow(uf7F: UF, fixture_test_grow):
    fixture_test_grow(uf7F)

def test_find(uf7F: UF, uvw):
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

def test_make_forest(dynamic_uf7F: UF, syndrome7F):
    dynamic_uf7F.validate(syndrome7F)
    forest = dynamic_uf7F._make_forest()
    assert len(forest) == len(dynamic_uf7F.erasure)

def test_make_tree(uf7F: UF, syndrome7F):
    uf7F.validate(syndrome7F)
    assert uf7F._make_tree((0, 2)) == []
    assert uf7F._make_tree((0, 0)) == [(((0, 0), (0, 1)), (0, 1))]
    first, *rest = uf7F._make_tree((2, -1))
    assert first == (((2, -1), (2, 0)), (2, 0))
    assert set(rest) == {
        (((1, 0), (2, 0)), (1, 0)),
        (((2, 0), (2, 1)), (2, 1)),
        (((2, 0), (3, 0)), (3, 0))
    }
    n_nodes_in_big_cluster = 26
    assert len(uf7F._make_tree((4, 6))) == n_nodes_in_big_cluster - 1


def test_weigh_correction(uf7F: UF):
    p = 0.5
    with mock.patch('localuf.noise.main._Uniform.get_edge_weights') as mock_get_edge_weights:
        mock_get_edge_weights.return_value = {
            ((0, 0), (0, 1)): (p, 1),
            ((0, 1), (0, 2)): (p, 2),
            ((0, 2), (0, 3)): (p, 3),
            ((0, 3), (0, 4)): (p, 4),
        }
        weight = uf7F._weigh_correction()
        mock_get_edge_weights.assert_called_once_with(None)
        assert weight == 0

        uf7F.correction = {
            ((0, 0), (0, 1)),
            ((0, 1), (0, 2)),
            ((0, 2), (0, 3)),
            ((0, 3), (0, 4)),
        }
        weight = uf7F._weigh_correction()
        assert weight == 10


def test_complementary_gap(uf7F: UF):
    uf7F.correction = set()
    with pytest.raises(ValueError):
        uf7F.complementary_gap()