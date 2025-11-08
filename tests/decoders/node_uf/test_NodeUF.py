from collections.abc import Callable
import itertools

import pytest

from localuf.type_aliases import Node
from localuf.constants import Growth
from localuf.decoders.node_uf import NodeUF, _NodeCluster

@pytest.fixture
def uf7F(sf7F):
    uf_graph = NodeUF(sf7F)
    uf_graph.history = []
    return uf_graph

@pytest.fixture
def dynamic_uf7F(sf7F):
    uf_graph = NodeUF(sf7F, dynamic=True)
    uf_graph.history = []
    return uf_graph

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
    uf7F: NodeUF,
    get_uf_after_union: Callable[[NodeUF], tuple[NodeUF, _NodeCluster, _NodeCluster]],
    syndrome7F: set[Node],
):
    uf7F.load(syndrome7F)
    return get_uf_after_union(uf7F)

def test_validate_static(
        uf7F: NodeUF,
        validated_static_erasure,
        syndrome7F,
):
    uf7F.validate(syndrome7F)
    assert len(uf7F.clusters) == 26
    assert uf7F.active_clusters == set()
    assert uf7F.erasure == validated_static_erasure

def test_validate_dynamic(
        dynamic_uf7F: NodeUF,
        validated_dynamic_erasure,
        syndrome7F,
):
    dynamic_uf7F.validate(syndrome7F)
    assert len(dynamic_uf7F.clusters) == 27
    assert dynamic_uf7F.active_clusters == set()
    assert dynamic_uf7F.erasure == validated_dynamic_erasure

def test_growth_round_static(
        uf7F: NodeUF,
        syndrome7F,
        first_changed_edges,
):
    uf7F.load(syndrome7F)
    log_history = True
    uf7F._growth_round(log_history)
    assert uf7F.history[0].changed_edges == first_changed_edges
    while uf7F.active_clusters:
        assert all(v in {
            Growth.UNGROWN,
            Growth.HALF,
            Growth.FULL
        } for v in uf7F.growth.values())
        uf7F._growth_round(log_history)

def test_growth_round_dynamic(
        dynamic_uf7F: NodeUF,
        syndrome7F,
        first_changed_edges,
):
    dynamic_uf7F.load(syndrome7F)
    log_history = True
    dynamic_uf7F._growth_round(log_history)
    assert dynamic_uf7F.history[0].changed_edges == first_changed_edges
    while dynamic_uf7F.active_clusters:
        assert all(v in {
            Growth.BURNT,
            Growth.UNGROWN,
            Growth.HALF,
            Growth.FULL
        } for v in dynamic_uf7F.growth.values())
        dynamic_uf7F._growth_round(log_history)

def test_grow(uf7F: NodeUF, fixture_test_grow):
    sq, c = fixture_test_grow(uf7F)

    # test ex-frontier nodes removed
    for _ in itertools.repeat(None, 2):
        uf7F._growth_round(
            log_history=False,
            clusters_to_grow={c}
        )
        c = uf7F.clusters[uf7F._find(sq)]
    assert c.frontier == {
        (1, 0),
        (2, 1),
        (3, 0),
    }

def test_union(uf_after_union, fixture_test_union):
    fixture_test_union(uf_after_union)
    _, larger, smaller = uf_after_union
    assert larger.frontier == {smaller.root}