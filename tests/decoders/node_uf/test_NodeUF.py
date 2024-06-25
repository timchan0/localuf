import itertools

import pytest

from localuf.constants import Growth
from localuf.decoders.node_uf import NodeUF

@pytest.fixture
def uf7F(sf7F):
    uf_graph = NodeUF(sf7F)
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
    get_uf_after_union,
    syndrome7F
):
    uf7F.load(syndrome7F)
    return get_uf_after_union(uf7F)

def test_validate_static(
        uf7F: NodeUF,
        validated_static_erasure,
        syndrome7F,
):
    uf7F.validate(syndrome7F, dynamic=False)
    assert len(uf7F.clusters) == 26
    assert uf7F.active_clusters == set()
    assert uf7F.erasure == validated_static_erasure

def test_validate_dynamic(
        uf7F: NodeUF,
        validated_dynamic_erasure,
        syndrome7F,
):
    uf7F.validate(syndrome7F, dynamic=True)
    assert len(uf7F.clusters) == 27
    assert uf7F.active_clusters == set()
    assert uf7F.erasure == validated_dynamic_erasure

def test_growth_round_static(
        uf7F: NodeUF,
        syndrome7F,
        first_changed_edges,
):
    uf7F.load(syndrome7F)
    dynamic, log_history = False, True
    uf7F._growth_round(dynamic, log_history)
    assert uf7F.history[0].changed_edges == first_changed_edges
    while uf7F.active_clusters:
        assert all(v in {
            Growth.UNGROWN,
            Growth.HALF,
            Growth.FULL
        } for v in uf7F.growth.values())
        uf7F._growth_round(dynamic, log_history)

def test_growth_round_dynamic(
        uf7F: NodeUF,
        syndrome7F,
        first_changed_edges,
):
    uf7F.load(syndrome7F)
    dynamic, log_history = True, True
    uf7F._growth_round(dynamic, log_history)
    assert uf7F.history[0].changed_edges == first_changed_edges
    while uf7F.active_clusters:
        assert all(v in {
            Growth.BURNT,
            Growth.UNGROWN,
            Growth.HALF,
            Growth.FULL
        } for v in uf7F.growth.values())
        uf7F._growth_round(dynamic, log_history)

def test_grow(uf7F: NodeUF, fixture_test_grow):
    sq, c = fixture_test_grow(uf7F)

    # test nonboundaries removed
    for _ in itertools.repeat(None, 2):
        uf7F._growth_round(
            dynamic=False,
            log_history=False,
            clusters_to_grow={c}
        )
        c = uf7F.clusters[uf7F._find(sq)]
    assert c.boundaries == {
        (1, 0),
        (2, 1),
        (3, 0),
    }

def test_union(uf_after_union, fixture_test_union):
    fixture_test_union(uf_after_union)
    _, larger, smaller = uf_after_union
    assert larger.boundaries == {smaller.root}