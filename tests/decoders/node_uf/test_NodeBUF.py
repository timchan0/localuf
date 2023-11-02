from unittest import mock

import pytest

from localuf.decoders.uf import UF, _Cluster
from localuf.decoders.node_uf import NodeBUF

@pytest.fixture
def buf(sf7F):
    buf_graph = NodeBUF(sf7F)
    buf_graph.history = []
    return buf_graph

@pytest.fixture
def buf_after_union(
    buf: NodeBUF,
    get_uf_after_union,
    syndrome7F,
) -> tuple[UF, _Cluster, _Cluster]:
    buf.load(syndrome7F)
    return get_uf_after_union(buf)

def test_reset(buf: NodeBUF):
    with mock.patch("localuf.decoders.node_uf.NodeUF.reset") as super_reset:
        buf.reset()
        super_reset.assert_called_once_with()
    # test correct number of buckets
    assert len(buf.buckets) == len(buf.CODE.NODES)
    # test all buckets empty
    assert all(bucket == set() for bucket in buf.buckets.values())

def test_load(buf, syndrome7F):
    with mock.patch("localuf.decoders.node_uf.NodeUF.load") as super_load:
        buf.load(syndrome7F)
        super_load.assert_called_once_with(syndrome7F)
    # test bucket 1 initially has all defects
    assert {cluster.root for cluster in buf.buckets[1]} == buf.syndrome
    # test active_clusters attribute replaced by buckets
    assert not hasattr(buf, 'active_clusters')

def test_validate_static(
        buf: NodeBUF,
        validated_static_erasure,
        syndrome7F,
):
    buf.validate(syndrome7F, dynamic=False)
    assert len(buf.clusters) == 26
    assert all(bucket == set() for bucket in buf.buckets.values())
    assert buf.erasure == validated_static_erasure

def test_validate_dynamic(
        buf: NodeBUF,
        validated_dynamic_erasure,
        syndrome7F,
):
    buf.validate(syndrome7F, dynamic=True)
    assert len(buf.clusters) == 27
    assert all(bucket == set() for bucket in buf.buckets.values())
    assert buf.erasure == validated_dynamic_erasure

def test_update_self_after_union(
        buf_after_union: NodeBUF,
        fixture_test_update_self_after_union,
):
    fixture_test_update_self_after_union(buf_after_union)