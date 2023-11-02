from unittest import mock

import pytest

from localuf.decoders.uf import BUF

@pytest.fixture
def buf(sf7F):
    buf_graph = BUF(sf7F)
    buf_graph.history = []
    return buf_graph

@pytest.fixture
def buf_after_union(
    buf: BUF,
    get_uf_after_union,
    syndrome7F,
):
    buf.load(syndrome7F)
    return get_uf_after_union(buf)

def test_reset(buf: BUF):
    with mock.patch("localuf.decoders.uf.UF.reset") as super_reset:
        buf.reset()
        super_reset.assert_called_once_with()
    # test correct number of buckets
    assert len(buf.buckets) == buf.CODE.N_EDGES
    # test all buckets empty
    assert all(bucket == set() for bucket in buf.buckets.values())
    # test mvl
    assert buf.mvl is None

def test_load(buf: BUF, syndrome7F):
    with (
        mock.patch("localuf.decoders.uf.UF.load") as super_load,
        mock.patch("localuf.decoders.uf.BUF._update_mvl") as mock_update_mvl,
    ):
        buf.load(syndrome7F)
        super_load.assert_called_once_with(syndrome7F)
        mock_update_mvl.assert_called_once_with()
    # test buckets 3 & 4 initially have all defects
    assert {
        cluster.root for cluster in buf.buckets[3].union(buf.buckets[4])
    } == buf.syndrome
    # test active_clusters attribute replaced by buckets
    assert not hasattr(buf, 'active_clusters')

def test_update_mvl(buf: BUF):
    buf.buckets[3] = {buf.clusters[0, 0]}
    buf._update_mvl()
    assert buf.mvl == 3

def test_validate_static(
        buf: BUF,
        validated_static_erasure,
        syndrome7F,
):
    buf.validate(syndrome7F, dynamic=False)
    assert len(buf.clusters) == 28
    assert all(bucket == set() for bucket in buf.buckets.values())
    assert buf.erasure == validated_static_erasure

def test_validate_dynamic(
        buf: BUF,
        validated_dynamic_erasure,
        syndrome7F,
):
    buf.validate(syndrome7F, dynamic=True)
    assert len(buf.clusters) == 29
    assert all(bucket == set() for bucket in buf.buckets.values())
    assert buf.erasure == validated_dynamic_erasure

def test_update_self_after_union(
        buf_after_union: BUF,
        fixture_test_update_self_after_union,
):
    fixture_test_update_self_after_union(buf_after_union)