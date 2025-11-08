from unittest import mock

import pytest

from localuf.decoders import BUF
from localuf.decoders.uf import _Cluster

@pytest.fixture
def buf_after_union(
    buf: BUF,
    get_uf_after_union,
    syndrome7F,
) -> tuple[BUF, _Cluster, _Cluster]:
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
        mock.patch("localuf.decoders.buf.BUF._update_mvl") as mock_update_mvl,
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

def test_validate(buf: BUF, syndrome7F):

    def set_mvl(mvl: None | int = None):
        buf.mvl = mvl

    not_none = 1
    set_mvl(not_none)
    with (
        mock.patch("localuf.decoders.buf.BUF.load") as mock_load,
        mock.patch("localuf.decoders.buf.BUF._growth_round") as mock_growth_round,
        mock.patch(
            "localuf.decoders.buf.BUF._update_mvl",
            side_effect=set_mvl,
        ) as mock_update_mvl,
    ):
        buf.validate(syndrome7F)
        mock_load.assert_called_once_with(syndrome7F)
        mock_growth_round.assert_called_once_with(
            False,
            clusters_to_grow=buf.buckets[not_none],
        )
        mock_update_mvl.assert_called_once_with()

def test_update_mvl(buf: BUF):
    buf.buckets[3] = {buf.clusters[0, 0]}
    buf._update_mvl()
    assert buf.mvl == 3

def test_update_self_after_union(
        buf_after_union,
        fixture_test_update_self_after_union,
):
    fixture_test_update_self_after_union(buf_after_union)