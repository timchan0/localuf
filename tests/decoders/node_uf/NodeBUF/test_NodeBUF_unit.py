from unittest import mock

import pytest

from localuf.decoders.uf import _Cluster
from localuf.decoders import NodeBUF

@pytest.fixture
def node_buf_after_union(
    node_buf: NodeBUF,
    get_uf_after_union,
    syndrome7F,
) -> tuple[NodeBUF, _Cluster, _Cluster]:
    node_buf.load(syndrome7F)
    return get_uf_after_union(node_buf)

def test_reset(node_buf: NodeBUF):
    with mock.patch("localuf.decoders.node_uf.NodeUF.reset") as super_reset:
        node_buf.reset()
        super_reset.assert_called_once_with()
    # test correct number of buckets
    assert len(node_buf.buckets) == len(node_buf.CODE.NODES)
    # test all buckets empty
    assert all(bucket == set() for bucket in node_buf.buckets.values())

def test_load(node_buf, syndrome7F):
    with mock.patch("localuf.decoders.node_uf.NodeUF.load") as super_load:
        node_buf.load(syndrome7F)
        super_load.assert_called_once_with(syndrome7F)
    # test bucket 1 initially has all defects
    assert {cluster.root for cluster in node_buf.buckets[1]} == node_buf.syndrome
    # test active_clusters attribute replaced by buckets
    assert not hasattr(node_buf, 'active_clusters')

def test_validate(node_buf, syndrome7F):

    bucket = {_Cluster(node_buf, (0, 0))}
    node_buf.buckets = {1: bucket}

    def fake_growth_round(log_history, clusters_to_grow: set[_Cluster]):
        clusters_to_grow.clear()

    with (
        mock.patch("localuf.decoders.node_buf.NodeBUF.load") as mock_load,
        mock.patch(
            "localuf.decoders.node_buf.NodeBUF._growth_round",
            side_effect=fake_growth_round,
        ) as mock_growth_round,
    ):
        node_buf.validate(syndrome7F)
        mock_load.assert_called_once_with(syndrome7F)
        mock_growth_round.assert_called_once_with(
            False,
            clusters_to_grow=bucket,
        )

def test_update_self_after_union(
        node_buf_after_union,
        fixture_test_update_self_after_union,
):
    fixture_test_update_self_after_union(node_buf_after_union)