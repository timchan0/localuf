import pytest

from localuf.decoders import NodeBUF


@pytest.fixture
def dynamic_node_buf(sf7F):
    nbuf = NodeBUF(sf7F, dynamic=True)
    nbuf.history = []
    return nbuf


def test_validate_static(
        node_buf: NodeBUF,
        validated_static_erasure,
        syndrome7F,
):
    node_buf.validate(syndrome7F)
    assert len(node_buf.clusters) == 26
    assert all(bucket == set() for bucket in node_buf.buckets.values())
    assert node_buf.erasure == validated_static_erasure


def test_validate_dynamic(
        dynamic_node_buf: NodeBUF,
        validated_dynamic_erasure,
        syndrome7F,
):
    dynamic_node_buf.validate(syndrome7F)
    assert len(dynamic_node_buf.clusters) == 27
    assert all(bucket == set() for bucket in dynamic_node_buf.buckets.values())
    assert dynamic_node_buf.erasure == validated_dynamic_erasure