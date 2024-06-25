from localuf.decoders import NodeBUF


def test_validate_static(
        node_buf: NodeBUF,
        validated_static_erasure,
        syndrome7F,
):
    node_buf.validate(syndrome7F, dynamic=False)
    assert len(node_buf.clusters) == 26
    assert all(bucket == set() for bucket in node_buf.buckets.values())
    assert node_buf.erasure == validated_static_erasure


def test_validate_dynamic(
        node_buf: NodeBUF,
        validated_dynamic_erasure,
        syndrome7F,
):
    node_buf.validate(syndrome7F, dynamic=True)
    assert len(node_buf.clusters) == 27
    assert all(bucket == set() for bucket in node_buf.buckets.values())
    assert node_buf.erasure == validated_dynamic_erasure