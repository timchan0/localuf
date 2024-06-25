from localuf.decoders import BUF


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