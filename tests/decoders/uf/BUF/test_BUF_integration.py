import pytest

from localuf.codes import Surface
from localuf.decoders import BUF

@pytest.fixture
def dynamic_buf(sf7F: Surface) -> BUF:
    decoder = BUF(sf7F, dynamic=True)
    decoder.history = []
    return decoder


def test_validate_static(
        buf: BUF,
        validated_static_erasure,
        syndrome7F,
):
    buf.validate(syndrome7F)
    assert len(buf.clusters) == 28
    assert all(bucket == set() for bucket in buf.buckets.values())
    assert buf.erasure == validated_static_erasure


def test_validate_dynamic(
        dynamic_buf: BUF,
        validated_dynamic_erasure,
        syndrome7F,
):
    dynamic_buf.validate(syndrome7F)
    assert len(dynamic_buf.clusters) == 29
    assert all(bucket == set() for bucket in dynamic_buf.buckets.values())
    assert dynamic_buf.erasure == validated_dynamic_erasure