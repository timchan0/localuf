import pytest

from localuf.decoders.uf import UF
from localuf.constants import Growth


def test_reset(uf7F: UF):  # should be BaseUF
    for e in uf7F.growth.keys():
        uf7F.growth[e] = Growth.FULL
    uf7F.erasure
    uf7F.reset()
    assert all(growth == 0 for growth in uf7F.growth.values())
    with pytest.raises(AttributeError, match="erasure"):
        del uf7F.erasure


def test_erasure_property(uf7F: UF, uvw):
    u, v, w = uvw
    assert uf7F.erasure == set()
    uf7F.growth[u, v] = Growth.HALF
    uf7F.growth[v, w] = Growth.FULL
    # erasure a cached_property so need delete attribute to run method again
    del uf7F.erasure
    assert uf7F.erasure == {(v, w)}