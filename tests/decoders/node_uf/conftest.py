import pytest

from localuf.type_aliases import Edge
from localuf.decoders.uf import UF

@pytest.fixture
def validated_static_erasure() -> set[Edge]:
    return {
        ((0, 0), (0, 1)),
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
        ((2, 3), (2, 4)),
        ((2, 3), (3, 3)),
        ((2, 4), (2, 5)),
        ((2, 4), (3, 4)),
        ((2, 5), (3, 5)),
        ((3, 2), (3, 3)),
        ((3, 2), (4, 2)),
        ((3, 3), (3, 4)),
        ((3, 3), (4, 3)),
        ((3, 4), (3, 5)),
        ((3, 4), (4, 4)),
        ((3, 5), (3, 6)),
        ((3, 5), (4, 5)),
        ((4, 1), (4, 2)),
        ((4, 2), (4, 3)),
        ((4, 2), (5, 2)),
        ((4, 3), (4, 4)),
        ((4, 3), (5, 3)),
        ((4, 4), (4, 5)),
        ((4, 4), (5, 4)),
        ((4, 5), (4, 6)),
        ((4, 5), (5, 5)),
        ((5, 2), (5, 3)),
        ((5, 3), (5, 4)),
        ((5, 3), (6, 3)),
        ((5, 4), (5, 5)),
        ((5, 4), (6, 4))
    }

@pytest.fixture
def validated_dynamic_erasure() -> set[Edge]:
    return {
        ((0, 0), (0, 1)),
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
        ((2, 3), (2, 4)),
        ((2, 4), (2, 5)),
        ((2, 4), (3, 4)),
        ((3, 2), (4, 2)),
        ((3, 3), (3, 4)),
        ((3, 4), (3, 5)),
        ((3, 4), (4, 4)),
        ((4, 1), (4, 2)),
        ((4, 2), (4, 3)),
        ((4, 2), (5, 2)),
        ((4, 3), (4, 4)),
        ((4, 3), (5, 3)),
        ((4, 4), (4, 5)),
        ((4, 4), (5, 4)),
        ((4, 5), (4, 6)),
        ((5, 3), (6, 3)),
        ((5, 4), (5, 5)),
        ((5, 4), (6, 4))
    }

@pytest.fixture
def get_uf_after_union():
    def f(uf: UF):
        u, v = (0, -1), (0, 0)
        larger = uf.clusters[u]
        smaller = uf.clusters[v]
        uf._union(larger, smaller)
        return uf, larger, smaller
    return f