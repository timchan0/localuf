import pytest

from localuf.codes import Surface
from localuf.decoders._base_uf import BaseUF
from localuf.constants import Growth
from localuf.type_aliases import Node


class ConcreteBaseUF(BaseUF):

    correction = set()
    
    @property
    def _FIG_HEIGHT(self): return 0
    
    @property
    def _FIG_WIDTH(self): return self._FIG_HEIGHT

    def decode(self):
        """Override abstract method."""

    def draw_decode(self):
        """Override abstract method."""


@pytest.fixture
def uf5F(sf5F: Surface):
    return ConcreteBaseUF(sf5F)


@pytest.fixture
def uf7F(sf7F: Surface):
    return ConcreteBaseUF(sf7F)


@pytest.fixture
def uf3T(sf3T: Surface):
    return ConcreteBaseUF(sf3T)


def test_reset(uf7F: BaseUF):
    for e in uf7F.growth.keys():
        uf7F.growth[e] = Growth.FULL
    uf7F.erasure
    uf7F.reset()
    assert all(growth == 0 for growth in uf7F.growth.values())
    with pytest.raises(AttributeError, match="erasure"):
        del uf7F.erasure


class TestErasureProperty:

    def test_empty(self, uf7F: BaseUF):
        assert uf7F.erasure == set()

    def test_with_full_growth(self, uf7F: BaseUF):
        uf7F.growth[(0, -1), (0, 0)] = Growth.FULL
        assert uf7F.erasure == {((0, -1), (0, 0))}

    def test_with_half_growth(self, uf7F: BaseUF):
        uf7F.growth[(0, -1), (0, 0)] = Growth.HALF
        assert uf7F.erasure == set()

    def test_with_mixed_growth(self, uf7F: BaseUF, uvw: tuple[Node, Node, Node]):
        u, v, w = uvw
        assert uf7F.erasure == set()
        uf7F.growth[u, v] = Growth.HALF
        uf7F.growth[v, w] = Growth.FULL
        # erasure a cached_property so need delete attribute to run method again
        del uf7F.erasure
        assert uf7F.erasure == {(v, w)}


class TestSwimDistance:

    def test_no_clusters_5F(self, uf5F: BaseUF):
        assert uf5F.swim_distance() == uf5F.CODE.D

    @pytest.mark.parametrize("i", range(5))
    @pytest.mark.parametrize("max_j", range(5))
    def test_percolated_5F(self, uf5F: BaseUF, i: int, max_j: int):
        for j in range(-1, max_j-1):
            uf5F.growth[(i, j), (i, j+1)] = Growth.FULL
        assert uf5F.swim_distance() == 5-max_j

    @pytest.mark.parametrize("i", range(5))
    @pytest.mark.parametrize("max_j", range(5))
    def test_half_percolated_5F(self, uf5F: BaseUF, i: int, max_j: int):
        for j in range(-1, max_j-1):
            uf5F.growth[(i, j), (i, j+1)] = Growth.HALF
        assert uf5F.swim_distance() == 5 - max_j/2

    def test_no_clusters_3T(self, uf3T: BaseUF):
        assert uf3T.swim_distance() == uf3T.CODE.D

    @pytest.mark.parametrize("i", range(3))
    @pytest.mark.parametrize("max_j", range(3))
    @pytest.mark.parametrize("t", range(3))
    def test_percolated_3T(self, uf3T: BaseUF, i: int, max_j: int, t: int):
        for j in range(-1, max_j-1):
            uf3T.growth[(i, j, t), (i, j+1, t)] = Growth.FULL
        assert uf3T.swim_distance() == 3-max_j

    @pytest.mark.parametrize("i", range(3))
    @pytest.mark.parametrize("max_j", range(3))
    @pytest.mark.parametrize("t", range(3))
    def test_half_percolated_3T(self, uf3T: BaseUF, i: int, max_j: int, t: int):
        for j in range(-1, max_j-1):
            uf3T.growth[(i, j, t), (i, j+1, t)] = Growth.HALF
        assert uf3T.swim_distance() == 3 - max_j/2