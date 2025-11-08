import pytest

from localuf.codes import Surface
from localuf.decoders.uf import UF, _Cluster


@pytest.fixture
def uf3T(sf3T: Surface):
    decoder = UF(sf3T)
    return decoder


@pytest.fixture
def two_clusters(uf3T: UF):
    east = (0, 2, 0)
    west = (0, -1, 0)
    larger = _Cluster(uf3T, east)
    smaller = _Cluster(uf3T, west)
    return larger, smaller