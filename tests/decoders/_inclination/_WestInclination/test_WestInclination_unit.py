import pytest

from localuf.codes import Surface
from localuf.decoders.uf import _WestInclination


@pytest.fixture
def west_inclination(sf: Surface):
    return _WestInclination(sf.LONG_AXIS)


def test_update_boundary(two_clusters, west_inclination: _WestInclination):
    larger, smaller = two_clusters
    east, west = larger.root, smaller.root
    
    larger.boundary = None
    smaller.boundary = None
    west_inclination.update_boundary(larger, smaller)
    assert larger.boundary is None
    assert smaller.boundary is None
    
    larger.boundary = east
    west_inclination.update_boundary(larger, smaller)
    assert larger.boundary == east
    assert smaller.boundary is None
    
    larger.boundary = None
    smaller.boundary = west
    west_inclination.update_boundary(larger, smaller)
    assert larger.boundary == west
    assert smaller.boundary == west

    larger.boundary = east
    west_inclination.update_boundary(larger, smaller)
    assert larger.boundary == west
    assert smaller.boundary == west