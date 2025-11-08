import pytest

from localuf.codes import Surface
from localuf.decoders.uf import _DefaultInclination


@pytest.fixture
def default_inclination(sf: Surface):
    return _DefaultInclination(sf.LONG_AXIS)


def test_update_boundary(two_clusters, default_inclination: _DefaultInclination):
    larger, smaller = two_clusters
    east, west = larger.root, smaller.root
    
    larger.boundary = None
    smaller.boundary = None
    default_inclination.update_boundary(larger, smaller)
    assert larger.boundary is None
    assert smaller.boundary is None
    
    larger.boundary = east
    default_inclination.update_boundary(larger, smaller)
    assert larger.boundary == east
    assert smaller.boundary is None
    
    larger.boundary = None
    smaller.boundary = west
    default_inclination.update_boundary(larger, smaller)
    assert larger.boundary == west
    assert smaller.boundary == west

    larger.boundary = east
    default_inclination.update_boundary(larger, smaller)
    assert larger.boundary == east
    assert smaller.boundary == west