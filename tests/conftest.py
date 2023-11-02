import itertools

import pytest

import localuf

@pytest.fixture(name="sf", params=itertools.product(
        range(3, 11, 2),
        (
    'code capacity',
    'phenomenological',
    'circuit-level',
        ),
), ids=lambda x: f"d{x[0]} {x[1]}")
def _sf(request):
    return localuf.Surface(*request.param)

@pytest.fixture
def sf3F():
    return localuf.Surface(3, 'code capacity')

@pytest.fixture
def sf3T():
    return localuf.Surface(3, 'phenomenological')

@pytest.fixture
def sf5F():
    return localuf.Surface(5, 'code capacity')

@pytest.fixture
def sf5T():
    return localuf.Surface(5, 'phenomenological')

@pytest.fixture
def sf7F():
    return localuf.Surface(7, 'code capacity')

@pytest.fixture
def sf9F():
    return localuf.Surface(9, 'code capacity')

@pytest.fixture
def test_property():
    def f(instance: object, prop: str):
        with pytest.raises(AttributeError):
            delattr(instance, prop)
        with pytest.raises(AttributeError):
            setattr(instance, prop, None)
    return f