import pytest

from localuf.error_models.main import _Uniform

@pytest.fixture
def uniform():
    return _Uniform((
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ))


@pytest.mark.parametrize("prop", [
    "EDGES",
])
def test_property_attributes(test_property, uniform: _Uniform, prop):
    test_property(uniform, prop)


def test_make_error(uniform: _Uniform):
    assert uniform.make_error(0) == set()
    assert uniform.make_error(1) == set(uniform.EDGES)


def test_get_edge_probabilities(uniform: _Uniform):
    p = 1e-1
    assert uniform.get_edge_probabilities(p) == tuple((e, p) for e in uniform.EDGES)