import pytest

from localuf.noise.main import _Uniform

class ConcreteUniform(_Uniform):
    """Instantiable version of `_Uniform`."""

    def __str__(self):
        return "concrete uniform"

@pytest.fixture
def uniform():
    return ConcreteUniform((
        ((0, 0), (0, 1)),
        ((0, 0), (1, 0)),
    ))


@pytest.mark.parametrize("prop", [
    "EDGES",
])
def test_property_attributes(test_property, uniform: ConcreteUniform, prop):
    test_property(uniform, prop)


def test_make_error(uniform: ConcreteUniform):
    assert uniform.make_error(0) == set()
    assert uniform.make_error(1) == set(uniform.EDGES)


def test_get_edge_weights(uniform: ConcreteUniform):
    p = 1e-1
    weight = uniform.log_odds_of_no_flip(p)
    assert uniform.get_edge_weights(p) == {e: (p, weight) for e in uniform.EDGES}


def test_force_error(uniform: ConcreteUniform):
    assert uniform.force_error(0) == set()
    assert len(uniform.force_error(1)) == 1
    assert uniform.force_error(len(uniform.EDGES)) == set(uniform.EDGES)