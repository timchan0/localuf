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
    "FRESH_EDGES",
])
def test_property_attributes(test_property, uniform: ConcreteUniform, prop):
    test_property(uniform, prop)


def test_make_error(uniform: ConcreteUniform):
    assert uniform.make_error(0) == set()
    assert uniform.make_error(1) == set(uniform.FRESH_EDGES)


def test_get_edge_weights(uniform: ConcreteUniform):
    flip_probability = 1e-1
    weight = uniform.log_odds_of_no_flip(flip_probability)
    assert uniform.get_edge_weights(flip_probability) == {e: (flip_probability, weight) for e in uniform.FRESH_EDGES}


def test_force_error(uniform: ConcreteUniform):
    assert uniform.force_error(0) == set()
    assert len(uniform.force_error(1)) == 1
    assert uniform.force_error(len(uniform.FRESH_EDGES)) == set(uniform.FRESH_EDGES)