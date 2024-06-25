import pytest
from localuf.noise import CircuitLevel

from localuf.type_aliases import Edge, EdgeType


@pytest.fixture
def e_westmost() -> tuple[Edge, Edge]:
    return tuple(((1, -1, t), (1, 0, t)) for t in (0, 1)) # type: ignore


@pytest.fixture
def toy_edge_dict(e_westmost: tuple[Edge, Edge]):
    f = ((1, -1, 0), (1, 0, 1))
    g = ((0, -1, 0), (1, 0, 1))
    return {
        'E westmost': e_westmost,
        'EU west corners': (f,),
        'SEU': (g,),
    }


@pytest.fixture
def toy_merges(toy_edge_dict: dict[EdgeType, tuple[Edge, ...]]):
    _, e1 = toy_edge_dict['E westmost']
    f, = toy_edge_dict['EU west corners']
    g, = toy_edge_dict['SEU']
    return {f: e1, g: e1}


@pytest.fixture
def toy_cl(
    toy_edge_dict: dict[EdgeType, tuple[Edge, ...]],
    toy_merges: dict[Edge, Edge],
):
    return CircuitLevel(
        edge_dict=toy_edge_dict,
        parametrization='balanced',
        demolition=False,
        monolingual=False,
        merges=toy_merges,
    )