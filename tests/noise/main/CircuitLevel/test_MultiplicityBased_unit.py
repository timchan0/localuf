import numpy as np
import pytest

from localuf.type_aliases import Edge, MultiplicityVector
from localuf.noise import CircuitLevel


@pytest.fixture
def ottf():
    return (1, 2, 3, 4)


@pytest.fixture
def cl_dc(e_westmost: tuple[Edge, Edge], ottf):
    e0, e1 = e_westmost
    f = ((1, -1, 0), (1, 0, 1))
    g = ((0, -1, 0), (1, 0, 1))

    return CircuitLevel._get_dc(
        edge_dict={
            'E westmost': (e0, e1),
            'EU west corners': (f,),
            'SEU': (g,),
        },
        multiplicities={
            'E westmost': np.array(ottf),
            'EU west corners': 10*np.array(ottf),
            'SEU': 100*np.array(ottf),
        },
        merges={f: e1, g: e1},
    )


def test_get_dc(
        cl_dc: dict[Edge, MultiplicityVector],
        e_westmost: tuple[Edge, Edge],
        ottf,
):
    e0, e1 = e_westmost

    assert len(cl_dc) == 2
    assert (cl_dc[e0] == np.array(ottf)).all()
    assert (cl_dc[e1] == 111*np.array(ottf)).all()


def test_change_edges(e_westmost: tuple[Edge, Edge]):
    e0, e1 = e_westmost
    dc = CircuitLevel._get_dc(
        edge_dict={'E westmost': (e0, e1)},
        multiplicities={'E westmost': np.array((1, 2, 3, 4))},
        merges={},
    )
    assert (dc[e0] == dc[e1]).all()
    assert dc[e0] is not dc[e1]
    dc[e0] *= 2
    assert (dc[e0] == 2*dc[e1]).all()