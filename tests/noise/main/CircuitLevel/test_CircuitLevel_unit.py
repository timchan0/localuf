import copy
from unittest import mock

import numpy as np
import pytest

from localuf.noise import CircuitLevel
from localuf.type_aliases import EdgeType, Edge, MultiplicityVector

@pytest.fixture(name="cl", params=[
    'standard',
    'balanced',
    'ion trap',
])
def _cl(request):
    return CircuitLevel(
        edge_dict={'S': (((0, 0, 0), (1, 0, 0)),)},
        parametrization=request.param,
        demolition=False,
        monolingual=False,
    )

def test_DEFAULT_MULTIPLICITIES():
    assert type(CircuitLevel._DEFAULT_MULTIPLICITIES) is dict
    assert len(CircuitLevel._DEFAULT_MULTIPLICITIES) == 12
    assert len(CircuitLevel._DEFAULT_MULTIPLICITIES['S']) == 4

def test_ALL_COEFFICIENTS():
    assert type(CircuitLevel._ALL_COEFFICIENTS) is dict
    assert len(CircuitLevel._ALL_COEFFICIENTS) == 3
    assert len(CircuitLevel._ALL_COEFFICIENTS['standard']) == 4

@pytest.mark.parametrize("parametrization", [
    'standard',
    'balanced',
    'ion trap',
])
def test_init(parametrization):
    edge_dict: dict[EdgeType, tuple[Edge, ...]] = {'S': (((0, 0, 0), (1, 0, 0)),)}
    with mock.patch(
        "localuf.noise.CircuitLevel._make_multiplicities",
        return_value=CircuitLevel._make_multiplicities(demolition=False, monolingual=False),
    ) as mock_gm:
        cl = CircuitLevel(
            edge_dict=edge_dict,
            parametrization=parametrization,
            demolition=False,
            monolingual=False,
        )
        mock_gm.assert_called_once_with(False, False)
    assert cl._COEFFICIENTS == CircuitLevel._ALL_COEFFICIENTS[parametrization]


def test_get_dc(
        toy_edge_dict: dict[EdgeType, tuple[Edge, ...]],
        toy_merges: dict[Edge, Edge],
):
    """`dc` should have two entries whose values are different."""
    e0, e1 = toy_edge_dict['E westmost']
    toy_multiplicities: dict[EdgeType, MultiplicityVector] = {
        et: np.array(m) for et, m
        in CircuitLevel._DEFAULT_MULTIPLICITIES.items()
    }
    dc = CircuitLevel._get_dc(
        toy_edge_dict,
        toy_multiplicities,
        toy_merges,
    )
    assert len(dc) == 2
    assert (dc[e0] == toy_multiplicities['E westmost']).all()
    assert (dc[e1] == sum((
            toy_multiplicities['E westmost'],
            toy_multiplicities['EU west corners'],
            toy_multiplicities['SEU'],
        ))).all()


def test_make_multiplicities(assert_these_multiplicities_unchanged):
    m = CircuitLevel._make_multiplicities(demolition=False, monolingual=False)
    assert_these_multiplicities_unchanged(m, CircuitLevel._DEFAULT_MULTIPLICITIES.keys())

def test_make_multiplicities_demolition(
        assert_these_multiplicities_unchanged,
        diagonals,
):
    m = CircuitLevel._make_multiplicities(demolition=True, monolingual=False)
    assert (m['S'] == np.array((4, 2, 2, 0))).all()
    assert (m['E westmost'] == np.array((1, 0, 3, 0))).all()
    assert (m['E bulk'] == np.array((2, 0, 2, 0))).all()
    assert (m['E eastmost'] == np.array((1, 0, 2, 0))).all()
    assert (m['U 3'] == np.array((3, 0, 1, 2))).all()
    assert (m['U 4'] == np.array((4, 0, 0, 2))).all()
    assert_these_multiplicities_unchanged(m, diagonals)

def test_make_multiplicities_monolingual(
        assert_these_multiplicities_unchanged,
        diagonals,
):
    m = CircuitLevel._make_multiplicities(demolition=False, monolingual=True)
    assert (m['S'] == np.array((4, 2, 3, 0))).all()
    assert (m['E westmost'] == np.array((1, 0, 4, 0))).all()
    assert (m['E bulk'] == np.array((2, 0, 3, 0))).all()
    assert (m['E eastmost'] == np.array((1, 0, 3, 0))).all()
    assert (m['U 3'] == np.array((3, 0, 3, 1))).all()
    assert (m['U 4'] == np.array((4, 0, 2, 1))).all()
    assert_these_multiplicities_unchanged(m, diagonals)

def test_DEFAULT_MULTIPLICITIES_unchanged(assert_these_multiplicities_unchanged):
    dc = copy.deepcopy(CircuitLevel._DEFAULT_MULTIPLICITIES)
    m = CircuitLevel._make_multiplicities(demolition=True, monolingual=True)
    assert_these_multiplicities_unchanged(dc, dc.keys())
    m['S'] += np.array((1, 1, 1, 1))
    assert_these_multiplicities_unchanged(dc, ('S',))


def test_make_error(toy_cl: CircuitLevel, e_westmost: tuple[Edge, Edge]):
    m0, m1 = toy_cl._EDGES.keys()
    for p in (0, 1):
        with mock.patch(
            'localuf.noise.CircuitLevel._get_flip_probabilities',
            return_value={m0: 1-p, m1: p},
        ) as mock_gep:
            error = toy_cl.make_error(p)
            mock_gep.assert_called_once_with(p)
            assert error == {e_westmost[p]}


def test_get__flip_probabilities(cl: CircuitLevel):
    with mock.patch('localuf.noise._multiset_handler.MultisetHandler.pr') as mock_pr:
        cl._get_flip_probabilities(0)
        cl._get_flip_probabilities(1)
        assert mock_pr.call_args_list == [
            mock.call(m, (0, 0, 0, 0))
            for m in cl._EDGES.keys()
        ] + [
            mock.call(m, cl._COEFFICIENTS)
            for m in cl._EDGES.keys()
        ]


def test_get_edge_weights(cl: CircuitLevel):
    p = 1e-1
    with mock.patch("localuf.noise.CircuitLevel._get_flip_probabilities") as mock_gfp:
        cl.get_edge_weights(p)
        mock_gfp.assert_called_once_with(p)