import itertools

import pytest

from localuf import Repetition

@pytest.fixture(name="rp", params=itertools.product(
        range(3, 11, 2),
        ('phenomenological',),
), ids=lambda x: f"d{x[0]}")
def _rp(request):
    return Repetition(*request.param)

@pytest.fixture(
        name="rpCC",
        params=range(3, 11, 2),
        ids=lambda x: f"d{x}",
)
def _rpCC(request):
    return Repetition(request.param, 'code capacity')

def test_N_EDGES_attribute(rp: Repetition):
    assert type(rp.N_EDGES) is int
    assert rp.N_EDGES == len(rp.EDGES)

def test_EDGES_attribute(rp: Repetition):
    assert type(rp.EDGES) is tuple

def test_NODES_code_capacity(rpCC: Repetition):
    assert type(rpCC.NODES) is tuple
    assert len(rpCC.NODES) == rpCC.D+1

def test_NODES_attribute(rp: Repetition):
    assert type(rp.NODES) is tuple
    match str(rp.SCHEME):
        case 'batch':
            assert len(rp.NODES) == rp.D * (rp.D+1)
        case 'overlapping':
            assert len(rp.NODES) == \
                (rp.D+1) * rp.SCHEME.WINDOW_HEIGHT + rp.D-1
        case 'streaming':
            assert len(rp.NODES) == (3*(rp.D//2)+1) * (rp.D+1) + rp.D-1
            
def test_LONG_AXIS_attribute(rp: Repetition):
    assert rp.LONG_AXIS == 0

def test_DIMENSION_attribute(rp: Repetition):
    assert rp.DIMENSION == len(rp.NODES[0])

def test_get_pos():
    rp = Repetition(3, 'code capacity')
    pos = rp._get_pos()
    assert type(pos) is dict