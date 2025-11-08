import itertools

import pytest

from localuf import Repetition
from localuf._schemes import Forward

@pytest.fixture(name="rp", params=itertools.product(
        range(3, 11, 2),
        ('phenomenological',),
        ('batch', 'forward', 'frugal'),
), ids=lambda x: f"d{x[0]} {x[2]}")
def _rp(request):
    d, noise, scheme = request.param
    return Repetition(d, noise, scheme=scheme)

@pytest.fixture(
        name="rpCC",
        params=range(3, 11, 2),
        ids=lambda x: f"d{x}",
)
def _rpCC(request):
    return Repetition(request.param, 'code capacity')


class TestPhenomenologicalEdges:

    @pytest.mark.parametrize("buffer_height", range(1, 4), ids=lambda x: f"buffer_height {x}")
    def test_t_start_consistency(self, forward_rp: Forward, buffer_height):
        code = forward_rp._CODE
        commit_height = forward_rp._COMMIT_HEIGHT
        n_commit_edges, commit_edges = code._phenomenological_edges(
            commit_height,
            True,
        )
        n_fresh_edges, fresh_edges = code._phenomenological_edges(
            commit_height,
            True,
            t_start=buffer_height,
        )
        assert n_commit_edges == len(commit_edges)
        assert n_fresh_edges == n_commit_edges
        assert fresh_edges == tuple((
            (*u[:-1], buffer_height+u[-1]),
            (*v[:-1], buffer_height+v[-1]),
        ) for u, v in commit_edges)
        nodes = set().union(*commit_edges)
        assert nodes == set(itertools.product(range(-1, code.D), range(commit_height))) | {
            (j, commit_height) for j in range(code.D-1)
        }

    def test_merge_equivalent_boundary_edges(self, forward_rp: Forward):
        code = forward_rp._CODE
        h = forward_rp.WINDOW_HEIGHT
        edge_count, edges = code._phenomenological_edges(
            h,
            False,
            merge_equivalent_boundary_nodes=True,
        )
        assert edge_count == len(edges)
        assert edge_count == code.D * h + (code.D-1) * (h-1)
        nodes = set().union(*edges)
        assert nodes == set(itertools.product(range(code.D-1), range(h))) | {
            (-1, 0), (code.D-1, 0),
        }


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
        case 'forward' | 'frugal':
            assert len(rp.NODES) == \
                (rp.D+1) * rp.SCHEME.WINDOW_HEIGHT + rp.D-1

def test_LONG_AXIS_attribute(rp: Repetition):
    assert rp.LONG_AXIS == 0

def test_DIMENSION_attribute(rp: Repetition):
    assert rp.DIMENSION == len(rp.NODES[0])

def test_get_pos():
    rp = Repetition(3, 'code capacity')
    pos = rp.get_pos()
    assert type(pos) is dict