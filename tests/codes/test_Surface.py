import itertools

import pytest

from localuf.codes import Surface
from localuf.error_models import CodeCapacity

@pytest.fixture(
        name="sfCL",
        params=itertools.product(range(3, 9, 2), range(2, 5)),
        ids=lambda x: f"d{x[0]} wh{x[1]}",
)
def _sfCL(request):
    d, wh = request.param
    return Surface(d=d, error_model='circuit-level', window_height=wh)


def test_N_EDGES_attribute(sf: Surface):
    assert type(sf.N_EDGES) is int
    assert sf.N_EDGES == len(sf.EDGES)

def test_EDGES_attribute(sf: Surface):
    assert type(sf.EDGES) is tuple

def test_NODES(sf: Surface):
    assert type(sf.NODES) is tuple
    if isinstance(sf.ERROR_MODEL, CodeCapacity):
        assert len(sf.NODES) == sf.D * (sf.D+1)
    else:  # _Phenomenological | _CircuitLevel
        assert len(sf.NODES) == sf.D**2 * (sf.D+1)

def test_LONG_AXIS_attribute(sf: Surface):
    assert sf.LONG_AXIS == 1

def test_DIMENSION_attribute(sf: Surface):
    assert sf.DIMENSION == len(sf.NODES[0])

def test_index_to_label(sf5F: Surface):
    assert sf5F.index_to_label((1, 1)) == 8
    # assert sf3.index_to_label((1, 0, 1)) == 37

def test_label_to_index(sf5F: Surface):
    assert sf5F.label_to_index(8) == (1, 1)

@pytest.mark.parametrize("sf", [
    "sf3F",
    "sf5F",
    "sf7F",
    "sf9F",
])
def test_index_label_consistency(sf: Surface, request):
    sf = request.getfixturevalue(sf)
    d = sf.D
    for a in range(d*(d+1)):
        assert sf.index_to_label(sf.label_to_index(a)) == a

def test_get_pos(sf3F: Surface):
    pos = sf3F._get_pos()
    assert type(pos) is dict


@pytest.fixture
def index_to_id_helper():
    def f(sf: Surface):
        d = sf.D
        boundary_IDs: set[int] = set()
        bulk_IDs: set[int] = set()
        for v in sf.NODES:
            id_ = sf.index_to_id(v)
            if sf.is_boundary(v):
                boundary_IDs.add(id_)
            else:
                bulk_IDs.add(id_)
        assert max(boundary_IDs) < min(bulk_IDs)
        return d, boundary_IDs, bulk_IDs
    return f


def test_index_to_id_2D(sf5F: Surface, index_to_id_helper):
    d, boundary_IDs, bulk_IDs = index_to_id_helper(sf5F)
    assert len(boundary_IDs) + len(bulk_IDs) == d * (d+1)
    # highest ID boundary node
    assert sf5F.index_to_id((d-1, d-1)) == 2*d - 1
    # highest ID bulk node
    assert sf5F.index_to_id((d-1, d-2)) == d * (d+1) - 1


def test_index_to_id_3D(sf5T: Surface, index_to_id_helper):
    d, boundary_IDs, bulk_IDs = index_to_id_helper(sf5T)
    assert len(boundary_IDs) + len(bulk_IDs) == d**2 * (d+1)
    assert sf5T.index_to_id((d-1, d-1, d-1)) == 2 * d**2 - 1
    assert sf5T.index_to_id((d-1, d-2, d-1)) == d**2 * (d+1) - 1


def test_make_circuit_level_inputs(sfCL: Surface):
    d, wh = sfCL.D, sfCL.SCHEME.WINDOW_HEIGHT

    edges, edge_dict, merges = sfCL._make_circuit_level_inputs(d=d, wh=wh, merge_redundant_edges=True)
    len_merges = 2 * (wh-1) * (2*d-1)

    assert type(edges) is tuple
    assert len(edges) == sfCL.N_EDGES
    assert len(set(edges)) == sfCL.N_EDGES

    assert type(edge_dict) is dict
    assert len(edge_dict) == 12
    assert sum(len(es) for es in edge_dict.values()) == sfCL.N_EDGES + len_merges

    assert type(merges) is dict
    assert len(merges) == len_merges

    assert set(edges) & set(merges.keys()) == set()

    eu_edge_NS = tuple(((i, j, t), (i, j+1, t+1)) for i in      (0, d-1) for j in range(d-2) for t in range(wh-1))
    eu_edge_EW = tuple(((i, j, t), (i, j+1, t+1)) for i in range(1, d-1) for j in  (-1, d-2) for t in range(wh-1))
    assert edge_dict['EU edge'][:len(eu_edge_NS)] == eu_edge_NS
    assert edge_dict['EU edge'][len(eu_edge_NS):] == eu_edge_EW

    assert set(edge_dict['SEU']) == {
        ((i, j, t), (i+1, j+1, t+1))
        for i in range(d-1)
        for j in range(-1, d-1)
        for t in range(wh-1)
    }
    assert len(edge_dict['SEU']) == (d-1) * d * (wh-1)

    edges, edge_dict_False, merges = sfCL._make_circuit_level_inputs(d=d, wh=wh, merge_redundant_edges=False)

    assert len(edges) == sfCL.N_EDGES + len_merges
    assert len(set(edges)) == sfCL.N_EDGES + len_merges
    assert merges is None
    assert edge_dict_False == edge_dict


def test_substitute(sf3F: Surface, sf3T: Surface):
    d = 3
    assert sf3F._substitute(d, ((0, -1), (1, 0))) == ((1, -1), (1, 0))
    assert sf3F._substitute(d, ((2, 1), (1, 2))) == ((2, 1), ((2, 2)))
    assert sf3T._substitute(d, ((0, -1, 0), (1, 0, 1))) == ((1, -1, 1), (1, 0, 1))
    assert sf3T._substitute(d, ((2, 1, 1), (2, 2, 2))) == ((2, 1, 1), (2, 2, 1))


def test_get_matching_graph(sfCL: Surface):
    p = 1e-1
    matching = sfCL.get_matching_graph(p)
    assert matching.num_edges == sfCL.N_EDGES
    assert matching.num_detectors == sfCL.D * (sfCL.D-1) * sfCL.SCHEME.WINDOW_HEIGHT
    assert matching.num_fault_ids == 1