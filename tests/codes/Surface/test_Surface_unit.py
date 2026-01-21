import itertools

import pytest

from localuf import Surface
from localuf.noise import CodeCapacity
from localuf._schemes import Forward

def test_N_EDGES_attribute(sf: Surface):
    assert type(sf.N_EDGES) is int
    assert sf.N_EDGES == len(sf.EDGES)

def test_EDGES_attribute(sf: Surface):
    assert type(sf.EDGES) is tuple


class TestNODES:


    @pytest.fixture(
            name="sf_forward",
            params=range(3, 11, 2),
            ids=lambda x: f"d{x}",
    )
    def _sf_forward(self, request):
        return Surface(
            d=request.param,
            noise='phenomenological',
            scheme='forward',
    )

    def test_batch(self, sf: Surface):
        assert type(sf.NODES) is tuple
        if isinstance(sf.NOISE, CodeCapacity):
            assert len(sf.NODES) == sf.D * (sf.D+1)
        else:  # Phenomenological | CircuitLevel
            assert len(sf.NODES) == sf.D**2 * (sf.D+1)


    def test_forward(self, sf_forward: Surface):
        assert type(sf_forward.NODES) is tuple
        assert len(sf_forward.NODES) == sf_forward.D * (
            (sf_forward.D+1) * sf_forward.SCHEME.WINDOW_HEIGHT
            + sf_forward.D-1
        )

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
    pos = sf3F.get_pos()
    assert type(pos) is dict


class TestIndexToId:


    @pytest.fixture
    def helper(self):
        def f(sf: Surface):
            d = sf.D
            boundary_IDs: set[int] = set()
            detector_IDs: set[int] = set()
            for v in sf.NODES:
                id_ = sf.index_to_id(v)
                if sf.is_boundary(v):
                    boundary_IDs.add(id_)
                else:
                    detector_IDs.add(id_)
            assert max(boundary_IDs) < min(detector_IDs)
            return d, boundary_IDs, detector_IDs
        return f


    def test_2D(self, sf5F: Surface, helper):
        d, boundary_IDs, detector_IDs = helper(sf5F)
        assert len(boundary_IDs) + len(detector_IDs) == d * (d+1)
        # highest ID boundary node
        assert sf5F.index_to_id((d-1, d-1)) == 2*d - 1
        # highest ID detector
        assert sf5F.index_to_id((d-1, d-2)) == d * (d+1) - 1


    def test_3D(self, sf5T: Surface, helper):
        d, boundary_IDs, detector_IDs = helper(sf5T)
        assert len(boundary_IDs) + len(detector_IDs) == d**2 * (d+1)
        assert sf5T.index_to_id((d-1, d-1, d-1)) == 2 * d**2 - 1
        assert sf5T.index_to_id((d-1, d-2, d-1)) == d**2 * (d+1) - 1


class TestCodeCapacityEdges:
    
    def test_correct_count(self, sf3F: Surface):
        count, edges = sf3F._code_capacity_edges(False)
        assert count == len(edges)
        nodes = set().union(*edges)
        assert nodes == set(itertools.product(range(3), range(-1, 3)))

    def test_merge_equivalent_boundary_nodes(self, sf3F: Surface):
        count, edges = sf3F._code_capacity_edges(True)
        assert count == len(edges)
        nodes = set().union(*edges)
        assert nodes == set(itertools.product(range(3), range(2))) | {
            (0, -1), (0, 2),
        }


class TestPhenomenologicalEdges:
    
    def test_correct_count(self, sf3T: Surface):
        count, edges = sf3T._phenomenological_edges(
            3,
            False,
            merge_equivalent_boundary_nodes=False,
        )
        assert count == len(edges)
        nodes = set().union(*edges)
        assert nodes == set(itertools.product(range(3), range(-1, 3), range(3)))

    def test_merge_equivalent_boundary_nodes(self, sf3T: Surface):
        count, edges = sf3T._phenomenological_edges(
            3,
            False,
            merge_equivalent_boundary_nodes=True,
        )
        assert count == len(edges)
        nodes = set().union(*edges)
        assert nodes == set(itertools.product(range(3), range(2), range(3))) | {
            (0, -1, 0), (0, 2, 0),
        }


class TestCircuitLevelEdges:


    def helper(
            self,
            surface: Surface,
            future_boundary: bool,
            merge_equivalent_boundary_nodes=False,
    ):
        """Output:
        * `d` code distance
        * `layer_count` number of layers in decoding graph
        * `edge_dict` maps from edge type
        (i.e. orientation and location)
        to tuple of all edges of that type.
        Always includes redundant edges.
        Inter-key order matters as used by `noise.forcers.ForceByEdge.force_error`.
        """
        d, h = surface.D, surface.SCHEME.WINDOW_HEIGHT
        layer_count = h if future_boundary else h-1
        len_merges = 2 * layer_count * (2*d-1)
        horizontal_edge_count = h * surface.DATA_QUBIT_COUNT
        u_per_layer = d*(d-1)
        sd_per_layer = (d-1)**2

        n_edges, edges, edge_dict, merges = surface._circuit_level_edges(
            h=h,
            future_boundary=future_boundary,
            _merge_redundant_edges=True,
            merge_equivalent_boundary_nodes=merge_equivalent_boundary_nodes,
        )

        eu_per_layer = d*(d-2)
        seu_per_layer = (d-1)*(d-2)
        timelike_edge_count = layer_count * (u_per_layer + sd_per_layer + eu_per_layer + seu_per_layer)
        assert n_edges == horizontal_edge_count + timelike_edge_count
        assert surface.N_EDGES == n_edges
        assert type(edges) is tuple
        assert len(edges) == n_edges
        assert len(set(edges)) == n_edges  # test uniqueness

        assert type(edge_dict) is dict
        assert len(edge_dict) == 12
        assert sum(len(es) for es in edge_dict.values()) == surface.N_EDGES + len_merges

        assert type(merges) is dict
        assert len(merges) == len_merges

        if not merge_equivalent_boundary_nodes:
            assert set(edges) & set(merges.keys()) == set()

            n_edges, edges, edge_dict_False, merges = surface._circuit_level_edges(
                h=h,
                future_boundary=future_boundary,
                _merge_redundant_edges=False,
            )

            eu_per_layer = d**2
            seu_per_layer = d*(d-1)
            timelike_edge_count = layer_count * (u_per_layer + sd_per_layer + eu_per_layer + seu_per_layer)
            assert n_edges == horizontal_edge_count + timelike_edge_count
            assert n_edges == surface.N_EDGES + len_merges
            assert len(edges) == n_edges
            assert len(set(edges)) == n_edges
            assert merges is None
            assert edge_dict_False == edge_dict

        return d, layer_count, edge_dict


    def test_future_boundary_False(self, sfCL: Surface):
        d, layer_count, edge_dict = self.helper(sfCL, False)

        eu_edge_NS = tuple(((i, j, t), (i, j+1, t+1)) for i in ( 0, d-1) for j in range(   d-2) for t in range(layer_count))
        eu_edge_EW = tuple(((i, j, t), (i, j+1, t+1)) for j in (-1, d-2) for i in range(1, d-1) for t in range(layer_count))
        assert edge_dict['EU edge'] == (*eu_edge_NS, *eu_edge_EW)

        seu_bulk     = tuple(((i, j, t), (i+1, j+1, t+1)) for i in range(d-1) for j in range(d-2) for t in range(layer_count))
        seu_boundary = tuple(((i, j, t), (i+1, j+1, t+1)) for j in  (-1, d-2) for i in range(d-1) for t in range(layer_count))
        assert edge_dict['SEU'] == (*seu_boundary, *seu_bulk)
        assert set(edge_dict['SEU']) == {
            ((i, j, t), (i+1, j+1, t+1))
            for i in range(d-1)
            for j in range(-1, d-1)
            for t in range(layer_count)
        }
        assert len(edge_dict['SEU']) == (d-1) * d * layer_count


    def test_future_boundary_True(self, sfCL_OL: Surface):
        d, layer_count, edge_dict = self.helper(sfCL_OL, True)

        eu_edge_NS = tuple(((i, j, t), (i, j+1, t+1)) for i in ( 0, d-1) for j in range(   d-2) for t in range(layer_count))
        eu_edge_W = tuple(((i,  -1, t), (i,   0, t+1)) for i in range(1, d-1) for t in range(-1, layer_count-1))
        eu_edge_E = tuple(((i, d-2, t), (i, d-1, t+1)) for i in range(1, d-1) for t in range(    layer_count  ))
        assert edge_dict['EU edge'] == (*eu_edge_NS, *eu_edge_W, *eu_edge_E)

        seu_bulk = tuple(((i, j, t), (i+1, j+1, t+1)) for i in range(d-1) for j in range(d-2) for t in range(layer_count))
        seu_W = tuple(((i, -1, t), (i+1, 0, t+1)) for i in range(d-1) for t in range(-1, layer_count-1))
        seu_E = tuple(((i, d-2, t), (i+1, d-1, t+1)) for i in range(d-1) for t in range(layer_count))
        assert edge_dict['SEU'] == (*seu_W, *seu_E, *seu_bulk)
        assert len(edge_dict['SEU']) == (d-1) * d * layer_count


    @pytest.mark.parametrize("buffer_height", range(1, 4), ids=lambda x: f"buffer_height {x}")
    @pytest.mark.parametrize("_merge_redundant_edges", [False, True], ids=lambda x: f"_merge_redundant_edges {x}")
    def test_t_start(self, sfCL_OL_scheme: Forward, buffer_height, _merge_redundant_edges):
        code = sfCL_OL_scheme._CODE
        commit_height = sfCL_OL_scheme._COMMIT_HEIGHT
        n_commit_edges, commit_edges, commit_edge_dict, commit_merges = code._circuit_level_edges(
            h=commit_height,
            future_boundary=True,
            _merge_redundant_edges=_merge_redundant_edges,
        )
        n_fresh_edges, fresh_edges, fresh_edge_dict, fresh_merges = code._circuit_level_edges(
            h=commit_height,
            future_boundary=True,
            _merge_redundant_edges=_merge_redundant_edges,
            t_start=buffer_height,
        )
        assert n_fresh_edges == n_commit_edges
        assert fresh_edges == tuple(code.raise_edge(e, buffer_height) for e in commit_edges)
        assert fresh_edge_dict == {edge_type: tuple(
            code.raise_edge(e, buffer_height) for e in edges
        ) for edge_type, edges in commit_edge_dict.items()}
        if _merge_redundant_edges:
            assert fresh_merges == {
                code.raise_edge(e, buffer_height):
                code.raise_edge(substitute, buffer_height)
                for e, substitute in commit_merges.items() # type: ignore
            }
        else:
            assert fresh_merges is None
            assert commit_merges is None

    
    def test_merge_equivalent_boundary_nodes(self, sfCL: Surface):
        self.helper(sfCL, False, True)


def test_substitute(sf3F: Surface, sf3T: Surface):
    assert sf3F._substitute(((0, -1), (1, 0))) == ((1, -1), (1, 0))
    assert sf3F._substitute(((2, 1), (1, 2))) == ((2, 1), ((2, 2)))
    assert sf3T._substitute(((0, -1, 0), (1, 0, 1))) == ((1, -1, 1), (1, 0, 1))
    assert sf3T._substitute(((2, 1, 1), (2, 2, 2))) == ((2, 1, 1), (2, 2, 1))