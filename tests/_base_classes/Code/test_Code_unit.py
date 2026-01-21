from unittest import mock

import networkx as nx
import pytest

from localuf import constants, Surface, Repetition
from localuf._schemes import Scheme
from localuf.noise import CodeCapacity
from localuf.noise.main import Phenomenological
from localuf.type_aliases import Edge
from localuf._base_classes import Code

@pytest.fixture
def error_2D():
    return {
        ((0, -1), (0, 0)),
        ((0, 0), (0, 1)),
    }

@pytest.fixture
def error_3D():
    return {
        ((0, -1, 1), (0, 0, 1)),
        ((0, 0, 1), (0, 1, 1)),
    }

@pytest.mark.parametrize("prop", [
    "D",
    "SCHEME",
    "N_EDGES",
    "EDGES",
    "NODES",
    "TIME_AXIS",
    "LONG_AXIS",
    "INCIDENT_EDGES",
])
def test_property_attributes(test_property, sf3F: Surface, prop: str):
    test_property(sf3F, prop)


class ConcreteCode(Code):

    _EDGES = ()
    _N_EDGES = 0
    _NODES = ()

    def _code_capacity_edges(self, merge_equivalent_boundary_nodes):
        return super()._code_capacity_edges(merge_equivalent_boundary_nodes)

    def _phenomenological_edges(
            self,
            h,
            future_boundary,
            merge_equivalent_boundary_nodes,
    ):
        return super()._phenomenological_edges(
            h,
            future_boundary,
            merge_equivalent_boundary_nodes,
        )

    def _circuit_level_edges(
            self,
            h,
            future_boundary,
            _merge_redundant_edges,
            merge_equivalent_boundary_nodes,
    ): pass

    def _future_boundary_nodes(self, h): pass

    def _redundant_boundary_nodes(self, h): pass

    def index_to_id(self, index): pass

    def get_pos(self, x_offset, nodelist): pass


@pytest.mark.parametrize("noise", ('code capacity', 'phenomenological'))
@pytest.mark.parametrize("scheme", ('batch', 'global batch'))
def test_init(noise, scheme):
    with (
        mock.patch(
            "localuf._base_classes.Code._code_capacity_edges",
            return_value=(0, (((0, 0), (0, 1)),))
        ) as mock_code_capacity_edges,
        mock.patch(
            "localuf._base_classes.Code._phenomenological_edges",
            return_value=(0, (((0, 0, 0), (0, 1, 0)),))
        ) as mock_phenomenological_edges,
        mock.patch(
            "localuf._schemes.Batch.__init__",
            return_value=None
        ) as mock_batch_init,
        mock.patch(
            "localuf._schemes.Global.__init__",
            return_value=None
        ) as mock_global_batch_init,
    ):
        ConcreteCode(3, noise, scheme)
        if noise == 'phenomenological':
            mock_phenomenological_edges.assert_called_once()
            mock_code_capacity_edges.assert_not_called()
        else:
            mock_phenomenological_edges.assert_not_called()
            mock_code_capacity_edges.assert_called_once()
        if scheme == 'batch':
            mock_batch_init.assert_called_once()
            mock_global_batch_init.assert_not_called()
        else:
            mock_batch_init.assert_not_called()
            mock_global_batch_init.assert_called_once()

def test_D_attribute(sf: Surface):
    assert type(sf.D) is int


def test_SCHEME_attribute(sf: Surface):
    assert issubclass(type(sf.SCHEME), Scheme)


def test_INCIDENT_EDGES_attribute(sf: Surface):
    assert type(sf.INCIDENT_EDGES) is dict
    assert len(sf.INCIDENT_EDGES) == len(sf.NODES)


def test_NOISE_attribute(sf3F: Surface, sf3T: Surface):
    assert type(sf3F.NOISE) is CodeCapacity
    assert type(sf3T.NOISE) is Phenomenological


def test_INCIDENT_EDGES(sf):
    """Test `INCIDENT_EDGES` consistent with the old, slower calculation of incident edges."""
    assert sf.INCIDENT_EDGES == {v: {
        e for e in sf.EDGES if v in e
    } for v in sf.NODES}


def test_INCIDENT_EDGES_code_capacity(sf5F: Surface):
    # boundary node
    assert sf5F.INCIDENT_EDGES[0, -1] == {((0, -1), (0, 0))}
    # weight-3 stabilizer
    assert sf5F.INCIDENT_EDGES[0, 1] == {
        ((0, 0), (0, 1)),
        ((0, 1), (0, 2)),
        ((0, 1), (1, 1))
    }
    # weight-4 stabilizer
    assert sf5F.INCIDENT_EDGES[1, 1] == {
        ((0, 1), (1, 1)),
        ((1, 0), (1, 1)),
        ((1, 1), (1, 2)),
        ((1, 1), (2, 1))
    }


def test_INCIDENT_EDGES_phenomenological(sf5T: Surface):
    # boundary node
    assert sf5T.INCIDENT_EDGES[0, -1, 0] == {((0, -1, 0), (0, 0, 0))}
    # degree-5 node
    assert sf5T.INCIDENT_EDGES[0, 1, 1] == {
        ((0, 1, 0), (0, 1, 1)),
        ((0, 0, 1), (0, 1, 1)),
        ((0, 1, 1), (0, 2, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((0, 1, 1), (0, 1, 2))
    }
    # degree-6 node
    assert sf5T.INCIDENT_EDGES[1, 1, 1] == {
        ((1, 1, 0), (1, 1, 1)),
        ((1, 0, 1), (1, 1, 1)),
        ((1, 1, 1), (1, 2, 1)),
        ((0, 1, 1), (1, 1, 1)),
        ((1, 1, 1), (2, 1, 1)),
        ((1, 1, 1), (1, 1, 2))
    }


def test_is_boundary(sf3F: Surface):
    with mock.patch("localuf._base_classes.Scheme.is_boundary") as m:
        v, *_ = sf3F.NODES
        sf3F.is_boundary(v)
        m.assert_called_once_with(v)


def test_neighbors(sf5F: Surface, sf5T: Surface):
    assert sf5F.neighbors((0, -1)) == {(0, 0)}  # boundary node
    assert sf5F.neighbors((0, 1)) == {(0, 0), (0, 2), (1, 1)}  # weight-3 stabilizer
    assert sf5F.neighbors((1, 1)) == {(0, 1), (1, 0), (1, 2), (2, 1)}  # weight-4 stabilizer
    assert sf5T.neighbors((0, -1, 0)) == {(0, 0, 0)}
    assert sf5T.neighbors((0, 1, 1)) == {
        (0, 1, 0),
        (0, 0, 1),
        (0, 2, 1),
        (1, 1, 1),
        (0, 1, 2)
    }
    assert sf5T.neighbors((1, 1, 1)) == {
        (1, 1, 0),
        (0, 1, 1),
        (1, 0, 1),
        (1, 2, 1),
        (2, 1, 1),
        (1, 1, 2)
    }


def test_traverse_edge(sf3F: Surface):
    u, v = (0, 0), (0, 1)
    assert sf3F.traverse_edge((u, v), u) == v
    assert sf3F.traverse_edge((u, v), v) == u
    u, v = (0, 0, 0), (1, 0, 0)
    assert sf3F.traverse_edge((u, v), u) == v
    assert sf3F.traverse_edge((u, v), v) == u


def test_raise_node(sf3T: Surface):
    for v in sf3T.NODES:
        di, dj, dt = sf3T.raise_node(v)
        assert v == (di, dj, dt-1)
        li, lj, lt = sf3T.raise_node(v, -3)
        assert v == (li, lj, lt+3)


def test_raise_edge(sf3T: Surface):
    for e in sf3T.EDGES:
        (*uij, ut), (*vij, vt) = sf3T.raise_edge(e)
        assert e == ((*uij, ut-1), (*vij, vt-1))
        (*uij, ut), (*vij, vt) = sf3T.raise_edge(e, -3)
        assert e == ((*uij, ut+3), (*vij, vt+3))


class TestMakeError:

    @pytest.mark.parametrize("noise_level", [0.0, 1e-3, 1e-1, 1.0])
    def test_inner_called(self, sf3F: Surface, noise_level: float):
        with mock.patch(
            "localuf.noise.main._Uniform.make_error",
            return_value=set(sf3F.EDGES),
        ) as m:
            error = sf3F.make_error(noise_level)
            m.assert_called_once_with(noise_level)
        assert error == set(sf3F.EDGES)

    def test_exclude_future_boundary_forward(self, rp3_forward: Repetition):
        self._exclude_temproal_boundary_edges_helper(rp3_forward)

    def test_exclude_future_boundary_frugal(self, rp3_frugal: Repetition):
        self._exclude_temproal_boundary_edges_helper(rp3_frugal)

    def _exclude_temproal_boundary_edges_helper(self, repetition: Repetition):
        NOISE_LEVEL = 1
        h = repetition.SCHEME.WINDOW_HEIGHT
        future_boundary_edges = {((j, h-1), (j, h)) for j in range(0, repetition.D-1)}
        with mock.patch(
            "localuf.noise.main._Uniform.make_error",
            return_value=future_boundary_edges,
        ) as m:
            error = repetition.make_error(NOISE_LEVEL, exclude_future_boundary=True)
            m.assert_called_once_with(NOISE_LEVEL)
        assert error == set()

    def test_exclude_future_boundary_surface(self, sfCL_OL: Surface):
        NOISE_LEVEL = 1
        h = sfCL_OL.SCHEME.WINDOW_HEIGHT
        future_boundary_edges = {e for e in sfCL_OL.EDGES if any(
            v[sfCL_OL.TIME_AXIS] == h for v in e)}
        with mock.patch(
            "localuf.noise.main.CircuitLevel.make_error",
            return_value=future_boundary_edges,
        ) as m:
            error = sfCL_OL.make_error(NOISE_LEVEL, exclude_future_boundary=True)
            m.assert_called_once_with(NOISE_LEVEL)
        assert error == set()


def test_get_syndrome(
        sf3F: Surface,
        sf3T: Surface,
        error_2D: set[Edge],
        error_3D: set[Edge],
):
    assert sf3F.get_syndrome(set()) == set()
    assert sf3F.get_syndrome(error_2D) == {(0, 1)}
    assert sf3T.get_syndrome(error_3D) == {(0, 1, 1)}


def test_compose_errors(sf3F: Surface, error_2D: set[Edge]):
    assert sf3F.compose_errors() == set()
    assert sf3F.compose_errors(error_2D) == error_2D
    assert sf3F.compose_errors(error_2D, error_2D) == set()
    # test error_2D not modified
    assert sf3F.compose_errors(error_2D, {((0, -1), (0, 0))}) == {((0, 0), (0, 1)),}


def test_get_logical_error(sf3F: Surface):
    with mock.patch("localuf._schemes.Batch.get_logical_error") as m:
        sf3F.get_logical_error(set())
        m.assert_called_once_with(set())


def test_graph_attribute(sf: Surface):
    with mock.patch("networkx.Graph") as mock_Graph:
        sf.GRAPH
        mock_Graph.assert_called_once_with(
            sf.EDGES,
            D=sf.D,
            SCHEME=str(sf.SCHEME),
        )
    del sf.GRAPH
    g = sf.GRAPH
    assert type(g) is nx.Graph
    assert len(sf.EDGES) == len(g.edges)
    assert set(g.nodes) == set(sf.NODES)
    assert g.graph['D'] == sf.D
    assert g.graph['SCHEME'] == str(sf.SCHEME)
    # test cached property
    with mock.patch("networkx.Graph") as mock_Graph:
        sf.GRAPH
        mock_Graph.assert_not_called()


def test_draw(sf3F: Surface):
    with (
        mock.patch("localuf.Surface.get_syndrome") as mock_get_syndrome,
        mock.patch("localuf.Surface.get_pos") as mock_get_pos,
        mock.patch("localuf.Surface.get_node_color") as mock_get_node_color,
        mock.patch("networkx.draw") as mock_draw,
    ):
        sf3F.draw()
        mock_get_syndrome.assert_called_once_with(set())
        mock_get_pos.assert_called_once_with(constants.DEFAULT_X_OFFSET)
        mock_get_node_color.assert_called_once()
        mock_draw.assert_called_once()


def test_get_node_color(sf3F: Surface):
    ls = sf3F.get_node_color(
        set(),
        boundary_color='a',
        defect_color='b',
        nondefect_color='c',
    )
    assert type(ls) is list
    assert len(ls) == len(sf3F.NODES)
    assert set(ls) == {'a', 'c'}
    ls = sf3F.get_node_color(
        {(0, 1)},
        boundary_color='d',
        defect_color='e',
        nondefect_color='f',
    )
    assert set(ls) == {'d', 'e', 'f'}
    ls = sf3F.get_node_color(
        {v for v in sf3F.NODES if sf3F.is_boundary(v)},
        boundary_color='g',
        defect_color='h',
        nondefect_color='i',
    )
    assert set(ls) == {'h', 'i'}
    ls = sf3F.get_node_color(
        set(),
        nodelist=[]
    )
    assert ls == []