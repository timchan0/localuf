from pymatching import Matching
import pytest

from localuf._base_classes import Code
from localuf.codes import Surface
from localuf.decoders import MWPM
from localuf.type_aliases import Node


@pytest.mark.parametrize("noise_level", [None, 1e-1, 1e-2])
def test_get_matching(sfCL: Surface, noise_level: float):
    decoder = MWPM(sfCL)
    matching = decoder.get_matching(noise_level=noise_level)
    assert matching.num_edges == sfCL.N_EDGES
    assert matching.num_detectors == sfCL.D * (sfCL.D-1) * sfCL.SCHEME.WINDOW_HEIGHT
    assert matching.num_fault_ids == 1
    assert_all_edge_data_correct(sfCL, matching, noise_level, decoder._DETECTOR_TO_INT)


def assert_all_edge_data_correct(
        code: Code,
        matching: Matching,
        noise_level: float,
        detector_to_int: dict[Node, int],
):
    """Assert all edge data are correct according to `noise_level`."""
    edge_weights = code.NOISE.get_edge_weights(noise_level)
    for (u, v), (p, weight) in edge_weights.items():
        if u in code.BOUNDARY_NODES:
            edge_data = matching.get_boundary_edge_data(detector_to_int[v])
        elif v in code.BOUNDARY_NODES:
            edge_data = matching.get_boundary_edge_data(detector_to_int[u])
        else:
            edge_data = matching.get_edge_data(
                node1=detector_to_int[u],
                node2=detector_to_int[v],
            )
        assert edge_data['fault_ids'] == ({0} if (u[code.LONG_AXIS] == -1) else set())
        assert edge_data['weight'] == weight
        assert edge_data['error_probability'] == p