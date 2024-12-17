from unittest import mock

import pytest

from localuf import Surface
from localuf.type_aliases import Edge
from localuf.decoders import UF
from localuf._schemes import Global


@pytest.fixture
def global3F():
    """Global scheme for distance-3 surface code under code capacity noise model."""
    code = Surface(3, 'code capacity', scheme='global batch')
    return code.SCHEME


@pytest.mark.parametrize("d", range(3, 7, 2), ids=lambda x: f"d{x}")
@pytest.mark.parametrize("n", range(1, 4), ids=lambda x: f"n{x}")
def test_run(d: int, n: int):
    global_scheme = Surface(
        d,
        'phenomenological',
        scheme='global batch',
        window_height=d*n,
    ).SCHEME
    decoder = UF(global_scheme._CODE)
    p = 0.5
    with (
        mock.patch('localuf._schemes.Global.reset') as mock_reset,
        mock.patch('localuf._schemes.Batch._sim_cycle_given_p') as scgp,
    ):
        assert global_scheme.run(decoder, p, n) == (scgp.return_value, n)
        mock_reset.assert_called_once_with()
        scgp.assert_called_once_with(decoder, p)


class TestGetLogicalError:

    def test_load_called(self, global3F: Global):
        e: Edge = ((0, 0), (0, 1))
        with mock.patch('localuf._pairs.Pairs.load') as mock_load:
            global3F.get_logical_error({e})
            mock_load.assert_called_once_with(e)

    def test_no_logical_error(self, global3F: Global):
        leftover: set[Edge] = {((0, j), (0, j+1)) for j in range(-1, 3-2)}
        assert global3F.get_logical_error(leftover) == 0

    def test_one_logical_error(self, global3F: Global):
        leftover: set[Edge] = {((0, j), (0, j+1)) for j in range(-1, 3-1)}
        assert global3F.get_logical_error(leftover) == 1

    def test_two_logical_errors(self, global3F: Global):
        leftover: set[Edge] = {((i, j), (i, j+1)) for j in range(-1, 3-1) for i in range(2)}
        assert global3F.get_logical_error(leftover) == 2