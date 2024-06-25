from unittest import mock

import pytest

from localuf import Surface
from localuf.type_aliases import Edge
from localuf.decoders import UF
from localuf._schemes import Global


@pytest.fixture
def global3F():
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


def test_get_logical_error(global3F: Global):
    e: Edge = ((0, 0), (0, 1))
    with mock.patch('localuf._pairs.Pairs.load') as mock_load:
        global3F.get_logical_error({e})
        mock_load.assert_called_once_with(e)