from unittest import mock

import pytest

from localuf import Surface
from localuf.decoders import UF
from localuf._determinants import SpaceDeterminant
from localuf._schemes import Batch

def test_DETERMINANT_attribute(batch3F: Batch):
    assert type(batch3F._DETERMINANT) is SpaceDeterminant

class TestRun:

    @pytest.mark.parametrize("n", range(1, 4))
    def test_code_capacity(self, sf3F: Surface, n: int):
        decoder = UF(sf3F)
        with mock.patch("localuf._schemes.Batch._sim_cycle_given_p", return_value=1) as mock_scgp:
            m, slenderness = sf3F.SCHEME.run(decoder, 1, n)
            assert mock_scgp.call_args_list == [mock.call(decoder, 1)] * n
            assert m == n
            assert slenderness == n

    @pytest.mark.parametrize("n", range(1, 4))
    def test_phenomenological(self, sf3T: Surface, n: int):
        decoder = UF(sf3T)
        fake_height = 2
        sf3T.SCHEME._WINDOW_HEIGHT = fake_height # type: ignore
        with mock.patch("localuf._schemes.Batch._sim_cycle_given_p", return_value=1) as mock_scgp:
            m, slenderness = sf3T.SCHEME.run(decoder, 1, n)
            assert mock_scgp.call_args_list == [mock.call(decoder, 1)] * n
            assert m == n
            assert slenderness == fake_height * n / 3