import pytest

from localuf.codes import Surface
from localuf.decoders.uf import UF
from localuf.decoders.node_uf import NodeUF


@pytest.mark.parametrize('noise', ('phenomenological', 'circuit-level'))
@pytest.mark.parametrize('decoder_class', (UF, NodeUF))
@pytest.mark.parametrize('row', (0, 1))
class TestUpdateBoundary:

    def test_d2(self, noise, decoder_class, row):
        code = Surface(2, noise)
        decoder = decoder_class(code, inclination='west')
        syndrome = {(row, 0, 0)}
        decoder.decode(syndrome)
        assert decoder.CODE.get_logical_error(decoder.correction)
        assert decoder.correction == {((row, -1, 0), (row, 0, 0))}

    def test_d4(self, noise, decoder_class, row):
        code = Surface(4, noise)
        decoder = decoder_class(code, inclination='west')
        syndrome = {(row, 1, 0)}
        decoder.decode(syndrome)
        assert decoder.CODE.get_logical_error(decoder.correction)
        assert decoder.correction == {
            ((row, -1, 0), (row, 0, 0)),
            ((row, 0, 0), (row, 1, 0)),
        }