import pytest

from localuf.codes import Surface
from localuf.decoders.uf import UF
from localuf.decoders.node_uf import NodeUF


@pytest.mark.parametrize('decoder_class', (UF, NodeUF))
@pytest.mark.parametrize('row', (0, 1))
class TestUpdateBoundary:

    def test_d2(self, decoder_class, row):
        code = Surface(2, 'phenomenological')
        decoder = decoder_class(code)
        syndrome = {(row, 0, 0)}
        decoder.decode(syndrome)
        assert not decoder.CODE.get_logical_error(decoder.correction)
        assert decoder.correction == {((row, 0, 0), (row, 1, 0))}