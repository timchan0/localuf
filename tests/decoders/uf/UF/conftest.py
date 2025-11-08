import pytest

from localuf.codes import Surface
from localuf.decoders.uf import UF


@pytest.fixture
def uf7F(sf7F: Surface):
    decoder = UF(sf7F)
    decoder.history = []
    return decoder