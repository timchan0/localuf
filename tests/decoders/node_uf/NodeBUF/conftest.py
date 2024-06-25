import pytest
from localuf.decoders import NodeBUF


@pytest.fixture
def node_buf(sf7F):
    nbuf = NodeBUF(sf7F)
    nbuf.history = []
    return nbuf