import pytest
from localuf.decoders import BUF


@pytest.fixture
def buf(sf7F):
    b = BUF(sf7F)
    b.history = []
    return b