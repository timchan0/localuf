import pytest
from localuf.decoders.luf import LUF, Controller


@pytest.fixture
def c3(sf3F):
    luf = LUF(sf3F)
    return Controller(luf)