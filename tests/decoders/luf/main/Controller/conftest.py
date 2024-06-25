import pytest
from localuf.decoders.luf import Macar, Controller


@pytest.fixture
def c3(sf3F):
    macar = Macar(sf3F)
    return Controller(macar)