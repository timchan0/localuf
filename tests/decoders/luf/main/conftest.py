import pytest

from localuf import Surface
from localuf.type_aliases import Node
from localuf.decoders.luf import Macar, Actis

@pytest.fixture(name="luf", params=[Macar, Actis], ids=["Macar", "Actis"])
def _luf(sf: Surface, request):
    return request.param(sf)

@pytest.fixture
def macar(sf: Surface):
    return Macar(sf)

@pytest.fixture
def actis(sf: Surface):
    return Actis(sf)

@pytest.fixture(name="luf3F", params=[Macar, Actis], ids=["Macar", "Actis"])
def _luf3F(sf3F: Surface, request):
    return request.param(sf3F)

@pytest.fixture(name="luf3T", params=[Macar, Actis], ids=["Macar", "Actis"])
def _luf3T(sf3T: Surface, request):
    return request.param(sf3T)

@pytest.fixture
def macar3F(sf3F: Surface):
    return Macar(sf3F)

@pytest.fixture
def actis3Fu(sf3F: Surface):
    return Actis(sf3F, _optimal=False)

@pytest.fixture
def macar3T(sf3T: Surface):
    return Macar(sf3T)

@pytest.fixture
def actis3Tu(sf3T: Surface):
    return Actis(sf3T, _optimal=False)

@pytest.fixture
def macar5F(sf5F: Surface):
    return Macar(sf5F)

@pytest.fixture
def actis5Fu(sf5F: Surface):
    return Actis(sf5F, _optimal=False)

@pytest.fixture
def actis5T(sf5T: Surface):
    return Actis(sf5T)

@pytest.fixture
def macar5T(sf5T: Surface):
    return Macar(sf5T)

@pytest.fixture(name="ns3F", params=[Macar, Actis], ids=["Macar", "Actis"])
def _ns3F(sf3F, request):
    decoder = request.param(sf3F)
    return decoder.NODES

@pytest.fixture(name="ns3T", params=[Macar, Actis], ids=["Macar", "Actis"])
def _ns3T(sf3T, request):
    decoder = request.param(sf3T)
    return decoder.NODES

@pytest.fixture
def actis_nodes(actis: Actis):
    return actis.NODES

@pytest.fixture
def actis5Fu_nodes(actis5Fu: Actis):
    return actis5Fu.NODES

@pytest.fixture
def actis5T_nodes(actis5T: Actis):
    return actis5T.NODES

@pytest.fixture
def syndrome5F() -> set[Node]:
    return {
        (0, 0),
        (0, 1),
        (0, 2),
        (4, 1),
    }

@pytest.fixture
def syndrome5T() -> set[Node]:
    return {
        (0, 0, 0),
        (0, 1, 1),
        (0, 2, 2),
        (4, 1, 3),
    }