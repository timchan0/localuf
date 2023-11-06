import pytest

from localuf import Surface
from localuf.type_aliases import Node
from localuf.decoders.luf import LUF

@pytest.fixture(name="luf", params=[True, False], ids=["vis", "invis"])
def _luf(sf: Surface, request):
    return LUF(sf, request.param)

@pytest.fixture
def macar(sf: Surface):
    return LUF(sf)

@pytest.fixture
def actis(sf: Surface):
    return LUF(code=sf, visible=False)

@pytest.fixture(name="luf3F", params=[True, False], ids=["vis", "invis"])
def _luf3F(sf3F: Surface, request):
    return LUF(sf3F, request.param)

@pytest.fixture(name="luf3T", params=[True, False], ids=["vis", "invis"])
def _luf3T(sf3T: Surface, request):
    return LUF(sf3T, request.param)

@pytest.fixture
def macar3F(sf3F: Surface):
    return LUF(sf3F)

@pytest.fixture
def actis3Fu(sf3F: Surface):
    return LUF(sf3F, False, False)

@pytest.fixture
def macar3T(sf3T: Surface):
    return LUF(sf3T)

@pytest.fixture
def actis3Tu(sf3T: Surface):
    return LUF(sf3T, False, False)

@pytest.fixture
def macar5F(sf5F: Surface):
    return LUF(sf5F)

@pytest.fixture
def actis5Fu(sf5F: Surface):
    return LUF(sf5F, False, False)

@pytest.fixture
def actis5T(sf5T: Surface):
    return LUF(sf5T, False)

@pytest.fixture
def macar5T(sf5T: Surface):
    return LUF(sf5T)

@pytest.fixture(name="ns3F", params=[True, False], ids=["vis", "invis"])
def _ns3F(sf3F, request):
    luf = LUF(sf3F, request.param)
    return luf.NODES

@pytest.fixture(name="ns3T", params=[True, False], ids=["vis", "invis"])
def _ns3T(sf3T, request):
    luf = LUF(sf3T, request.param)
    return luf.NODES

@pytest.fixture
def actis_nodes(actis: LUF):
    return actis.NODES

@pytest.fixture
def actis5Fu_nodes(actis5Fu: LUF):
    return actis5Fu.NODES

@pytest.fixture
def actis5T_nodes(actis5T: LUF):
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