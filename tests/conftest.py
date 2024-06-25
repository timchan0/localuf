import itertools

import pytest

import localuf
from localuf import Repetition, Surface

@pytest.fixture(name="sf", params=itertools.product(
    range(3, 11, 2),
    (
        'code capacity',
        'phenomenological',
        'circuit-level',
    ),
), ids=lambda x: f"d{x[0]} {x[1]}")
def _sf(request):
    return localuf.Surface(*request.param)

@pytest.fixture
def sf3F():
    return localuf.Surface(3, 'code capacity')

@pytest.fixture
def sf3T():
    return localuf.Surface(3, 'phenomenological')

@pytest.fixture  # unused
def sf3O():
    return localuf.Surface(
        d=3,
        noise='phenomenological',
        scheme='forward',
    )

@pytest.fixture
def sf5F():
    return localuf.Surface(5, 'code capacity')

@pytest.fixture
def sf5T():
    return localuf.Surface(5, 'phenomenological')

@pytest.fixture
def sf7F():
    return localuf.Surface(7, 'code capacity')

@pytest.fixture
def sf9F():
    return localuf.Surface(9, 'code capacity')

@pytest.fixture
def test_property():
    def f(instance: object, prop: str):
        with pytest.raises(AttributeError):
            delattr(instance, prop)
        with pytest.raises(AttributeError):
            setattr(instance, prop, None)
    return f


@pytest.fixture
def rp3_forward():
    return Repetition(
        3,
        noise='phenomenological',
        scheme='forward',
    )


@pytest.fixture
def rp3_frugal():
    return Repetition(
        3,
        noise='phenomenological',
        scheme='frugal',
    )


@pytest.fixture
def batch3F(sf3F: Surface):
    return sf3F.SCHEME


@pytest.fixture
def forward_rp(rp_forward: Repetition):
    return rp_forward.SCHEME


@pytest.fixture(params=range(3, 11, 2), ids=lambda x: f"d{x}")
def rp_forward(request):
    return Repetition(
        request.param,
        noise='phenomenological',
        scheme='forward',
    )


@pytest.fixture(
        name="sfCL",
        params=itertools.product(range(3, 9, 2), range(2, 5)),
        ids=lambda x: f"d{x[0]} h{x[1]}",
)
def _sfCL(request):
    d, h = request.param
    return Surface(d=d, noise='circuit-level', window_height=h)


@pytest.fixture(
        name="sfCL_OL",
        params=range(3, 9, 2),
        ids=lambda x: f"d{x}",
)
def _sfCL_OL(request):
    return Surface(
        d=request.param,
        noise='circuit-level',
        scheme='forward',
    )

@pytest.fixture
def sfCL_OL_scheme(sfCL_OL: Surface):
    return sfCL_OL.SCHEME