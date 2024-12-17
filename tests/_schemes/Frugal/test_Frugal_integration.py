import itertools

import pytest

from localuf import Surface
from localuf.decoders import Snowflake
from localuf._schemes import Frugal


@pytest.fixture
def sf_frugal3():
    sf3 = Surface(
        3,
        noise='phenomenological',
        scheme='frugal',
    )
    return sf3.SCHEME


def test_get_logical_error(sf_frugal3: Frugal):
    
    decoder = Snowflake(sf_frugal3._CODE)
    h = sf_frugal3.WINDOW_HEIGHT
    m = 0
    for error in itertools.chain(
        [
            {
                ((1, 1, h-1), (1, 1, h)),
                ((0, 1, h-1), (0, 1, h)),
            },
            {
                ((1, 1, h-1), (1, 2, h-1)),
            },
        ],
        itertools.repeat(set(), 2*h),
    ):
        sf_frugal3._raise_window()
        syndrome = sf_frugal3._load(error)
        decoder.decode(syndrome)
        m += sf_frugal3.get_logical_error()
    assert m == 0
    # below failed if pairs w/ zero pair separation (in j) are not added to `new_pairs`
    assert sf_frugal3.pairs._dc == {}