from unittest import mock

import pytest

from localuf.constants import Growth
from localuf.decoders import Snowflake
from localuf.decoders.snowflake import _Edge


@pytest.fixture
def sfe3a(snowflake3: Snowflake):
    """`_Edge` instance with `EdgeContact`."""
    return snowflake3.EDGES[(0, 1), (1, 1)]


@pytest.mark.parametrize("prop", [
    "DROPEE",
])
def test_property_attributes(test_property, sfe3a: _Edge, prop):
    test_property(sfe3a.CONTACT, prop)


def test_init(sfe3a: _Edge):
    with mock.patch("localuf.decoders.snowflake._Contact.__init__") as mock_init:
        sfe3a.CONTACT.__init__(sfe3a)
        mock_init.assert_called_once_with(sfe3a)


def test_drop(snowflake3: Snowflake):

    edge_1 = snowflake3.EDGES[(0, 1), (1, 1)]
    edge_0 = snowflake3.EDGES[(0, 0), (1, 0)]

    edge_1.growth = Growth.FULL
    edge_1.correction = True

    edge_1.CONTACT.drop()
    
    assert edge_0.next_growth is Growth.FULL
    assert edge_0.next_correction is True