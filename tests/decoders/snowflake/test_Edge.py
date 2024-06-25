from unittest import mock

import pytest

from localuf.constants import Growth
from localuf.decoders.snowflake import Snowflake, _Edge, FloorContact, EdgeContact

@pytest.mark.parametrize("prop", [
    "SNOWFLAKE",
    "INDEX",
    "CONTACT",
])
def test_property_attributes(test_property, sfe3: _Edge, prop):
    test_property(sfe3, prop)


def test_init(sfe3: _Edge):
    with mock.patch("localuf.decoders.snowflake._Edge.reset") as mock_reset:
        sfe3.__init__(sfe3.SNOWFLAKE, sfe3.INDEX)
        mock_reset.assert_called_once_with()


def test_SNOWFLAKE_property(sfe3: _Edge):
    assert type(sfe3.SNOWFLAKE) is Snowflake
    assert sfe3.SNOWFLAKE.EDGES[sfe3.INDEX] == sfe3


def test_INDEX_property(sfe3: _Edge, fixture_test_INDEX_property):
    fixture_test_INDEX_property(sfe3)


def test_CONTACT_property(snowflake: Snowflake):
    for (u, v), edge in snowflake.EDGES.items():
        ut = u[snowflake.CODE.TIME_AXIS]
        vt = v[snowflake.CODE.TIME_AXIS]
        if ut != 0 and vt != 0:
            assert type(edge.CONTACT) is EdgeContact
        else:
            assert type(edge.CONTACT) is FloorContact


def test_reset(sfe3: _Edge):

    sfe3.growth = Growth.FULL
    sfe3.correction = True

    sfe3.next_growth = Growth.FULL
    sfe3.next_correction = True

    sfe3.reset()

    assert sfe3.growth is Growth.UNGROWN
    assert sfe3.correction is False

    assert sfe3.next_growth is Growth.UNGROWN
    assert sfe3.next_correction is False


def test_update_after_drop(sfe3: _Edge):
    sfe3.next_growth = Growth.FULL
    sfe3.next_correction = True
    sfe3.update_after_drop()
    assert sfe3.growth is Growth.FULL
    assert sfe3.correction is True
    assert sfe3.next_growth is Growth.FULL
    assert sfe3.next_correction is True