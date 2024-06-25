from unittest import mock

import pytest

from localuf.decoders import Snowflake
from localuf.decoders.snowflake import _Node


@pytest.fixture
def sfn3a(snowflake3: Snowflake):
    """`_Node` instance with `NodeFriendship`."""
    return snowflake3.NODES[0, 1]


@pytest.mark.parametrize("prop", [
    "DROPEE",
])
def test_property_attributes(test_property, sfn3a: _Node, prop):
    test_property(sfn3a.FRIENDSHIP, prop)


def test_init(sfn3a: _Node):
    with mock.patch("localuf.decoders.snowflake.Friendship.__init__") as mock_init:
        sfn3a.FRIENDSHIP.__init__(sfn3a)
        mock_init.assert_called_once_with(sfn3a)


def test_drop(snowflake3: Snowflake):

    node_0 = snowflake3.NODES[0, 0]
    node_1 = snowflake3.NODES[0, 1]

    node_1.defect = True
    node_1.active = True

    with mock.patch("localuf.decoders.snowflake.Friendship.drop") as mock_drop:
        node_1.FRIENDSHIP.drop()
        mock_drop.assert_called_once_with()

    assert node_0.next_defect is True
    assert node_0.next_active is True