from unittest import mock

import pytest

from localuf.decoders import Snowflake
from localuf.decoders.snowflake import _Node


@pytest.fixture
def sfn3b(snowflake3: Snowflake):
    """`_Node` instance with `TopSheetFriendship`."""
    h = snowflake3.CODE.SCHEME.WINDOW_HEIGHT
    return snowflake3.NODES[0, h-1]


def test_drop(sfn3b: _Node):
    sfn3b.next_defect = True
    sfn3b.next_active = True
    with mock.patch("localuf.decoders.snowflake.NodeFriendship.drop") as mock_drop:
        sfn3b.FRIENDSHIP.drop()
        mock_drop.assert_called_once_with()
    assert sfn3b.next_defect is False
    assert sfn3b.next_active is False