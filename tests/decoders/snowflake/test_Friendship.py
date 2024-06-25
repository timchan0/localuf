import pytest

from localuf.decoders.snowflake import _Node, Friendship

@pytest.mark.parametrize("prop", [
    "NODE",
])
def test_property_attributes(test_property, sfn3: _Node, prop):
    test_property(sfn3.FRIENDSHIP, prop)


def test_drop(sfn3: _Node):
    fs = Friendship(sfn3)
    sfn3.unrooted = True
    sfn3.next_unrooted = True
    fs.drop()
    assert sfn3.unrooted is False
    assert sfn3.next_unrooted is False