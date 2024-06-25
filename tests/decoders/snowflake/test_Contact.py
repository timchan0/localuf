import pytest

from localuf.decoders.snowflake import _Edge

@pytest.mark.parametrize("prop", [
    "EDGE",
])
def test_property_attributes(test_property, sfe3: _Edge, prop):
    test_property(sfe3.CONTACT, prop)