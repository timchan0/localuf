import pytest

from localuf import Surface


@pytest.fixture
def batch3F(sf3F: Surface):
    return sf3F.SCHEME