import pytest

from localuf._pairs import Pairs
from localuf.type_aliases import Node


@pytest.fixture
def pairs():
    return Pairs()


@pytest.fixture
def uvwx() -> tuple[Node, Node, Node, Node]:
    return (0, 0), (1, 0), (0, 1), (1, 1)