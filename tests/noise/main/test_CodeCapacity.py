from localuf import Surface
from localuf.noise import CodeCapacity


def test_EDGES(sf3F: Surface):
    cc: CodeCapacity = sf3F.NOISE # type: ignore
    assert cc.EDGES == sf3F.EDGES