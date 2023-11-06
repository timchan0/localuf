import pytest

from localuf.type_aliases import Node
from localuf.decoders.luf import LUF

# edit be independent of luf!
@pytest.mark.parametrize("luf, syndrome", [
    ("macar5F", "syndrome5F"),
    ("macar5T", "syndrome5T"),
])
def test_load(luf: LUF, syndrome: set[Node], request):
    luf = request.getfixturevalue(luf)
    syndrome = request.getfixturevalue(syndrome)
    luf.NODES.load(syndrome)
    for v, node in luf.NODES.dc.items():
        if v in syndrome:
            assert node.defect
            assert node.anyon
            assert node.active
        else:
            assert not node.defect
            assert not node.anyon
            assert not node.active