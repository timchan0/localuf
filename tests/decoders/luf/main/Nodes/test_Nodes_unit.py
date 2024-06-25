import pytest

from localuf.type_aliases import Node
from localuf.decoders.luf import Macar

# edit be independent of luf!
@pytest.mark.parametrize("decoder, syndrome", [
    ("macar5F", "syndrome5F"),
    ("macar5T", "syndrome5T"),
])
def test_load(decoder: Macar, syndrome: set[Node], request):
    decoder = request.getfixturevalue(decoder)
    syndrome = request.getfixturevalue(syndrome)
    decoder.NODES.load(syndrome)
    for v, node in decoder.NODES.dc.items():
        if v in syndrome:
            assert node.defect
            assert node.anyon
            assert node.active
        else:
            assert not node.defect
            assert not node.anyon
            assert not node.active