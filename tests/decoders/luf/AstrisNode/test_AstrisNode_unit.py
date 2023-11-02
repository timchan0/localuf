from unittest import mock

import pytest

from localuf.decoders.luf import LUF, AstrisNode
from localuf.decoders.luf.constants import Stage

@pytest.fixture
def vn(astris: LUF):
    v = (0, 0) if astris.NODES.LUF.CODE.DIMENSION==2 else (0, 0, 0)
    return AstrisNode(astris.NODES, v)

def test_advance(vn: AstrisNode):
    # from https://stackoverflow.com/a/63690318/20887677
    with mock.patch("localuf.decoders.luf.AstrisNode.growing") as growing_mock:
        vn.advance()
        growing_mock.assert_called_once_with()
    vn.NODES.LUF.CONTROLLER.stage = Stage.MERGING
    with mock.patch("localuf.decoders.luf.AstrisNode.merging") as merging_mock:
        vn.advance()
        merging_mock.assert_called_once_with()
    vn.NODES.LUF.CONTROLLER.stage = Stage.PRESYNCING
    with mock.patch("localuf.decoders.luf.AstrisNode.presyncing") as presyncing_mock:
        vn.advance()
        presyncing_mock.assert_called_once_with()
    vn.NODES.LUF.CONTROLLER.stage = Stage.SYNCING
    with mock.patch("localuf.decoders.luf.AstrisNode.syncing") as syncing_mock:
        vn.advance()
        syncing_mock.assert_called_once_with()