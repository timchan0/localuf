from unittest import mock

import pytest

from localuf.decoders.luf import Macar, MacarNode
from localuf.decoders.luf.constants import Stage

@pytest.fixture
def vn(macar: Macar):
    v = (0, 0) if macar.NODES.LUF.CODE.DIMENSION==2 else (0, 0, 0)
    return MacarNode(macar.NODES, v)

def test_advance(vn: MacarNode):
    # from https://stackoverflow.com/a/63690318/20887677
    with mock.patch("localuf.decoders.luf.MacarNode.growing") as growing_mock:
        vn.advance()
        growing_mock.assert_called_once_with()
    vn.NODES.LUF.CONTROLLER.stage = Stage.MERGING
    with mock.patch("localuf.decoders.luf.MacarNode.merging") as merging_mock:
        vn.advance()
        merging_mock.assert_called_once_with()
    vn.NODES.LUF.CONTROLLER.stage = Stage.PRESYNCING
    with mock.patch("localuf.decoders.luf.MacarNode.presyncing") as presyncing_mock:
        vn.advance()
        presyncing_mock.assert_called_once_with()
    vn.NODES.LUF.CONTROLLER.stage = Stage.SYNCING
    with mock.patch("localuf.decoders.luf.MacarNode.syncing") as syncing_mock:
        vn.advance()
        syncing_mock.assert_called_once_with()