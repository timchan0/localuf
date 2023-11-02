from unittest import mock

from localuf.decoders.luf import ActisNodes, UnoptimalWaiter
from localuf.decoders.luf.constants import Stage

def test_advance(actis_nodes: ActisNodes):
    actis_nodes._WAITER = UnoptimalWaiter(actis_nodes)
    iw = actis_nodes._WAITER
    with mock.patch("localuf.decoders.luf.ActisNodes.update_valid") as mock_update_valid:
        iw.advance()

        mock_update_valid.assert_not_called()
        assert actis_nodes.countdown == actis_nodes.SPAN + 1

        actis_nodes.countdown = 0
        actis_nodes.LUF.CONTROLLER.stage = Stage.MERGING
        iw.advance()

        mock_update_valid.assert_not_called()
        assert actis_nodes.countdown == actis_nodes.SPAN
        
        actis_nodes.countdown = 0
        actis_nodes.LUF.CONTROLLER.stage = Stage.PRESYNCING
        iw.advance()

        mock_update_valid.assert_not_called()
        assert actis_nodes.countdown == actis_nodes.SPAN

        actis_nodes.countdown = 0
        actis_nodes.LUF.CONTROLLER.stage = Stage.SYNCING
        iw.advance()

        mock_update_valid.assert_called_once_with()
        assert actis_nodes.countdown == actis_nodes.SPAN

    actis_nodes.busy_signal = True
    iw.advance()

    assert actis_nodes.countdown == actis_nodes.SPAN + 1

    actis_nodes.busy_signal = False
    iw.advance()

    assert actis_nodes.countdown == actis_nodes.SPAN