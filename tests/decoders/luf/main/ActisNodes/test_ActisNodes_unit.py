from unittest import mock

from localuf.decoders.luf import ActisNode, ActisNodes, Waiter

def test_init(actis_nodes: ActisNodes):

    assert type(actis_nodes._dc) is dict
    assert len(actis_nodes._dc) == len(actis_nodes.LUF.CODE.NODES)
    v, node = next(iter(actis_nodes._dc.items()))
    assert type(v) is tuple
    assert type(node) is ActisNode

    d = actis_nodes.LUF.CODE.D
    noise = actis_nodes.LUF.CODE.NOISE
    if str(noise) == 'code capacity':
        assert actis_nodes._SPAN == 1 + d-1 + d
    elif str(noise) == 'phenomenological':
        assert actis_nodes._SPAN == 1 + d-1 + d + d-1
    elif str(noise) == 'circuit-level':
        assert actis_nodes._SPAN == 1 + d
    else:
        raise ValueError(f"Unknown noise model: {noise}")

    assert isinstance(actis_nodes._WAITER, Waiter)

    with (
        mock.patch("localuf.decoders.luf.Nodes.__init__") as mock_init,
        mock.patch("localuf.decoders.luf.ActisNodes.reset") as mock_reset,
    ):
        actis_nodes.__init__(actis_nodes.LUF)
        mock_init.assert_called_once_with(actis_nodes.LUF)
        mock_reset.assert_called_once_with()

def test_reset(actis5Fu_nodes: ActisNodes):
    actis5Fu_nodes.busy = True
    actis5Fu_nodes.valid = False
    actis5Fu_nodes.countdown = 1
    actis5Fu_nodes.busy_signal = True
    actis5Fu_nodes.next_busy_signal = True
    actis5Fu_nodes.active_signal = True
    actis5Fu_nodes.next_active_signal = True

    actis5Fu_nodes.reset()

    assert not actis5Fu_nodes.busy
    assert actis5Fu_nodes.valid
    assert actis5Fu_nodes.countdown == 0
    assert not actis5Fu_nodes.busy_signal
    assert not actis5Fu_nodes.next_busy_signal
    assert not actis5Fu_nodes.active_signal
    assert not actis5Fu_nodes.next_active_signal

def test_update_valid(actis_nodes: ActisNodes):
    assert actis_nodes.valid
    actis_nodes.update_valid()
    assert actis_nodes.valid
    actis_nodes.active_signal = True
    actis_nodes.update_valid()
    assert not actis_nodes.valid
    assert not actis_nodes.active_signal

def test_advance(actis_nodes: ActisNodes):
    with (
        mock.patch("localuf.decoders.luf.Nodes.advance") as mock_ufn,
        mock.patch("localuf.decoders.luf.OptimalWaiter.advance") as mock_ew,
    ):
        actis_nodes.advance()
        mock_ufn.assert_called_once_with()
        mock_ew.assert_called_once_with()