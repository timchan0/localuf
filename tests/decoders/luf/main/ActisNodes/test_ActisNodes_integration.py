from localuf.decoders.luf import ActisNodes

def test_busy_signal_causality(actis5Fu_nodes: ActisNodes):
    """Ensure `busy_signal` of node whose ID is 0 resets not countdown of
    `ActisNodes` in same timestep when `ActisNodes.advance()` called.
    """
    rep = actis5Fu_nodes.dc[0, -1]
    rep.busy_signal = True
    rep.FRIENDSHIP.relay_signals()

    assert not actis5Fu_nodes.busy_signal
    assert actis5Fu_nodes.next_busy_signal

    actis5Fu_nodes._update_unphysicals_for_actis()

    assert actis5Fu_nodes.busy_signal
    assert not actis5Fu_nodes.next_busy_signal