"""Module for local Union--Find decoder."""

from localuf.codes import Surface
from localuf._schemes import Batch, Frugal
from localuf.decoders.luf.constants import Stage
from localuf.decoders.luf.main import \
    LUF, Controller, Nodes, MacarNodes, ActisNodes, \
    Waiter, OptimalWaiter, UnoptimalWaiter, \
    _Node, MacarNode, ActisNode, \
    Friendship, ControllerFriendship, NodeFriendship


class Macar(LUF):
    """Local UF where controller directly connects to each node.
    
    Extends ``LUF``.
    Compatible only with surface code, and all schemes save frugal.
    """

    def __init__(self, code: Surface):
        """Input: ``code`` the code to be decoded."""
        if isinstance(code.SCHEME, Frugal):
            raise ValueError('Macar incompatible with frugal scheme.')
        super().__init__(code, visible=True)


class Actis(LUF):
    """Strictly local UF i.e. controller only connects to node whose ID is 0.
    
    Extends ``LUF``.
    Compatible only with surface code and batch scheme.
    """

    def __init__(self, code: Surface, _optimal=True):
        """
        :param code: the code to be decoded.
        :param _optimal: whether management of ``self.NODES.countdown`` is optimal.
        """
        if not isinstance(code.SCHEME, Batch):
            raise ValueError('Actis compatible only with batch scheme.')
        super().__init__(code, visible=False, _optimal=_optimal)