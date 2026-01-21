from localuf.decoders.luf import LUF, MacarNode, Nodes
from localuf.decoders.luf.constants import Stage


class AlternativeMacarNodes(Nodes):
    """Alternative to ``MacarNodes`` where ``valid``
    is an attribute rather than a property
    and is updated after every PRESYNCING stage.
    
    Extends ``Nodes``.
    Designed to emulate faster than ``MacarNodes`` but empirically not.
    Hence currently unused.
    """

    def __init__(self, luf: LUF) -> None:
        super().__init__(luf)
        self._dc = {v: MacarNode(nodes=self, index=v)
                    for v in luf.CODE.NODES}
        self.reset()

    @property
    def dc(self): return self._dc

    @property
    def busy(self):
        return any(node.busy for node in self.dc.values())

    def reset(self):
        self.valid = True
        return super().reset()

    def load(self, syndrome):
        super().load(syndrome)
        self.valid = False

    def update_valid(self):
        """Update ``valid`` attribute. Call after every PRESYNCING stage."""
        self.valid = not any(node.active for node in self.dc.values())

    def advance(self):
        super().advance()
        if self.LUF.CONTROLLER.stage is Stage.PRESYNCING:
            self.update_valid()