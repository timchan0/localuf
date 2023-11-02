import abc
import itertools

from localuf.type_aliases import Node, Edge
from localuf.codes import Code


class BaseDecoder(abc.ABC):
    """Base class for decoders.
    
    Instance attributes (1 constant):
    * `CODE` the code to be decoded.
    * `correction` a set of edges comprising the correction.
    """

    @property
    def CODE(self): return self._CODE

    def __init__(self, code: Code):
        """Input:
        * `code` the code to be decoded.
        """
        self._CODE = code
        self.correction: set[Edge]

    def reset(self):
        """Factory reset."""
        try: del self.correction
        except AttributeError: pass

    @abc.abstractmethod
    def decode(
            self,
            syndrome: set[Node],
            **kwargs,
    ):
        """Decode syndrome.
        
        Inputs:
        * `syndrome` the set of defects.
        * `draw` whether to draw.

        Correction stored in `self.correction`.
        """
    
    @abc.abstractmethod
    def _draw_decode(self, **kwargs):
        """Draw all stages of decoding."""

    def _sim_cycle_given_error(self, error: set[Edge]):
        """Simulate a decoding cycle given `error`.
        
        Inputs:
        * `error` the set of bitflipped edges.

        Outputs:
        * `0` if success else `1`.
        """
        syndrome = self.CODE.get_syndrome(error)
        self.reset()
        self.decode(syndrome)
        leftover = error ^ self.correction
        return self.CODE.get_logical_error(leftover)
    
    def sim_cycle_given_p(self, p: float):
        """Simulate a decoding cycle given `p`.
        
        Inputs:
        * `p` physical error probability.

        Outputs:
        * `0` if success else `1`.
        """
        error = self.CODE.make_error(p)
        return self._sim_cycle_given_error(error)
    
    def sim_cycles_given_p(self, p: float, n: int):
        """Simulate `n` decoding cycles given `p`.
        
        Inputs:
        * `p` physical error probability.
        * `n` number of decoding cycles.

        Outputs:
        * tuple of (number of failures, `n`).
        """
        # itertools.repeat faster than range
        m = sum(self.sim_cycle_given_p(p) for _ in itertools.repeat(None, n))
        return m, n