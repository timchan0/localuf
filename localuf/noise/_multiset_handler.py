from math import prod

from localuf.type_aliases import FourFloats, FourInts, MultiplicityVector


class MultisetHandler:

    @staticmethod
    def pr(multiplicities: FourInts | MultiplicityVector, pi: FourFloats) -> float:
        """Return probability of odd number of faults in multiset
        defined by ``multiplicities``, ``pi``.
        
        
        :param multiplicities: a multiplicity vector.
        :param pi: a 4-tuple of fault probabilities.
        """
        f1 = prod((1-p)**m for p, m in zip(pi, multiplicities))
        f2 = sum(m*p/(1-p) for p, m in zip(pi, multiplicities))
        return f1 * f2