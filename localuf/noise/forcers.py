"""Classes for ``noise.CircuitLevel.force_error``.

This method samples an error from the set of all errors of a given weight
where weight can be an integer or a vector of integers
depending on the noise model.
"""


import abc
import itertools
import random
from functools import cached_property
from typing import Iterable

import numpy as np
from scipy.stats import binom

from localuf.type_aliases import Edge, FourInts, FourFloats


class _BaseForcer(abc.ABC):
    """Abstract base class for all forcers.
    
    Instance attributes:
    * ``_EDGES`` maps each multiplicity 4-vector
        to the corresponding subset as a list of edges.
    """

    def __init__(self, edges: dict[FourInts, list[Edge]]) -> None:
        self._EDGES = edges

    @abc.abstractmethod
    def force_error(self, weight: tuple[int, ...]) -> set[Edge]:
        """Make error whose weight in pair/edge subset ``k`` is ``weight[k]``."""

    @property
    @abc.abstractmethod
    def ALL_WEIGHTS(self) -> tuple[tuple[int, ...], ...]:
        """All possible ``force_error`` inputs."""
    
    @abc.abstractmethod
    def subset_probability(
        self,
        weights: tuple[tuple[int, ...], ...],
        pi: FourFloats,
    ) -> Iterable[float]:
        """Compute probability of each subset characterized by ``weight`` in ``weights``.
        
        See ``Noise.subset_probability``.
        
        
        :param weights: a tuple of inputs to ``force_error``.
        :param pi: a 4-tuple of probabilities.
        
        
        :returns: ``probs`` an iterable of probabilities where each corresponds to a ``weight`` in ``weights``.
        
        Example for ``ForceByPairBalanced``:
        * ``weights = ((0, 0), (0, 1), (1, 0), (1, 1))``
        * ``B_k = B(n=len(self.PAIR_POPULATIONS[k]), p=pi[k])``
        * ``probs`` = (
        
        ``B_0(0) * B_1(0)``,
        
        ``B_0(0) * B_1(1)``,
        
        ``B_0(1) * B_1(0)``,
        
        ``B_0(1) * B_1(1)``,
        )
        """


class _BaseForceByPair(_BaseForcer):
    """Base class for forcers whose error subsets are distinguished by the weight of each pair type.
    
    Extends ``_BaseForcer``.
    """

    PAIR_POPULATIONS: tuple[list[Edge], ...]
    """Population for each pair type."""

    def force_error(self, weight: tuple[int, ...]):
        """Make error whose weight in pair subset ``k`` is ``weight[k]``."""
        error: set[Edge] = set()
        for pairs, w in zip(self.PAIR_POPULATIONS, weight, strict=True):
            sample = random.sample(pairs, w)
            for e in sample:
                if e in error:
                    error.remove(e)
                else:
                    error.add(e)
        return error
    
    @property
    def ALL_WEIGHTS(self) -> tuple[tuple[int, ...], ...]:
        return tuple(itertools.product(*(
            range(len(pairs)+1) for pairs in self.PAIR_POPULATIONS
        )))
    
    def subset_probability(self, weights, pi) -> Iterable[float]:
        probs = np.ones(len(weights))
        weights = np.array(weights).T
        # truncate pi to same length as PAIR_POPULATIONS
        pairs_pi = zip(self.PAIR_POPULATIONS, pi)
        for row, (pairs, p) in zip(weights, pairs_pi, strict=True):
            probs *= binom.pmf(
                k=row,
                n=len(pairs),
                p=p,
            )
        return probs

    @property
    def _PAIR_POPULATIONS(self):
        """Helper property for ``PAIR_POPULATIONS``."""
        populations: list[list[Edge]] = []
        for k in range(4):
            pairs = []
            for m, edges in self._EDGES.items():
                pairs += m[k] * edges
            populations.append(pairs)
        return tuple(populations)


class ForceByPair(_BaseForceByPair):
    """Forcer whose error subsets are distinguished by the weight of each of the 4 pair types.
    
    Extends ``_BaseForceByPair``.
    Use when ``parametrization != 'balanced'``.
    In this class, each weight is a 4-tuple of integers.
    """
    
    @cached_property
    def PAIR_POPULATIONS(self):
        return self._PAIR_POPULATIONS


class ForceByPairBalanced(_BaseForceByPair):
    """Same as ``ForceByPair`` but aggregating the last 3 pair subsets.
    
    Extends ``_BaseForceByPair``.
    Use when ``parametrization == 'balanced'``.
    In this class, each weight is a 2-tuple of integers.
    """
    
    @cached_property
    def PAIR_POPULATIONS(self):
        p0, *p_rest = self._PAIR_POPULATIONS
        return (p0, sum(p_rest, start=[]))


class ForceByEdge(_BaseForcer):
    """Forcer whose error subsets are distinguished by the weight of each of the 13 edge types.
    
    Slower than ``ForceByPair`` as there are 13 edge types but only 4 pair types.
    Hence, use is discouraged.
    """

    def force_error(self, weight: tuple[int, ...]):
        """Make error whose weight in edge subset ``k`` is ``weight[k]``."""
        error: set[Edge] = set()
        try:
            for subset, w in zip(self._EDGES.values(), weight, strict=True):
                error.update(random.sample(subset, w))
        except ValueError as e:
            if str(e) == 'Sample larger than population or is negative':
                raise e
            else:
                raise ValueError(
                    f'len(weight)={len(weight)} != {len(self._EDGES)}=number of edge subsets'
                ) from e
        return error
    
    @property
    def ALL_WEIGHTS(self) -> tuple[tuple[int, ...], ...]:
        return tuple(itertools.product(*(
            range(len(subset)+1) for subset in self._EDGES.values()
        )))
    
    def subset_probability(self, weights, pi) -> Iterable[float]:
        raise NotImplementedError