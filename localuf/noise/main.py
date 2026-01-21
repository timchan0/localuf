"""Classes for noise models."""

import abc
from collections import defaultdict
from functools import cache
import random
from typing import Literal, Iterable

import numpy as np
import pandas as pd
from scipy.stats import binom

from localuf.noise.forcers import ForceByEdge, ForceByPair, ForceByPairBalanced
from localuf.type_aliases import Edge, EdgeType, Parametrization, FourFloats, MultiplicityVector, FourInts
from localuf.noise._multiset_handler import MultisetHandler


class Noise(abc.ABC):
    """Abstract base class for noise models."""

    @abc.abstractmethod
    def __str__(self) -> str: raise NotImplementedError

    @property
    @abc.abstractmethod
    def ALL_WEIGHTS(self) -> Iterable[int] | tuple[tuple[int, ...], ...]:
        """All possible ``force_error`` inputs."""
    
    @property
    @abc.abstractmethod
    def ALL_WEIGHTS_INDEX(self) -> pd.Index:
        """``ALL_WEIGHTS`` as pandas Index."""

    @abc.abstractmethod
    def make_error(self, p: float) -> set[Edge]:
        """See ``Code.make_error``."""

    @abc.abstractmethod
    def force_error(self, weight: int | tuple[int, ...]) -> set[Edge]:
        """Make error of weight ``weight``."""
    
    @abc.abstractmethod
    def subset_probability(
        self,
        weights: Iterable[int] | tuple[tuple[int, ...], ...],
        p: float,
    ) -> Iterable[float]:
        """Return probability of any error of weight ``weight``, for ``weight`` in ``weights``."""

    def subset_probabilities(self, p: float, survival: bool = True):
        """Return DataFrame containing probabilities of each subset.
        
        
        :param p: noise level.
        :param survival: whether to compute survival probability column.
        
        
        :returns: DataFrame indexed by subset weight, with columns ``['subset prob', 'survival prob']``.
        """
        subset_prob = self.subset_probability(self.ALL_WEIGHTS, p)
        df = pd.DataFrame({'subset prob': subset_prob}, index=self.ALL_WEIGHTS_INDEX)
        if survival:
            df.sort_values(by='subset prob', inplace=True)
            df['survival prob'] = df['subset prob'].cumsum().shift(fill_value=0)
        df.sort_values(by='subset prob', ascending=False, inplace=True)
        return df

    @abc.abstractmethod
    def get_edge_weights(
        self,
        noise_level: None | float,
    ) -> dict[Edge, tuple[float, float]]:
        """Return map from edge to its flip probability and weight.
        
        
        :param noise_level: a probability that represents the noise strength.
            This is needed to define nonuniform edge weights of the decoding graph
        in the circuit-level noise model.
        If ``None``, all edges have flip probability 0 and weight 1.
        
        
        :returns: map from edge to the pair (flip probability, weight).
        """

    @staticmethod
    def log_odds_of_no_flip(p: float) -> float:
        """Convert flip probability to log odds of no flip."""
        return np.log10((1-p)/p)


class _Uniform(Noise):
    """Base class for noise model where each edge has flip probability ``p``.
    
    Extends ``Noise``.
    
    Attributes:
    * ``EDGES`` the edges of the freshly discovered region after a window raise
        if scheme is streaming; else,
    the edges of G.
    """

    def __init__(self, edges: tuple[Edge, ...]):
        """
        ``edges`` the edges of the freshly discovered region after a window raise
        if scheme is streaming; else,
        the edges of G.
        """
        self._EDGES = edges

    @property
    def EDGES(self): return self._EDGES

    def make_error(self, p):
        return {e for e in self.EDGES if random.random() < p}
    
    def force_error(self, weight: int):
        """Bitlfip exactly ``weight`` edges in G.
        
        Input: ``weight`` between 0 and ``len(self.EDGES)``.
        Output: The set of bitflipped edges.
        """
        return set(random.sample(self.EDGES, weight))

    @property
    def ALL_WEIGHTS(self) -> Iterable[int]:
        return range(len(self.EDGES)+1)
    
    @property
    def ALL_WEIGHTS_INDEX(self):
        return pd.Index(self.ALL_WEIGHTS, name='weight')
    
    def subset_probability(self, weights: Iterable[int], p: float) -> Iterable[float]:
        probs = binom.pmf(
            k=weights,
            n=len(self.EDGES),
            p=p,
        )
        return probs
    
    @cache
    def get_edge_weights(self, noise_level: None | float) -> dict[Edge, tuple[float, float]]:
        if noise_level is None:
            flip_probability = 0
            weight = 1
        else:
            flip_probability = noise_level
            weight = self.log_odds_of_no_flip(noise_level)
        return {e: (flip_probability, weight) for e in self.EDGES}


class CodeCapacity(_Uniform):
    """Code capacity noise model.
    
    Extends ``_Uniform``.
    """

    def __str__(self) -> str:
        return 'code capacity'


class Phenomenological(_Uniform):
    """Phenomenological noise model.
    
    Extends ``_Uniform``.
    """

    def __str__(self) -> str:
        return 'phenomenological'


class CircuitLevel(Noise):
    """Circuit-level depolarizing noise model.
    
    Class attributes:
    * ``_DEFAULT_MULTIPLICITIES`` maps from edge type to multiplicity vector
        e.g. ``{'S': (4, 2, 1, 0), ...}``.
    * ``_ALL_COEFFICIENTS`` maps from parametrization name
        to 4-vector c such that pi = c*p.
    
    Private attributes:
    * ``_EDGES`` maps from multiplicity vector (as a tuple) to tuple of edges
        e.g. ``{(4, 2, 1, 0): (e1, ...), ...}``.
    Order matters as used by ``ForceByEdge.force_error``.
    The union of all edge tuples in ``_EDGES`` is the freshly discovered region after a window raise
    if scheme is streaming; else, the edges of G.
    * ``_COEFFICIENTS`` a 4-vector c such that pi = c*p.
    * ``_FORCER`` method for forcing error.
    """

    _DEFAULT_MULTIPLICITIES: dict[EdgeType, FourInts] = {
        'S':               (4, 2, 1, 0),
        'E westmost':      (1, 0, 2, 0),
        'E bulk':          (2, 0, 1, 0),
        'E eastmost':      (1, 0, 1, 0),
        'U 3':             (3, 0, 1, 1),
        'U 4':             (4, 0, 0, 1),
        'SD':              (2, 0, 0, 0),
        'EU west corners': (2, 0, 1, 0),
        'EU east corners': (2, 0, 2, 0),
        'EU edge':         (3, 0, 1, 0),
        'EU centre':       (4, 0, 0, 0),
        'SEU':             (2, 0, 0, 0),
    }

    _ALL_COEFFICIENTS: dict[Parametrization, FourFloats] = {
        'standard': (4/15, 8/15, 2/3, 1),
        'balanced': (4/15, 8/15, 8/15, 8/15),
        'ion trap': (4/15, 8/15, (2e-3)/3, 1e-2),
    }

    def __str__(self) -> str:
        return 'circuit-level'

    def __init__(
            self,
            edge_dict: dict[EdgeType, tuple[Edge, ...]],
            parametrization: Parametrization,
            demolition: bool,
            monolingual: bool,
            merges: dict[Edge, Edge] | None = None,
            force_by: Literal['pair', 'edge'] = 'pair',
    ):
        """
        :param edge_dict: maps from edge type to tuple of edges.
            The union of all edge tuples in ``edge_dict`` is the freshly discovered region after a window raise
        if scheme is streaming; else, the edges of G.
        :param parametrization: name of parametrization.
        :param demolition: whether ancilla qubit measurement demolishes state.
        :param monolingual: whether measurements are native in only one basis.
        :param merges: maps from each redundant edge to its substitute.
        :param force_by: whether ``force_error()`` makes an error based on the weight of each pair or edge type.
            See ``noise.forcers`` for more details.
        """
        self._EDGES = self._make_edges(edge_dict, demolition, monolingual, merges)
        self._COEFFICIENTS = self._ALL_COEFFICIENTS[parametrization]
        self._FORCER = ForceByEdge(self._EDGES) if force_by == 'edge' \
            else ForceByPairBalanced(self._EDGES) if parametrization == 'balanced' \
            else ForceByPair(self._EDGES)

    @classmethod
    def _make_edges(
        cls,
        edge_dict: dict[EdgeType, tuple[Edge, ...]],
        demolition: bool,
        monolingual: bool,
        merges: dict[Edge, Edge] | None,
    ):
        """Return ``_EDGES`` instance attribute."""
        multiplicities = cls._make_multiplicities(demolition, monolingual)
        edges: defaultdict[FourInts, list[Edge]] = defaultdict(list)
        if merges is None:
            for edge_type, es in edge_dict.items():
                m: FourInts = tuple(multiplicities[edge_type]) # type: ignore
                edges[m] += list(es)
        else:
            dc = cls._get_dc(edge_dict, multiplicities, merges)
            for e, mv in dc.items():
                m: FourInts = tuple(mv) # type: ignore
                edges[m].append(e)
        return dict(edges)

    @classmethod
    def _make_multiplicities(cls, demolition: bool, monolingual: bool):
        """Return map from edge type to multiplicity vector.
        
        E.g. ``{'S': np.array((4, 2, 1, 0)), ...}``.
        """
        multiplicities: dict[EdgeType, MultiplicityVector] = {
            edge_type: np.array(m)
            for edge_type, m
            in cls._DEFAULT_MULTIPLICITIES.items()
        }
        horizontals: list[EdgeType] = [
            'S',
            'E westmost',
            'E bulk',
            'E eastmost',
        ]
        verticals: list[EdgeType] = ['U 3', 'U 4']
        if demolition:
            for edge_type in horizontals:
                multiplicities[edge_type][2] += 1
            for edge_type in verticals:
                multiplicities[edge_type][3] += 1
        if monolingual:
            for edge_type in horizontals+verticals:
                multiplicities[edge_type][2] += 2
        return multiplicities

    @staticmethod
    def _get_dc(
        edge_dict: dict[EdgeType, tuple[Edge, ...]],
        multiplicities: dict[EdgeType, MultiplicityVector],
        merges: dict[Edge, Edge],
    ):
        """Return map from edge to multiplicity 4-vector."""
        # IMPORTANT: instantiate NEW array for each edge!
        dc: defaultdict[Edge, MultiplicityVector] = defaultdict(lambda: np.zeros(4, dtype=int))
        for edge_type, edges in edge_dict.items():
            m = multiplicities[edge_type]
            for e in edges:
                if e in merges:
                    substitute = merges[e]
                    dc[substitute] += m
                else:
                    dc[e] += m
        return dc
    
    def make_error(self, p):
        bitflip_probs = self._get_flip_probabilities(p)
        error: set[Edge] = set()
        for m, pr in bitflip_probs.items():
            error |= {e for e in self._EDGES[m] if random.random() < pr}
        return error
    
    def force_error(self, weight: tuple[int, ...]):
        return self._FORCER.force_error(weight)
    
    @property
    def ALL_WEIGHTS(self):
        return self._FORCER.ALL_WEIGHTS
    
    @property
    def ALL_WEIGHTS_INDEX(self):
        return pd.MultiIndex.from_tuples(self.ALL_WEIGHTS)
    
    def subset_probability(self, weights: tuple[tuple[int, ...], ...], p: float):
        pi = self._pi(p)
        return self._FORCER.subset_probability(weights, pi)
    
    @cache
    def get_edge_weights(self, noise_level: None | float):
        result: dict[Edge, tuple[float, float]] = {}
        if noise_level is None:
            for m, edges in self._EDGES.items():
                for e in edges:
                    result[e] = (0, 1)
        else:
            flip_probabilities = self._get_flip_probabilities(noise_level)
            for m, edges in self._EDGES.items():
                p = flip_probabilities[m]
                weight = self.log_odds_of_no_flip(p)
                for e in edges:
                    result[e] = (p, weight)
        return result

    def _pi(self, noise_level: float) -> FourFloats:
        """Return 4-vector probability ``pi = c*noise_level``."""
        return tuple(c*noise_level for c in self._COEFFICIENTS) # type: ignore
    
    @cache
    def _get_flip_probabilities(self, noise_level: float) -> dict[FourInts, float]:
        """Return map from multiplicity to flip probability.
        
        
        :param noise_level: a probability that represents the noise strength.
        
        
        :returns: map from multiplicity to a flip probability.
        """
        pi = self._pi(noise_level)
        return {m: MultisetHandler.pr(m, pi) for m in self._EDGES.keys()}
