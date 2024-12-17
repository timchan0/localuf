"""Classes for noise models."""

import abc
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

    @property
    @abc.abstractmethod
    def ALL_WEIGHTS(self) -> Iterable[int] | tuple[tuple[int, ...], ...]:
        """All possible `force_error` inputs."""
    
    @property
    @abc.abstractmethod
    def ALL_WEIGHTS_INDEX(self) -> pd.Index:
        """`ALL_WEIGHTS` as pandas Index."""

    @abc.abstractmethod
    def make_error(self, p: float) -> set[Edge]:
        """See `Code.make_error`."""

    @abc.abstractmethod
    def force_error(self, weight: int | tuple[int, ...]) -> set[Edge]:
        """Make error of weight `weight`."""
    
    @abc.abstractmethod
    def subset_probability(
        self,
        weights: Iterable[int] | tuple[tuple[int, ...], ...],
        p: float,
    ) -> Iterable[float]:
        """Return probability of any error of weight `weight`, for `weight` in `weights`."""

    def subset_probabilities(self, p: float, survival: bool = True):
        """Return DataFrame containing probabilities of each subset.

        Input:
        * `p` noise level.
        * `survival` whether to compute survival probability column.

        Output:
        * DataFrame indexed by subset weight,
        with columns `['subset prob', 'survival prob']`.
        """
        subset_prob = self.subset_probability(self.ALL_WEIGHTS, p)
        df = pd.DataFrame({'subset prob': subset_prob}, index=self.ALL_WEIGHTS_INDEX)
        if survival:
            df.sort_values(by='subset prob', inplace=True)
            df['survival prob'] = df['subset prob'].cumsum().shift(fill_value=0)
        df.sort_values(by='subset prob', ascending=False, inplace=True)
        return df

    @abc.abstractmethod
    def get_edge_probabilities(self, p: float) -> tuple[tuple[Edge, float], ...]:
        """Return edges and their bitflip probabilities.
        
        Used only in `Code.get_matching_graph`.
        
        Input:
        * `p` noise level.

        Output:
        * |E|-tuple of pairs (edge, bitflip probability).
        """


class _Uniform(Noise):
    """Base class for noise model where each edge has bitflip probability `p`.

    Extends `Noise`.

    Attributes:
    * `EDGES` the edges of the freshly discovered region after a window raise
    if scheme is streaming; else,
    the edges of G.
    """

    def __init__(self, edges: tuple[Edge, ...]):
        """Input:
        `edges` the edges of the freshly discovered region after a window raise
        if scheme is streaming; else,
        the edges of G.
        """
        self._EDGES = edges

    @property
    def EDGES(self): return self._EDGES

    def make_error(self, p):
        return {e for e in self.EDGES if random.random() < p}
    
    def force_error(self, weight: int):
        """Bitlfip exactly `weight` edges in G.

        Input: `weight` between 0 and `len(self.EDGES)`.
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
    
    def get_edge_probabilities(self, p: float):
        return tuple((e, p) for e in self.EDGES)


class CodeCapacity(_Uniform):
    """Code capacity noise model.

    Extends `_Uniform`.
    """

    def __str__(self) -> str:
        return 'code capacity'


class Phenomenological(_Uniform):
    """Phenomenological noise model.

    Extends `_Uniform`.
    """

    def __str__(self) -> str:
        return 'phenomenological'


class CircuitLevel(Noise):
    """Circuit-level depolarizing noise model.

    Class attributes:
    * `_DEFAULT_MULTIPLICITIES` maps from edge type to multiplicity vector
    e.g. `{'S': (4, 2, 1, 0), ...}`.
    * `_ALL_COEFFICIENTS` maps from parametrization name
    to 4-vector c such that pi = c*p.

    Private attributes:
    * `_EDGES` maps from multiplicity vector (as a tuple) to tuple of edges
    e.g. `{(4, 2, 1, 0): (e1, ...), ...}`.
    Order matters as used by `ForceByEdge.force_error`.
    The union of all edge tuples in `_EDGES` is the freshly discovered region after a window raise
    if scheme is streaming; else, the edges of G.
    * `_COEFFICIENTS` a 4-vector c such that pi = c*p.
    * `_FORCER` method for forcing error.
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
        """Input:
        * `edge_dict` maps from edge type to tuple of edges.
        The union of all edge tuples in `edge_dict` is the freshly discovered region after a window raise
        if scheme is streaming; else, the edges of G.
        * `parametrization` name of parametrization.
        * `demolition` whether ancilla qubit measurement demolishes state.
        * `monolingual` whether measurements are native in only one basis.
        * `merges` maps from each redundant edge to its substitute.
        * `force_by` whether `force_error()` makes an error based on the weight of each pair or edge type.
        See `noise.forcers` for more details.
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
        """Return `_EDGES` instance attribute."""
        multiplicities = cls._make_multiplicities(demolition, monolingual)
        edges: dict[FourInts, list[Edge]] = {}
        if merges is None:
            for edge_type, es in edge_dict.items():
                m: FourInts = tuple(multiplicities[edge_type]) # type: ignore
                if m in edges:
                    edges[m] += list(es)
                else:
                    edges[m] = list(es)
        else:
            dc = cls._get_dc(edge_dict, multiplicities, merges)
            for e, mv in dc.items():
                m: FourInts = tuple(mv) # type: ignore
                if m in edges:
                    edges[m].append(e)
                else:
                    edges[m] = [e]
        return edges

    @classmethod
    def _make_multiplicities(cls, demolition: bool, monolingual: bool):
        """Return map from edge type to multiplicity vector.
        
        E.g. `{'S': np.array((4, 2, 1, 0)), ...}`.
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
        dc: dict[Edge, MultiplicityVector] = {}
        for edge_type, es in edge_dict.items():
            m = multiplicities[edge_type]
            for e in es:
                if e in merges:
                    substitute = merges[e]
                    dc[substitute] += m
                else:
                    # EXTREMELY IMPORTANT: instantiate NEW array for each edge!
                    dc[e] = np.array(m)
        return dc
    
    def make_error(self, p):
        bitflip_probs = self._get_bitflip_probabilities(p)
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
    
    def get_edge_probabilities(self, p):
        bitflip_probs = self._get_bitflip_probabilities(p)
        ls: list[tuple[Edge, float]] = []
        for m, edges in self._EDGES.items():
            bp = bitflip_probs[m]
            for e in edges:
                ls.append((e, bp))
        return tuple(ls)
    
    def _pi(self, p: float) -> FourFloats:
        """Return 4-vector probability pi = c*p."""
        return tuple(c*p for c in self._COEFFICIENTS) # type: ignore
    
    @cache
    def _get_bitflip_probabilities(self, p: float) -> dict[FourInts, float]:
        """Return map from multiplicity to bitflip probability."""
        pi = self._pi(p)
        return {m: MultisetHandler.pr(m, pi) for m in self._EDGES.keys()}