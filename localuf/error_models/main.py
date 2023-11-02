"""Classes for error models."""

import abc
from functools import cache
from math import prod
import random

import numpy as np

from localuf.type_aliases import Edge, EdgeType, Parametrization, FourFloats, MultiplicityVector, FourInts


class _ErrorModel(abc.ABC):
    """Abstract base class for error models."""

    @abc.abstractmethod
    def get_edge_probabilities(self, p: float) -> tuple[tuple[Edge, float], ...]:
        """Return edges and their bitflip probabilities.
        
        Used only in `Code.get_matching_graph`.
        
        Input:
        * `p` characteristic physical error probability.

        Output:
        * |E|-tuple of pairs (edge, bitflip probability).
        """


class _Uniform(_ErrorModel):
    """Error model where each edge has bitflip probability `p`.

    Extends `_ErrorModel`.

    Attributes:
    * `EDGES` a tuple of edges of G.
    """

    def __init__(self, edges: tuple[Edge, ...]) -> None:
        self._EDGES = edges

    @property
    def EDGES(self): return self._EDGES

    def make_error(self, p: float) -> set[Edge]:
        """Sample edges from G with probability `p`."""
        return {e for e in self.EDGES if random.random() < p}
    
    def get_edge_probabilities(self, p: float):
        return tuple((e, p) for e in self.EDGES)


class CodeCapacity(_Uniform):
    """Code capacity error model.

    Extends `_Uniform`.
    """

    def __str__(self) -> str:
        return 'code capacity'


class Phenomenological(_Uniform):
    """Phenomenological error model.

    Extends `_Uniform`.
    """

    def __str__(self) -> str:
        return 'phenomenological'


class CircuitLevel(_ErrorModel):
    """Circuit-level depolarizing error model.

    Class attributes:
    * `_DEFAULT_MULTIPLICITIES` maps from edge type to multiplicity vector.
    * `_ALL_COEFFICIENTS` maps from parametrization name
    to 4-vector c such that pi = c*p.

    Private attributes:
    * `_EDGES` maps from multiplicity vector as a tuple to tuple of edges.
    * `_COEFFICIENTS` a 4-vector c such that pi = c*p.
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
    ) -> None:
        """Construct `multiplicities` then feed into `make_edges()`.
        
        Inputs:
        * `edge_dict` maps from edge type to tuple of edges.
        * `parametrization` name of parametrization.
        * `demolition` whether ancilla qubit measurement demolishes state.
        * `monolingual` whether measurements are native in only one basis.
        * `merges` maps from each redundant edge to its substitute.
        """
        multiplicities = self._make_multiplicities(demolition, monolingual)
        self._EDGES = self._make_edges(edge_dict, multiplicities, merges)
        self._COEFFICIENTS = self._ALL_COEFFICIENTS[parametrization]

    @classmethod
    def _make_multiplicities(cls, demolition: bool, monolingual: bool):
        """Return map from edge type to multiplicity vector."""
        multiplicities: dict[EdgeType, MultiplicityVector] = {
            et: np.array(m)
            for et, m
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
            for et in horizontals:
                multiplicities[et][2] += 1
            for et in verticals:
                multiplicities[et][3] += 1
        if monolingual:
            for et in horizontals+verticals:
                multiplicities[et][2] += 2
        return multiplicities

    @classmethod
    def _make_edges(
        cls,
        edge_dict: dict[EdgeType, tuple[Edge, ...]],
        multiplicities: dict[EdgeType, MultiplicityVector],
        merges: dict[Edge, Edge] | None,
    ):
        """Return `_EDGES` instance attribute."""
        edges: dict[FourInts, list[Edge]] = {}
        if merges is not None:
            dc = cls._get_dc(edge_dict, multiplicities, merges)
            for e, m in dc.items():
                key: FourInts = tuple(m) # type: ignore
                if key in edges:
                    edges[key].append(e)
                else:
                    edges[key] = [e]
        else:
            for et, es in edge_dict.items():
                m = tuple(multiplicities[et])
                key: FourInts = tuple(m) # type: ignore
                if key in edges:
                    edges[key] += list(es)
                else:
                    edges[key] = list(es)
        return edges

    @staticmethod
    def _get_dc(
        edge_dict: dict[EdgeType, tuple[Edge, ...]],
        multiplicities: dict[EdgeType, MultiplicityVector],
        merges: dict[Edge, Edge],
    ):
        """Return map from edge to multiplicity 4-vector."""
        dc: dict[Edge, MultiplicityVector] = {}
        for et, es in edge_dict.items():
            m = multiplicities[et]
            for e in es:
                if e in merges:
                    substitute = merges[e]
                    dc[substitute] += m
                else:
                    # EXTREMELY IMPORTANT: instantiate NEW array for each edge!
                    dc[e] = np.array(m)
        return dc
    
    def make_error(self, p: float):
        """Sample edges from G with probability defined by its multiplicity.

        Input:
        * `p` characteristic probability.
        """
        bitflip_probs = self._get_bitflip_probabilities(p)
        error: set[Edge] = set()
        for m, pr in bitflip_probs.items():
            error |= {e for e in self._EDGES[m] if random.random() < pr}
        return error
    
    def get_edge_probabilities(self, p: float):
        bitflip_probs = self._get_bitflip_probabilities(p)
        ls: list[tuple[Edge, float]] = []
        for m, edges in self._EDGES.items():
            bp = bitflip_probs[m]
            for e in edges:
                ls.append((e, bp))
        return tuple(ls)
    
    def pi(self, p: float) -> FourFloats:
        """Return 4-vector probability pi = c*p."""
        return tuple(c*p for c in self._COEFFICIENTS) # type: ignore
    
    @cache
    def _get_bitflip_probabilities(self, p: float) -> dict[FourInts, float]:
        """Return map from multiplicity to bitflip probability."""
        pi = self.pi(p)
        return {m: _MultisetHandler.pr(m, pi) for m in self._EDGES.keys()}


class _MultisetHandler:

    @staticmethod
    def pr(multiplicities: FourInts | MultiplicityVector, pi: FourFloats) -> float:
        """Return probability of odd number of faults in multiset
        defined by `multiplicities`, `pi`.

        Input:
        * `multiplicities` a multiplicity vector.
        * `pi` a 4-tuple of fault probabilities.
        """
        f1 = prod((1-p)**m for p, m in zip(pi, multiplicities))
        f2 = sum(m*p/(1-p) for p, m in zip(pi, multiplicities))
        return f1 * f2