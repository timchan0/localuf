import numpy as np
from pymatching import Matching

from localuf._base_classes import Code, Decoder
from localuf._schemes import Batch, Forward
from localuf.type_aliases import Node, Edge
from localuf.constants import Boundary


class MWPM(Decoder):
    """Minimum weight perfect matching decoder using PyMatching implementation.
    
    Extends ``Decoder``.
    Compatible so far only with the batch and forward decoding schemes.
    """

    correction_vector: np.ndarray[tuple[int], np.dtype[np.uint8]]
    """The stored correction as a binary vector whose length is the number of fault IDs."""
    correction_weight: float = 0
    """The weight of the stored correction."""

    def __init__(
            self,
            code: Code,
            detector_to_int: None | dict[Node, int] = None,
            _edge_weight_modifier: None | dict[Edge, float] = None,
    ):
        """
        :param code: The code to be decoded.
        :param detector_to_int: An optional map from each detector node index to a unique integer.
            If ``None``, use the default map from index to a unique integer in 0..detector_count.
        :param _edge_weight_modifier: Maps each edge to a multiplicative factor
            that scales the weight of that particular edge.
            Designed for implementing non-uniform weights for phenomenological noise model.
            If not specified, no scaling is done i.e. all edges have modifier 1.
        """
        if type(code.SCHEME) is Batch:
            self._MERGE_STRATEGY = 'disallow'
        elif type(code.SCHEME) is Forward:
            self._MERGE_STRATEGY = 'independent'
        else:
            raise NotImplementedError
        super().__init__(code)
        self.correction = set()
        self.DETECTOR_TO_INT = {v: k for k, v in enumerate(code.DETECTORS)} if detector_to_int is None else detector_to_int
        """Map from each detector node index to a unique integer in 0..detector_count."""
        self._WEST_BOUNDARY_NODES = {v for v in code.BOUNDARY_NODES if v[code.LONG_AXIS] == -1}
        """The set of west boundary nodes."""
        self._edge_weight_modifier: dict[Edge, float] = {
            e: 1.0 for e in code.EDGES
        } if _edge_weight_modifier is None else _edge_weight_modifier
        """Map from each edge to a multiplicative factor
        that scales the weight of that particular edge.
        Designed for implementing non-uniform weights for phenomenological noise model.
        """

        _int_pair_to_edge: dict[frozenset[int], Edge] = {}
        _boundary_detectors: dict[Boundary, set[Node]] = {
            Boundary.WEST: set(),
            Boundary.EAST: set(),
            Boundary.FUTURE: set(),
        }
        for u, v in code.EDGES:
            
            detector = None
            if u[code.LONG_AXIS] == -1:  # west boundary edge
                detector = v
                _boundary_detectors[Boundary.WEST].add(detector)
            elif v[code.LONG_AXIS] == code.D - 1:  # east boundary edge
                detector = u
                _boundary_detectors[Boundary.EAST].add(detector)
            elif u[code.TIME_AXIS] == code.SCHEME.WINDOW_HEIGHT:  # future boundary edge
                detector = v
                _boundary_detectors[Boundary.FUTURE].add(detector)
            elif v[code.TIME_AXIS] == code.SCHEME.WINDOW_HEIGHT:  # future boundary edge
                detector = u
                _boundary_detectors[Boundary.FUTURE].add(detector)
            
            if detector is None:
                _int_pair_to_edge[frozenset({self.DETECTOR_TO_INT[u], self.DETECTOR_TO_INT[v]})] = (u, v)
            else:
                _int_pair_to_edge[frozenset({self.DETECTOR_TO_INT[detector], -1})] = (u, v)
        self._INT_PAIR_TO_EDGE = _int_pair_to_edge
        """Map from each unordered pair of integers representing PyMatching nodes
        to the edge in the decoding graph.
        """
        self._BOUNDARY_DETECTORS = _boundary_detectors
        """Map from each boundary type to the set of detectors connecting that boundary."""
        
        self.set_noise_level(None)
        self.noise_level: None | float
        """A probability representing the noise strength.
        This is needed to define nonuniform edge weights of the decoding graph
        in the circuit-level noise model.
        If `None`, all edges are assumed to have weight 1.
        """
        self._matching: Matching
        """A PyMatching matching graph with 1 fault ID, whose...
            1. boundary nodes are all virtual,
            2. edges connected to west boundary node have fault ID 0.
        """
        self._complementary_gap_matchings: tuple[Matching, Matching]
        """A pair of PyMatching matching graphs for complementary gap calculation.
        
        The first is a PyMatching matching graph with 2 virtual boundary nodes:
        all edges connected to the west (east) virtual boundary node have fault ID 0 (1).

        The second is the same as the first but the virtual boundary nodes are now detectors,
        numbered in the same order as their fault IDs.
        E.g. if the detector count is 6, then the virtual boundary node corresponding to fault ID 0 is numbered 6,
        and that corresponding to fault ID 1 is numbered 7.
        """

    def _get_matching(
            self,
            noise_level: None | float = None,
            detector_to_int: None | dict[Node, int] = None,
    ):
        """Return a PyMatching matching graph.
        
        :param noise_level: A probability that represents the noise strength.
            This is passed to ``self.CODE.NOISE.get_edge_weights()``.
            If ``None``, all edges in ``matching`` have error probability 0 and weight 1
            (before being scaled by ``self._edge_weight_modifier``).
        :param detector_to_int: An optional map from each detector node index to a unique integer.
            If ``None``, use ``self.DETECTOR_TO_INT``.
        
        :return matching: a PyMatching matching graph whose...
            1. edge weights and error probabilities depend on ``noise_level`` and ``self._edge_weight_modifier``,
            2. detectors are numbered according to ``detector_to_int``,
            3. boundary nodes are all virtual;
            all edges connected to the west boundary node have fault ID 0.
        """
        if detector_to_int is None:
            detector_to_int = self.DETECTOR_TO_INT
        edge_weights = self.CODE.NOISE.get_edge_weights(noise_level)
        matching = Matching()
        for (u, v), (flip_probability, weight) in edge_weights.items():
            modified_weight = self._edge_weight_modifier[u, v] * weight
            int_u = detector_to_int.get(u, None)
            int_v = detector_to_int.get(v, None)
            fault_u = 0 if u in self._WEST_BOUNDARY_NODES else None
            fault_v = 0 if v in self._WEST_BOUNDARY_NODES else None
            if int_u is None:
                # type(int_v) is int
                matching.add_boundary_edge(
                    node=int_v, # type: ignore
                    fault_ids=fault_u, # type: ignore
                    weight=modified_weight,
                    error_probability=flip_probability,
                    merge_strategy=self._MERGE_STRATEGY,
                )
            elif int_v is None:
                matching.add_boundary_edge(
                    node=int_u,
                    fault_ids=fault_v, # type: ignore
                    weight=modified_weight,
                    error_probability=flip_probability,
                    merge_strategy=self._MERGE_STRATEGY,
                )
            else:
                matching.add_edge(
                    node1=int_u,
                    node2=int_v,
                    weight=modified_weight,
                    error_probability=flip_probability,
                    # following line required for
                    # self._complementary_gap_matchings[1] in the forward decoding scheme
                    merge_strategy=self._MERGE_STRATEGY,
                )
        return matching
    
    def reset(self):
        super().reset()
        self.correction_vector = np.zeros(self.CODE.DETECTOR_COUNT, dtype=np.uint8)
        self.correction_weight = 0

    def get_binary_vector(self, syndrome: set[Node]):
        """Convert a set of defect coordinates to a binary vector.
        
        :param syndrome: The set of defect coordinates.
        
        :return binary_vector: A binary vector of length equal to detector count
            whose ordering is given by ``self.DETECTOR_TO_INT``.
            If not specified, uses ordering given by ``self.CODE.DETECTORS``.
        """
        binary_vector = np.zeros(self.CODE.DETECTOR_COUNT, dtype=np.uint8)
        for v in syndrome:
            binary_vector[self.DETECTOR_TO_INT[v]] = 1
        return binary_vector
        
    def decode(self, syndrome: set[Node]):
        """Decode syndrome.
        
        :param syndrome: The set of defect coordinates.
        
        Side effects:
        * Update ``self.correction_vector`` and ``self.correction_weight``.
        """
        syndrome_vector = self.get_binary_vector(syndrome)
        self.correction_vector, self.correction_weight = self._matching.decode(
            z=syndrome_vector,
            return_weight=True,
        )

    def decode_to_edge_set(self, syndrome_vector: np.ndarray[tuple[int], np.dtype[np.uint8]]):
        """Find the set of edges in the minimum-weight correction.
        
        :param syndrome_vector: A binary vector of length equal to detector count
            representing the set of defects.

        Side effect: Set ``self.correction`` to the set of edges in the minimum-weight correction.
        """
        int_pairs: np.ndarray[tuple[int, int], np.dtype[np.int64]] \
            = self._matching.decode_to_edges_array(syndrome_vector)
        self.correction = {self._INT_PAIR_TO_EDGE[frozenset(int_pair)] for int_pair in int_pairs}
    
    def draw_decode(self, **kwargs_for_networkx_draw):
        raise NotImplementedError
    
    def complementary_gap(self, syndrome: set[Node]) -> tuple[np.uint8, float]:
        """Decode ``syndrome`` and calculate the complementary gap.
        
        Does not require ``self.decode()`` to be called first.
        
        :param syndrome: the set of defects.
        
        :return correction_bit: The parity of number of edges in
            the minimum-weight correction that connect to the west boundary node.
        :return gap: The complementary gap.
        
        :raises ValueError: if ``self.CODE.MERGED_EQUIVALENT_BOUNDARY_NODES`` is ``False``.
        """
        if not self.CODE.MERGED_EQUIVALENT_BOUNDARY_NODES:
            raise ValueError('Complementary gap requires all equivalent boundary nodes in the decoding graph be merged.')
        syndrome_vector = self.get_binary_vector(syndrome)
        matching_1, matching_2 = self._complementary_gap_matchings
        correction_1, weight_1 = matching_1.decode(syndrome_vector, return_weight=True)
        correction_1: np.ndarray[tuple[int], np.dtype[np.uint8]]
        complementary_syndrome_vector = np.concatenate((syndrome_vector, correction_1 ^ 1))
        _, weight_2 = matching_2.decode(complementary_syndrome_vector, return_weight=True)
        correction_bit, = correction_1
        return correction_bit, weight_2 - weight_1

    def set_noise_level(self, noise_level: None | float = None):
        """Set the noise level of the decoder and preconstruct matching objects.

        :param noise_level: A probability that represents the noise strength.
            This is passed to ``self.CODE.NOISE.get_edge_weights()``.
            If ``None``, all edges in the matching objects will have error probability 0 and weight 1
            (before being scaled by ``self._edge_weight_modifier``).

        Side effects:
        * Update ``self.noise_level``.
        * Update ``self.matching``.
        * Update ``self._complementary_gap_matchings``.
        """
        self.noise_level = noise_level
        self._matching = self._get_matching(noise_level=noise_level)
        
        matching_1 = self._get_matching(noise_level=noise_level)
        detector_to_int_2 = self.DETECTOR_TO_INT | {
            v: self.CODE.DETECTOR_COUNT for v in self._WEST_BOUNDARY_NODES}
        matching_2 = self._get_matching(
            noise_level=noise_level,
            detector_to_int=detector_to_int_2,
        )
        self._complementary_gap_matchings = matching_1, matching_2