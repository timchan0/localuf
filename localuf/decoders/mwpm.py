from collections.abc import Mapping
from collections import defaultdict

import numpy as np
import numpy.typing as npt
from pymatching import Matching

from localuf._base_classes import Code, Decoder
from localuf._schemes import Batch
from localuf.type_aliases import Node, Edge


class MWPM(Decoder):
    """Minimum weight perfect matching decoder using PyMatching implementation.
    
    Extends ``Decoder``.
    Compatible so far only with the batch decoding scheme.
    
    Additional instance constants:
    * ``_DETECTOR_TO_INT`` maps each detector node index to a unique integer in 0..detector_count.
    * ``_WEST_LOGICAL`` maps the west (east) boundary node index to fault ID 0 (None).
    * ``_WEST_EAST_LOGICALS`` maps the west (east) boundary node index to fault ID 0 (1).
    * ``_edge_weight_modifier`` maps each edge to a multiplicative factor
        that scales the weight of that particular edge.
    Designed for implementing non-uniform weights for phenomenological noise model.
    
    Additional instance attributes:
    * ``correction_vector`` the stored correction as a binary vector whose length is the number of fault IDs.
    * ``correction_weight`` the weight of the stored correction.
    """

    def __init__(self, code: Code, _edge_weight_modifier: None | dict[Edge, float] = None):
        """
        :param code: the code to be decoded.
        :param _edge_weight_modifier: maps each edge to a multiplicative factor
            that scales the weight of that particular edge.
        Designed for implementing non-uniform weights for phenomenological noise model.
        If not specified, no scaling is done i.e. all edges have modifier 1.
        """
        if not isinstance(code.SCHEME, Batch):
            raise NotImplementedError
        super().__init__(code)
        self._DETECTOR_TO_INT = {v: k for k, v in enumerate(code.DETECTORS)}
        self._WEST_LOGICAL: dict[Node, int | None] = {
            v: 0 if (v[code.LONG_AXIS] == -1) else None
            for v in code.BOUNDARY_NODES
        }
        self._WEST_EAST_LOGICALS: dict[Node, int] = {
            v: 0 if (v[code.LONG_AXIS] == -1) else 1
            for v in code.BOUNDARY_NODES
        }
        self._edge_weight_modifier: dict[Edge, float] = {
            e: 1.0 for e in code.EDGES
        } if _edge_weight_modifier is None else _edge_weight_modifier

    def get_matching(
            self,
            noise_level: None | float = None,
            detector_to_int: None | dict[Node, int] = None,
            boundary_node_to_fault_id: None | Mapping[Node, int | None] = None,
    ):
        """Return a PyMatching matching graph.
        
        
        :param noise_level: a probability that represents the noise strength.
            This is passed to ``self.CODE.NOISE.get_edge_weights()``.
        If ``None``, all edges in ``matching`` have error probability 0 and weight 1
        (before being scaled by ``self._edge_weight_modifier``).
        :param detector_to_int: maps each detector node index to a unique integer.
            If ``None``, use the default map from each index to a unique integer in 0..detector_count.
        :param boundary_node_to_fault_id: maps each boundary node index to its fault ID.
            If ``None``, use the default map from the west (east) boundary node index to fault ID 0 (None).
        
        Output: ``matching`` a PyMatching matching graph whose...
        * edge weights and error probabilities depend on ``noise_level`` and ``self._edge_weight_modifier``,
        * detectors are numbered according to ``detector_to_int``,
        * boundary nodes are all virtual;
            all edges connected to each boundary node have the fault ID given by ``boundary_node_to_fault_id``.
        """
        if detector_to_int is None:
            detector_to_int = self._DETECTOR_TO_INT
        if boundary_node_to_fault_id is None:
            boundary_node_to_fault_id = self._WEST_LOGICAL
        edge_weights = self.CODE.NOISE.get_edge_weights(noise_level)
        matching = Matching()
        for (u, v), (p, weight) in edge_weights.items():
            modified_weight = self._edge_weight_modifier[u, v] * weight
            int_u = detector_to_int.get(u, None)
            int_v = detector_to_int.get(v, None)
            fault_u = boundary_node_to_fault_id.get(u, None)
            fault_v = boundary_node_to_fault_id.get(v, None)
            if int_u is None:
                # type(int_v) is int
                matching.add_boundary_edge(
                    node=int_v, # type: ignore
                    fault_ids=fault_u, # type: ignore
                    weight=modified_weight,
                    error_probability=p,
                )
            elif int_v is None:
                matching.add_boundary_edge(
                    node=int_u,
                    fault_ids=fault_v, # type: ignore
                    weight=modified_weight,
                    error_probability=p,
                )
            else:
                matching.add_edge(
                    node1=int_u,
                    node2=int_v,
                    weight=modified_weight,
                    error_probability=p,
                )
        return matching

    # unused
    def _get_matching_with_real_boundaries(
            self,
            noise_level: None | float = None,
            detector_to_int: None | dict[Node, int] = None,
            boundary_node_to_fault_id: None | Mapping[Node, int | None] = None,
    ):
        """Same as ``get_matching`` but the boundary nodes of the output are real
        and are numbered consecutively starting from ``detector_count``.
        """
        if detector_to_int is None:
            detector_to_int = self._DETECTOR_TO_INT
        if boundary_node_to_fault_id is None:
            boundary_node_to_fault_id = self._WEST_LOGICAL
        edge_weights = self.CODE.NOISE.get_edge_weights(noise_level)
        matching = Matching()
        # `boundary_nodes` maps each fault ID to a boundary node index
        boundary_nodes: defaultdict[int | None, int] = defaultdict(
            lambda: self.CODE.DETECTOR_COUNT + len(boundary_nodes))
        for (u, v), (p, weight) in edge_weights.items():
            int_u = detector_to_int.get(u, None)
            int_v = detector_to_int.get(v, None)
            fault_u = boundary_node_to_fault_id.get(u, None)
            fault_v = boundary_node_to_fault_id.get(v, None)
            if int_u is None:
                # type(int_v) is int
                matching.add_edge(
                    node1=boundary_nodes[fault_u],
                    node2=int_v, # type: ignore
                    fault_ids=fault_u, # type: ignore
                    weight=weight,
                    error_probability=p,
                )
            elif int_v is None:
                matching.add_edge(
                    node1=int_u,
                    node2=boundary_nodes[fault_v],
                    fault_ids=fault_v, # type: ignore
                    weight=weight,
                    error_probability=p,
                )
            else:
                matching.add_edge(
                    node1=int_u,
                    node2=int_v,
                    weight=weight,
                    error_probability=p,
                )
        matching.set_boundary_nodes(set(boundary_nodes.values()))
        return matching
    
    def reset(self):
        self.correction_vector: npt.NDArray[np.bool_] = np.zeros(self.CODE.DETECTOR_COUNT, dtype=bool)
        self.correction_weight: float = 0

    def get_binary_vector(self, syndrome: set[Node], detector_to_int: None | dict[Node, int] = None):
        """Convert ``syndrome`` to a binary vector.
        
        
        :param syndrome: the set of defects.
        :param detector_to_int: maps each detector node index to a unique integer in 0..detector_count.
        
        
        :returns: ``binary_vector`` a binary vector of length equal to detector count whose ordering is given by ``detector_to_int``. If not specified, uses ordering given by ``self.CODE.DETECTORS``.
        """
        if detector_to_int is None:
            detector_to_int = self._DETECTOR_TO_INT
        binary_vector: npt.NDArray[np.bool_] = np.zeros(self.CODE.DETECTOR_COUNT, dtype=bool)
        for v in syndrome:
            binary_vector[detector_to_int[v]] = 1
        return binary_vector
        
    def decode(self, syndrome, **kwargs):
        """Decode syndrome.
        
        
        :param syndrome: the set of defects.
        :param kwargs: passed to ``self.get_matching()``.
        
        Side effects:
        * Update ``self.correction_vector`` and ``self.correction_weight``.
        """
        syndrome_vector = self.get_binary_vector(syndrome)
        matching = self.get_matching(**kwargs)
        self.correction_vector, self.correction_weight = matching.decode(
            z=syndrome_vector,
            return_weight=True,
        )
    
    def draw_decode(self, **kwargs_for_networkx_draw):
        raise NotImplementedError
    
    def complementary_gap(
            self,
            syndrome: set[Node],
            noise_level: None | float = None,
    ) -> tuple[npt.NDArray[np.bool_], float]:
        """Decode ``syndrome`` and calculate the complementary gap.
        
        Does not require ``self.decode()`` to be called first.
        If ``self.CODE.MERGED_EQUIVALENT_BOUNDARY_NODES`` is ``False``,
        raises a ``ValueError``.
        
        
        :param syndrome: the set of defects.
        :param noise_level: a probability representing the noise strength.
            This is needed to define nonuniform edge weights of the decoding graph
        in the circuit-level noise model.
        If ``None``, all edges are assumed to have weight 1.
        
        
        :returns:
        * ``[correction_west, correction_east]`` a 2-vector whose first (second) element is the parity of number of edges in the minimum-weight correction that connect to the west (east) boundary node.
        * The complementary gap.
        """
        if not self.CODE.MERGED_EQUIVALENT_BOUNDARY_NODES:
            raise ValueError('Complementary gap requires all equivalent boundary nodes in the decoding graph be merged.')
        syndrome_vector = self.get_binary_vector(syndrome)
        matching_1, matching_2 = self._get_complementary_gap_matchings(noise_level=noise_level)
        correction_1, weight_1 = matching_1.decode(syndrome_vector, return_weight=True)
        complementary_syndrome_vector = np.concatenate((syndrome_vector, correction_1 ^ True))
        _, weight_2 = matching_2.decode(complementary_syndrome_vector, return_weight=True)
        return correction_1, weight_2 - weight_1

    def _get_complementary_gap_matchings(self, noise_level: None | float = None):
        """Return two PyMatching matching graphs for complementary gap calculation.
        
        
        :param noise_level: a probability that represents the noise strength.
            This is passed to ``self.NOISE.get_edge_weights()``.
        If ``None``, all edges have error probability 0 and weight 1.
        
        
        :returns:
        * ``matching_1`` a PyMatching matching graph with 2 virtual boundary nodes: all edges connected to the west (east) virtual boundary node have fault ID 0 (1).
        * ``matching_2`` same as ``matching_1`` but the virtual boundary nodes are now detectors, numbered in the same order as their fault IDs. E.g. if the detector count is 6, then the virtual boundary node corresponding to fault ID 0 is numbered 6, and that corresponding to fault ID 1 is numbered 7.
        """
        detector_to_int = self._DETECTOR_TO_INT | {
            v: self.CODE.DETECTOR_COUNT + fault_id
            for v, fault_id in self._WEST_EAST_LOGICALS.items()
        }
        matching_1 = self.get_matching(
            noise_level=noise_level,
            boundary_node_to_fault_id=self._WEST_EAST_LOGICALS,
        )
        matching_2 = self.get_matching(
            noise_level=noise_level,
            detector_to_int=detector_to_int,
        )
        return matching_1, matching_2