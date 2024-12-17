import itertools

from localuf._base_classes import Code
from localuf._schemes import Batch, Global, Forward, Frugal
from localuf.noise import CircuitLevel, CodeCapacity, Phenomenological
from localuf.type_aliases import Parametrization, NoiseModel, Node, DecodingScheme


class InnerInitHelper:
    """Helper for `Code._inner_init`."""

    @staticmethod
    def help_(
            code: Code,
            node_ranges: list[range],
            noise: NoiseModel,
            scheme: DecodingScheme,
            window_height: int | None,
            commit_height: int | None,
            buffer_height: int | None,
            parametrization: Parametrization,
            demolition: bool,
            monolingual: bool,
            merge_redundant_edges: bool,
    ):
        d = code.D
        if 'batch' in scheme:
            for height in (commit_height, buffer_height):
                if height is not None:
                    raise ValueError(f"Cannot specify `{height=}` for `{scheme}` scheme.")
            if noise == 'code capacity':
                h = 1
                code._N_EDGES, code._EDGES = code._code_capacity_edges()
                code._NOISE = CodeCapacity(code.EDGES)
            else:
                h = d if window_height is None else window_height
                node_ranges.append(range(h))
                code._DIMENSION += 1
                if noise == 'phenomenological':
                    code._N_EDGES, code._EDGES = code._phenomenological_edges(h, False)
                    code._NOISE = Phenomenological(code.EDGES)
                else:  # noise == 'circuit-level'
                    code._N_EDGES, code._EDGES, edge_dict, merges = code._circuit_level_edges(
                        h=h,
                        temporal_boundary=False,
                        merge_redundant_edges=merge_redundant_edges,
                    )
                    code._NOISE = CircuitLevel(
                        edge_dict=edge_dict,
                        parametrization=parametrization,
                        demolition=demolition,
                        monolingual=monolingual,
                        merges=merges,
                    )
            code._SCHEME = Batch(code, h) if scheme == 'batch' else Global(code, h)
            code._NODES = tuple(itertools.product(*node_ranges))
        else:
            if window_height is not None:
                raise ValueError(f"Cannot specify `window_height` for the {scheme} decoding scheme.")
            code._DIMENSION += 1
            if scheme == 'forward':
                if commit_height is None: commit_height = d
                if buffer_height is None: buffer_height = d
                scheme_class = Forward
            else:  # scheme == 'frugal'
                if commit_height is None: commit_height = 1
                if buffer_height is None: buffer_height = 2*(d//2)
                scheme_class = Frugal
            h = commit_height + buffer_height
            node_ranges.append(range(h))
            nodes: list[Node] = list(itertools.product(*node_ranges))
            nodes += code._temporal_boundary_nodes(h)

            if noise == 'code capacity':
                raise TypeError(f"Code capacity incompatible with the {scheme} decoding scheme.")
            elif noise == 'phenomenological':
                code._N_EDGES, code._EDGES = code._phenomenological_edges(h, True)
                _, commit_edges = code._phenomenological_edges(commit_height, True)
                _,  fresh_edges = code._phenomenological_edges(commit_height, True, t_start=buffer_height)
                code._NOISE = Phenomenological(fresh_edges)
            else:  # noise == 'circuit-level'
                if not merge_redundant_edges:
                    nodes += code._redundant_boundary_nodes(h)
                code._N_EDGES, code._EDGES, *_ = code._circuit_level_edges(
                    h=h,
                    temporal_boundary=True,
                    merge_redundant_edges=merge_redundant_edges,
                )
                _, commit_edges, *_ = code._circuit_level_edges(
                    h=commit_height,
                    temporal_boundary=True,
                    merge_redundant_edges=merge_redundant_edges,
                )
                *_, fresh_edge_dict, fresh_merges = code._circuit_level_edges(
                    h=commit_height,
                    temporal_boundary=True,
                    merge_redundant_edges=merge_redundant_edges,
                    t_start=buffer_height,
                )
                code._NOISE = CircuitLevel(
                    edge_dict=fresh_edge_dict,
                    parametrization=parametrization,
                    demolition=demolition,
                    monolingual=monolingual,
                    merges=fresh_merges,
                )
            
            code._NODES = tuple(nodes)
            code._SCHEME = scheme_class(
                    code,
                    commit_height,
                    buffer_height,
                    commit_edges,
                )