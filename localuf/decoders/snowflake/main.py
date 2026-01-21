import abc
from collections.abc import Iterable, Sequence
from functools import cache
from typing import Literal

import networkx as nx

from localuf import constants, Repetition
from localuf.decoders.snowflake.constants import RESET, Stage
from localuf.noise import CodeCapacity
from localuf.type_aliases import Edge, Node, Coord
from localuf.constants import Growth
from localuf.decoders.policies import DecodeDrawer
from localuf._base_classes import Code
from localuf._schemes import Frugal
from localuf.decoders._base_uf import direction, BaseUF


class Snowflake(BaseUF):
    """Snowflake decoder based on UF.
    
    Extends ``BaseUF``.
    Incompatible with code capacity noise model.
    Compatible only with frugal scheme.
    
    Class constants:
    ``BW_DEFAULT_NODE_SIZE`` the default node size for black-and-white drawing.
    
    Additional instance attributes:
    * ``NODES`` a dictionary of nodes.
    * ``EDGES`` ditto for edges.
    * ``_pointer_digraph`` a NetworkX digraph representing the fully grown edges used by pointers,
        the set of its edges as directed edges,
    the set of its edges as undirected edges.
    * ``_stage`` the current stage of the decoder.
        Only used for ``draw_growth`` when ``show_global = True``.
    * ``_LOWEST_EDGES`` a tuple of the edges in the bottom layer of the viewing window.
        For repetition code, the edges are ordered
    first by type (up, east)
    then by y-coordinate (west to east).
    For surface code, the edges are ordered
    first by type (up, south_down, east_up, south_east_up, south, east),
    then by x-coordinate (north to south), then by y-coordinate (west to east).
    If ``__init__`` was called with ``_include_timelike_lowest_edges = False``,
    the purely timelike (i.e. 'up') edges are excluded.
    * ``floor_history`` a list of bitstrings representing
        the correction output from the bottom layer at each drop.
    The order of the bits is given by ``self._LOWEST_EDGES``.
    
    Overriden methods:
    * ``reset``.
    * ``decode``.
    * ``draw_growth``.
    * ``draw_decode``.
    
    Uses not:
    * ``_growth`` attribute.
    * ``erasure`` attribute.
    """

    BW_DEFAULT_NODE_SIZE = 360

    def __init__(
            self,
            code: Code,
            merger: Literal['fast', 'slow'] = 'fast',
            schedule: Literal['2:1', '1:1'] = '2:1',
            unrooter: Literal['full', 'simple'] = 'full',
            _neighbor_order: Iterable[direction] | None = None,
            _include_timelike_lowest_edges: bool = True,
    ):
        """
        :param code: the code to be decoded.
        :param merger: decides whether nodes flood before syncing (fast) or vice versa (slow) in a merging step.
            Setting this to ``'slow'`` helps break down merging for visualisation.
        :param schedule: the cluster growth schedule.
        :param unrooter: the type of unrooting process to use.
            If ``'full'``, each node in the amputated cluster resets its CID and pointer
        so that the pointer tree structure can be rebuilt from scratch,
        via further merging timesteps.
        If ``'simple'``, the node at breaking point only
        establishes the shortest path to a boundary.
        :param _neighbor_order: optionally customizes the order in which each node checks its neighbors.
            For the repetition code, this should be an iterable of the four directions
        ('W', 'E', 'D', 'U') in some order.
        For the surface code, this should be an iterable of the twelve directions
        ('NWD', 'N', 'NU', 'WD', 'W', 'D', 'U', 'E', 'EU', 'SD', 'S', 'SEU') in some order.
        Technically you can exclude the ones which are not part of the decoding the graph
        e.g. the diagonal directions (comprising >=2 characters) for phenomenological noise,
        but if you accidentally exclude a relevant direction
        then no node will ever check that direction for growing, flooding, nor syncing.
        So it is safest to include all twelve directions even if some are not used.
        If ``_neighbor_order`` is not specified, the orders listed above are used.
        :param _include_timelike_lowest_edges: whether to include
            the purely timelike (i.e. 'up') edges in ``self._LOWEST_EDGES``.
        """
        if isinstance(code.NOISE, CodeCapacity):
            raise ValueError('Snowflake incompatible with code capacity noise model.')
        if not isinstance(code.SCHEME, Frugal):
            raise ValueError('Snowflake only compatible with frugal scheme i.e. `code.SCHEME` must be `Frugal`.')
        super().__init__(code)
        self._SCHEDULE: _Schedule = (_TwoOne if schedule == '2:1' else _OneOne)(self)
        window_height = self.CODE.SCHEME.WINDOW_HEIGHT
        self._EDGES = {
            e: _Edge(self, e) for e in self.CODE.EDGES
            if all(v[self.CODE.TIME_AXIS] < window_height for v in e)
        }
        if _neighbor_order is None:
            if isinstance(code, Repetition):
                _neighbor_order = ('W', 'E', 'D', 'U')
            else:  # Surface
                _neighbor_order = ('NWD', 'N', 'NU', 'WD', 'W', 'D', 'U', 'E', 'EU', 'SD', 'S', 'SEU')
        self._NEIGHBOR_ORDER: Iterable[direction] = _neighbor_order
        self._NODES = {
            v: _Node(
                self,
                v,
                merger=merger,
                unrooter=unrooter,
            ) for v in self.CODE.NODES
            if v[self.CODE.TIME_AXIS] < window_height
        }
        self._DECODE_DRAWER = DecodeDrawer(self._FIG_WIDTH, fig_height=self._FIG_HEIGHT)
        self._stage = Stage.DROP
        self._BITSTRING_CONVERTER = (_Repetition if type(code) is Repetition else _Surface)(code.D, window_height)
        self._LOWEST_EDGES = tuple(self.EDGES[e] for e in self._BITSTRING_CONVERTER.lowest_edges(
            str(code.NOISE),
            _include_timelike_lowest_edges=_include_timelike_lowest_edges,
        ))
    
    def __repr__(self) -> str:
        return f'decoders.Snowflake({self.CODE})'

    @property
    def NODES(self): return self._NODES

    @property
    def EDGES(self): return self._EDGES

    @property
    def syndrome(self):
        return self.verbose_syndrome.difference(self.CODE.BOUNDARY_NODES)

    @property
    def verbose_syndrome(self):
        return {v for v, node in self.NODES.items() if node.defect}
    
    @property
    def growth(self):
        return {e: edge.growth for e, edge in self.EDGES.items()}

    @property
    def correction(self):
        return {e for e, edge in self.EDGES.items() if edge.correction}

    @cache
    def index_to_id(self, index: Node):
        """Return ID of node at index (j, t) or (i, j, t)."""
        d = self.CODE.D
        h = self.CODE.SCHEME.WINDOW_HEIGHT
        if len(index) == 2:
            j, t = index
            if j == -1:  # on west boundary
                return 2 * (h-1 - t)
            elif j == d-1:  # on east boundary
                return 2 * (h-1 - t) + 1
            else:
                return 2*h + (d-1)*(h-1 - t) + j
        else:  # 3
            i, j, t = index
            if j == -1:
                return 2*d * (h-1 - t) + i
            elif j == d-1:
                return 2*d * (h-1 - t) + d + i
            else:
                return 2*d*h + d*(d-1)*(h-1 - t) + d*j + i
            
    def init_history(self):
        """Initialize ``history`` and ``floor_history`` attributes."""
        super().init_history()
        self.floor_history: list[str] = []
        
    @cache
    def id_below(self, id_: int):
        """Return ID of node below node with ID ``id_``.
        
        Answer valid only for nodes not on bottom layer of viewing window.
        In implementation this function would be stored in each node.
        """
        d = self.CODE.D
        h = self.CODE.SCHEME.WINDOW_HEIGHT
        if isinstance(self.CODE, Repetition):
            delta = 2 if id_<2*h else d-1
        else:  # Surface
            delta = 2*d if id_<2*d*h else d*(d-1)
        return id_ + delta
        
    def reset(self):
        super().reset()
        for node in self.NODES.values():
            node.reset()
        for edge in self.EDGES.values():
            edge.reset()
        self._stage = Stage.DROP
        try: del self.history
        except AttributeError: pass
        try: del self.floor_history
        except AttributeError: pass
    
    def decode(
            self,
            syndrome: set[Node],
            log_history: Literal[False, 'fine', 'coarse'] = False,
            log_floor_history: bool = False,
            time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
            defects_possible: bool = True,
        ):
        """Perform a decoding cycle i.e. a growth round.
        
        
        :param syndrome: the syndrome in the new region discovered by the window raise
            i.e. all defects in ``syndrome`` have the time coordinate ``self.CODE.SCHEME.WINDOW_HEIGHT-1``.
        :param log_history: whether to populate ``history`` attribute --
            'fine' logs each timestep;
        'coarse', only the final timestep of the growth round.
        :param log_floor_history: whether to populate ``floor_history`` attribute.
        :param time_only: whether runtime includes a timestep
            for each drop, each grow, and each merging step ('all');
        each merging step only ('merging');
        or each unrooting step only ('unrooting').
        :param defects_possible: whether to expect there may be defects in the viewing window
            in the current or any future timestep.
        If ``False``, the decoder will perform only 'drop',
        and will skip 'grow' and 'merge' stages.
        This is useful at the end of the memory experiment after the final syndrome data has come in.
        
        
        :returns: ``t`` number of timesteps to complete decoding cycle. Equals the increase in ``len(self.history)`` if ``log_history`` is 'fine' and ``time_only`` is ``'all'``.
        """
        self._stage = Stage.DROP
        if log_floor_history:
            self.floor_history.append(
                ''.join(str(int(e.correction)) for e in self._LOWEST_EDGES))
        self.drop(syndrome)
        if log_history == 'fine': self.append_history()
        if defects_possible:
            return self._SCHEDULE.finish_decode(
                log_history=log_history,
                time_only=time_only,
            )
        else:
            return 1 if time_only == 'all' else 0
    
    def drop(self, syndrome: set[Node]):
        """Make all nodes perform a ``drop`` i.e. raise window by a layer."""
        # SLIDE EDGES DOWN
        for edge in self.EDGES.values():
            edge.CONTACT.drop()
        for edge in self.EDGES.values():
            edge.update_after_drop()
        # SLIDE NODES DOWN
        for node in self.NODES.values():
            node.FRIENDSHIP.drop()
        self._load(syndrome)
        for node in self.NODES.values():
            node.update_after_drop()

    def _load(self, syndrome: set[Node]):
        """Load ``syndrome`` onto decoder.
        
        I.e. top sheet inherits the new measurement round results.
        Note do not set ``active`` as ``defect``
        else nodes in top sheet of viewing window will grow.
        This growth would be premature as there are no edges above these nodes,
        leading to uneven cluster shapes.
        """
        for v in syndrome:
            self.NODES[v].next_defect ^= True

    def merge(
        self,
        whole: bool,
        log_history: Literal[False, 'fine', 'coarse'],
        time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
    ):
        """Make all nodes perform ``merging`` until none are busy.
        
        Emergent effect: merge touching clusters, push defects to roots.
        
        
        :param whole: whether to perform ``MERGING_WHOLE`` or ``MERGING_HALF`` stage.
        :param log_history: as in ``decode`` inputs.
        :param time_only: as in ``decode`` inputs.
        
        Output: ``t`` number of timesteps to complete growth round.
        """
        t = -1 if time_only == 'merging' else 1 if time_only == 'all' else 0

        while True:
            for node in self.NODES.values():
                node.merging(whole)
            for node in self.NODES.values():
                node.update_after_merging()
            t += (time_only!='unrooting') or any(node.cid==RESET for node in self.NODES.values())
            if not any(node.busy for node in self.NODES.values()):
                break
            if log_history == 'fine': self.append_history()

        if log_history == 'coarse': self.append_history()
        return t

    # DRAWERS

    def _labels(
            self,
            show_global=True,
            show_2_1_schedule_variables=True,
    ):
        """Return the labels dictionary for the drawer.
        
        
        :param show_global: whether to prepend the global label to the top-left node label.
        :param show_2_1_schedule_variables: whether to show
            node variables specific to the 2:1 cluster growth schedule.
        
        
        :returns: ``result`` a dictionary where each key a node index as a tuple; value, the label for the node at that index.
        """
        result = {v: node.label(show_2_1_schedule_variables)
                  for v, node in self.NODES.items()}
        if show_global:
            t = self.CODE.SCHEME.WINDOW_HEIGHT - 1
            top_left = (0, -1, t) if self.CODE.DIMENSION == 3 else (-1, t)
            result[top_left] = str(self._stage) + result[top_left]
        return result

    def draw_growth(
        self,
        highlighted_edges: set[Edge] | None = None,
        highlighted_edge_color='k',
        unhighlighted_edge_color=constants.DARK_GRAY,
        x_offset=constants.STREAM_X_OFFSET,
        with_labels=True,
        labels: dict[Node, str] | None = None,
        show_global=True,
        show_2_1_schedule_variables: None | bool = None,
        node_size: int | None = None,
        linewidths: float | None = None,
        active_shape='s',
        width: float | None = None,
        arrows: bool | None = None,
        show_boundary_defects=True,
        black_and_white=False,
        # following 2 kwargs are only for `black_and_white = True`
        bw_unhighlighted_width=constants.MEDIUM_THIN,
        bw_unrooted_color=constants.GRAY,
        **kwargs_for_networkx_draw,
    ):
        g = self.CODE.GRAPH
        dig, dig_diedges, dig_edges = self._pointer_digraph
        pos = self.CODE.get_pos(x_offset)
        unrooted_nodes = {v for v, node in self.NODES.items() if node.unrooted}
        if highlighted_edges is None:
            highlighted_edges = self.correction
        if show_2_1_schedule_variables is None:
            show_2_1_schedule_variables = isinstance(self._SCHEDULE, _TwoOne)
        if labels is None:
            labels = self._labels(
                show_global=show_global,
                show_2_1_schedule_variables=show_2_1_schedule_variables,
            )
        if node_size is None:
            node_size = self.BW_DEFAULT_NODE_SIZE if black_and_white else constants.DEFAULT
        if linewidths is None:
            linewidths = constants.THIN if black_and_white else constants.MEDIUM
        if width is None:
            width = constants.WIDE if black_and_white else constants.WIDE_MEDIUM

        if black_and_white:
            return self._draw_growth_black_and_white(
                g, dig, dig_diedges, dig_edges,
                pos, unrooted_nodes,
                highlighted_edges, with_labels, labels,
                node_size=node_size,
                unrooted_color=bw_unrooted_color,
                linewidths=linewidths,
                active_shape=active_shape,
                width=width,
                arrows=arrows,
                unhighlighted_width=bw_unhighlighted_width,
                show_boundary_defects=show_boundary_defects,
                **kwargs_for_networkx_draw,
            )

        # DRAW INACTIVE NODES AND EDGES UNUSED BY POINTERS
        
        # node-related kwargs
        inactive_nodelist = [v for v, node in self.NODES.items() if not node.active]
        inactive_node_color, inactive_edgecolors = self._get_node_color_and_edgecolors(
            outlined_nodes=unrooted_nodes,
            nodelist=inactive_nodelist,
            show_boundary_defects=show_boundary_defects,
        )
        
        # edge-related kwargs
        edgelist = [
            e for e, edge in self.EDGES.items()
            if edge.growth in {Growth.HALF, Growth.FULL}
            and e not in dig_edges
        ]
        edge_color, style = self._get_edge_color_and_style(
            edgelist,
            highlighted_edges,
            highlighted_edge_color,
            unhighlighted_edge_color,
        )
        nx.draw(
            g,
            pos=pos,
            with_labels=with_labels,
            labels=labels,
            nodelist=inactive_nodelist,
            node_size=node_size,
            node_color=inactive_node_color,
            linewidths=linewidths,
            edgecolors=inactive_edgecolors,
            edgelist=edgelist,
            width=width,
            edge_color=edge_color,
            style=style,
            **kwargs_for_networkx_draw,
        )

        # DRAW ACTIVE NODES
        active_nodelist = [v for v, node in self.NODES.items() if node.active]
        active_node_color, active_edgecolors = self._get_node_color_and_edgecolors(
            outlined_nodes=unrooted_nodes,
            nodelist=active_nodelist,
            show_boundary_defects=show_boundary_defects,
        )
        # need not actually pass `show_boundary_defects`
        # as boundary nodes are never active
        # but include in case need test correctness
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=active_nodelist,
            node_size=node_size,
            node_color=active_node_color, # type: ignore
            node_shape=active_shape,
            linewidths=linewidths,
            edgecolors=active_edgecolors,
        )

        # DRAW EDGES USED BY POINTERS
        dig_edge_color = [
            highlighted_edge_color if e in highlighted_edges else
            unhighlighted_edge_color for e in dig_edges
        ]
        nx.draw_networkx_edges(
            dig,
            pos,
            node_size=node_size,
            edgelist=dig_diedges,
            width=width,
            edge_color=dig_edge_color, # type: ignore
            arrows=arrows,
        )

    def _draw_growth_black_and_white(
        self,
        g: nx.Graph, dig: nx.DiGraph, dig_diedges: list[Edge], dig_edges: list[Edge],
        pos: dict[Node, Coord], unrooted_nodes: set[Node],
        highlighted_edges: set[Edge], with_labels: bool, labels: dict[Node, str],
        node_size=BW_DEFAULT_NODE_SIZE,
        unrooted_color=constants.GRAY,
        linewidths=constants.THIN,
        active_shape='s',
        width=constants.WIDE,
        unhighlighted_width=constants.MEDIUM_THIN,
        arrows: bool | None = None,
        show_boundary_defects=True,
        **kwargs_for_networkx_draw,
    ):

        # DRAW INACTIVE DETECTORS AND EDGES UNUSED BY POINTERS
        
        # node-related kwargs
        inactive_detector_list = [v for v, node in self.NODES.items() if not (node.active or node._IS_BOUNDARY)]
        inactive_detector_linewidths = [width if v in self.verbose_syndrome else linewidths for v in inactive_detector_list]
        inactive_detector_color = [unrooted_color if v in unrooted_nodes else 'w' for v in inactive_detector_list]
        
        # edge-related kwargs
        edgelist = [
            e for e, edge in self.EDGES.items()
            if edge.growth in {Growth.HALF, Growth.FULL}
            and e not in dig_edges
        ]
        edge_width = [
            width if e in highlighted_edges else
            unhighlighted_width for e in edgelist
        ]
        style = [
            ':' if self.growth[e] is Growth.HALF else
            '-' for e in edgelist
        ]
        nx.draw(
            g,
            pos=pos,
            with_labels=with_labels,
            labels=labels,
            nodelist=inactive_detector_list,
            node_size=node_size,
            node_color=inactive_detector_color,
            linewidths=inactive_detector_linewidths,
            edgecolors='k',
            edgelist=edgelist,
            width=edge_width,
            edge_color='k',
            style=style,
            **kwargs_for_networkx_draw,
        )

        # DRAW ACTIVE DETECTORS
        active_detector_list = [v for v, node in self.NODES.items() if node.active]
        active_detector_linewidths = [width if v in self.verbose_syndrome else linewidths for v in active_detector_list]
        active_detector_color = [unrooted_color if v in unrooted_nodes else 'w' for v in active_detector_list]
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=active_detector_list,
            node_size=node_size,
            node_color=active_detector_color, # type: ignore
            node_shape=active_shape,
            linewidths=active_detector_linewidths,
            edgecolors='k',
        )

        # DRAW BOUNDARY NODES
        boundary_nodelist = [v for v, node in self.NODES.items() if node._IS_BOUNDARY]
        boundary_linewidths = [width if v in self.verbose_syndrome else linewidths for v in boundary_nodelist] \
            if show_boundary_defects else linewidths
        nx.draw_networkx_nodes(
            g,
            pos,
            nodelist=boundary_nodelist,
            node_size=node_size,
            node_color='w',
            node_shape='8',
            linewidths=boundary_linewidths,
            edgecolors='k',
        )

        # DRAW EDGES USED BY POINTERS
        dig_edge_width = [
            width if e in highlighted_edges else
            unhighlighted_width for e in dig_edges
        ]
        nx.draw_networkx_edges(
            dig,
            pos,
            node_size=node_size,
            edgelist=dig_diedges,
            width=dig_edge_width, # type: ignore
            edge_color='k',
            arrows=arrows,
        )

    @property
    def _pointer_digraph(self):
        """Return a NetworkX digraph representing the fully grown edges used by pointers,
        the set of its edges as directed edges,
        the set of its edges as undirected edges.
        TODO: this is a temporary fix of ``_DigraphMaker.pointer_digraph``.
        i.e. ``self.__init__`` used to have the line
        ``self._DIGRAPH_MAKER = DigraphMaker(self.NODES, self.growth)``.
        """
        dig = nx.DiGraph()
        dig.add_nodes_from(self.NODES.keys())
        dig_diedges: list[Edge] = []
        dig_edges: list[Edge] = []
        for u, node in self.NODES.items():
            if node.pointer != 'C':
                try:
                    e, index = node.NEIGHBORS[node.pointer]
                    if self.EDGES[e].growth is Growth.FULL:
                        v = e[index]
                        dig.add_edge(u, v)
                        dig_diedges.append((u, v))
                        dig_edges.append(e)
                except KeyError: pass  # node in bottom sheet points down
        return dig, dig_diedges, dig_edges
    
    def draw_decode(self, **kwargs_for_networkx_draw):
        self._DECODE_DRAWER.draw(self.history, **kwargs_for_networkx_draw)

    @property
    def _FIG_WIDTH(self):
        return max(1, self.CODE.D * self._FIG_FACTOR)
    
    @property
    def _FIG_HEIGHT(self):
        return max(1, (self.CODE.SCHEME.WINDOW_HEIGHT-1) * self._FIG_FACTOR)

    @property
    def _FIG_FACTOR(self):
        n = self.CODE.DIMENSION
        return 3*n*(n-1) / 10
    

    def generate_output(
            self,
            syndromes: list[str] | list[set[Node]],
            output_to_csv_file: str | None = None,
            draw: Literal[False, 'fine', 'coarse'] = False,
            margins=(0.2, 0.2),
            style: Literal['interactive', 'horizontal', 'vertical'] = 'interactive',
            **kwargs_for_draw_decode,
    ):
        """Generate output in the form of bitstrings.
        
        
        :param syndromes: the input of Snowflake.
            This can either be a list of sets of coordinates (each set representing a syndrome;
            each coordinate specifying a defect), or a list of syndrome vectors (strings of '0's
            and '1's). The vertical (i.e. time) coordinate of each defect must be
            ``self.CODE.SCHEME.WINDOW_HEIGHT-1``.

            For syndrome vectors, the ordering of the nodes is:

            - repetition code: west to east.
            - surface code: west to east along each row, then from south to north.
        :param output_to_csv_file: the CSV file path to save the data in e.g. 'snowflake_data.csv'.
            Defaults to ``None``, meaning no CSV file is saved.
        :param draw: whether to skip drawing the decoding process,
            draw it finely or draw it coarsely.
        :param margins: margins for the drawing.
        :param style: how different drawing frames are laid out.
            Can be 'interactive', 'horizontal', or 'vertical'.
        :param kwargs_for_draw_decode: additional keyword arguments for ``decoder.draw_decode()``.
        
        
        :returns: The output of Snowflake. This is a list of strings, each representing the edges in the bottom layer that are flipped just before each drop. The ordering of the edges is given by ``self._LOWEST_EDGES``. INCONSISTENCY: for the repetition code, the purely timelike edges in the last layer are excluded from ``self._LOWEST_EDGES``; the spacelike edges are ordered from west to east.
        
        Side effects:
        * If ``output_to_csv_file`` is not None, the input and output of Snowflake
            are saved in a CSV file at path ``output_to_csv_file``.
        * If ``draw`` is True, the decoding process is drawn.
        """
        first_syndrome, *_ = syndromes
        if type(first_syndrome) is str:
            # pad `syndrome_vectors` with zero vectors
            syndrome_vectors: list[str] = syndromes + self.CODE.SCHEME.WINDOW_HEIGHT * ['0'*self._BITSTRING_CONVERTER.LENGTH] # type: ignore
            syndrome_sets = self._BITSTRING_CONVERTER.syndrome_vectors_to_sets(syndrome_vectors)
        else:
            # pad `syndrome_sets` with empty sets
            syndrome_sets: list[set[Node]] = syndromes + self.CODE.SCHEME.WINDOW_HEIGHT * [set()] # type: ignore
            syndrome_vectors = self._BITSTRING_CONVERTER.sets_to_syndrome_vectors(syndrome_sets)
        
        self.init_history()
        for syndrome in syndrome_sets:
            self.decode(syndrome, log_history=draw, log_floor_history=True)
        if draw:
            self.draw_decode(
                margins=margins,
                style=style,
                **kwargs_for_draw_decode,
            )
        if output_to_csv_file is not None:
            import csv
            # Save syndrome vectors and corresponding floor history to CSV
            with open(output_to_csv_file, 'w', newline='') as csvfile:
                writer = csv.writer(csvfile)
                # Write header
                writer.writerow(['syndrome_vector', 'floor_history'])
                # Write data rows
                for syndrome_vector, floor_vector in zip(syndrome_vectors, self.floor_history, strict=True):
                    writer.writerow([syndrome_vector, floor_vector])
        return self.floor_history


class _BitstringConverter(abc.ABC):
    """Converts syndromes from bitstring format to sets of coordinates format."""

    @property
    @abc.abstractmethod
    def LENGTH(self):
        """The expected length of the syndrome vector."""

    def __init__(self, d: int, window_height: int):
        """
        :param d: the code distance.
        :param window_height: the height of the viewing window.
        """
        self._D = d
        self._WINDOW_HEIGHT = window_height

    def syndrome_vectors_to_sets(
            self,
            vectors: Sequence[str],
        ) -> list[set[Node]]:
        """Convert a sequence of syndrome vectors to a list of sets of defects.
        
        
        :param vectors: a sequence of syndrome vectors, each a string of '0's and '1's.
        
        
        :returns: A list of sets of defects, each set corresponding to a syndrome vector.
        """
        return [self._syndrome_vector_to_set(vector) for vector in vectors]

    @abc.abstractmethod
    def _syndrome_vector_to_set(
            self,
            vector: str,
        ) -> set[Node]:
        """Convert a syndrome vector to a set of defects.
        
        The ordering of the nodes is stated in ``Snowflake.generate_output`` docstring.
        """

    def sets_to_syndrome_vectors(
            self,
            syndromes: Sequence[set[Node]],
        ) -> list[str]:
        """Convert a sequence of sets of defects to a list of syndrome vectors.
        
        
        :param syndromes: a sequence of sets of defects.
        
        
        :returns: A list of syndrome vectors, each a string of '0's and '1's, each corresponding to a set of defects.
        """
        return [self._set_to_syndrome_vector(syndrome) for syndrome in syndromes]

    @abc.abstractmethod
    def _set_to_syndrome_vector(
            self,
            syndrome: set[Node],
    ) -> str:
        """Convert a set of defects to a syndrome vector.
        
        The ordering of the nodes is stated in ``Snowflake.generate_output`` docstring.
        """

    @abc.abstractmethod
    def lowest_edges(
            self,
            noise_model: str,
            _include_timelike_lowest_edges: bool = True,
    ) -> tuple[Edge, ...]:
        """Return the edges in the bottom layer of the viewing window
        in the order given by ``Snowflake._LOWEST_EDGES``.
        
        
        :param noise_model: the noise model of the code.
        :param _include_timelike_lowest_edges: whether to include the purely timelike edges.
        """


class _Repetition(_BitstringConverter):

    @property
    def LENGTH(self):
        return self._D - 1
    
    def _syndrome_vector_to_set(self, vector):
        if len(vector) != self.LENGTH:
            raise ValueError(f'Repetition code of distance {self._D} expects syndrome vector of length {self.LENGTH} but received length {len(vector)}.')
        return {(j, self._WINDOW_HEIGHT-1) for j, bit in enumerate(vector) if bit == '1'}
    
    def _set_to_syndrome_vector(self, syndrome):
        ones = {j for j, _ in syndrome}
        return ''.join('1' if j in ones else '0' for j in range(self.LENGTH))
    
    def lowest_edges(self, noise_model, _include_timelike_lowest_edges=True):
        up = tuple(((j, 0), (j, 1)) for j in range(self._D-1)) \
            if _include_timelike_lowest_edges else ()
        east = tuple(((j, 0), (j+1, 0)) for j in range(-1, self._D-1))
        if noise_model == 'phenomenological':
            return up + east
        else:
            raise NotImplementedError(f"Noise model {noise_model} not implemented for repetition code.")


class _Surface(_BitstringConverter):

    @property
    def LENGTH(self):
        return self._D * (self._D - 1)

    def _syndrome_vector_to_set(self, vector):
        if len(vector) != self.LENGTH:
            raise ValueError(f'Surface code of distance {self._D} expects syndrome vector of length {self.LENGTH} but received length {len(vector)}.')
        return {self._bit_position_to_defect(position) for position, bit in enumerate(vector) if bit == '1'}
    
    def _bit_position_to_defect(self, position: int) -> Node:
        """Convert bit position ``position`` to defect coordinate (i, j, t)."""
        return (
            self._D-1 - (position // (self._D-1)),
            position % (self._D-1),
            self._WINDOW_HEIGHT-1,
        )
    
    def _set_to_syndrome_vector(self, syndrome):
        ones = {self._defect_to_bit_position(defect) for defect in syndrome}
        return ''.join('1' if position in ones else '0' for position in range(self.LENGTH))
    
    def _defect_to_bit_position(self, defect: Node) -> int:
        """Convert defect coordinate (i, j, t) to a bit position."""
        i, j, t = defect
        if not (0 <= i < self._D and 0 <= j < self._D-1 and t == self._WINDOW_HEIGHT-1):
            raise ValueError(f'Defect {defect} out of range for surface code of distance {self._D} with window height {self._WINDOW_HEIGHT}.')
        return (self._D-1 - i) * (self._D-1) + j
    
    def lowest_edges(self, noise_model, _include_timelike_lowest_edges=True):
        d = self._D
        up = tuple(
            ((i, j, 0), (i, j, 1)) for i in range(d) for j in range(d-1)) \
            if _include_timelike_lowest_edges else ()
        south_down = tuple(
            ((i, j, 1), (i+1, j, 0)) for i in range(d-1) for j in range(d-1))
        east_up = tuple(
            ((i, j, 0), (i, j+1, 1)) for i in range(d) for j in range(d-2))
        south_east_up = tuple(
            ((i, j, 0), (i+1, j+1, 1)) for i in range(d-1) for j in range(d-2))
        south = tuple(
            ((i, j, 0), (i+1, j, 0)) for i in range(d-1) for j in range(d-1))
        east = tuple(
            ((i, j, 0), (i, j+1, 0)) for i in range(d) for j in range(-1, d-1))
        if noise_model == 'phenomenological':
            return up + south + east
        elif noise_model == 'circuit-level':
            return up + south_down + east_up + south_east_up + south + east
        else:
            raise NotImplementedError(f"Noise model {noise_model} not implemented for surface code.")


class NodeEdgeMixin(abc.ABC):
    """Mixin class for ``_Node`` and ``_Edge``.
    
    Instance attributes:
    * ``SNOWFLAKE`` the decoder the node or edge belongs to.
    """

    _SNOWFLAKE: Snowflake

    @property
    def SNOWFLAKE(self): return self._SNOWFLAKE


class _Node(NodeEdgeMixin):
    """Node for Snowflake.
    
    Extends ``NodeEdgeMixin``.
    
    Additional instance attributes:
    * ``INDEX`` index of node.
    * ``ID`` the unique ID of node.
    * ``FRIENDSHIP`` the type of connection the node has for communicating.
    * ``NEIGHBORS`` a dictionary where each
        key a pointer string;
    value, a tuple of the form (edge, index of edge which is neighbor).
    * ``_IS_BOUNDARY`` whether node is a boundary node.
    * ``_MERGER`` the provider for the ``merging`` method.
    * ``UNROOTER`` the type of unrooting process used.
    * ``active, whole, cid, defect, pointer, grown, unrooted, busy``
        explained in [arXiv:2406.01701, C.1].
    * The variables in the above bulletpoint, save ``busy``,
        each have ``next_`` versions for next timestep.
    * ``access`` refers to neighbors along fully grown edges.
        In the form of a dictionary of where each
    key a pointer string;
    value, the ``_Node`` object in the direction of that pointer.
    """

    def __init__(
            self,
            snowflake: Snowflake,
            index: Node,
            merger: Literal['fast', 'slow'] = 'fast',
            unrooter: Literal['full', 'simple'] = 'full',
        ) -> None:
        """
        :param snowflake: the decoder the node belongs to.
        :param index: index of node.
        :param merger: decides whether to flood before syncing (fast) or vice versa (slow) in a merging step.
        :param unrooter: the type of unrooting process to use.
        """
        self._SNOWFLAKE = snowflake
        self._INDEX = index
        self._ID = snowflake.index_to_id(index)
        self._FRIENDSHIP = NothingFriendship(self) if index[snowflake.CODE.TIME_AXIS] == 0 \
            else TopSheetFriendship(self) if index[snowflake.CODE.TIME_AXIS] == snowflake.CODE.SCHEME.WINDOW_HEIGHT-1 \
            else NodeFriendship(self)
        if isinstance(snowflake.CODE, Repetition):
            j, t = index
            provisional_neighbors: dict[direction, tuple[Edge, int]] = {
                'W': (((j-1, t), index), 0),
                'E': ((index, (j+1, t)), 1),
                'D': (((j, t-1), index), 0),
                'U': ((index, (j, t+1)), 1),
            }
        else:  # Surface
            i, j, t = index
            provisional_neighbors: dict[direction, tuple[Edge, int]] = {
                'NWD': (((i-1, j-1, t-1), index), 0),
                'N': (((i-1, j, t), index), 0),
                'NU': (((i-1, j, t+1), index), 0),
                'WD': (((i, j-1, t-1), index), 0),
                'W': (((i, j-1, t), index), 0),
                'D': (((i, j, t-1), index), 0),
                'U': ((index, (i, j, t+1)), 1),
                'E': ((index, (i, j+1, t)), 1),
                'EU': ((index, (i, j+1, t+1)), 1),
                'SD': ((index, (i+1, j, t-1)), 1),
                'S': ((index, (i+1, j, t)), 1),
                'SEU': ((index, (i+1, j+1, t+1)), 1),
            }
        self._NEIGHBORS: dict[direction, tuple[Edge, int]] = {
            pointer: provisional_neighbors[pointer]
            for pointer in snowflake._NEIGHBOR_ORDER
            if provisional_neighbors[pointer][0] in snowflake.EDGES
        }
        self._IS_BOUNDARY = snowflake.CODE.is_boundary(index)
        self._MERGER = _FastMerger(self) if merger == 'fast' else _SlowMerger(self)
        self._UNROOTER = _FullUnrooter(self) if unrooter == 'full' else _SimpleUnrooter(self)
        self.reset()

    def __repr__(self) -> str:
        return f'decoders.snowflake._Node({self._SNOWFLAKE}, {self.INDEX})'
    
    def __str__(self) -> str:
        return str(self.INDEX)

    @property
    def INDEX(self): return self._INDEX

    @property
    def ID(self): return self._ID

    @property
    def FRIENDSHIP(self): return self._FRIENDSHIP

    @property
    def NEIGHBORS(self): return self._NEIGHBORS

    @property
    def UNROOTER(self): return self._UNROOTER

    def label(self, show_2_1_schedule_variables=True):
        cid = str(self.cid) if self.cid != RESET else 'R'
        if show_2_1_schedule_variables:
            if self.grown:
                grown_half = "'" if self.whole else ":"
            else:
                grown_half = "" if self.whole else "."
        else:
            grown_half = ""
        return cid + grown_half

    def reset(self):
        """Factory reset."""

        self.active = False
        self.whole = True
        self.cid = self.ID
        self.defect = False
        self.pointer: direction = 'C'
        self.grown = False
        self.unrooted = False

        self.next_active = False
        self.next_whole = True
        self.next_cid = self.ID
        self.next_defect = False
        self.next_pointer: direction = 'C'
        self.next_grown = False
        self.next_unrooted = False

        self.busy = False
        self.access: dict[direction, _Node] = {}

    def update_after_drop(self):
        """Update unphysicals after a drop.
        
        Set ``[attribute]`` := ``next_[attribute]``.
        For nodes in top sheet of viewing window, this equals ``reset``.
        """
        self.defect = self.next_defect
        self.active = self.next_active
        self.cid = self.next_cid
        self.pointer = self.next_pointer
        self.whole = self.next_whole
        # NEED NOT RESET...
        # `next_defect` as value correct for next use in `syncing`
        # `next_active` as always overwritten in `syncing` before next used in `update_after_merging`
        # `next_cid` as value correct for next use in `flooding`
        # `next_pointer` as always overwritten in `drop` before next used in `update_after_drop`
        # `next_whole` as always overwritten in `drop` before next used in `update_after_drop`

    def grow(self):
        """Call ``grow`` if active, then find broken pointers.
        
        Used only in 1:1 schedule.
        """
        if self.active:
            self._grow()
        self.FRIENDSHIP.find_broken_pointers()

    def grow_whole(self):
        """Call ``grow`` if active and whole, then find broken pointers."""
        if self.active and self.whole:
            self._grow()
            self.grown = True
            self.next_grown = True
        self.FRIENDSHIP.find_broken_pointers()

    def grow_half(self):
        """Call ``grow`` if active and half and not yet grown in current decoding cycle."""
        self.unrooted = False
        self.next_unrooted = False
        if self.active and not self.whole and not self.grown:
            self._grow()
    
    def _grow(self):
        """Growth-increment incident edges."""
        s = self.SNOWFLAKE
        edges_to_grow = {
            e for e, _ in self.NEIGHBORS.values()
            if s.EDGES[e].growth in s._ACTIVE_GROWTH_VALUES
        }
        for e in edges_to_grow:
            s.EDGES[e].growth += Growth.INCREMENT
        self.whole ^= True

    def update_access(self):
        """Update access.
        
        Call after growth has changed. This method is unphysical.
        """
        self.access = {
            pointer: self.SNOWFLAKE.NODES[e[index]]
            for pointer, (e, index) in self.NEIGHBORS.items()
            if self.SNOWFLAKE.EDGES[e].growth is Growth.FULL
        }

    def merging(self, whole: bool):
        """Advance 1 merging timestep.
        
        The emergent effect of each node running this method repeatedly
        is the merging of clusters.
        """
        self.busy = False
        self._MERGER.merging(whole)

    def syncing(self):
        """Update ``active`` depending on, and push defect to, pointee."""
        detector_defect = not self._IS_BOUNDARY and self.defect
        if self.pointer == 'C':
            self.next_active = detector_defect
        else:
            self.next_active = self.access[self.pointer].active
            if detector_defect:  # PUSH DEFECT
                self.busy = True
                # relay along pointer
                self.access[self.pointer].next_defect ^= True
                self.next_defect ^= True
                # flip edge used by pointer
                e, _ = self.NEIGHBORS[self.pointer]
                self.SNOWFLAKE.EDGES[e].correction ^= True
        if self.active != self.next_active:
            self.busy = True

    def flooding(self, whole: bool):
        """Update ``pointer, cid, unrooted, grown`` depending on access."""
        if whole:
            self.UNROOTER.flooding_whole()
        else:
            self.UNROOTER.flooding_half()

    def update_after_merging(self):
        """``next_{cid, defect, active, unrooted, grown}`` -> ``{cid, defect, active, unrooted, grown}``."""
        self.cid = self.next_cid
        self.defect = self.next_defect
        self.active = self.next_active
        self.unrooted = self.next_unrooted
        self.grown = self.next_grown
        # NEED NOT RESET...
        # `next_cid` as value correct for next use in `flooding`
        # `next_defect` as value correct for next use in `syncing`
        # `next_active` as always overwritten in `syncing` before next used in `update_after_merging`
        # `next_unrooted` as value correct (unless edited) for next use in `update_after_merging`
        # `next_grown` as value correct for next use in `flooding`


class Friendship(abc.ABC):
    """The type of connection for communicating
    ``defect, active, cid, pointer`` information.
    
    Instance attributes (1 constant):
    * ``NODE`` the node which has this friendship.
    """

    def __init__(self, node: _Node) -> None:
        self._NODE = node

    @property
    def NODE(self): return self._NODE

    def drop(self):
        """Drop information to node immediately below."""
        self.NODE.grown = False
        self.NODE.next_grown = False
        self.NODE.unrooted = False
        self.NODE.next_unrooted = False

    def find_broken_pointers(self):
        """Start unrooting if in bottom sheet of viewing window and point downward."""


class NodeFriendship(Friendship):
    """Friendship with node immediately below.
    
    Extends ``Friendship``.
    
    Instance attributes (1 constant):
    * ``DROPEE`` the node immediately below.
    """

    def __init__(self, node: _Node) -> None:
        self._DROPEE = node.SNOWFLAKE.CODE.raise_node(
            node.INDEX,
            delta_t=-1,
        )
        super().__init__(node)

    @property
    def DROPEE(self): return self._DROPEE

    def drop(self):
        super().drop()
        r = self.NODE.SNOWFLAKE.NODES[self.DROPEE]
        r.next_defect = self.NODE.defect
        r.next_active = self.NODE.active
        r.next_cid = r.SNOWFLAKE.id_below(self.NODE.cid)
        r.next_pointer = self.NODE.pointer
        r.next_whole = self.NODE.whole

class TopSheetFriendship(NodeFriendship):
    """Friendship for nodes in top sheet of viewing window.
    
    Extends ``NodeFriendship``.
    After a drop, these nodes must reset ``next_active`` and ``next_defect``.
    """

    def drop(self):
        super().drop()
        self.NODE.next_defect = False
        self.NODE.next_active = False


class NothingFriendship(Friendship):
    """Friendship with nothing immediately below.
    
    Extends ``Friendship``.
    Only nodes whose time index is 0 have this friendship.
    """

    def find_broken_pointers(self):
        if 'D' in self.NODE.pointer:
            self.NODE.UNROOTER.start()


class _Merger(abc.ABC):
    """The class providing ``_Node.merging``."""

    def __init__(self, node: _Node) -> None:
        self._NODE = node

    @abc.abstractmethod
    def merging(self):
        """See ``_Node.merging``."""


class _SlowMerger(_Merger):
    """Provider for ``_Node.merging`` which syncs before it floods.
    
    Extends ``_Merger``.
    """

    def merging(self, whole: bool):
        self._NODE.syncing()
        self._NODE.flooding(whole)


class _FastMerger(_Merger):
    """Provider for ``_Node.merging`` which floods before it syncs.
    
    Extends ``_Merger``.
    """

    def merging(self, whole: bool):
        self._NODE.flooding(whole)
        self._NODE.syncing()


class _Schedule(abc.ABC):
    """Abstract base class for the cluster growth schedule."""

    def __init__(self, snowflake: Snowflake) -> None:
        self._SNOWFLAKE = snowflake

    @abc.abstractmethod
    def finish_decode(
        self,
        log_history: Literal[False, 'fine', 'coarse'] = False,
        time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
    ) -> int:
        """Perform the rest of the decoding cycle after drop.
        
        
        :param log_history: as in ``decode`` inputs.
        :param time_only: as in ``decode`` inputs.
        
        
        :returns: ``t`` number of timesteps to complete decoding cycle.
        """

    @abc.abstractmethod
    def grow(self):
        """Make all nodes perform a ``grow``."""


class _OneOne(_Schedule):
    """The 1:1 cluster growth schedule.
    
    Extends ``_Schedule``.
    """

    def finish_decode(self, log_history, time_only):
        self._SNOWFLAKE._stage += Stage.INCREMENT
        self.grow()
        if log_history == 'fine': self._SNOWFLAKE.append_history()
        self._SNOWFLAKE._stage += Stage.INCREMENT
        return self._SNOWFLAKE.merge(
            whole=True,
            log_history=log_history,
            time_only=time_only,
        )
    
    def grow(self):
        for node in self._SNOWFLAKE.NODES.values():
            node.grow()
        for node in self._SNOWFLAKE.NODES.values():
            node.update_access()  # TODO: speed up by changing just the accesses affected


class _TwoOne(_Schedule):
    """The 2:1 cluster growth schedule.
    
    Extends ``_Schedule``.
    """

    def finish_decode(self, log_history, time_only):
        t = 0
        for whole in (True, False):
            self._SNOWFLAKE._stage += Stage.INCREMENT
            self.grow(whole)
            if log_history == 'fine': self._SNOWFLAKE.append_history()
            self._SNOWFLAKE._stage += Stage.INCREMENT
            t += self._SNOWFLAKE.merge(
                whole,
                log_history,
                time_only=time_only,
            )
        return t
    
    def grow(self, whole: bool):
        if whole:
            for node in self._SNOWFLAKE.NODES.values():
                node.grow_whole()
        else:
            for node in self._SNOWFLAKE.NODES.values():
                node.grow_half()
        for node in self._SNOWFLAKE.NODES.values():
            node.update_access()  # TODO: speed up by changing just the accesses affected


class _Unrooter(abc.ABC):
    """Abstract base class for the type of unrooting process to use.
    
    Instance constants:
    * ``_NODE`` the node which has this unrooter.
    """

    def __init__(self, node: _Node):
        """Input: ``node`` the node the unrooter belongs to."""
        self._NODE = node

    @abc.abstractmethod
    def start(self):
        """Start unrooting the node."""

    @abc.abstractmethod
    def flooding_whole(self):
        """Update ``pointer, cid, unrooted, grown`` depending on access."""

    def flooding_half(self):
        """Update ``pointer, cid`` depending on access."""
        for pointer, neighbor in self._NODE.access.items():
            self._compare_cid(pointer, neighbor)

    def _compare_cid(self, pointer: direction, neighbor: _Node):
        """Update ``next_cid`` depending on ``neighbor.cid``."""
        if neighbor.cid < self._NODE.next_cid:
            self._NODE.busy = True
            self._NODE.pointer = pointer
            self._NODE.next_cid = neighbor.cid

    def _check_grown(self, neighbor: _Node):
        """Update ``next_grown`` depending on ``neighbor.grown``."""
        if neighbor.grown and not self._NODE.next_grown:
            self._NODE.busy = True
            self._NODE.next_grown = True


class _FullUnrooter(_Unrooter):
    """Full unrooting process where each node in the amputated cluster resets its CID and pointer.
    
    This is so that the pointer tree structure can be rebuilt from scratch,
    via further merging timesteps.
    
    Extends ``_Unrooter``.
    """

    def start(self):
        self._NODE.cid = RESET
        self._NODE.pointer = 'C'
        # NEED NOT RESET...
        # `next_cid` as always overwritten in `NODE.flooding`
        # `next_pointer` as always overwritten in `drop` before next used in `NODE.update_after_drop`

    def flooding_whole(self):
        if self._NODE.cid == RESET:  # finish unrooting `self._NODE`
            self._NODE.busy = True
            self._NODE.next_cid = self._NODE.ID
            self._NODE.next_unrooted = True
        else:
            for pointer, neighbor in self._NODE.access.items():
                if neighbor.cid == RESET:
                    if not self._NODE.unrooted:  # start unrooting `self._NODE`
                        self._NODE.busy = True
                        self._NODE.next_cid = RESET
                        self._NODE.pointer = 'C'
                        break
                else:
                    self._compare_cid(pointer, neighbor)
                self._check_grown(neighbor)


class _SimpleUnrooter(_Unrooter):
    """Simple unrooting process where the node at breaking point only establishes the shortest path to a boundary.
    
    Extends ``_Unrooter``.
    
    Additional instance constants:
    * ``_CLOSEST_BOUNDARY_DIRECTION`` the direction toward the closest boundary.
    """

    def __init__(self, node: _Node):
        super().__init__(node)
        code = node.SNOWFLAKE.CODE
        self._CLOSEST_BOUNDARY_DIRECTION = 'W' if node.INDEX[code.LONG_AXIS] < (code.D-1)/2 else 'E'

    def start(self):
        self._NODE.cid = RESET
        self._NODE.next_cid = RESET

    def flooding_whole(self):
        if self._NODE.cid == RESET:
            if not self._NODE.unrooted and not self._NODE._IS_BOUNDARY:
                self._wave()
        else:
            for pointer, neighbor in self._NODE.access.items():
                if neighbor.cid != RESET:
                    self._compare_cid(pointer, neighbor)
                self._check_grown(neighbor)

    def _wave(self):
        """Propagate unroot wave toward nearest boundary."""
        self._NODE.next_unrooted = True
        self._NODE.pointer = self._CLOSEST_BOUNDARY_DIRECTION
        e, index = self._NODE.NEIGHBORS[self._NODE.pointer]
        u = self._NODE.SNOWFLAKE.NODES[e[index]]
        self._NODE.SNOWFLAKE.EDGES[e].growth = Growth.FULL  # physical
        self._NODE.access[self._NODE.pointer] = u  # unphysical
        u.next_cid = RESET


class _Edge(NodeEdgeMixin):
    """Edge for Snowflake.
    
    Extends ``NodeEdgeMixin``.
    
    Additional instance attributes:
    * ``INDEX`` index of edge.
    * ``CONTACT`` the type of connection the edge has for communicating.
        Analogous to ``FRIENDSHIP`` for nodes.
    * ``growth`` its growth value.
    * ``correction`` whether it is in the correction.
    """

    def __init__(self, snowflake: Snowflake, index: Edge) -> None:
        """
        :param snowflake: the decoder the edge belongs to.
        :param INDEX: index of edge.
        """
        self._SNOWFLAKE = snowflake
        self._INDEX = index
        lowness = sum(v[snowflake.CODE.TIME_AXIS] == 0 for v in index)
        self._CONTACT = EdgeContact(self) if lowness == 0 else FloorContact(self)
        self.reset()

    def __repr__(self) -> str:
        return f'decoders.snowflake._Edge({self.SNOWFLAKE}, {self.INDEX})'
    
    def __str__(self) -> str:
        return str(self.INDEX)

    @property
    def INDEX(self): return self._INDEX

    @property
    def CONTACT(self): return self._CONTACT

    def reset(self):
        """Factory reset."""

        self.growth = Growth.UNGROWN
        self.correction = False

        self.next_growth = Growth.UNGROWN
        self.next_correction = False

    def update_after_drop(self):
        """Update unphysicals after a drop.
        
        Set ``[attribute]`` := ``next_[attribute]``.
        For edges in ``self.CODE.SCHEME.BUFFER_EDGES``,
        this equals ``reset``.
        """
        self.growth = self.next_growth
        self.correction = self.next_correction


class _Contact(abc.ABC):
    """The type of connection for communicating
    ``growth, correction`` information.
    
    Instance attributes (1 constant):
    * ``EDGE`` the edge which has this contact.
    """

    def __init__(self, edge: _Edge) -> None:
        self._EDGE = edge

    @property
    def EDGE(self): return self._EDGE

    @abc.abstractmethod
    def drop(self):
        """Drop information to edge or floor immediately below."""


class EdgeContact(_Contact):
    """Contact with edge immediately below.
    
    Extends ``_Contact``.
    
    Instance attributes (1 constant):
    * ``DROPEE`` the edge immediately below.
    """

    def __init__(self, edge: _Edge) -> None:
        code = edge.SNOWFLAKE.CODE
        self._DROPEE = code.raise_edge(edge.INDEX, delta_t=-1)
        super().__init__(edge)

    @property
    def DROPEE(self): return self._DROPEE

    def drop(self):
        r = self.EDGE.SNOWFLAKE.EDGES[self.DROPEE]
        r.next_growth = self.EDGE.growth
        r.next_correction = self.EDGE.correction


class FloorContact(_Contact):
    """Contact for edges in commit region.
    
    Extends ``_Contact``.
    Only edges in the bottom sheet of the viewing window
    have this contact.
    """

    def drop(self):
        frugal: Frugal = self.EDGE.SNOWFLAKE.CODE.SCHEME # type: ignore
        if self.EDGE.correction:
            frugal.pairs.load(self.EDGE.INDEX)