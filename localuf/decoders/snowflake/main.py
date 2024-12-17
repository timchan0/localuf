import abc
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

    Extends `BaseUF`.
    Incompatible with code capacity noise model.
    Compatible only with frugal scheme.

    Class constants:
    `BW_DEFAULT_NODE_SIZE` the default node size for black-and-white drawing.
    
    Additional instance attributes:
    * `NODES` a dictionary of nodes.
    * `EDGES` ditto for edges.
    * `_pointer_digraph` a NetworkX digraph representing the fully grown edges used by pointers,
    the set of its edges as directed edges,
    the set of its edges as undirected edges.
    * `_stage` the current stage of the decoder.
    Only used for `draw_growth` when `show_global = True`.

    Overriden methods:
    * `reset`.
    * `decode`.
    * `draw_growth`.
    * `draw_decode`.
    
    Uses not:
    * `_growth` attribute.
    * `erasure` attribute.
    """

    BW_DEFAULT_NODE_SIZE = 360

    def __init__(
            self,
            code: Code,
            merger: Literal['fast', 'slow'] = 'fast',
            schedule: Literal['2:1', '1:1'] = '2:1',
            unrooter: Literal['full', 'simple'] = 'full',
    ):
        """Input:
        * `code` the code to be decoded.
        * `merger` decides whether nodes flood before syncing (fast) or vice versa (slow) in a merging step.
        Setting this to `'slow'` helps break down merging for visualisation.
        * `schedule` the cluster growth schedule.
        * `unrooter` the type of unrooting process to use.
        If `'full'`, each node in the amputated cluster resets its CID and pointer
        so that the pointer tree structure can be rebuilt from scratch,
        via further merging timesteps.
        If `'simple'`, the node at breaking point only
        establishes the shortest path to a boundary.
        """
        if isinstance(code.NOISE, CodeCapacity):
            raise ValueError('Snowflake incompatible with code capacity noise model.')
        if not isinstance(code.SCHEME, Frugal):
            raise ValueError('Snowflake only compatible with frugal scheme.')
        super().__init__(code)
        self._SCHEDULE: _Schedule = _TwoOne(self) if schedule == '2:1' else _OneOne(self)
        self._EDGES = {
            e: _Edge(self, e) for e in self.CODE.EDGES
            if all(v[self.CODE.TIME_AXIS] < self.CODE.SCHEME.WINDOW_HEIGHT for v in e)
        }
        self._NODES = {
            v: _Node(
                self,
                v,
                merger=merger,
                unrooter=unrooter,
            ) for v in self.CODE.NODES
            if v[self.CODE.TIME_AXIS] < self.CODE.SCHEME.WINDOW_HEIGHT
        }
        self._DECODE_DRAWER = DecodeDrawer(self._FIG_WIDTH, fig_height=self._FIG_HEIGHT)
        self._stage = Stage.DROP
    
    def __repr__(self) -> str:
        return f'decoders.snowflake.Snowflake(code={self.CODE})'

    @property
    def NODES(self): return self._NODES

    @property
    def EDGES(self): return self._EDGES

    @property
    def syndrome(self):
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
        
    @cache
    def id_below(self, id_: int):
        """Return ID of node below node with ID `id_`.
        
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
    
    def decode(
            self,
            syndrome: set[Node],
            log_history: Literal[False, 'fine', 'coarse'] = False,
            time_only: Literal['all', 'merging', 'unrooting'] = 'merging',
        ):
        """Perform a decoding cycle i.e. a growth round.
        
        Input:
        * `syndrome` the syndrome in the new region discovered by the window raise
        i.e. all defects in `syndrome` have time coordinate as `WINDOW_HEIGHT-1`.
        * `log_history` whether to populate `history` attribute --
        'fine' logs each timestep;
        'coarse', only the final timestep of the growth round.
        * `time_only` whether runtime includes a timestep
        for each drop, each grow, and each merging step ('all');
        each merging step only ('merging');
        or each unrooting step only ('unrooting').

        Output:
        * `t` number of timesteps to complete decoding cycle.
        Equals the increase in `len(self.history)` if
        `log_history` is 'fine' and `time_only` is `'all'`.
        """
        
        self._stage = Stage.DROP
        self.drop(syndrome)
        if log_history == 'fine': self.append_history()
        return self._SCHEDULE.finish_decode(
            log_history=log_history,
            time_only=time_only,
        )
    
    def drop(self, syndrome: set[Node]):
        """Make all nodes perform a `drop` i.e. raise window by a layer."""
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
        """Load `syndrome` onto decoder.
        
        I.e. top sheet inherits the new measurement round results.
        Note do not set `active` as `defect`
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
        """Make all nodes perform `merging` until none are busy.
        
        Emergent effect: merge touching clusters, push defects to roots.
        
        Input:
        * `whole` whether to perform `MERGING_WHOLE` or `MERGING_HALF` stage.
        * `log_history, time_only` as in `decode` inputs.
        
        Output: `t` number of timesteps to complete growth round.
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
        
        Input:
        * `show_global` whether to prepend the global label to the top-left node label.
        * `show_2_1_schedule_variables` whether to show
        node variables specific to the 2:1 cluster growth schedule.
        
        Output:
        `result` a dictionary where each
        key a node index as a tuple;
        value, the label for the node at that index.
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
        inactive_detector_linewidths = [width if v in self.syndrome else linewidths for v in inactive_detector_list]
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
        active_detector_linewidths = [width if v in self.syndrome else linewidths for v in active_detector_list]
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
        boundary_linewidths = [width if v in self.syndrome else linewidths for v in boundary_nodelist] \
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
        TODO: this is a temporary fix of `_DigraphMaker.pointer_digraph`.
        i.e. `self.__init__` used to have the line
        `self._DIGRAPH_MAKER = DigraphMaker(self.NODES, self.growth)`.
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


class NodeEdgeMixin(abc.ABC):
    """Mixin class for `_Node` and `_Edge`.
    
    Instance attributes:
    * `SNOWFLAKE` the decoder the node or edge belongs to.
    """

    _SNOWFLAKE: Snowflake

    @property
    def SNOWFLAKE(self): return self._SNOWFLAKE


class _Node(NodeEdgeMixin):
    """Node for Snowflake.

    Extends `NodeEdgeMixin`.
    
    Additional instance attributes:
    * `INDEX` index of node.
    * `ID` the unique ID of node.
    * `FRIENDSHIP` the type of connection the node has for communicating.
    * `NEIGHBORS` a dictionary where each
    key a pointer string;
    value, a tuple of the form (edge, index of edge which is neighbor).
    * `_IS_BOUNDARY` whether node is a boundary node.
    * `_MERGER` the provider for the `merging` method.
    * `UNROOTER` the type of unrooting process used.
    * `active, whole, cid, defect, pointer, grown, unrooted, busy`
    explained in [arXiv:2406.01701, C.1].
    * The variables in the above bulletpoint, save `busy`,
    each have `next_` versions for next timestep.
    * `access` refers to neighbors along fully grown edges.
    In the form of a dictionary of where each
    key a pointer string;
    value, the `_Node` object in the direction of that pointer.
    """

    def __init__(
            self,
            snowflake: Snowflake,
            index: Node,
            merger: Literal['fast', 'slow'] = 'fast',
            unrooter: Literal['full', 'simple'] = 'full',
        ) -> None:
        """Input:
        * `snowflake` the decoder the node belongs to.
        * `index` index of node.
        * `merger` decides whether to flood before syncing (fast) or vice versa (slow) in a merging step.
        * `unrooter` the type of unrooting process to use.
        """
        self._SNOWFLAKE = snowflake
        self._INDEX = index
        self._ID = snowflake.index_to_id(index)
        self._FRIENDSHIP = NothingFriendship(self) if index[snowflake.CODE.TIME_AXIS] == 0 \
            else TopSheetFriendship(self) if index[snowflake.CODE.TIME_AXIS] == snowflake.CODE.SCHEME.WINDOW_HEIGHT-1 \
            else NodeFriendship(self)
        if isinstance(snowflake.CODE, Repetition):
            j, t = index
            provisional_neighbors = (
                ('W', ((j-1, t), index), 0),
                ('E', (index, (j+1, t)), 1),
                ('D', ((j, t-1), index), 0),
                ('U', (index, (j, t+1)), 1),
            )
        else:  # Surface
            i, j, t = index
            provisional_neighbors = (
                ('NWD', ((i-1, j-1, t-1), index), 0),
                ('N', ((i-1, j, t), index), 0),
                ('NU', ((i-1, j, t+1), index), 0),
                ('WD', ((i, j-1, t-1), index), 0),
                ('W', ((i, j-1, t), index), 0),
                ('D', ((i, j, t-1), index), 0),
                ('U', (index, (i, j, t+1)), 1),
                ('E', (index, (i, j+1, t)), 1),
                ('EU', (index, (i, j+1, t+1)), 1),
                ('SD', (index, (i+1, j, t-1)), 1),
                ('S', (index, (i+1, j, t)), 1),
                ('SEU', (index, (i+1, j+1, t+1)), 1),
            )
        self._NEIGHBORS: dict[direction, tuple[Edge, int]] = { # type: ignore
            pointer: (e, e_index)
            for pointer, e, e_index in provisional_neighbors
            if e in snowflake.EDGES
        }
        self._IS_BOUNDARY = snowflake.CODE.is_boundary(index)
        self._MERGER = _FastMerger(self) if merger == 'fast' else _SlowMerger(self)
        self._UNROOTER = _FullUnrooter(self) if unrooter == 'full' else _SimpleUnrooter(self)
        self.reset()

    def __repr__(self) -> str:
        return f'decoders.snowflake._Node(snowflake={self._SNOWFLAKE}, index={self.INDEX})'

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
        
        Set `[attribute]` := `next_[attribute]`.
        For nodes in top sheet of viewing window, this equals `reset`.
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
        """Call `grow` if active, then find broken pointers.
        
        Used only in 1:1 schedule.
        """
        if self.active:
            self._grow()
        self.FRIENDSHIP.find_broken_pointers()

    def grow_whole(self):
        """Call `grow` if active and whole, then find broken pointers."""
        if self.active and self.whole:
            self._grow()
            self.grown = True
            self.next_grown = True
        self.FRIENDSHIP.find_broken_pointers()

    def grow_half(self):
        """Call `grow` if active and half and not yet grown in current decoding cycle."""
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
        """Update `active` depending on, and push defect to, pointee."""
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
        """Update `pointer, cid, unrooted, grown` depending on access."""
        if whole:
            self.UNROOTER.flooding_whole()
        else:
            self.UNROOTER.flooding_half()

    def update_after_merging(self):
        """`next_{cid, defect, active, unrooted, grown}` -> `{cid, defect, active, unrooted, grown}`."""
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
    `defect, active, cid, pointer` information.
    
    Instance attributes (1 constant):
    * `NODE` the node which has this friendship.
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
    
    Extends `Friendship`.

    Instance attributes (1 constant):
    * `DROPEE` the node immediately below.
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
        
    Extends `NodeFriendship`.
    After a drop, these nodes must reset `next_active` and `next_defect`.
    """

    def drop(self):
        super().drop()
        self.NODE.next_defect = False
        self.NODE.next_active = False


class NothingFriendship(Friendship):
    """Friendship with nothing immediately below.
    
    Extends `Friendship`.
    Only nodes whose time index is 0 have this friendship.
    """

    def find_broken_pointers(self):
        if 'D' in self.NODE.pointer:
            self.NODE.UNROOTER.start()


class _Merger(abc.ABC):
    """The class providing `_Node.merging`."""

    def __init__(self, node: _Node) -> None:
        self._NODE = node

    @abc.abstractmethod
    def merging(self):
        """See `_Node.merging`."""


class _SlowMerger(_Merger):
    """Provider for `_Node.merging` which syncs before it floods.
    
    Extends `_Merger`.
    """

    def merging(self, whole: bool):
        self._NODE.syncing()
        self._NODE.flooding(whole)


class _FastMerger(_Merger):
    """Provider for `_Node.merging` which floods before it syncs.
    
    Extends `_Merger`.
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
        
        Output:
        `t` number of timesteps to complete decoding cycle.
        """

    @abc.abstractmethod
    def grow(self):
        """Make all nodes perform a `grow`."""


class _OneOne(_Schedule):
    """The 1:1 cluster growth schedule.
    
    Extends `_Schedule`.
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
    
    Extends `_Schedule`.
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
    * `_NODE` the node which has this unrooter.
    """

    def __init__(self, node: _Node):
        """Input: `node` the node the unrooter belongs to."""
        self._NODE = node

    @abc.abstractmethod
    def start(self):
        """Start unrooting the node."""

    @abc.abstractmethod
    def flooding_whole(self):
        """Update `pointer, cid, unrooted, grown` depending on access."""

    def flooding_half(self):
        """Update `pointer, cid` depending on access."""
        for pointer, neighbor in self._NODE.access.items():
            self._compare_cid(pointer, neighbor)

    def _compare_cid(self, pointer: direction, neighbor: _Node):
        """Update `next_cid` depending on `neighbor.cid`."""
        if neighbor.cid < self._NODE.next_cid:
            self._NODE.busy = True
            self._NODE.pointer = pointer
            self._NODE.next_cid = neighbor.cid

    def _check_grown(self, neighbor: _Node):
        """Update `next_grown` depending on `neighbor.grown`."""
        if neighbor.grown and not self._NODE.next_grown:
            self._NODE.busy = True
            self._NODE.next_grown = True


class _FullUnrooter(_Unrooter):
    """Full unrooting process where each node in the amputated cluster resets its CID and pointer.
    
    This is so that the pointer tree structure can be rebuilt from scratch,
    via further merging timesteps.
    
    Extends `_Unrooter`.
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
    
    Extends `_Unrooter`.

    Additional instance constants:
    * `_CLOSEST_BOUNDARY_DIRECTION` the direction toward the closest boundary.
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

    Extends `NodeEdgeMixin`.
    
    Additional instance attributes:
    * `INDEX` index of edge.
    * `CONTACT` the type of connection the edge has for communicating.
    Analogous to `FRIENDSHIP` for nodes.
    * `growth` its growth value.
    * `correction` whether it is in the correction.
    """

    def __init__(self, snowflake: Snowflake, index: Edge) -> None:
        """Input:
        * `snowflake` the decoder the edge belongs to.
        * `INDEX` index of edge.
        """
        self._SNOWFLAKE = snowflake
        self._INDEX = index
        lowness = sum(v[snowflake.CODE.TIME_AXIS] == 0 for v in index)
        self._CONTACT = EdgeContact(self) if lowness == 0 else FloorContact(self)
        self.reset()

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
        
        Set `[attribute]` := `next_[attribute]`.
        For edges in `self.CODE.SCHEME.BUFFER_EDGES`,
        this equals `reset`.
        """
        self.growth = self.next_growth
        self.correction = self.next_correction


class _Contact(abc.ABC):
    """The type of connection for communicating
    `growth, correction` information.
    
    Instance attributes (1 constant):
    * `EDGE` the edge which has this contact.
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
    
    Extends `_Contact`.

    Instance attributes (1 constant):
    * `DROPEE` the edge immediately below.
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
    
    Extends `_Contact`.
    Only edges in the bottom sheet of the viewing window
    have this contact.
    """

    def drop(self):
        frugal: Frugal = self.EDGE.SNOWFLAKE.CODE.SCHEME # type: ignore
        if self.EDGE.correction:
            frugal.pairs.load(self.EDGE.INDEX)