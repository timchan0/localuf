"""Classes for local Union--Find decoder."""

import abc
from functools import cached_property
from typing import Literal

import networkx as nx

from localuf.decoders.luf.constants import Stage, BUSY_SIGNAL_SYMBOLS, ACTIVE_SIGNAL_SYMBOLS
from localuf.type_aliases import Edge, Node
from localuf.codes import Surface
from localuf import noise
from localuf import constants
from localuf.constants import Growth
from localuf.decoders.policies import DigraphMaker, DecodeDrawer
from localuf.decoders._base_uf import direction, BaseUF


class LUF(BaseUF):
    """The graph used by local UF decoder.
    
    Extends ``_BaseUF``.
    
    Class attributes:
    * ``DEFAULT_X_OFFSET`` default ``x_offset``.
        Slightly larger than ``constants.DEFAULT_X_OFFSET``
    as the drawer for this class shows more information at each node.
    
    Instance attributes (1--5 constant):
    * ``CONTROLLER`` a global controller object.
    * ``NODES`` a Nodes object.
    * ``VISIBLE`` whether controller has direct connection w/ each node or only node w/ ID 0
        i.e. strictly local iff ``VISIBLE`` is ``False``.
    Only computed if ``syndrome`` not yet an attribute.
    * ``_DIGRAPH_MAKER`` maker of NetworkX digraph.
    * ``_DECODE_DRAWER`` provides ``draw_decode``.
    * ``_FIG_WIDTH`` figure width used by drawer.
    * ``correction`` only exists after calling ``decode()``.
    * ``_pointer_digraph`` a NetworkX digraph representing the fully grown edges used by pointers,
        the set of its edges as directed edges,
    the set of its edges as undirected edges.
    
    In drawings of this graph:
    * active nodes are square-shaped
    * inactive nodes are circular
    * CID is shown as a label
    * nodes with anyons are outlined in black
    * pointers are shown by arrows on edges
    * edges so far added to the correction are in red
    * the top-left node also shows the controller stage (PS stands for presyncing etc.)
    """

    DEFAULT_X_OFFSET = 0.3

    def __init__(self, code: Surface, visible=True, _optimal=True):
        """
        :param code: the code to be decoded.
        :param visible: whether controller has direct connection w/ each node or only node w/ ID 0
            i.e. strictly local iff ``visible`` is ``False``.
        :param _optimal: whether management of ``self.NODES.countdown`` is optimal.
            Relevant only when ``visible`` is ``False``.
        """
        self.correction = set()
        super().__init__(code)
        self._CONTROLLER = Controller(self)
        self._NODES = MacarNodes(self) if visible else ActisNodes(self, _optimal)
        self._VISIBLE = visible
        self._DIGRAPH_MAKER = DigraphMaker(self.NODES.dc, self.growth)
        self._DECODE_DRAWER = DecodeDrawer(self._FIG_WIDTH, fig_height=self._FIG_HEIGHT)

    @property
    def CONTROLLER(self): return self._CONTROLLER

    @property
    def NODES(self): return self._NODES

    @property
    def VISIBLE(self): return self._VISIBLE
    
    def decode(
            self,
            syndrome: set[Node],
            draw=False,
            log_history=False,
            **kwargs_for_draw_decode,
    ):
        """Additional inputs over those of ``Decoder.decode()``:
        * ``log_history`` whether to populate ``history`` attribute.
        * ``kwargs_for_draw_decode`` passed to ``self.draw_decode()``.
        
        
        :returns:
        * ``tSV`` # timesteps to validate syndrome.
        * ``tBP`` # timesteps to burn and peel.
        
        If ``log_history`` is ``True`` then ``tSV + tBP = len(self.history)``.
        """
        log_history |= draw
        tSV = self.validate(
            syndrome,
            draw=False,
            log_history=log_history,
        )
        tBP = self.peel(log_history)
        if draw:
            self.draw_decode(**kwargs_for_draw_decode)
        return tSV, tBP

    def validate(
            self,
            syndrome: set[Node],
            draw=False,
            log_history=False,
            **kwargs_for_draw_decode,
    ):
        """Validate syndrome.
        
        
        :param syndrome: the set of defects.
        :param draw: whether to draw.
        :param log_history: whether to populate ``history`` attribute.
        :param kwargs_for_draw_decode: passed to ``self.draw_decode()``.
        
        
        :returns: ``tSV`` # timesteps to validate syndrome. Equals ``len(self.history)`` if ``log_history`` is ``True``.
        """
        self.NODES.load(syndrome)
        tSV = 0
        log_history |= draw
        if log_history:
            self.init_history()
            while self.CONTROLLER.stage is not Stage.BURNING:
                self._advance()
                self.append_history()
                tSV += 1
        else:
            while self.CONTROLLER.stage is not Stage.BURNING:
                self._advance()
                tSV += 1
        if draw:
            self.draw_decode(**kwargs_for_draw_decode)
        return tSV

    def peel(self, log_history: bool):
        """Burn & peel after syndrome validation.
        
        
        ``log_history`` whether to populate ``history`` attribute.
        
        
        :returns: ``tBP`` # timesteps to burn and peel.
        """
        tBP = 0
        if log_history:
            while self.CONTROLLER.stage is not Stage.DONE:
                self._advance()
                self.append_history()
                tBP += 1
        else:
            while self.CONTROLLER.stage is not Stage.DONE:
                self._advance()
                tBP += 1
        return tBP

    @cached_property
    def syndrome(self):
        """Syndrome computed from ``self.NODES``."""
        return self.NODES.syndrome

    def reset(self):
        super().reset()
        self.CONTROLLER.reset()
        self.NODES.reset()
        try: del self.syndrome
        except AttributeError: pass
        try: del self.history
        except AttributeError: pass

    def _advance(self):
        """Advance 1 timestep."""
        self.NODES.advance()  # physical
        self.NODES.update_unphysicals()  # nonphysical
        growth_changed = self.CONTROLLER.advance()  # physical
        if growth_changed:  # nonphsyical
            self.NODES.update_access()

    # DRAWERS

    def draw_growth(
        self,
        highlighted_edges: set[Edge] | None = None,
        highlighted_edge_color='k',
        unhighlighted_edge_color=constants.DARK_GRAY,
        x_offset=DEFAULT_X_OFFSET,
        with_labels=True,
        labels: dict[Node, str] | None = None,
        show_global=True,
        node_size=constants.DEFAULT,
        linewidths=constants.WIDE_MEDIUM,
        anyon_color='k',
        active_shape='s',
        width=constants.WIDE_MEDIUM,
        arrows: bool | None = None,
        boundary_color=constants.BLUE,
        defect_color=constants.RED,
        nondefect_color=constants.GREEN,
        show_boundary_defects=True,
        defect_label_color='k',
        **kwargs_for_networkx_draw,
    ):
        g = self.CODE.GRAPH
        dig, dig_diedges, dig_edges = self._pointer_digraph
        pos = self.CODE.get_pos(x_offset)
        if highlighted_edges is None:
            highlighted_edges = self.correction
        if labels is None:
            labels = self.NODES.labels(show_global)
        nodes_w_anyons = {v for v, node in self.NODES.dc.items() if node.anyon}

        # DRAW INACTIVE NODES AND EDGES UNUSED BY POINTERS
        
        # node-related kwargs
        inactive_nodelist = [v for v, node in self.NODES.dc.items() if not node.active]
        inactive_node_color, inactive_edgecolors = \
            self._get_node_color_and_edgecolors(
                outlined_nodes=nodes_w_anyons,
                nodelist=inactive_nodelist,
                outline_color=anyon_color,
                boundary_color=boundary_color,
                defect_color=defect_color,
                nondefect_color=nondefect_color,
                show_boundary_defects=show_boundary_defects,
            )
        
        # edge-related kwargs
        edgelist = [
            e for e in self.CODE.EDGES
            if self.growth[e] in {Growth.HALF, Growth.FULL}
            and e not in dig_edges
        ] + list(highlighted_edges)
        edge_color, style = self._get_edge_color_and_style(
            edgelist,
            highlighted_edges,
            highlighted_edge_color,
            unhighlighted_edge_color,
        )
        nx.draw(
            g,
            pos=pos,
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
        active_nodelist = [v for v, node in self.NODES.dc.items() if node.active]
        active_node_color, active_edgecolors = \
            self._get_node_color_and_edgecolors(
                outlined_nodes=nodes_w_anyons,
                nodelist=active_nodelist,
                outline_color=anyon_color,
                boundary_color=boundary_color,
                defect_color=defect_color,
                nondefect_color=nondefect_color,
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

        if with_labels:
            # DRAW LABELS (defect labels `defect_label_color`; nondefect, black)
            defect_labels = {v: label for v, label in labels.items() if v in self.syndrome}
            nondefect_labels = {v: label for v, label in labels.items() if v not in self.syndrome}
            nx.draw_networkx_labels(g, pos, labels=defect_labels, font_color=defect_label_color)
            nx.draw_networkx_labels(g, pos, labels=nondefect_labels)


    @property
    def _pointer_digraph(self): return self._DIGRAPH_MAKER.pointer_digraph
    
    def draw_decode(self, **kwargs_for_networkx_draw):
        self._DECODE_DRAWER.draw(self.history, **kwargs_for_networkx_draw)

    @property
    def _FIG_WIDTH(self):
        return max(1, self.CODE.D * self._FIG_FACTOR)

    @property
    def _FIG_HEIGHT(self):
        return max(1, (self.CODE.D-1) * self._FIG_FACTOR)
    
    @property
    def _FIG_FACTOR(self):
        return 3*(self.CODE.DIMENSION-1) / (1+self.VISIBLE) / 2


class Controller:
    """Global controller for LUF.
    
    Instance attributes (1 constant):
    * ``LUF`` the LUF object which has this controller.
    * ``stage`` the global stage.
    """

    def __init__(self, luf: LUF) -> None:
        """Input: ``luf`` the LUF object which has this controller."""
        self._LUF = luf
        self.reset()

    @property
    def LUF(self): return self._LUF

    def reset(self):
        """Factory reset."""
        self.stage = Stage.GROWING

    def advance(self):
        """Advance 1 timestep.
        
        Output: ``growth_changed`` which is ``True``
        if any growths have changed
        (happens after growth and burning stages, and peeling steps);
        else, ``False``.
        
        Side effect: Update ``stage``.
        """
        growth_changed = self.stage is Stage.PEELING
        if not self.LUF.NODES.busy:  # stage complete
            if self.stage < Stage.SV_STAGE_COUNT:  # in syndrome validation
                if self.stage is Stage.SYNCING and self.LUF.NODES.valid:
                    # syndrome validation done
                    self.stage = Stage.BURNING
                else:
                    if self.stage is Stage.GROWING:
                        growth_changed = True
                    self.stage += Stage.INCREMENT
                    self.stage %= Stage.SV_STAGE_COUNT
            else:
                growth_changed = True
                self.stage += Stage.INCREMENT
        return growth_changed


class Nodes(abc.ABC):
    """Collection of all LUF nodes.
    
    Instance attributes (1--2 constant):
    * ``LUF`` the LUF object which has these nodes.
    * ``dc`` a dictionary where each
        key an index tuple;
    value, the node object at that index.
    * ``syndrome`` the set of defects.
    * ``busy`` is ``False`` iff we are sure all nodes are ready for next stage.
    * ``valid`` is ``True`` iff we are sure no node is active after presyncing stage
        i.e. ``syndrome`` has been validated.
    * ``_global_label`` the label of global information used by drawer.
    """
    
    def __init__(self, luf: LUF) -> None:
        """Input: ``luf`` the LUF object which has these nodes."""
        self._LUF = luf
        self.dc: dict[Node, _Node]

    @property
    def LUF(self): return self._LUF

    def load(self, syndrome: set[Node]):
        """Load syndrome onto nodes collection."""
        for v in syndrome:
            self.dc[v].make_defect()

    @property
    def syndrome(self):
        """Compute syndrome from nodes in ``dc``."""
        return {v for v, node in self.dc.items() if node.defect}

    def reset(self):
        """Factory reset."""
        for node in self.dc.values():
            node.reset()

    def advance(self):
        """Advance 1 timestep."""
        for node in self.dc.values():
            node.advance()

    def update_unphysicals(self):
        """After each merging or syncing step, set ``attribute`` := ``next_attribute`` for all nodes."""
        if self.LUF.CONTROLLER.stage is Stage.MERGING:
            for node in self.dc.values():
                node.update_after_merge_step()
        elif self.LUF.CONTROLLER.stage is Stage.SYNCING:
            for node in self.dc.values():
                node.update_after_sync_step()

    def update_access(self):
        """Update access for all nodes.
        
        Call after growth has changed.
        """
        for node in self.dc.values():
            node.update_access()
    
    def labels(self, show_global=True):
        """Return the labels dictionary for the drawer.
        
        
        ``show_global`` whether to prepend the global label to the top-left node label.
        
        
        :returns: ``result`` a dictionary where each key a node index as a tuple; value, the label for the node at that index.
        """
        result = {
            v: node.get_label()
            for v, node in self.dc.items()
        }
        if show_global:
            code = self.LUF.CODE
            top_left = (0, -1, code.D-1) if code.DIMENSION == 3 else (0, -1)
            result[top_left] = self._global_label + result[top_left]
        return result

    @property
    def _global_label(self):
        return str(self.LUF.CONTROLLER.stage)


class MacarNodes(Nodes):
    """Collection of all LUF nodes, where controller directly connects to each node.
    
    Extends ``Nodes``.
    """

    def __init__(self, luf: LUF) -> None:
        super().__init__(luf)
        self._dc = {v: MacarNode(nodes=self, index=v)
                    for v in luf.CODE.NODES}
        
    @property
    def dc(self): return self._dc

    @property
    def busy(self):
        return any(node.busy for node in self.dc.values())

    @property
    def valid(self):
        return not any(node.active for node in self.dc.values())


class ActisNodes(Nodes):
    """Collection of all LUF nodes, where controller only sees node whose ID is 0.
    
    Extends ``Nodes``.
    
    Additional instance attributes (1--2 constant):
    * ``SPAN`` distance from controller to furthest node (boundary node w/ highest ID).
    * ``WAITER`` object which decides how long controller must wait until
        it is sure no more info is being relayed (towards it).
    * ``countdown`` tracker used by ``WAITER``.
    * ``busy_signal`` whether collection receives a busy signal in current timestep.
    * ``active_signal`` whether collection has received an active signal during the current syncing stage.
    * ``next_{busy, active}_signal`` the provisional ``{busy, active}_signal`` for the next timestep.
    * ``{busy, active}_signal`` have string representations via ``[attribute]_signal_symbol`` property.
    
    Notes:
    This decoder starts in the GROWING stage.
    Alternatively, could start in the SYNCING stage and in ``load()``,
    call ``self.update_valid()`` instead of ``self.valid = False``.
    This would be beneficial if noise level low enough.
    """
    
    def __init__(self, luf: LUF, optimal=True):
        super().__init__(luf)
        self._dc = {v: ActisNode(nodes=self, index=v)
                    for v in luf.CODE.NODES}

        d = luf.CODE.D
        if isinstance(luf.CODE.NOISE, noise.CodeCapacity):
            self._SPAN = 2*d  # 1 + (d-1) + d
        elif isinstance(luf.CODE.NOISE, noise.Phenomenological):
            self._SPAN = 3*d - 1  # 1 + (d-1) + d + (d-1)
        else: # CircuitLevel
            self._SPAN = d + 1  # 1 + d
        self._WAITER = OptimalWaiter(self) if optimal else UnoptimalWaiter(self)

        self.reset()

    @property
    def dc(self): return self._dc

    @property
    def SPAN(self): return self._SPAN
    
    @property
    def WAITER(self): return self._WAITER
    
    @property
    def busy_signal_symbol(self):
        return BUSY_SIGNAL_SYMBOLS[self.busy_signal]

    @property
    def active_signal_symbol(self):
        return ACTIVE_SIGNAL_SYMBOLS[self.active_signal]

    def reset(self):
        self.busy = False
        self.valid = True
        self.countdown = 0
        self.busy_signal = False
        self.next_busy_signal = False
        self.active_signal = False
        self.next_active_signal = False
        super().reset()

    def load(self, syndrome):
        super().load(syndrome)
        self.valid = False

    def update_valid(self):
        """Update ``valid`` attribute. Call after finish SYNCING stage."""
        self.valid = not self.active_signal
        self.active_signal = False
        self.next_active_signal = False
    
    def advance(self):
        super().advance()
        self.WAITER.advance()

    def update_unphysicals(self):
        super().update_unphysicals()
        for node in self.dc.values():
            node.update_unphysicals_for_actis()
        self._update_unphysicals_for_actis()

    def _update_unphysicals_for_actis(self):
        """``next_{busy, active}_signal`` -> ``{busy, active}_signal``."""

        self.busy_signal = self.next_busy_signal
        self.next_busy_signal = False

        self.active_signal = self.next_active_signal

    @property
    def _global_label(self):
        label = super()._global_label
        label += f'{self.countdown}{self.busy_signal_symbol}{self.active_signal_symbol}\n'
        return label


class Waiter(abc.ABC):
    """Base class for ``WAITER`` attribute of ``ActisNodes`` instance.
    
    Instance attributes:
    * ``NODES`` the ``ActisNodes`` object the waiter belongs to.
    * ``RECEIVING_START`` the value the waiter sets countdown to
        if it receives a busy signal during the receiving window.
    """

    def __init__(self, nodes: ActisNodes) -> None:
        """Input: ``nodes`` the ``ActisNodes`` object the waiter belongs to."""
        self._NODES = nodes
        self.RECEIVING_START: int

    @property
    def NODES(self): return self._NODES

    @property
    @abc.abstractmethod
    def received_busy_signal(self) -> bool:
        """Whether to set ``countdown`` to ``RECEIVING_START``."""

    def advance(self):
        """Advance 1 timestep.
        
        Countdown start for controller stage:
        * GROWING or PRESYNCING is ``span`` as it takes ``span`` timesteps
            (``span-1`` .. -1 .. 0)
        for information to go from controller to furthest node.
        * MERGING is ``span+1`` as it takes ``span`` timesteps
            (``span`` .. -1 .. 1)
        for information to go from furthest node to controller
        [and furthest node (which is a boundary) can be busy].
        * SYNCING is ``span`` as it takes ``span-1`` timesteps
            (``span-1`` .. -1 .. 1)
        for information to go from furthest detector to controller
        (furthest boundary node never busy nor active).
        """
        self.NODES.busy = True
        if self.NODES.countdown == 0:  # stage complete
            self.NODES.busy = False
            self.NODES.countdown = self.NODES.SPAN
            match self.NODES.LUF.CONTROLLER.stage:
                case Stage.GROWING:
                    self.NODES.countdown += 1
                case Stage.SYNCING:
                    self.NODES.update_valid()
        # only ever happens when controller stage in {MERGING, SYNCING}
        elif self.received_busy_signal:
            self.NODES.countdown = self.RECEIVING_START
        else:
            self.NODES.countdown -= 1


class OptimalWaiter(Waiter):
    """Optimal waiter for ``ActisNodes``.
    
    Extends ``Waiter``.
    
    Additional class constants:
    * ``RECEIVING_WINDOW`` the countdown values
        during which the waiter considers busy signals.
    """

    RECEIVING_WINDOW = {1, 2}
    RECEIVING_START = 2

    @property
    def received_busy_signal(self):
        return self.NODES.busy_signal and self.NODES.countdown in self.RECEIVING_WINDOW


class UnoptimalWaiter(Waiter):
    """Unoptimal waiter for ``ActisNodes``.
    
    Extends ``Waiter``.
    
    Receiving window is the entire countdown i.e. always.
    """
    
    def __init__(self, nodes: ActisNodes):
        super().__init__(nodes)
        self.RECEIVING_START = nodes.SPAN + 1

    @property
    def received_busy_signal(self):
        return self.NODES.busy_signal


class _Node(abc.ABC):
    """Node for LUF.
    
    Class attributes:
    * ``OPPOSITE`` a dictionary where each
        key a possible ``pointer`` value;
    value, the value in the opposite direction.
    
    Instance attributes (1--4 constant,
    5 constant within a decoding cycle,
    6--14 variable within a decoding cycle):
    * ``NODES`` the Nodes object the node belongs to.
    * ``INDEX`` the index of the node.
    * ``ID`` the unique ID of the node.
        ``ID`` bijects ``INDEX``.
    * ``NEIGHBORS`` a dictionary where each
        key a pointer string;
    value, a tuple of the form (edge, index of edge which is neighbor).
    * ``_IS_BOUNDARY`` whether node is a boundary node.
    * ``defect`` whether the node is a defect.
    * ``active`` whether the cluster the node is in is active.
    * ``cid`` the ID of the cluster the node belongs to.
    * ``next_cid`` the provisional ``cid`` for the next timestep.
        It depends on the ``cid`` of the nodes around it.
    At the end of the timestep, each node sets ``cid = next_cid``.
    This ensures causality i.e. info travels at 1 edge per timestep.
    * ``anyon`` whether the node has an anyon.
    * ``next_anyon`` the provisional ``anyon`` for the next timestep.
    * ``pointer`` the direction the node sends anyons along.
    * ``busy`` whether the node changes any of its attributes in the current timestep.
    * ``access`` refers to neighbors along fully grown edges.
        In the form of a dictionary of where each
    key a pointer string;
    value, the ``_Node`` object in the direction of that pointer.
    """

    OPPOSITE: dict[direction, direction] = {
        'N': 'S',
        'W': 'E',
        'E': 'W',
        'S': 'N',
        'D': 'U',
        'U': 'D',
        'SD': 'NU',
        'NU': 'SD',
        'EU': 'WD',
        'WD': 'EU',
        'SEU': 'NWD',
        'NWD': 'SEU',
    }

    def __init__(self, nodes: Nodes, index: Node) -> None:
        """
        :param nodes: the Nodes object the node belongs to.
        :param index: the index of the node.
        """
        self._NODES = nodes
        self._INDEX = index
        self._ID = nodes.LUF.CODE.index_to_id(index)
        provisional_neighbors: tuple[tuple[direction, Edge, Literal[0, 1]], ...]
        """During merging,
        node will look at neighbors in order of this list,
        so order matters!
        """
        if isinstance(nodes.LUF.CODE.NOISE, noise.CodeCapacity):
            i, j = index
            provisional_neighbors = (
                ('N', ((i-1, j), index), 0),
                ('W', ((i, j-1), index), 0),
                ('E', (index, (i, j+1)), 1),
                ('S', (index, (i+1, j)), 1),
            )
        else:
            i, j, t = index
            n = ('N', ((i-1, j, t), index), 0)
            w = ('W', ((i, j-1, t), index), 0)
            d = ('D', ((i, j, t-1), index), 0)
            u = ('U', (index, (i, j, t+1)), 1)
            e = ('E', (index, (i, j+1, t)), 1)
            s = ('S', (index, (i+1, j, t)), 1)
            if isinstance(nodes.LUF.CODE.NOISE, noise.Phenomenological):
                provisional_neighbors = (n, w, d, u, e, s)
            else: # 'circuit-level'
                provisional_neighbors = (
                    ('NWD', ((i-1, j-1, t-1), index), 0),
                    n,
                    ('NU', ((i-1, j, t+1), index), 0),
                    ('WD', ((i, j-1, t-1), index), 0),
                    w, d, u, e,
                    ('EU', (index, (i, j+1, t+1)), 1),
                    ('SD', (index, (i+1, j, t-1)), 1),
                    s,
                    ('SEU', (index, (i+1, j+1, t+1)), 1),
                )
        # match case using CODE.INCIDENT_EDGES?
        self._NEIGHBORS: dict[direction, tuple[Edge, Literal[0, 1]]] = {
            pointer: (e, e_index)
            for pointer, e, e_index in provisional_neighbors
            if e in nodes.LUF.CODE.INCIDENT_EDGES[index]
        }
        self._IS_BOUNDARY = nodes.LUF.CODE.is_boundary(index)
        self.reset()

    @property
    def NODES(self): return self._NODES

    @property
    def INDEX(self): return self._INDEX

    @property
    def ID(self): return self._ID

    @property
    def NEIGHBORS(self): return self._NEIGHBORS

    def reset(self):
        """Factory reset."""
        self.defect = False
        self.active = False
        self.cid = self.ID
        self.next_cid = self.ID
        self.anyon = False
        self.next_anyon = False
        self.pointer: direction = 'C'
        self.busy = False
        self.access: dict[direction, _Node] = {}

    def make_defect(self):
        """Make the node a defect."""
        self.defect = True
        self.active = True
        self.anyon = True

    def growing(self):
        """Growth-increment all active incident edges."""
        if self.active:
            g = self.NODES.LUF
            edges_to_grow = {
                e for e in g.CODE.INCIDENT_EDGES[self.INDEX]
                if g.growth[e] in g._ACTIVE_GROWTH_VALUES
            }
            for e in edges_to_grow:
                g.growth[e] += Growth.INCREMENT

    def update_access(self):
        """Update access.
        
        Call after growth has changed.
        """
        self.access = {
            pointer: self.NODES.dc[e[index]]
            for pointer, (e, index) in self.NEIGHBORS.items()
            if self.NODES.LUF.growth[e] is Growth.FULL
        }

    def merging(self):
        """Relay anyon, and update CID and pointer depending on access."""
        self.busy = False
        if all((
            not self._IS_BOUNDARY,
            self.pointer != 'C',
            self.anyon,
        )):  # relay along pointer
            self.busy = True
            self.access[self.pointer].next_anyon ^= True
            self.anyon = False
        for pointer, neighbor in self.access.items():
            if neighbor.cid < self.next_cid:
                self.busy = True
                self.pointer = pointer
                self.next_cid = neighbor.cid

    def presyncing(self):
        """Initialise for syncing stage i.e. update after merge stage.
        
        If node the root of an active cluster, set ``active = True``;
        else, set ``active = False``.
        """
        self.active = all((
            not self._IS_BOUNDARY,
            self.pointer == 'C',
            self.anyon,
        ))

    @abc.abstractmethod
    def advance(self):
        """Advance 1 timestep."""

    def update_after_merge_step(self):
        """``next_{cid, anyon}`` -> ``{cid, anyon}``."""
        self.cid = self.next_cid
        # if replace ^= by = then a nonroot boundary node w/ anyon
        # will incorrectly lose its anyon in next timestep
        self.anyon ^= self.next_anyon
        self.next_anyon = False
    
    def syncing(self):
        """Update ``active`` depending on access."""
        self.busy = not self.active and any(
            neighbor.active for neighbor in self.access.values()
        )

    def update_after_sync_step(self):
        """``active`` := ``active`` OR ``busy``."""
        self.active |= self.busy

    def burning(self):
        """Burn edges whose endpoints point not along edge."""
        owned_edges: dict[direction, tuple[Edge, Literal[1]]] = {pointer: (e, index)
                       for pointer, (e, index) in self.NEIGHBORS.items()
                       if index == 1}
        for pointer, (e, index) in owned_edges.items():
            if self.NODES.LUF.growth[e] is Growth.FULL:
                if self.pointer != pointer:
                    if self.NODES.dc[e[index]].pointer != self.OPPOSITE[pointer]:
                        self.NODES.LUF.growth[e] = Growth.UNGROWN

    def peeling(self):
        """Peel self if leaf.
        
        Note the root must not be peeled
        even if it is incident to exactly one fully grown edge.
        """
        self.busy = False
        if self.pointer != 'C' and len(self.access) == 1:
            self.busy = True
            pointer, neighbor = next(iter(self.access.items()))
            e, _ = self.NEIGHBORS[pointer]
            self.NODES.LUF.growth[e] = Growth.UNGROWN
            if self.defect:
                self.defect = False
                self.NODES.LUF.correction.add(e)
                neighbor.defect ^= True
    
    def get_label(self):
        """Return node information for drawer."""
        return str(self.cid)


class MacarNode(_Node):
    """Node for LUF which directly accesses global controller.
    
    Extends ``_Node``.
    """

    def advance(self):
        match self.NODES.LUF.CONTROLLER.stage:
            case Stage.GROWING:
                self.growing()
            case Stage.MERGING:
                self.merging()
            case Stage.PRESYNCING:
                self.presyncing()
            case Stage.SYNCING:
                self.syncing()
            case Stage.BURNING:
                self.burning()
            case Stage.PEELING:
                self.peeling()


class ActisNode(_Node):
    """Node for LUF w/o direct access to blind global controller.
    
    Extends ``_Node``.
    
    Additional instance attributes (1--2 constant):
    * ``SPAN`` the countdown start of the node.
    * ``FRIENDSHIP`` the type of connection the node has for communicating
        ``stage`` and signal information.
    * ``countdown`` tracks how long node must wait until staging is done
        i.e. when all nodes have same stage, when ``stage`` in {GROWING, PRESYNCING}.
    * ``stage`` node stage.
    * ``busy_signal`` if in the current timestep the node either
        is busy
    or receives a busy signal from another node.
    * ``active_signal`` if in the current timestep the node either
        has ``stage`` as ``presyncing`` and becomes active,
    or receives an active signal from another node.
    * ``next_{stage, {busy, active}_signal}`` the provisional ``{stage, {busy, active}_signal}`` for the next timestep.
    * ``stage, {busy, active}_signal`` have string representations via ``[attribute]_symbol`` property.
    """

    def __init__(self, nodes: ActisNodes, index: Node) -> None:

        d = nodes.LUF.CODE.D
        if isinstance(nodes.LUF.CODE.NOISE, noise.CircuitLevel):
            i, j, t = index
            depth = max(i, j+1, t)  # distance to node 0
            self._SPAN = d - depth  # `d` the staging & signalling tree height
        else:  # CodeCapacity or Phenomenological
            self._SPAN = nodes.LUF.CODE.DIMENSION * (d-1) - sum(index)

        super().__init__(nodes, index)
        self._FRIENDSHIP = ControllerFriendship(self) if self.ID == 0 else NodeFriendship(self)

    @property
    def NODES(self):
        self._NODES: ActisNodes
        return self._NODES

    @property
    def SPAN(self): return self._SPAN

    @property
    def FRIENDSHIP(self): return self._FRIENDSHIP

    @property
    def stage_symbol(self):
        return str(self.stage) if self.stage is not self.NODES.LUF.CONTROLLER.stage else ''

    @property
    def busy_signal_symbol(self):
        return BUSY_SIGNAL_SYMBOLS[self.busy_signal]

    @property
    def active_signal_symbol(self):
        return ACTIVE_SIGNAL_SYMBOLS[self.active_signal]

    def reset(self):
        super().reset()
        self.countdown = 0
        self.stage = Stage.GROWING
        self.next_stage = self.stage
        self.busy_signal = False
        self.next_busy_signal = False
        self.active_signal = False
        self.next_active_signal = False

    def advance(self):
        match self.stage:
            case Stage.GROWING:
                self.advance_definite('growing')
            case Stage.MERGING:
                self.advance_indefinite('merging')
            case Stage.PRESYNCING:
                self.advance_definite('presyncing')
            case Stage.SYNCING:
                self.advance_indefinite('syncing')

    def advance_definite(self, stage: str):
        """Advance when ``stage`` in `{'growing', 'presyncing'}."""
        if self.countdown == 0:
            getattr(self, stage)()
            self.next_stage += Stage.INCREMENT
        else:
            self.countdown -= 1

    def advance_indefinite(self, stage: str):
        """Advance when ``stage`` in `{'merging', 'syncing'}."""
        changed = self.FRIENDSHIP.update_stage()
        if changed:
            self.countdown = self.SPAN
        else:
            getattr(self, stage)()
            self.FRIENDSHIP.relay_signals()

    def presyncing(self):
        """Initialise for syncing stage i.e. update after merge stage.
        
        If node the root of an active cluster, set ``active``, ``active_signal`` to ``True``;
        else, set both to ``False``.
        """
        super().presyncing()
        self.next_active_signal = self.active

    def update_unphysicals_for_actis(self):
        """``next_{stage, {busy, active}_signal}`` -> ``stage, {busy, active}_signal``."""
        self.stage = self.next_stage

        self.busy_signal = self.next_busy_signal
        self.next_busy_signal = False

        self.active_signal = self.next_active_signal
        self.next_active_signal = False

    def get_label(self):
        north = super().get_label()
        sw = self.countdown if self.stage in {
            Stage.GROWING,
            Stage.PRESYNCING,
        } else self.busy_signal_symbol
        label = f'{north}\n{self.stage_symbol}{sw} {self.active_signal_symbol}'
        return label


class Friendship(abc.ABC):
    """The type of connection for communicating ``stage, {busy, active}_signal`` information.
    
    Instance attributes (1 constant):
    * ``NODE`` the node which has this friendship.
    """

    def __init__(self, node: ActisNode) -> None:
        """Input: ``node`` node which has this friendship."""
        self._NODE = node

    @property
    def NODE(self): return self._NODE

    @abc.abstractmethod
    def update_stage(self) -> bool:
        """Update stage depending on neighbors and return whether stage changed."""

    def update_stage_helper(self, relayee: Controller | ActisNode):
        """Update stage based on stage of component ``relayee``."""
        changed = False
        # could just check if self.NODE.next_stage == relayee.stage?
        if (
            self.NODE.next_stage is Stage.MERGING and
            (rs:=relayee.stage) is Stage.PRESYNCING
        ) or (
            self.NODE.next_stage is Stage.SYNCING and
            (rs:=relayee.stage) is Stage.GROWING
        ):
            changed = True
            self.NODE.next_stage = rs
        return changed

    @abc.abstractmethod
    def relay_signals(self) -> None:
        """Relay ``{busy, active}_signal`` toward controller."""


class ControllerFriendship(Friendship):
    """Friendship w/ controller.
    
    Extends ``Friendship``.
    Only node whose ID is 0 has this friendship.
    Note this node never busy.
    """

    def update_stage(self):
        changed = self.update_stage_helper(self.NODE.NODES.LUF.CONTROLLER)
        return changed

    def relay_signals(self):
        self.NODE.NODES.next_busy_signal = self.NODE.busy_signal
        self.NODE.NODES.next_active_signal |= self.NODE.active_signal


class NodeFriendship(Friendship):
    """Friendship w/ neighboring nodes.
    
    Extends ``Friendship``.
    
    Additional instance attributes (1 constant):
    * ``RELAYEE`` the neighboring node that ``busy_signal`` is sent to.
        Equals the neighbor in the direction toward node of ID 0.
    """

    def __init__(self, node: ActisNode) -> None:
        if isinstance(node.NODES.LUF.CODE.NOISE, noise.CircuitLevel):
            i, j, t = node.INDEX
            self._RELAYEE = (
                i-1 if i > 0 else i,
                j-1 if j >-1 else j,
                t-1 if t > 0 else t,
            )
        else:  # CodeCapacity or Phenomenological
            match node.INDEX:
                case i, j:
                    if j > -1:
                        self._RELAYEE = i, j-1
                    elif i > 0:
                        self._RELAYEE = i-1, j
                    else:
                        self._RELAYEE = 0, -2
                case i, j, t:
                    if t > 0:
                        self._RELAYEE = i, j, t-1
                    elif j > -1:
                        self._RELAYEE = i, j-1, t
                    elif i > 0:
                        self._RELAYEE = i-1, j, t
                    else:
                        self._RELAYEE = 0, -2, 0
        super().__init__(node)

    @property
    def RELAYEE(self) -> Node: return self._RELAYEE

    def update_stage(self):
        changed = self.update_stage_helper(self.NODE.NODES.dc[self.RELAYEE])
        return changed

    def relay_signals(self):
        self.NODE.busy_signal |= self.NODE.busy
        self.NODE.NODES.dc[self.RELAYEE].next_busy_signal |= self.NODE.busy_signal
        self.NODE.NODES.dc[self.RELAYEE].next_active_signal |= self.NODE.active_signal

