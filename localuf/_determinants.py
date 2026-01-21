from localuf.type_aliases import Node


class Determinant:
    """Base class for determining whether a node is a boundary.
    
    Instance attributes:
    * ``D`` code distance.
    * ``LONG_AXIS`` that whose index runs from -1 to d-1 inclusive.
    """

    def __init__(self, d: int, long_axis: int) -> None:
        self._D = d
        self._LONG_AXIS = long_axis

    @property
    def D(self): return self._D

    @property
    def LONG_AXIS(self): return self._LONG_AXIS

    def is_boundary(self, v: Node):
        """See ``_base_classes.Code.is_boundary``."""
        # `node` either (i, j) or (j, t) or (i, j, t).
        return v[self.LONG_AXIS] in {-1, self.D-1}


class SpaceDeterminant(Determinant):
    """Determines only space boundaries.
    
    Extends ``Determinant``.
    """


class SpaceTimeDeterminant(Determinant):
    """Determines space and time boundaries.
    
    Extends ``Determinant``.
    
    Additional instance attributes:
    * ``TIME_AXIS`` that which represents time.
    * ``WINDOW_HEIGHT`` total height of sliding window.
    
    Overriden methods:
    * ``__init__``
    * ``is_boundary``
    """

    def __init__(
            self,
            d: int,
            long_axis: int,
            time_axis: int,
            window_height: int
    ):
        super().__init__(d, long_axis)
        self._TIME_AXIS = time_axis
        self._WINDOW_HEIGHT = window_height

    @property
    def TIME_AXIS(self): return self._TIME_AXIS
    
    @property
    def WINDOW_HEIGHT(self): return self._WINDOW_HEIGHT

    def is_boundary(self, v: Node):
        is_space_boundary = super().is_boundary(v)
        is_time_boundary = v[self.TIME_AXIS]==self.WINDOW_HEIGHT
        return is_space_boundary or is_time_boundary