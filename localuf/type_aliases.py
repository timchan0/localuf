"""Module containing type aliases."""

from typing import Literal
from collections.abc import Sequence

import numpy as np
import numpy.typing as npt


Node = tuple[int, ...]
"""Node index."""
Edge = tuple[Node, Node]
"""Edge index."""
Coord = tuple
"""Positional coordinate."""
DecodingScheme = Literal[
    'batch',
    'global batch',
    'forward',
    'frugal',
]
"""Decoding scheme."""
NoiseModel = Literal[
    'code capacity',
    'phenomenological',
    'circuit-level',
]
StreamingNoiseModel = Literal[
    'phenomenological',
    'circuit-level',
]
"""Noise model."""
Parametrization = Literal[
    'standard',
    'balanced',
    'ion trap',
]
"""Affects only circuit-level noise model."""
EdgeType = Literal[
    'S',
    'E westmost',
    'E bulk',
    'E eastmost',
    'U 3',
    'U 4',
    'SD',
    'EU west corners',
    'EU east corners',
    'EU edge',
    'EU centre',
    'SEU',
]
"""Edge type in circuit-level decoding graph for surface code."""
FourFloats = tuple[
    float,
    float,
    float,
    float,
]
"""Tuple of 4 floats."""
FourInts = tuple[
    int,
    int,
    int,
    int,
]
"""Tuple of 4 integers."""
MultiplicityVector = npt.NDArray[np.int_]
"""NumPy array of 4 integers."""
FloatSequence = Sequence[float] | npt.NDArray[np.float64]
IntSequence = Sequence[int] | npt.NDArray[np.int_]