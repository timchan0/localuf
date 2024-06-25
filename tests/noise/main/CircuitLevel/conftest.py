from typing import Iterable

import pytest

from localuf.noise import CircuitLevel
from localuf.type_aliases import EdgeType, MultiplicityVector

@pytest.fixture
def diagonals():
    return (
        'SD',
        'EU west corners',
        'EU east corners',
        'EU edge',
        'EU centre',
        'SEU',
    )


@pytest.fixture
def assert_these_multiplicities_unchanged():
    def f(
            m: dict[EdgeType, MultiplicityVector],
            edge_types: Iterable[EdgeType],
    ):
        for et in edge_types:
            assert tuple(m[et]) \
                == CircuitLevel._DEFAULT_MULTIPLICITIES[et]
    return f


@pytest.fixture
def ftto():
    return (4, 3, 2, 1)