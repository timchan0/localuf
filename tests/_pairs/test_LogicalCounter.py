import pytest
from localuf._pairs import Pairs, LogicalCounter

@pytest.fixture
def repetition_counter():
    """Counter for distance-3 repetition code where node indices are (j, t)."""
    return LogicalCounter(
        d=3,
        commit_height=2,
        long_axis=0,
        time_axis=1,
    )

@pytest.fixture
def surface_counter():
    """Counter for distance-3 surface code where node indices are (i, j, t)."""
    return LogicalCounter(
        d=3,
        commit_height=2,
        long_axis=1,
        time_axis=2,
    )


def test_init():
    d = 5
    commit_height = 10
    long_axis = 1
    time_axis = 2
    logical_counter = LogicalCounter(d, commit_height, long_axis, time_axis)
    assert logical_counter._D == d
    assert logical_counter._COMMIT_HEIGHT == commit_height
    assert logical_counter._LONG_AXIS == long_axis
    assert logical_counter._TIME_AXIS == time_axis


class TestCount:

    def test_no_pairs(self, repetition_counter: LogicalCounter):
        pairs = Pairs()
        error_count, new_pairs = repetition_counter.count(pairs)
        assert error_count == 0
        assert new_pairs._dc == {}

    def test_single_logical_error(self, repetition_counter: LogicalCounter):
        pairs = Pairs()
        pairs.add((-1, 0), (2, 0))
        error_count, new_pairs = repetition_counter.count(pairs)
        assert error_count == 1
        assert new_pairs._dc == {}

    def test_multiple_logical_errors(self, repetition_counter: LogicalCounter):
        pairs = Pairs()
        pairs.add((-1, 0), (2, 0))
        pairs.add((-1, 1), (2, 1))
        error_count, new_pairs = repetition_counter.count(pairs)
        assert error_count == 2
        assert new_pairs._dc == {}

    def test_no_logical_errors(self, repetition_counter: LogicalCounter):
        pairs = Pairs()
        pairs.add((0, 2), (1, 2))
        error_count, new_pairs = repetition_counter.count(pairs)
        assert error_count == 0
        assert new_pairs.as_set == {((0, 0), (1, 0))}

    def test_mixed_pairs(self, repetition_counter: LogicalCounter):
        pairs = Pairs()
        pairs.add((-1, 0), (2, 0))
        pairs.add((-1, 1), (1, 2))
        error_count, new_pairs = repetition_counter.count(pairs)
        assert error_count == 1
        assert new_pairs.as_set == {((-1, -1), (1, 0))}

    @pytest.mark.parametrize("j", (-1, 2))
    def test_ignore_pair_on_same_boundary(self, repetition_counter: LogicalCounter, j: int):
        pairs = Pairs()
        pairs.add((j, 0), (j, 1))
        error_count, new_pairs = repetition_counter.count(pairs)
        assert error_count == 0
        assert new_pairs._dc == {}

    def test_keep_zero_separation_pair(self, surface_counter: LogicalCounter):
        pairs = Pairs()
        pairs.add((0, 0, 2), (1, 0, 2))
        error_count, new_pairs = surface_counter.count(pairs)
        assert error_count == 0
        assert new_pairs.as_set == {((0, 0, 0), (1, 0, 0))}