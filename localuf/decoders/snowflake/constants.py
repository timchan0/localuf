"""Constants for Snowflake decoder."""

from enum import IntEnum

RESET = -1
"""`cid` value for nodes to be unrooted."""

class Stage(IntEnum):
    """Constants for stage values."""

    INCREMENT = 1
    STAGE_COUNT = 3
    """Stage count."""
    DROP, GROW, MERGING = range(STAGE_COUNT)

    def __iadd__(self, other: 'Stage'):
        return Stage(self.value + other.value)

    def __imod__(self, other: 'Stage'):
        return Stage(self.value % other.value)

    def __str__(self):
        match self:
            case Stage.DROP:    return 'D'
            case Stage.GROW:    return 'G'
            case Stage.MERGING: return 'M'