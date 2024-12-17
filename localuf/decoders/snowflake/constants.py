"""Constants for Snowflake decoder."""

from enum import IntEnum

RESET = -1
"""`cid` value for nodes to be unrooted."""

class Stage(IntEnum):
    """Constants for stage values."""

    INCREMENT = 1
    STAGE_COUNT = 5
    """Stage count."""
    DROP, GROW_WHOLE, MERGING_WHOLE, GROW_HALF, MERGING_HALF = range(STAGE_COUNT)

    def __iadd__(self, other: 'Stage'):
        return Stage(self.value + other.value)

    def __imod__(self, other: 'Stage'):
        return Stage(self.value % other.value)

    def __str__(self):
        match self:
            case Stage.DROP:          return 'D'
            case Stage.GROW_WHOLE:    return 'G'
            case Stage.MERGING_WHOLE: return 'M'
            case Stage.GROW_HALF:     return 'g'
            case Stage.MERGING_HALF:  return 'm'