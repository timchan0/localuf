"""Constants for local Union--Find decoder."""

from enum import IntEnum

BUSY_SIGNAL_SYMBOLS = {
    True: 'b',
    False: ' ',
}
"""How drawer should label a node with(out) a busy signal."""
ACTIVE_SIGNAL_SYMBOLS = {
    True: 'A',
    False: ' ',
}
"""How drawer should label a node with(out) an active signal."""


class Stage(IntEnum):
    """Constants for stage values."""

    INCREMENT = 1
    SV_STAGE_COUNT = 4
    """Syndrome validation stage count."""
    BP_STAGE_COUNT = 3
    """Burning & peeling stage count."""
    GROWING, MERGING, PRESYNCING, SYNCING = range(SV_STAGE_COUNT)
    BURNING, PEELING, DONE = range(SV_STAGE_COUNT, SV_STAGE_COUNT+BP_STAGE_COUNT)

    def __iadd__(self, other: 'Stage'):
        return Stage(self.value + other.value)

    def __imod__(self, other: 'Stage'):
        return Stage(self.value % other.value)

    def __str__(self):
        match self:
            case Stage.GROWING:    return 'G'
            case Stage.MERGING:    return 'M'
            case Stage.PRESYNCING: return 'PS'
            case Stage.SYNCING:    return 'S'
            case Stage.BURNING:    return 'B'
            case Stage.PEELING:    return 'P'
            case Stage.DONE:       return 'D'