"""Module for constants."""

from enum import IntEnum

from scipy.stats import norm

RED, GREEN, BLUE = '#Ff6666', '#Bef4be', '#9cdaed'
DARK_GRAY, GRAY, LIGHT_GRAY = '#b6b6b6', '#c6cbd3', '#ecedf0'
V_WIDE, WIDE, WIDE_MEDIUM, MEDIUM, MEDIUM_THIN, THIN = 5.0, 3.0, 2.0, 1.5, 1.0, 0.5
DEFAULT, SMALL = 300, 50
DEFAULT_X_OFFSET, STREAM_X_OFFSET = 0.2, 0.25

STANDARD_ERROR_ALPHA: float = 2 * (1-norm.cdf(1)) # type: ignore
"""Significance level corresponding to 1 SE. Roughly 32%."""


class Growth(IntEnum):
    """Constants for growth values."""

    BURNT, UNGROWN, HALF, FULL = range(-1, 3)
    INCREMENT = 1

    def __iadd__(self, other: 'Growth'):
        return Growth(self.value + other.value)
    
    @property
    def as_float(self) -> float:
        """Convert growth value to float. Burnt takes the value of 1."""
        if self is Growth.BURNT:
            return 1
        return self.value/2