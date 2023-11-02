"""Module for constants."""

from enum import IntEnum

from scipy.stats import norm

RED, PASTEL_GREEN, GREEN, BLUE = '#ee8989', '#d3ffce', '#a0dc99', '#96d8ec'
DARK_GRAY, GRAY, LIGHT_GRAY = '#b6b6b6', '#c6cbd3', '#ecedf0'
V_WIDE, WIDE, WIDE_MEDIUM, MEDIUM, THIN = 5.0, 3.0, 2.0, 1.0, 0.5
DEFAULT, SMALL = 300, 50
DEFAULT_X_OFFSET = 0.2
N_WINDOWS = 10

STANDARD_ERROR_ALPHA: float = 2 * (1-norm.cdf(1)) # type: ignore
"""Significance level corresponding to 1 SE. Roughly 32%."""


class Growth(IntEnum):
    """Constants for growth values."""

    BURNT, UNGROWN, HALF, FULL = range(-1, 3)
    INCREMENT = 1

    def __iadd__(self, other: 'Growth'):
        return Growth(self.value + other.value)