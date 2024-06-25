import pytest
from localuf import Repetition


@pytest.fixture
def forward3F():
    rp = Repetition(
        3,
        noise='phenomenological',
        scheme='forward',
    )
    return rp.SCHEME