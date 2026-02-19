import itertools

import pytest

from localuf.codes import Surface
from localuf.decoders import MWPM


@pytest.fixture(
        name="sfCL_merged",
        params=itertools.product(range(3, 9, 2), range(2, 5)),
        ids=lambda x: f"d{x[0]} h{x[1]}",
)
def _sfCL_merged(request):
    d, h = request.param
    return Surface(
        d=d,
        noise='circuit-level',
        window_height=h,
        merge_equivalent_boundary_nodes=True,
    )


@pytest.mark.parametrize("i", (0, 1, 2))
@pytest.mark.parametrize("intended_correction_bit", (0, 1))
def test_complementary_gap(sfCL_merged: Surface, i: int, intended_correction_bit: int):
    """Test that the complementary gap is zero for all distances."""
    decoder = MWPM(sfCL_merged)
    correction_bit, gap = decoder.complementary_gap({
        (i, sfCL_merged.D//2-intended_correction_bit, 0)})
    assert correction_bit == intended_correction_bit
    assert gap == 1