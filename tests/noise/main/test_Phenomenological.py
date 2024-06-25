import pytest


from localuf import Surface, Repetition
from localuf._schemes import _Streaming


class TestEdges:


    def test_batch(self, sf3T: Surface):
        assert sf3T.NOISE.EDGES == sf3T.EDGES # type: ignore


    @pytest.mark.parametrize("rp", ["rp3_forward", "rp3_frugal"])
    def test_nonbatch(self, rp: Repetition, request):
        rp = request.getfixturevalue(rp)
        scheme: _Streaming = rp.SCHEME # type: ignore
        buffer_height = scheme.WINDOW_HEIGHT - scheme._COMMIT_HEIGHT
        _, fresh_edges = rp._phenomenological_edges(
            scheme._COMMIT_HEIGHT,
            True,
            t_start=buffer_height,
        )
        assert rp.NOISE.EDGES == fresh_edges # type: ignore