from localuf.codes import Surface
from localuf.decoders.uf import UF


class TestSwimDistance:

    def test_Meister2024_example(self):
        """The example in [arXiv:2405.07433, Figure 2]."""
        decoder = UF(Surface(6, 'code capacity'))
        syndrome = {(2, 1), (3, 3)}
        decoder.decode(syndrome)
        assert decoder.swim_distance() == 1

    def test_syndrome7F(self, uf7F: UF, syndrome7F):
        uf7F.validate(syndrome7F)
        assert uf7F.swim_distance() == 2


class TestWeighCorrection:

    def test_no_noise_level(self, uf7F: UF, syndrome7F):
        uf7F.decode(syndrome7F)
        result = uf7F._weigh_correction()

        assert len(uf7F.correction) == 6
        assert result == len(uf7F.correction)

    def test_a_noise_level(self, uf7F: UF, syndrome7F):
        uf7F.decode(syndrome7F)
        result = uf7F._weigh_correction(noise_level=0.5)

        assert len(uf7F.correction) == 6
        weight = uf7F.CODE.NOISE.log_odds_of_no_flip(0.5)
        assert result == len(uf7F.correction) * weight