from jaxqualin.qnmode import str_to_mode
import numpy as np

import pytest

def test_prograde_mode_value(test_waveform_tuple, omega220r_test, omega220i_test):
    _, Mf, af = test_waveform_tuple
    mode220_test = str_to_mode('2.2.0', Mf, af)
    assert np.isclose(float(mode220_test.omegar), omega220r_test)
    assert np.isclose(float(mode220_test.omegai), omega220i_test)

def test_retrograde_mode_value(test_waveform_tuple, omegan220r_test, omegan220i_test):
    _, Mf, af = test_waveform_tuple
    moden220_test = str_to_mode('-2.2.0', Mf, af)
    assert np.isclose(float(moden220_test.omegar), omegan220r_test)
    assert np.isclose(float(moden220_test.omegai), omegan220i_test)

class TestRetroModes:

    @pytest.fixture(autouse=True)
    def set_omegas(self, test_waveform_tuple):
        _, Mf, af = test_waveform_tuple

        self.mode220 = str_to_mode('2.2.0', Mf, af)
        self.moden220 = str_to_mode('-2.2.0', Mf, af)
        self.mode2n20 = str_to_mode('2.-2.0', Mf, af)
        self.moden2n20 = str_to_mode('-2.-2.0', Mf, af)

        self.mode220_na = str_to_mode('2.2.0', Mf, -af)
        self.moden220_na = str_to_mode('-2.2.0', Mf, -af)
        self.mode2n20_na = str_to_mode('2.-2.0', Mf, -af)
        self.moden2n20_na = str_to_mode('-2.-2.0', Mf, -af)

        self.mode220_rem = str_to_mode('2.2.0', Mf, af, retro_def_orbit = False)
        self.moden220_rem = str_to_mode('-2.2.0', Mf, af, retro_def_orbit = False)
        self.mode2n20_rem = str_to_mode('2.-2.0', Mf, af, retro_def_orbit = False)
        self.moden2n20_rem = str_to_mode('-2.-2.0', Mf, af, retro_def_orbit = False)

        self.mode220_na_rem = str_to_mode('2.2.0', Mf, -af, retro_def_orbit = False)
        self.moden220_na_rem = str_to_mode('-2.2.0', Mf, -af, retro_def_orbit = False)
        self.mode2n20_na_rem = str_to_mode('2.-2.0', Mf, -af, retro_def_orbit = False)
        self.moden2n20_na_rem = str_to_mode('-2.-2.0', Mf, -af, retro_def_orbit = False)

    def test_omega_nm(self):
        assert np.isclose(float(self.mode2n20.omegar), -float(self.mode220.omegar))
        assert np.isclose(float(self.mode2n20.omegai), float(self.mode220.omegai))
        assert np.isclose(float(self.moden2n20.omegar), -float(self.moden220.omegar))
        assert np.isclose(float(self.moden2n20.omegai), float(self.moden220.omegai))

    def test_omega_na(self):    
        assert np.isclose(float(self.mode220_na.omegar), -float(self.moden220.omegar))
        assert np.isclose(float(self.mode220_na.omegai), float(self.moden220.omegai))
        assert np.isclose(float(self.moden220_na.omegar), -float(self.mode220.omegar))
        assert np.isclose(float(self.moden220_na.omegai), float(self.mode220.omegai))

    def test_omega_nm_na(self):
        assert np.isclose(float(self.mode2n20_na.omegar), float(self.moden220.omegar))
        assert np.isclose(float(self.mode2n20_na.omegai), float(self.moden220.omegai))
        assert np.isclose(float(self.moden2n20_na.omegar), float(self.mode220.omegar))
        assert np.isclose(float(self.moden2n20_na.omegai), float(self.mode220.omegai))

    def test_omega_rem(self):
        assert np.isclose(float(self.mode220_rem.omegar), float(self.mode220.omegar))
        assert np.isclose(float(self.mode220_rem.omegai), float(self.mode220.omegai))
        assert np.isclose(float(self.moden220_rem.omegar), float(self.moden220.omegar))
        assert np.isclose(float(self.moden220_rem.omegai), float(self.moden220.omegai))

    def test_omega_nm_rem(self):
        assert np.isclose(float(self.mode2n20_rem.omegar), -float(self.mode220.omegar))
        assert np.isclose(float(self.mode2n20_rem.omegai), float(self.mode220.omegai))
        assert np.isclose(float(self.moden2n20_rem.omegar), -float(self.moden220.omegar))
        assert np.isclose(float(self.moden2n20_rem.omegai), float(self.moden220.omegai))

    def test_omega_na_rem(self):
        assert np.isclose(float(self.mode220_na_rem.omegar), -float(self.mode220.omegar))
        assert np.isclose(float(self.mode220_na_rem.omegai), float(self.mode220.omegai))
        assert np.isclose(float(self.moden220_na_rem.omegar), -float(self.moden220.omegar))
        assert np.isclose(float(self.moden220_na_rem.omegai), float(self.moden220.omegai))

    def test_omega_nm_na_rem(self):
        assert np.isclose(float(self.mode2n20_na_rem.omegar), float(self.mode220.omegar))
        assert np.isclose(float(self.mode2n20_na_rem.omegai), float(self.mode220.omegai))
        assert np.isclose(float(self.moden2n20_na_rem.omegar), float(self.moden220.omegar))
        assert np.isclose(float(self.moden2n20_na_rem.omegai), float(self.moden220.omegai))
