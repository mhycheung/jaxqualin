from jaxqualin.qnmode import str_to_mode
import numpy as np

def test_mode_value(test_waveform_tuple, omega220r_test, omega220i_test):
    _, Mf, af, retro = test_waveform_tuple
    mode220_test = str_to_mode('2.2.0', Mf, af, retro = retro)
    assert np.isclose(float(mode220_test.omegar), omega220r_test)
    assert np.isclose(float(mode220_test.omegai), omega220i_test)
